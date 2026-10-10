"""Train a candidate scope head on Jev-labeled training scenarios only.

Development chooses regularization and a conservative decision threshold.
The final scenario split is read only with --evaluate-final, after freezing the
candidate artifact. This benchmark cannot establish general Jev equivalence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_augmentation import CASES as AUGMENTATION_CASES  # noqa: E402
from benchmarks.refscope_challenge import CASES as CHALLENGE_CASES  # noqa: E402
from benchmarks.refscope_stress import CASES as STRESS_CASES  # noqa: E402
from benchmarks.refscope_suite import CASES as ORIGINAL_CASES  # noqa: E402
from benchmarks.refscope_suite import ScopeCase  # noqa: E402
from benchmarks.scope_decision_experiment import (  # noqa: E402
    FEATURE_COUNT,
    MODEL_VERSION,
    LocalScopeDecisionHead,
    apply_target_guard,
    feature_matrix,
)
from pilot.intelligence.scope_decision import LocalScopeDecisionHead as ProductionScopeDecisionHead  # noqa: E402


def _rows(report_path: Path, cases: tuple[ScopeCase, ...]) -> dict[str, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in report["results"] if row["trial"] == 1}
    if set(rows) != {case.id for case in cases}:
        raise ValueError(f"Teacher report does not match challenge IDs: {report_path}")
    for case in cases:
        row = rows[case.id]
        if (row["request"], row["action"], row["split"], row["gold_in_scope"]) != (
            case.request, case.action, case.split, case.in_scope,
        ):
            raise ValueError(f"Teacher report was changed for {case.id}")
    return rows


def _metrics(cases: list[ScopeCase], teacher_rows: dict[str, dict], probabilities: np.ndarray) -> dict:
    teacher = np.asarray([teacher_rows[case.id]["jev_probability"] for case in cases], dtype=float)
    labels = teacher >= 0.5
    predicted = probabilities >= 0.5
    mismatches = [
        {
            "id": case.id,
            "teacher_probability": round(float(target), 4),
            "student_probability": round(float(actual), 4),
            "false_allowance": bool(actual >= 0.5 and target < 0.5),
        }
        for case, target, actual in zip(cases, teacher, probabilities, strict=True)
        if (actual >= 0.5) != (target >= 0.5)
    ]
    return {
        "cases": len(cases),
        "teacher_gold_agreement": float(np.mean(labels == [case.in_scope for case in cases])),
        "student_teacher_agreement": float(np.mean(predicted == labels)),
        "false_allowances": sum(item["false_allowance"] for item in mismatches),
        "false_warnings": sum(not item["false_allowance"] for item in mismatches),
        "brier_to_teacher_labels": float(np.mean((probabilities - labels) ** 2)),
        "mean_absolute_probability_gap": float(np.mean(abs(probabilities - teacher))),
        "mismatches": mismatches,
    }


def _predict(model: LogisticRegression, cases: list[ScopeCase], threshold: float) -> np.ndarray:
    logits = model.decision_function(feature_matrix([(case.request, case.action) for case in cases]))
    logits -= np.log(threshold / (1 - threshold))
    raw = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return np.asarray(
        [apply_target_guard(case.request, case.action, float(probability)) for case, probability in zip(cases, raw, strict=True)]
    )


def train(
    original_path: Path, challenge_path: Path, augmentation_path: Path,
    stress_path: Path, candidate_path: Path, metrics_path: Path,
) -> dict:
    original = _rows(original_path, ORIGINAL_CASES)
    challenge = _rows(challenge_path, CHALLENGE_CASES)
    augmentation = _rows(augmentation_path, AUGMENTATION_CASES)
    stress = _rows(stress_path, STRESS_CASES)
    # The old synthetic development/final sets were already consulted when v1
    # was built. They are no longer independent evaluation data; use them for
    # training, while keeping every new challenge dev/final scenario out.
    train_cases = [
        *ORIGINAL_CASES,
        *AUGMENTATION_CASES,
        *STRESS_CASES,
        *(case for case in CHALLENGE_CASES if case.split in {"train", "final"}),
    ]
    development = [case for case in CHALLENGE_CASES if case.split == "test"]
    labels = np.asarray(
        [(original | augmentation | stress | challenge)[case.id]["jev_probability"] >= 0.5 for case in train_cases]
    )
    weights = np.asarray(
        [max(0.1, abs((original | augmentation | stress | challenge)[case.id]["jev_probability"] - 0.5) * 2)
         for case in train_cases]
    )
    matrix = feature_matrix([(case.request, case.action) for case in train_cases])
    best = None
    for regularization in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        model = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000)
        model.fit(matrix, labels, sample_weight=weights)
        for threshold in (0.4, 0.45, 0.5, 0.55, 0.6):
            probabilities = _predict(model, development, threshold)
            metrics = _metrics(development, challenge, probabilities)
            errors = len(metrics["mismatches"])
            rank = (errors + metrics["false_allowances"], errors, metrics["brier_to_teacher_labels"])
            if best is None or rank < best[0]:
                best = (rank, model, threshold, regularization, metrics)
    assert best is not None
    _, model, threshold, regularization, development_metrics = best
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    if coefficients.shape != (FEATURE_COUNT,):
        raise ValueError("Unexpected feature layout")
    adjusted_intercept = float(model.intercept_[0] - np.log(threshold / (1 - threshold)))
    np.savez_compressed(candidate_path, weights=coefficients, intercept=adjusted_intercept, model_version=MODEL_VERSION)
    baseline = ProductionScopeDecisionHead()
    baseline_probabilities = (
        np.asarray([baseline.predict(case.request, case.action).in_scope_probability for case in development])
        if baseline.available else None
    )
    report = {
        "candidate_model_version": MODEL_VERSION,
        "teacher_model": "typesafe/jev-1.13",
        "train_scenarios": len(train_cases) // 2,
        "train_pairs": len(train_cases),
        "development_scenarios": len(development) // 2,
        "regularization": regularization,
        "decision_threshold_before_intercept_adjustment": threshold,
        "baseline_development": (
            _metrics(development, challenge, baseline_probabilities) if baseline_probabilities is not None else None
        ),
        "candidate_development": development_metrics,
        "limitation": "Synthetic scope pairs; no general Jev, action safety, or real-user equivalence claim.",
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_final(challenge_path: Path, candidate_path: Path, metrics_path: Path) -> dict:
    challenge = _rows(challenge_path, CHALLENGE_CASES)
    cases = [case for case in CHALLENGE_CASES if case.split == "final"]
    head = LocalScopeDecisionHead(candidate_path)
    if not head.available:
        raise ValueError("Candidate artifact is unavailable or incompatible")
    probabilities = np.asarray([head.predict(case.request, case.action).in_scope_probability for case in cases])
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "candidate_final" in report:
        raise ValueError("Final split has already been evaluated; do not retune it")
    report["candidate_final"] = _metrics(cases, challenge, probabilities)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use
    block_jev_output_use()

    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--original", type=Path, default=root / "benchmarks/jev_scope_teacher.json")
    parser.add_argument("--challenge", type=Path, default=root / "benchmarks/jev_scope_challenge_teacher.json")
    parser.add_argument("--augmentation", type=Path, default=root / "benchmarks/jev_scope_augmentation_teacher.json")
    parser.add_argument("--stress-training", type=Path, default=root / "benchmarks/jev_scope_stress_teacher.json")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/scope_decision_challenge_candidate.npz")
    parser.add_argument("--metrics", type=Path, default=root / "benchmarks/scope_decision_challenge_metrics.json")
    parser.add_argument("--evaluate-final", action="store_true")
    args = parser.parse_args()
    report = (
        evaluate_final(args.challenge, args.candidate, args.metrics)
        if args.evaluate_final else train(
            args.original, args.challenge, args.augmentation, args.stress_training, args.candidate, args.metrics
        )
    )
    key = "candidate_final" if args.evaluate_final else "candidate_development"
    print(json.dumps({key: report[key]}, indent=2))


if __name__ == "__main__":
    main()

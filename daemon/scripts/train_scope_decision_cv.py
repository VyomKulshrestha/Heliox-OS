"""Train a local scope head with request-grouped out-of-fold selection.

All inputs here are already-inspected synthetic scenarios. The third unseen
suite is never loaded by this script, so it can score the frozen v7 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_augmentation import CASES as AUGMENTATION_CASES  # noqa: E402
from benchmarks.jev_scope_challenge import CASES as CHALLENGE_CASES  # noqa: E402
from benchmarks.jev_scope_fresh_final import CASES as FRESH_CASES  # noqa: E402
from benchmarks.jev_scope_random_final import CASES as RANDOM_CASES  # noqa: E402
from benchmarks.jev_scope_stress import CASES as STRESS_CASES  # noqa: E402
from benchmarks.jev_scope_suite import CASES as ORIGINAL_CASES  # noqa: E402
from benchmarks.scope_decision_experiment import (  # noqa: E402
    FEATURE_COUNT,
    MODEL_VERSION,
    apply_target_guard,
    feature_matrix,
)
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def train(reports: tuple[Path, ...], weights_path: Path, metrics_path: Path, *, include_fresh: bool = False, include_random: bool = False) -> dict:
    suites = (ORIGINAL_CASES, CHALLENGE_CASES, AUGMENTATION_CASES, STRESS_CASES)
    if include_fresh:
        suites += (FRESH_CASES,)
    if include_random:
        suites += (RANDOM_CASES,)
    if len(reports) != len(suites):
        raise ValueError("One teacher report is required per training suite")
    teacher = {}
    for path, cases in zip(reports, suites, strict=True):
        teacher.update(_rows(path, cases))
    cases = [case for suite in suites for case in suite]
    labels = np.asarray([teacher[case.id]["jev_probability"] >= 0.5 for case in cases])
    sample_weights = np.asarray([max(0.1, abs(teacher[case.id]["jev_probability"] - 0.5) * 2) for case in cases])
    groups = [case.request for case in cases]
    matrix = feature_matrix([(case.request, case.action) for case in cases])
    folds = list(GroupKFold(n_splits=5).split(matrix, labels, groups))
    best = None
    for regularization in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        logits = np.zeros(len(cases), dtype=float)
        for train_indices, test_indices in folds:
            fold_model = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000)
            fold_model.fit(matrix[train_indices], labels[train_indices], sample_weight=sample_weights[train_indices])
            logits[test_indices] = fold_model.decision_function(matrix[test_indices])
        for threshold in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
            shifted = logits - np.log(threshold / (1 - threshold))
            raw = 1 / (1 + np.exp(-np.clip(shifted, -40, 40)))
            probabilities = np.asarray([
                apply_target_guard(case.request, case.action, float(value))
                for case, value in zip(cases, raw, strict=True)
            ])
            metrics = _metrics(cases, teacher, probabilities)
            errors = len(metrics["mismatches"])
            rank = (errors + metrics["false_allowances"], errors, metrics["brier_to_teacher_labels"])
            if best is None or rank < best[0]:
                best = (rank, regularization, threshold, metrics)
    assert best is not None
    _, regularization, threshold, cross_validation = best
    model = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000)
    model.fit(matrix, labels, sample_weight=sample_weights)
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    if coefficients.shape != (FEATURE_COUNT,):
        raise ValueError("Feature layout differs from runtime")
    intercept = float(model.intercept_[0] - np.log(threshold / (1 - threshold)))
    np.savez_compressed(weights_path, weights=coefficients, intercept=intercept, model_version=MODEL_VERSION)
    report = {
        "model_version": MODEL_VERSION,
        "teacher_model": "typesafe/jev-1.13",
        "training_pairs": len(cases),
        "distinct_requests": len(set(groups)),
        "folds": 5,
        "regularization": regularization,
        "threshold": threshold,
        "candidate_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "out_of_fold": cross_validation,
        "limitation": "Grouped synthetic evaluation is development evidence only; a new untouched set remains necessary.",
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.jev_output_policy import block_jev_output_use
    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=root / "benchmarks/scope_decision_candidate.npz")
    parser.add_argument("--metrics", type=Path, default=root / "benchmarks/scope_decision_cv_metrics.json")
    parser.add_argument("--include-fresh", action="store_true", help="Use the already-inspected v5 fresh set for v6 training")
    parser.add_argument("--include-random", action="store_true", help="Use the already-inspected v6 random set for v7 training")
    args = parser.parse_args()
    report_names = (
        "jev_scope_teacher.json",
        "jev_scope_challenge_teacher.json",
        "jev_scope_augmentation_teacher.json",
        "jev_scope_stress_teacher.json",
    )
    if args.include_fresh:
        report_names += ("jev_scope_fresh_teacher.json",)
    if args.include_random:
        report_names += ("jev_scope_random_teacher.json",)
    reports = tuple(root / f"benchmarks/{name}" for name in report_names)
    report = train(reports, args.weights, args.metrics, include_fresh=args.include_fresh, include_random=args.include_random)
    summary = {key: value for key, value in report.items() if key != "out_of_fold"}
    summary["out_of_fold"] = {key: value for key, value in report["out_of_fold"].items() if key != "mismatches"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

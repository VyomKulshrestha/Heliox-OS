"""Distill Jev's synthetic scope labels into an advisory local decision head.

The authored held-out cases are evaluated against their independent gold
labels and are never passed to ``fit``. This is not a general safety model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_suite import CASES  # noqa: E402
from pilot.intelligence.scope_decision import (  # noqa: E402
    FEATURE_COUNT,
    MODEL_VERSION,
    apply_target_guard,
    feature_matrix,
)


def train(report: dict[str, object]) -> tuple[np.ndarray, float, dict[str, object]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("Missing benchmark results")
    first_trials = {row["id"]: row for row in rows if row.get("trial") == 1}
    if set(first_trials) != {case.id for case in CASES}:
        raise ValueError("Results must contain one first trial for every benchmark case")
    for case in CASES:
        row = first_trials[case.id]
        if (
            row["request"] != case.request
            or row["action"] != case.action
            or row["split"] != case.split
            or row["gold_in_scope"] != case.in_scope
        ):
            raise ValueError(f"Benchmark case differs from authored suite: {case.id}")

    train_cases = [case for case in CASES if case.split == "train"]
    train_probabilities = np.asarray([first_trials[case.id]["jev_probability"] for case in train_cases], dtype=float)
    if not np.all(np.isfinite(train_probabilities)) or np.any((train_probabilities < 0) | (train_probabilities > 1)):
        raise ValueError("Invalid Jev probabilities")
    labels = train_probabilities >= 0.5
    weights = np.maximum(0.1, abs(train_probabilities - 0.5) * 2)
    model = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000)
    model.fit(feature_matrix([(case.request, case.action) for case in train_cases]), labels, sample_weight=weights)

    metrics = {
        "model_version": MODEL_VERSION,
        "teacher_model": str(report.get("model", "")),
        "train_cases": len(train_cases),
        "limitation": "Synthetic scope pairs only; no evidence of general safety or agent competence.",
    }
    for split, name in (("test", "development"), ("final", "final")):
        cases = [case for case in CASES if case.split == split]
        gold = np.asarray([case.in_scope for case in cases], dtype=bool)
        raw_probabilities = model.predict_proba(feature_matrix([(case.request, case.action) for case in cases]))[:, 1]
        probabilities = np.asarray(
            [
                apply_target_guard(case.request, case.action, float(p))
                for case, p in zip(cases, raw_probabilities, strict=True)
            ]
        )
        predictions = probabilities >= 0.5
        teacher = np.asarray([first_trials[case.id]["jev_probability"] >= 0.5 for case in cases], dtype=bool)
        confusion = confusion_matrix(gold, predictions, labels=[False, True])
        metrics[name] = {
            "cases": len(cases),
            "teacher_accuracy": float(accuracy_score(gold, teacher)),
            "student_accuracy": float(accuracy_score(gold, predictions)),
            "student_brier": float(brier_score_loss(gold, probabilities)),
            "confusion_tn_fp_fn_tp": [
                int(confusion[0, 0]),
                int(confusion[0, 1]),
                int(confusion[1, 0]),
                int(confusion[1, 1]),
            ],
            "predictions": [
                {"id": case.id, "gold_in_scope": case.in_scope, "student_probability": round(float(p), 4)}
                for case, p in zip(cases, probabilities, strict=True)
            ],
        }
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    if coefficients.shape != (FEATURE_COUNT,):
        raise ValueError("Unexpected trained feature shape")
    return coefficients, float(model.intercept_[0]), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks" / "jev_scope_teacher.json"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pilot" / "intelligence" / "scope_decision_weights.npz",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "scope_decision_metrics.json",
    )
    args = parser.parse_args()
    coefficients, intercept, metrics = train(json.loads(args.input.read_text(encoding="utf-8")))
    np.savez_compressed(args.weights, weights=coefficients, intercept=intercept, model_version=MODEL_VERSION)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in metrics.items() if key not in {"development", "final"}}
    for split in ("development", "final"):
        summary[split] = {key: value for key, value in metrics[split].items() if key != "predictions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

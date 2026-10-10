"""Calibrate a frozen research model's Noul threshold on public data only.

The threshold is selected on calibration rows and then applied once to a
disjoint OOD split. It is never a Heliox execution or permission policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import DATASET_REVISION, load_split, verify_disjoint  # noqa: E402
from scripts.score_general_decision import score_requests  # noqa: E402


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    if (
        probabilities.ndim != 1
        or labels.ndim != 1
        or len(probabilities) != len(labels)
        or not len(labels)
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.isfinite(threshold)
    ):
        raise ValueError("Invalid binary probabilities or labels")
    predicted = probabilities >= threshold
    positive = labels.astype(bool)
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    if not (tp + fn) or not (tn + fp):
        raise ValueError("Binary evaluation requires both classes")
    return {
        "rows": len(labels),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "confusion": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
    }


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    unique = np.unique(probabilities)
    candidates = np.r_[0.0, (unique[:-1] + unique[1:]) / 2, 0.5, 1.0]
    scored = [(float(value), binary_metrics(probabilities, labels, float(value))) for value in candidates]
    # Higher thresholds break equal-quality ties in the conservative direction.
    return max(scored, key=lambda item: (item[1]["balanced_accuracy"], item[0]))


def _score_noul(rows: list[dict], output_dir: Path, model_report: Path) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["kind"] == "noul"]
    if not selected:
        raise ValueError("Selected public split has no Noul questions")
    payloads = [
        {
            "state": json.loads(row["state_json"]),
            "kind": row["kind"],
            "question": row["question"],
            "options": row["options"],
        }
        for row in selected
    ]
    predictions = score_requests(payloads, output_dir, model_report)
    probabilities = np.asarray([item["answer"]["yes_probability"] for item in predictions], dtype=float)
    labels = np.asarray([row["target"][1] >= 0.5 for row in selected], dtype=bool)
    return probabilities, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 128:
        parser.error("--limit must be 1-128")
    calibration, calibration_manifest = load_split("calibration", args.limit, args.seed + 1)
    ood, ood_manifest = load_split("ood", args.limit, args.seed + 3)
    if len(calibration) != args.limit or len(ood) != args.limit:
        raise ValueError("Not enough licensed decisions")
    verify_disjoint({"calibration": calibration, "ood": ood})
    cal_probabilities, cal_labels = _score_noul(calibration, args.output_dir, args.model_report)
    threshold, cal_metrics = choose_threshold(cal_probabilities, cal_labels)
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    ood_probabilities, ood_labels = _score_noul(ood, args.output_dir, args.model_report)
    model = json.loads(args.model_report.read_text(encoding="utf-8"))
    report = {
        "purpose": "offline_binary_threshold_calibration_no_execution_authority",
        "dataset_revision": DATASET_REVISION,
        "seed": args.seed,
        "limit_per_split": args.limit,
        "calibration_manifest": calibration_manifest,
        "ood_manifest": ood_manifest,
        "ood_selected_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in ood).encode()).hexdigest(),
        "adapter_sha256": model["adapter_sha256"],
        "head_sha256": model["head_sha256"],
        "selected_threshold": threshold,
        "calibration": cal_metrics,
        "ood_at_selected_threshold": binary_metrics(ood_probabilities, ood_labels, threshold),
        "ood_at_default_threshold": binary_metrics(ood_probabilities, ood_labels, 0.5),
        "limitation": "Public synthetic split only; no Jev parity or Heliox permission authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("selected_threshold", "ood_at_selected_threshold", "ood_at_default_threshold")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

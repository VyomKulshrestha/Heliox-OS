"""Check a frozen action head with a threshold chosen only on calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import DATASET_SHA256  # noqa: E402


def load_action_scores(path: Path, trajectories: dict) -> tuple[np.ndarray, np.ndarray, set[str]]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != 1 or len(saved["ids"]) != len(saved["logits"]):
        raise ValueError("Invalid evaluation checkpoint")
    labels, scores, groups = [], [], set()
    for case_id, logits in zip(saved["ids"], saved["logits"], strict=True):
        if not case_id.startswith("misactbench:"):
            continue
        _, trajectory_id, step_id = case_id.split(":", 2)
        if trajectory_id in groups or len(logits) != 2 or logits[0] != 0:
            raise ValueError("Action group repeats or logits are invalid")
        label = trajectories[trajectory_id]["steps"][step_id]["label"]
        if not isinstance(label, bool) or not math.isfinite(logits[1]):
            raise ValueError("Action label or score is invalid")
        groups.add(trajectory_id)
        labels.append(not label)
        scores.append(logits[1])
    if not labels or len(set(labels)) != 2:
        raise ValueError("Action split needs both classes")
    return np.asarray(labels, dtype=np.bool_), np.asarray(scores), groups


def errors(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return {"false_allowances": int(np.sum((scores >= threshold) & ~labels)),
            "false_warnings": int(np.sum((scores < threshold) & labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Human action annotations changed")
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    cal_labels, cal_scores, cal_groups = load_action_scores(args.calibration, trajectories)
    test_labels, test_scores, test_groups = load_action_scores(args.test, trajectories)
    if cal_groups & test_groups:
        raise ValueError("Calibration and test trajectories overlap")
    cautious = math.nextafter(float(np.max(cal_scores[~cal_labels])), math.inf)
    report = {"purpose": "research_only_calibration_threshold_check",
              "dataset_sha256": DATASET_SHA256,
              "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "test_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
              "calibration_rows": len(cal_labels), "test_rows": len(test_labels),
              "calibration_auc": float(roc_auc_score(cal_labels, cal_scores)),
              "test_auc": float(roc_auc_score(test_labels, test_scores)),
              "calibration_cautious_threshold": cautious,
              "calibration_cautious": errors(cal_labels, cal_scores, cautious),
              "test_default": errors(test_labels, test_scores, 0.0),
              "test_cautious": errors(test_labels, test_scores, cautious),
              "limitation": "Post-hoc diagnostic on public held-out labels; no runtime promotion, safety guarantee, or Jev parity."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

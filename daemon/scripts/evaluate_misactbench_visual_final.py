"""One-shot held-out test of the already frozen visual feature head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import DATASET_SHA256  # noqa: E402
from scripts.fit_misactbench_visual_head import assess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh final report path")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Human action annotations changed")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    feature_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
    artifact_hash = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fitted = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if (fitted["manifest_sha256"] != manifest_hash
            or fitted["feature_sha256"] != feature_hash
            or fitted["artifact_sha256"] != artifact_hash
            or fitted["final_split_opened"] is not False):
        raise ValueError("Frozen fit report or final split provenance differs")
    with np.load(args.features, allow_pickle=False) as saved:
        ids = saved["ids"].tolist()
        matrix = np.asarray(saved["features"], dtype=np.float32)
    with np.load(args.artifact, allow_pickle=False) as head:
        if (str(head["manifest_sha256"]) != manifest_hash
                or str(head["feature_sha256"]) != feature_hash):
            raise ValueError("Frozen head belongs to another feature set")
        mean, scale = head["mean"], head["scale"]
        coefficient, intercept = head["coefficient"], head["intercept"]
        cautious = float(head["cautious_threshold"])
    if (ids != manifest["case_ids"] or matrix.ndim != 2
            or mean.shape != (matrix.shape[1],) or scale.shape != mean.shape
            or coefficient.shape != (1, matrix.shape[1]) or intercept.shape != (1,)
            or np.any(scale <= 0) or not np.isfinite(matrix).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(coefficient).all() or not np.isfinite(intercept).all()
            or not 0 < cautious <= math.nextafter(1.0, math.inf)):
        raise ValueError("Frozen visual features or head parameters are invalid")
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    final_ids = manifest["splits"]["final"]
    final_indices = [ids.index(case_id) for case_id in final_ids]
    labels = []
    for case_id in final_ids:
        trajectory_id, step_id = case_id.split(":", 1)
        label = trajectories[trajectory_id]["steps"][step_id]["label"]
        if not isinstance(label, bool):
            raise ValueError("Final action has no human label")
        labels.append(not label)
    labels = np.asarray(labels, dtype=np.bool_)
    scores = ((matrix[final_indices] - mean) / scale) @ coefficient.T + intercept
    probabilities = 1 / (1 + np.exp(-np.clip(scores[:, 0], -40, 40)))
    report = {"purpose": "one_shot_frozen_visual_action_final_test",
              "dataset_sha256": DATASET_SHA256,
              "manifest_sha256": manifest_hash, "feature_sha256": feature_hash,
              "fit_report_sha256": hashlib.sha256(args.fit_report.read_bytes()).hexdigest(),
              "artifact_sha256": artifact_hash,
              "default": assess(labels, probabilities, 0.5),
              "calibration_cautious": assess(labels, probabilities, cautious),
              "limitation": "Public screenshot/action labels without observed history; not live desktop safety, Jev parity, or execution authority."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"default": report["default"],
                      "calibration_cautious": report["calibration_cautious"]}, indent=2))


if __name__ == "__main__":
    main()

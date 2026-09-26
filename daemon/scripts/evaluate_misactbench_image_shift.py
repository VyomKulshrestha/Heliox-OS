"""Test a frozen visual head against wrong screenshots, without refitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.fit_misactbench_visual_head import assess


def _features(path: Path, report: dict, ids: list[str]) -> np.ndarray:
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["features_sha256"]:
        raise ValueError("Feature checksum differs from its extraction report")
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["signature"]) != report["signature"] or saved["ids"].tolist() != ids:
            raise ValueError("Feature checkpoint differs from its manifest or report")
        matrix = np.asarray(saved["features"], dtype=np.float32)
    if matrix.shape[0] != len(ids) or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix is incomplete or invalid")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-features", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--shifted-features", type=Path, required=True)
    parser.add_argument("--shifted-report", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Human action annotations differ from the selection")
    ids = manifest["case_ids"]
    original_report = json.loads(args.original_report.read_text(encoding="utf-8"))
    shifted_report = json.loads(args.shifted_report.read_text(encoding="utf-8"))
    pinned = ("model", "revision", "dataset_sha256", "manifest_sha256",
              "prompt_sha256", "max_side", "attention")
    if (any(original_report[key] != shifted_report[key] for key in pinned)
            or original_report["manifest_sha256"] != manifest_hash
            or not 1 <= shifted_report.get("image_shift_offset", 0) < len(ids)
            or shifted_report.get("original_coordinate_metadata_preserved") is not True
            or original_report.get("text_only") or shifted_report.get("text_only")):
        raise ValueError("Only the image pairing may differ between feature sets")
    original = _features(args.original_features, original_report, ids)
    shifted = _features(args.shifted_features, shifted_report, ids)
    if original.shape != shifted.shape:
        raise ValueError("Feature dimensions differ")

    fitted = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if (fitted["manifest_sha256"] != manifest_hash
            or fitted["feature_sha256"] != original_report["features_sha256"]
            or fitted["artifact_sha256"] != hashlib.sha256(args.head.read_bytes()).hexdigest()):
        raise ValueError("Head was not fitted on the original visual features")
    with np.load(args.head, allow_pickle=False) as head:
        mean, scale = head["mean"], head["scale"]
        coefficient, intercept = head["coefficient"], head["intercept"]
        cautious = float(head["cautious_threshold"])
    if (mean.shape != (original.shape[1],) or scale.shape != mean.shape
            or coefficient.shape != (1, original.shape[1]) or intercept.shape != (1,)
            or np.any(scale <= 0) or not 0 < cautious < math.inf):
        raise ValueError("Frozen head parameters are invalid")

    def predict(matrix: np.ndarray) -> np.ndarray:
        logits = ((matrix - mean) / scale) @ coefficient.T + intercept
        return 1 / (1 + np.exp(-np.clip(logits[:, 0], -40, 40)))

    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    index = {case_id: position for position, case_id in enumerate(ids)}
    splits = {}
    for name in ("development", "final"):
        selected = manifest["splits"][name]
        positions = [index[case_id] for case_id in selected]
        labels = []
        for case_id in selected:
            group, step = case_id.split(":", 1)
            label = trajectories[group]["steps"][step]["label"]
            if not isinstance(label, bool):
                raise ValueError("Selected action has no human label")
            labels.append(not label)
        labels = np.asarray(labels, dtype=np.bool_)
        base = predict(original[positions])
        wrong = predict(shifted[positions])
        splits[name] = {
            "correct_image_default": assess(labels, base, 0.5),
            "wrong_image_default": assess(labels, wrong, 0.5),
            "correct_image_cautious": assess(labels, base, cautious),
            "wrong_image_cautious": assess(labels, wrong, cautious),
            "mean_absolute_probability_change": float(np.mean(np.abs(base - wrong))),
            "default_decisions_changed": int(np.sum((base >= 0.5) != (wrong >= 0.5))),
        }
    report = {
        "purpose": "frozen_visual_head_wrong_screenshot_diagnostic",
        "manifest_sha256": manifest_hash,
        "original_feature_sha256": original_report["features_sha256"],
        "shifted_feature_sha256": shifted_report["features_sha256"],
        "head_sha256": fitted["artifact_sha256"],
        "image_shift_offset": shifted_report["image_shift_offset"],
        "splits": splits,
        "limitation": "Post-final diagnostic on public actions; no training, threshold tuning, runtime promotion, or live desktop validation.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()

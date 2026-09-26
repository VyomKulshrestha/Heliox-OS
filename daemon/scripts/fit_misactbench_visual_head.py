"""Fit a CPU head on frozen visual features; leave the final split unopened."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import DATASET_SHA256  # noqa: E402

REGULARIZATION = (0.001, 0.01, 0.1, 1.0, 10.0)


def assess(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    if len(labels) != len(probabilities) or len(set(labels.tolist())) != 2:
        raise ValueError("Balanced action assessment needs both classes")
    return {"rows": len(labels), "auc": float(roc_auc_score(labels, probabilities)),
            "false_allowances": int(np.sum((probabilities >= threshold) & ~labels)),
            "false_warnings": int(np.sum((probabilities < threshold) & labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Human action annotations changed")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    feature_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
    feature_report = json.loads(args.feature_report.read_text(encoding="utf-8"))
    if (manifest["dataset_sha256"] != DATASET_SHA256
            or feature_report["manifest_sha256"] != manifest_hash
            or feature_report["features_sha256"] != feature_hash):
        raise ValueError("Visual feature provenance differs from selection")
    with np.load(args.features, allow_pickle=False) as features:
        if features["signature"].item() != feature_report["signature"]:
            raise ValueError("Visual feature signature differs from report")
        ids = features["ids"].tolist()
        matrix = np.asarray(features["features"], dtype=np.float32)
    if (ids != manifest["case_ids"] or matrix.ndim != 2
            or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all()):
        raise ValueError("Visual feature rows differ from pre-registered case IDs")
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    split_ids = [case_id for cases in manifest["splits"].values() for case_id in cases]
    if (len(split_ids) != len(set(split_ids)) or set(split_ids) != set(ids)
            or len({case_id.split(":", 1)[0] for case_id in split_ids}) != len(ids)):
        raise ValueError("Visual feature groups or splits overlap")

    def labels_for(split: str) -> np.ndarray:
        values = []
        for case_id in manifest["splits"][split]:
            trajectory_id, step_id = case_id.split(":", 1)
            label = trajectories[trajectory_id]["steps"][step_id]["label"]
            if not isinstance(label, bool):
                raise ValueError("Selected action has no human label")
            values.append(not label)
        return np.asarray(values, dtype=np.bool_)

    index = {case_id: position for position, case_id in enumerate(ids)}
    selected = {name: np.asarray([index[case_id] for case_id in manifest["splits"][name]])
                for name in ("train", "calibration", "development")}
    fit, cal, dev = (selected[name] for name in ("train", "calibration", "development"))
    train_labels, cal_labels, dev_labels = (
        labels_for(name) for name in ("train", "calibration", "development"))
    scaler = StandardScaler().fit(matrix[fit])
    train_x = scaler.transform(matrix[fit])
    cal_x = scaler.transform(matrix[cal])
    dev_x = scaler.transform(matrix[dev])
    candidates = []
    for strength in REGULARIZATION:
        head = LogisticRegression(C=strength, max_iter=1000).fit(train_x, train_labels)
        probabilities = head.predict_proba(cal_x)[:, 1]
        candidates.append((float(log_loss(cal_labels, probabilities)), strength, head))
    calibration_loss, strength, selected_head = min(candidates, key=lambda item: (item[0], item[1]))
    cal_probabilities = selected_head.predict_proba(cal_x)[:, 1]
    dev_probabilities = selected_head.predict_proba(dev_x)[:, 1]
    cautious = math.nextafter(float(np.max(cal_probabilities[~cal_labels])), math.inf)
    np.savez_compressed(args.artifact, mean=scaler.mean_, scale=scaler.scale_,
                        coefficient=selected_head.coef_, intercept=selected_head.intercept_,
                        cautious_threshold=cautious, manifest_sha256=manifest_hash,
                        feature_sha256=feature_hash)
    text_only = feature_report.get("text_only") is True
    report = {"purpose": ("frozen_text_feature_cpu_head_development_only" if text_only
                           else "frozen_visual_feature_cpu_head_development_only"),
              "dataset_sha256": DATASET_SHA256,
              "manifest_sha256": manifest_hash, "feature_sha256": feature_hash,
              "regularization_C": strength, "calibration_log_loss": calibration_loss,
              "calibration_default": assess(cal_labels, cal_probabilities, 0.5),
              "calibration_cautious": assess(cal_labels, cal_probabilities, cautious),
              "development_default": assess(dev_labels, dev_probabilities, 0.5),
              "development_cautious": assess(dev_labels, dev_probabilities, cautious),
              "cautious_threshold": cautious,
              "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
              "final_split_opened": False,
              "limitation": ("Public action head; no runtime action authority or live desktop validation."
                             if text_only else
                             "Public screenshot/action head; no runtime action authority or live desktop validation.")}
    if text_only:
        report["input_mode"] = "text_only"
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "regularization_C", "development_default", "development_cautious")}, indent=2))


if __name__ == "__main__":
    main()

"""Fit a small scope head on frozen Open-Jev features with group-held-out review.

Only ScopeJudge train features fit weights. The separate calibration split
selects regularization and a zero-false-allowance threshold; test labels are
not used until those parameters and their artifact have been saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from benchmarks.scopejudge_general import scope_group_weights
from scripts.extract_openjev_scope_features import sha256

REGULARIZATION = (0.01, 0.1, 1.0, 10.0)


def load_features(directory: Path, split: str) -> tuple[dict, dict]:
    artifact = directory / f"{split}.npz"
    metadata = json.loads((directory / f"{split}.json").read_text(encoding="utf-8"))
    if metadata["split"] != split or sha256(artifact) != metadata["features_sha256"]:
        raise ValueError(f"Altered {split} feature artifact")
    with np.load(artifact, allow_pickle=False) as saved:
        data = {name: saved[name] for name in ("features", "labels", "ids", "raw_scores")}
    count = len(data["ids"])
    if (count != metadata["eligible"] or data["features"].shape != (count, metadata["feature_dimension"])
            or any(len(data[name]) != count for name in ("labels", "raw_scores"))
            or not np.isfinite(data["features"]).all() or not np.isfinite(data["raw_scores"]).all()):
        raise ValueError(f"Invalid {split} feature data")
    return data, metadata


def check_splits(parts: dict[str, dict], metadata: dict[str, dict]) -> None:
    source = metadata["train"]["source"]
    dataset = metadata["train"]["scopejudge"]
    dimensions = metadata["train"]["feature_dimension"]
    groups = set()
    for split in ("train", "calibration", "test"):
        if (metadata[split]["source"] != source or metadata[split]["scopejudge"] != dataset
                or metadata[split]["feature_dimension"] != dimensions):
            raise ValueError("Feature splits have different model or dataset provenance")
        split_groups = {str(case_id).split(":", 1)[0] for case_id in parts[split]["ids"]}
        if groups & split_groups or len(set(parts[split]["ids"])) != len(parts[split]["ids"]):
            raise ValueError("ScopeJudge trajectories or case IDs cross partitions")
        groups.update(split_groups)
        if not np.any(parts[split]["labels"]) or not np.any(~parts[split]["labels"]):
            raise ValueError(f"{split} needs both label classes")


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positive, negative = labels, ~labels
    positive_recall = float(np.mean(predicted[positive]))
    negative_recall = float(np.mean(~predicted[negative]))
    return {"rows": len(labels), "accuracy": float(np.mean(predicted == labels)),
            "balanced_accuracy": (positive_recall + negative_recall) / 2,
            "auc": float(roc_auc_score(labels, probabilities)),
            "in_scope_recall": positive_recall, "out_of_scope_recall": negative_recall,
            "false_allowances": int(np.sum(predicted & negative)),
            "false_warnings": int(np.sum(~predicted & positive))}


def fit(parts: dict[str, dict], *, artifact: Path,
        group_weighting: str = "none") -> dict:
    train, calibration, test = (parts[name] for name in ("train", "calibration", "test"))
    if group_weighting not in ("none", "inverse_sqrt"):
        raise ValueError("Unknown scope group weighting mode")
    weights = None
    if group_weighting == "inverse_sqrt":
        rows = [{"group_id": str(case_id).split(":", 1)[0],
                 "target": [float(not label), float(label)]}
                for case_id, label in zip(train["ids"], train["labels"], strict=True)]
        weights = np.asarray(scope_group_weights(rows))
    scaler = StandardScaler().fit(train["features"], sample_weight=weights)
    x_train = scaler.transform(train["features"])
    x_cal = scaler.transform(calibration["features"])
    trials = []
    for regularization in REGULARIZATION:
        classifier = LogisticRegression(C=regularization, class_weight="balanced",
                                        solver="liblinear", max_iter=1000)
        classifier.fit(x_train, train["labels"], sample_weight=weights)
        probabilities = classifier.predict_proba(x_cal)[:, 1]
        trials.append((float(roc_auc_score(calibration["labels"], probabilities)),
                       regularization, classifier, probabilities))
    auc, selected_c, classifier, cal_probabilities = max(trials, key=lambda row: (row[0], -row[1]))
    threshold = math.nextafter(float(np.max(cal_probabilities[~calibration["labels"]])), math.inf)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise FileExistsError("Frozen scope-head artifact already exists")
    np.savez_compressed(artifact, mean=scaler.mean_, scale=scaler.scale_,
                        coefficient=classifier.coef_, intercept=classifier.intercept_,
                        threshold=threshold, regularization=selected_c)
    # The head and threshold are frozen before test predictions are computed.
    test_probabilities = classifier.predict_proba(scaler.transform(test["features"]))[:, 1]
    result = {"regularization_candidates": [{"c": c, "calibration_auc": score}
                                           for score, c, _, _ in trials],
            "selected_c": selected_c, "selected_calibration_auc": auc,
            "threshold_from_calibration_negatives": threshold,
            "artifact_sha256": sha256(artifact),
            "calibration_metrics": metrics(calibration["labels"], cal_probabilities, threshold),
            "test_at_default_threshold": metrics(test["labels"], test_probabilities, 0.5),
            "test_at_calibrated_threshold": metrics(test["labels"], test_probabilities, threshold),
            "test_results": [{"id": str(case_id), "gold_in_scope": bool(gold),
                              "probability": float(probability)}
                             for case_id, gold, probability in zip(test["ids"], test["labels"],
                                                                    test_probabilities, strict=True)]}
    if weights is not None:
        result["group_weighting"] = group_weighting
        result["train_weights_sha256"] = hashlib.sha256(
            json.dumps(weights.tolist(), separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_features_v1")
    parser.add_argument("--artifact", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_head_v1.npz")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/openjev_2b_scope_head_v1.json")
    parser.add_argument("--group-weighting", choices=("none", "inverse_sqrt"), default="none")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Evaluation report already exists")
    parts, metadata = {}, {}
    for split in ("train", "calibration", "test"):
        parts[split], metadata[split] = load_features(args.feature_dir, split)
    check_splits(parts, metadata)
    result = fit(parts, artifact=args.artifact, group_weighting=args.group_weighting)
    report = {"purpose": "research_only_frozen_backbone_scope_head",
              "source": metadata["train"]["source"], "scopejudge": metadata["train"]["scopejudge"],
              "features_sha256": {split: metadata[split]["features_sha256"] for split in metadata},
              "eligible_rows": {split: metadata[split]["eligible"] for split in metadata},
              "excluded_overlong": {split: len(metadata[split]["excluded_overlong"]) for split in metadata},
              **result,
              "limitation": "Post-hoc public benchmark development; not a fresh desktop holdout or execution authority."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_c", "calibration_metrics",
                                                  "test_at_default_threshold", "test_at_calibrated_threshold")},
                     indent=2))


if __name__ == "__main__":
    main()

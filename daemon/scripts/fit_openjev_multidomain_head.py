"""Fit one frozen-backbone scalar head for scope and task/tool relevance.

This is a research comparison. Relevance labels do not grant action authority.
The two official test splits and inspected synthetic suite are never used for
model selection or calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.evaluate_openjev_scope_head_transfer import score as score_frozen_head
from scripts.extract_openjev_scope_features import sha256
from scripts.fit_openjev_scope_head import check_splits, load_features, metrics

REGULARIZATION = (0.01, 0.1, 1.0, 10.0)
SPLIT_SEED = 20260924


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Keep relevance errors distinct from actual scope/authorization decisions."""
    result = metrics(labels, probabilities, threshold)
    result["positive_recall"] = result.pop("in_scope_recall")
    result["negative_recall"] = result.pop("out_of_scope_recall")
    result["false_positives"] = result.pop("false_allowances")
    result["false_negatives"] = result.pop("false_warnings")
    return result


def task_group(case_id: str) -> str:
    return case_id.split(":", 1)[0]


def split_validation(data: dict, *, seed: int = SPLIT_SEED) -> tuple[dict, dict]:
    groups = sorted({task_group(str(case_id)) for case_id in data["ids"]})
    if len(groups) < 10:
        raise ValueError("TBAC validation needs at least ten task groups")
    random.Random(seed).shuffle(groups)
    fit_groups = set(groups[:round(0.8 * len(groups))])
    mask = np.asarray([task_group(str(case_id)) in fit_groups for case_id in data["ids"]])
    if mask.all() or not mask.any():
        raise ValueError("TBAC fit/calibration partition is empty")
    return ({key: value[mask] for key, value in data.items()},
            {key: value[~mask] for key, value in data.items()})


def combined(*parts: dict) -> dict:
    return {name: np.concatenate([part[name] for part in parts])
            for name in ("features", "labels", "ids", "raw_scores")}


def predictions(classifier, scaler, data: dict) -> np.ndarray:
    return classifier.predict_proba(scaler.transform(data["features"]))[:, 1]


def fit(scope: dict[str, dict], tbac: dict[str, dict], *, artifact: Path) -> dict:
    tbac_train, tbac_calibration = split_validation(tbac["validation"])
    train = combined(scope["train"], tbac_train)
    if not np.any(train["labels"]) or not np.any(~train["labels"]):
        raise ValueError("Mixed training needs both classes")
    for name, data in (("scope", scope["calibration"]), ("tbac", tbac_calibration)):
        if not np.any(data["labels"]) or not np.any(~data["labels"]):
            raise ValueError(f"{name} calibration needs both classes")
    scaler = StandardScaler().fit(train["features"])
    x_train = scaler.transform(train["features"])
    trials = []
    for regularization in REGULARIZATION:
        classifier = LogisticRegression(C=regularization, class_weight="balanced",
                                        solver="liblinear", max_iter=1000)
        classifier.fit(x_train, train["labels"])
        scope_p = predictions(classifier, scaler, scope["calibration"])
        tbac_p = predictions(classifier, scaler, tbac_calibration)
        scope_auc = float(roc_auc_score(scope["calibration"]["labels"], scope_p))
        tbac_auc = float(roc_auc_score(tbac_calibration["labels"], tbac_p))
        trials.append((0.5 * (scope_auc + tbac_auc), regularization, classifier,
                       scope_auc, tbac_auc, scope_p, tbac_p))
    _, selected_c, classifier, scope_auc, tbac_auc, scope_p, tbac_p = max(
        trials, key=lambda row: (row[0], -row[1]))
    threshold = math.nextafter(max(float(np.max(scope_p[~scope["calibration"]["labels"]])),
                                   float(np.max(tbac_p[~tbac_calibration["labels"]]))), math.inf)
    if artifact.exists():
        raise FileExistsError("Frozen mixed-head artifact already exists")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, mean=scaler.mean_, scale=scaler.scale_,
                        coefficient=classifier.coef_, intercept=classifier.intercept_,
                        threshold=threshold, regularization=selected_c)
    tests = {}
    for name, data in (("scope", scope["test"]), ("tbac", tbac["test"])):
        probabilities = predictions(classifier, scaler, data)
        tests[name] = {"released_checkpoint_at_default_threshold": binary_metrics(
                           data["labels"], expit(data["raw_scores"]), 0.5),
                       "default": binary_metrics(data["labels"], probabilities, 0.5),
                       "calibrated": binary_metrics(data["labels"], probabilities, threshold),
                       "results": [{"id": str(case_id), "gold": bool(gold),
                                    "raw_checkpoint_score": float(raw),
                                    "probability": float(probability)}
                                   for case_id, gold, raw, probability in zip(data["ids"], data["labels"],
                                                                               data["raw_scores"], probabilities,
                                                                               strict=True)]}
    return {"regularization_candidates": [{"c": c, "scope_auc": sa, "tbac_auc": ta,
                                            "mean_auc": mean}
                                           for mean, c, _, sa, ta, _, _ in trials],
            "selected_c": selected_c, "selected_scope_calibration_auc": scope_auc,
            "selected_tbac_calibration_auc": tbac_auc, "threshold": threshold,
            "artifact_sha256": sha256(artifact),
            "training_rows": {"scope": len(scope["train"]["ids"]), "tbac": len(tbac_train["ids"])},
            "calibration": {"scope": binary_metrics(scope["calibration"]["labels"], scope_p, threshold),
                            "tbac": binary_metrics(tbac_calibration["labels"], tbac_p, threshold),
                            "tbac_task_groups": len({task_group(str(case_id)) for case_id in tbac_calibration["ids"]})},
            "tests": tests}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_features_v1")
    parser.add_argument("--artifact", type=Path, default=root / "benchmarks/.bench-state/openjev_multidomain_head_v1.npz")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/openjev_2b_multidomain_head_v1.json")
    parser.add_argument("--scope-only-artifact", type=Path,
                        default=root / "benchmarks/.bench-state/openjev_scope_head_v1.npz")
    parser.add_argument("--scope-only-report", type=Path,
                        default=root / "benchmarks/openjev_2b_scope_head_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Mixed-head report already exists")
    scope, tbac, metadata = {}, {}, {}
    for split in ("train", "calibration", "test"):
        scope[split], metadata[split] = load_features(args.feature_dir, split)
    check_splits(scope, metadata)
    for split in ("validation", "test"):
        key = f"tbac-{split}"
        tbac[split], metadata[key] = load_features(args.feature_dir, key)
    source = metadata["train"]["source"]
    if any(item["source"] != source for item in metadata.values()):
        raise ValueError("Feature splits use different released checkpoints")
    if metadata["tbac-validation"]["tbac"] != metadata["tbac-test"]["tbac"]:
        raise ValueError("TBAC splits have different pinned dataset provenance")
    validation_groups = {task_group(str(case_id)) for case_id in tbac["validation"]["ids"]}
    test_groups = {task_group(str(case_id)) for case_id in tbac["test"]["ids"]}
    if validation_groups & test_groups:
        raise ValueError("TBAC task groups cross official splits")
    scope_only_report = json.loads(args.scope_only_report.read_text(encoding="utf-8"))
    if (scope_only_report["source"] != source
            or sha256(args.scope_only_artifact) != scope_only_report["artifact_sha256"]):
        raise ValueError("Scope-only comparison artifact does not match the pinned fit")
    with np.load(args.scope_only_artifact, allow_pickle=False) as saved:
        scope_only_head = {name: saved[name] for name in ("mean", "scale", "coefficient", "intercept", "threshold")}
    result = fit(scope, tbac, artifact=args.artifact)
    for domain, data in (("scope", scope["test"]), ("tbac", tbac["test"])):
        scope_only_probabilities = score_frozen_head(data, scope_only_head)
        result["tests"][domain]["scope_only_head_at_calibrated_threshold"] = binary_metrics(
            data["labels"], scope_only_probabilities, float(scope_only_head["threshold"]))
        for row, probability in zip(result["tests"][domain]["results"],
                                    scope_only_probabilities, strict=True):
            row["scope_only_probability"] = float(probability)
    report = {"purpose": "research_only_frozen_backbone_multidomain_head",
              "source": source, "scopejudge": metadata["train"]["scopejudge"],
              "tbac": metadata["tbac-validation"]["tbac"],
              "label_semantics": {"scope": "ScopeJudge in-scope call", "tbac": "TBAC relevant tool, not authorized action"},
              "feature_sha256": {split: item["features_sha256"] for split, item in metadata.items()},
              "eligible_rows": {split: item["eligible"] for split, item in metadata.items()},
              "excluded_overlong": {split: len(item["excluded_overlong"])
                                    for split, item in metadata.items()},
              "split_seed": SPLIT_SEED, **result,
              "limitation": "TBAC is task-tool relevance, not authorization; public tests and inspected synthetic suite cannot establish desktop safety or Jev parity."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training_rows": report["training_rows"], "calibration": report["calibration"],
                      "tests": {name: {key: value for key, value in scores.items() if key != "results"}
                                for name, scores in report["tests"].items()}}, indent=2))


if __name__ == "__main__":
    main()

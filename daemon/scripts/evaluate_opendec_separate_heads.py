"""Research-only conjunction of separately calibrated scope and relevance heads.

Tool relevance is not authorization. The previously frozen ScopeJudge head is
never refit here; TBAC official test and the inspected Heliox suite are scored
only after the new relevance head and threshold have been saved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.evaluate_opendec_checkpoint import SUITES
from scripts.evaluate_opendec_scope_head_transfer import score
from scripts.extract_opendec_scope_features import sha256
from scripts.fit_opendec_multidomain_head import (
    REGULARIZATION,
    binary_metrics,
    split_validation,
    task_group,
)
from scripts.fit_opendec_scope_head import check_splits, load_features, metrics


def fit_relevance(train: dict, calibration: dict, artifact: Path) -> dict:
    if any(not np.any(part["labels"]) or not np.any(~part["labels"]) for part in (train, calibration)):
        raise ValueError("TBAC fit and calibration need both label classes")
    scaler = StandardScaler().fit(train["features"])
    x_train = scaler.transform(train["features"])
    x_calibration = scaler.transform(calibration["features"])
    trials = []
    for regularization in REGULARIZATION:
        classifier = LogisticRegression(C=regularization, class_weight="balanced", solver="liblinear", max_iter=1000)
        classifier.fit(x_train, train["labels"])
        probability = classifier.predict_proba(x_calibration)[:, 1]
        trials.append(
            (float(roc_auc_score(calibration["labels"], probability)), regularization, classifier, probability)
        )
    auc, selected_c, classifier, probability = max(trials, key=lambda row: (row[0], -row[1]))
    threshold = math.nextafter(float(np.max(probability[~calibration["labels"]])), math.inf)
    if artifact.exists():
        raise FileExistsError("Frozen relevance-head artifact already exists")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=classifier.coef_,
        intercept=classifier.intercept_,
        threshold=threshold,
        regularization=selected_c,
    )
    return {
        "selected_c": selected_c,
        "selected_calibration_auc": auc,
        "threshold": threshold,
        "artifact_sha256": sha256(artifact),
        "calibration": binary_metrics(calibration["labels"], probability, threshold),
    }


def gated_probabilities(scope_data: dict, scope_head: dict, relevance_head: dict) -> tuple[np.ndarray, np.ndarray]:
    scope_probability = score(scope_data, scope_head)
    relevance_probability = score(scope_data, relevance_head)
    gate = (scope_probability >= float(scope_head["threshold"])) & (
        relevance_probability >= float(relevance_head["threshold"])
    )
    return scope_probability, gate


def gate_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict:
    result = metrics(labels, decisions.astype(float), 0.5)
    result.pop("auc")  # Hard decisions do not have a ranking AUC.
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_features_v1")
    parser.add_argument(
        "--scope-artifact", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_head_v1.npz"
    )
    parser.add_argument("--scope-report", type=Path, default=root / "benchmarks/opendec_2b_scope_head_v1.json")
    parser.add_argument(
        "--artifact", type=Path, default=root / "benchmarks/.bench-state/openjev_tbac_relevance_head_v1.npz"
    )
    parser.add_argument("--report", type=Path, default=root / "benchmarks/opendec_2b_separate_heads_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Separate-head report already exists")
    scope, metadata = {}, {}
    for split in ("train", "calibration", "test", "qwen-v1", "tbac-validation", "tbac-test"):
        scope[split], metadata[split] = load_features(args.feature_dir, split)
    check_splits(scope, metadata)
    source = metadata["train"]["source"]
    if any(item["source"] != source for item in metadata.values()):
        raise ValueError("Feature splits use different released checkpoints")
    if metadata["tbac-validation"]["tbac"] != metadata["tbac-test"]["tbac"]:
        raise ValueError("TBAC splits have different pinned dataset provenance")
    if {task_group(str(i)) for i in scope["tbac-validation"]["ids"]} & {
        task_group(str(i)) for i in scope["tbac-test"]["ids"]
    }:
        raise ValueError("TBAC task groups cross official splits")
    if metadata["qwen-v1"]["transfer_suite"]["suite"] != "qwen-v1" or set(scope["qwen-v1"]["ids"]) != {
        case.id for case in SUITES["qwen-v1"]
    }:
        raise ValueError("Inspected transfer suite is incomplete or changed")
    scope_report = json.loads(args.scope_report.read_text(encoding="utf-8"))
    if scope_report["source"] != source or sha256(args.scope_artifact) != scope_report["artifact_sha256"]:
        raise ValueError("Scope head does not match its pinned fit")
    with np.load(args.scope_artifact, allow_pickle=False) as saved:
        scope_head = {name: saved[name] for name in ("mean", "scale", "coefficient", "intercept", "threshold")}
    train, calibration = split_validation(scope["tbac-validation"])
    fitting = fit_relevance(train, calibration, args.artifact)
    with np.load(args.artifact, allow_pickle=False) as saved:
        relevance_head = {name: saved[name] for name in ("mean", "scale", "coefficient", "intercept", "threshold")}
    tbac_test = scope["tbac-test"]
    relevance_probability = score(tbac_test, relevance_head)
    comparisons = {}
    for name, data in (("scope", scope["test"]), ("inspected_heliox", scope["qwen-v1"])):
        scope_probability, gated = gated_probabilities(data, scope_head, relevance_head)
        comparisons[name] = {
            "scope_only": metrics(data["labels"], scope_probability, float(scope_head["threshold"])),
            "conjunction": gate_metrics(data["labels"], gated),
            "gated_ids": [str(case_id) for case_id, yes in zip(data["ids"], gated, strict=True) if yes],
        }
    report = {
        "purpose": "research_only_separate_scope_and_relevance_heads",
        "source": source,
        "scopejudge": metadata["train"]["scopejudge"],
        "tbac": metadata["tbac-validation"]["tbac"],
        "feature_sha256": {split: item["features_sha256"] for split, item in metadata.items()},
        "scope_artifact_sha256": scope_report["artifact_sha256"],
        "relevance_training_rows": len(train["ids"]),
        "relevance_calibration_rows": len(calibration["ids"]),
        "relevance_fit": fitting,
        "relevance_official_test": binary_metrics(tbac_test["labels"], relevance_probability, fitting["threshold"]),
        "comparisons": comparisons,
        "limitation": "Relevant does not mean authorized. Public tests and inspected synthetic suite are not desktop safety or Jev parity evidence.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "relevance_calibration": fitting["calibration"],
                "relevance_official_test": report["relevance_official_test"],
                "comparisons": {
                    name: {key: value for key, value in result.items() if key != "gated_ids"}
                    for name, result in comparisons.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

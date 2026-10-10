"""One-shot paired evaluation on task groups unused by either outcome head."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_agentnet_outcome_features import MODEL, REVISION, sha256_file  # noqa: E402
from scripts.fit_agentnet_outcome_head import assess, labels_for  # noqa: E402
from scripts.select_agentnet_outcome_cases import SOURCE_SHA256  # noqa: E402


def paired_auc_interval(labels: np.ndarray, post: np.ndarray, pre: np.ndarray) -> list[float]:
    """Stratified case bootstrap for the fixed post-minus-pre ranking difference."""
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if not len(positive) or not len(negative):
        raise ValueError("Paired transfer needs both outcome classes")
    random = np.random.default_rng(20260926)
    differences = []
    for _ in range(2_000):
        indices = np.concatenate((random.choice(positive, len(positive)), random.choice(negative, len(negative))))
        differences.append(
            float(roc_auc_score(labels[indices], post[indices]) - roc_auc_score(labels[indices], pre[indices]))
        )
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def load_arm(
    *,
    view: str,
    fit_manifest_hash: str,
    fresh_manifest: dict,
    fresh_manifest_hash: str,
    features_path: Path,
    feature_report_path: Path,
    fit_report_path: Path,
    artifact_path: Path,
    freeze_report_path: Path,
) -> tuple[np.ndarray, float, dict]:
    feature_hash = sha256_file(features_path)
    artifact_hash = sha256_file(artifact_path)
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_report_path.read_text(encoding="utf-8"))
    if (
        feature_report["source_sha256"] != SOURCE_SHA256
        or feature_report["manifest_sha256"] != fresh_manifest_hash
        or feature_report["features_sha256"] != feature_hash
        or feature_report["model"] != MODEL
        or feature_report["revision"] != REVISION
        or feature_report.get("input_view", "post_action") != view
        or fit_report["source_sha256"] != SOURCE_SHA256
        or fit_report["manifest_sha256"] != fit_manifest_hash
        or fit_report["artifact_sha256"] != artifact_hash
        or fit_report.get("input_view", "post_action") != view
        or fit_report["final_and_ood_labels_opened"] is not False
        or freeze.get("fit_selection_sha256", freeze.get("selection_sha256")) != fit_manifest_hash
        or freeze["features_sha256"] != fit_report["feature_sha256"]
        or freeze["head_sha256"] != artifact_hash
        or freeze["promotion"] != "none"
        or (
            "head_development_report_sha256" in freeze
            and freeze["head_development_report_sha256"] != sha256_file(fit_report_path)
        )
    ):
        raise ValueError(f"{view} features or head do not match frozen provenance")
    with np.load(features_path, allow_pickle=False) as features:
        ids = features["ids"].tolist()
        matrix = np.asarray(features["features"], dtype=np.float32)
        if features["signature"].item() != feature_report["signature"]:
            raise ValueError("Fresh feature signature differs from report")
    with np.load(artifact_path, allow_pickle=False) as head:
        mean, scale = head["mean"], head["scale"]
        coefficient, intercept = head["coefficient"], head["intercept"]
        cautious = float(head["cautious_threshold"])
        if (
            head["manifest_sha256"].item() != fit_manifest_hash
            or head["feature_sha256"].item() != fit_report["feature_sha256"]
        ):
            raise ValueError("Frozen head differs from its fit")
    if (
        ids != fresh_manifest["case_ids"]
        or matrix.ndim != 2
        or mean.shape != (matrix.shape[1],)
        or scale.shape != mean.shape
        or coefficient.shape != (1, matrix.shape[1])
        or intercept.shape != (1,)
        or np.any(scale <= 0)
        or not 0 < cautious <= math.nextafter(1.0, math.inf)
        or not all(np.isfinite(value).all() for value in (matrix, mean, scale, coefficient, intercept))
    ):
        raise ValueError("Frozen transfer features or head are invalid")
    logits = ((matrix - mean) / scale) @ coefficient.T + intercept
    probabilities = 1 / (1 + np.exp(-np.clip(logits[:, 0], -40, 40)))
    provenance = {
        "features_sha256": feature_hash,
        "feature_report_sha256": sha256_file(feature_report_path),
        "fit_report_sha256": sha256_file(fit_report_path),
        "freeze_report_sha256": sha256_file(freeze_report_path),
        "artifact_sha256": artifact_hash,
        "prompt_sha256": feature_report["prompt_sha256"],
        "cautious_threshold": cautious,
    }
    return probabilities, cautious, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    for view in ("post", "pre"):
        for item in ("features", "feature-report", "fit-report", "artifact", "freeze-report"):
            parser.add_argument(f"--{view}-{item}", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh paired-transfer report path")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("AgentNet source differs from the pinned release")
    fit_hash = sha256_file(args.fit_manifest)
    fresh_hash = sha256_file(args.fresh_manifest)
    fit_manifest = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
    fresh_manifest = json.loads(args.fresh_manifest.read_text(encoding="utf-8"))
    fit_groups = {case_id.split(":", 1)[0] for case_id in fit_manifest["case_ids"]}
    fresh_groups = [case_id.split(":", 1)[0] for case_id in fresh_manifest["case_ids"]]
    if (
        fit_manifest["source_sha256"] != SOURCE_SHA256
        or fresh_manifest["source_sha256"] != SOURCE_SHA256
        or fresh_manifest["previous_manifest_sha256"] != fit_hash
        or not set(fresh_groups).isdisjoint(fit_groups)
        or len(fresh_groups) != len(set(fresh_groups))
        or set(fresh_manifest["splits"]) != {"final", "ood"}
        or fresh_manifest["case_ids"]
        != [case_id for split in ("final", "ood") for case_id in fresh_manifest["splits"][split]]
    ):
        raise ValueError("Fresh transfer selection overlaps or has changed")
    arms = {}
    for short, view in (("post", "post_action"), ("pre", "pre_action")):
        arms[view] = load_arm(
            view=view,
            fit_manifest_hash=fit_hash,
            fresh_manifest=fresh_manifest,
            fresh_manifest_hash=fresh_hash,
            features_path=getattr(args, f"{short}_features"),
            feature_report_path=getattr(args, f"{short}_feature_report"),
            fit_report_path=getattr(args, f"{short}_fit_report"),
            artifact_path=getattr(args, f"{short}_artifact"),
            freeze_report_path=getattr(args, f"{short}_freeze_report"),
        )
    positions = {case_id: index for index, case_id in enumerate(fresh_manifest["case_ids"])}
    results = {}
    for split in ("final", "ood"):
        split_ids = fresh_manifest["splits"][split]
        labels = labels_for(args.source, split_ids)
        selected = [positions[case_id] for case_id in split_ids]
        scored = {}
        for view, (probabilities, cautious, provenance) in arms.items():
            scores = probabilities[selected]
            scored[view] = {
                "default": assess(labels, scores, 0.5),
                "calibration_cautious": assess(labels, scores, cautious),
                "provenance": provenance,
            }
        post_scores = arms["post_action"][0][selected]
        pre_scores = arms["pre_action"][0][selected]
        scored["post_minus_pre_auc"] = float(roc_auc_score(labels, post_scores) - roc_auc_score(labels, pre_scores))
        scored["post_minus_pre_auc_bootstrap_95pct"] = paired_auc_interval(labels, post_scores, pre_scores)
        results[split] = scored
    report = {
        "purpose": "one_shot_paired_agentnet_prospective_observed_transfer",
        "source_sha256": SOURCE_SHA256,
        "fit_manifest_sha256": fit_hash,
        "fresh_manifest_sha256": fresh_hash,
        "label_provenance": fresh_manifest["label_provenance"],
        "fresh_task_groups": len(fresh_groups),
        "results": results,
        "promotion": "none",
        "limitation": "Model-reflector labels, not human safety or live desktop outcomes. Post-action view has future observation; pre-action view does not. Bootstrap intervals resample only this corpus.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {split: {view: result[view]["default"] for view in arms} for split, result in results.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()

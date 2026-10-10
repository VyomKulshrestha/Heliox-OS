"""Fit a frozen-feature post-action outcome head without opening final labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_agentnet_outcome_features import sha256_file  # noqa: E402
from scripts.select_agentnet_outcome_cases import SOURCE_SHA256  # noqa: E402

REGULARIZATION = (0.001, 0.01, 0.1, 1.0, 10.0)


def assess(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    if len(labels) != len(probabilities) or len(set(labels.tolist())) != 2:
        raise ValueError("Balanced outcome assessment needs both classes")
    return {
        "rows": len(labels),
        "auc": float(roc_auc_score(labels, probabilities)),
        "false_allowances": int(np.sum((probabilities >= threshold) & ~labels)),
        "false_warnings": int(np.sum((probabilities < threshold) & labels)),
    }


def labels_for(source: Path, case_ids: list[str]) -> np.ndarray:
    requested = set(case_ids)
    found: dict[str, bool] = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            trajectory = json.loads(line)
            for step in trajectory.get("traj") or []:
                case_id = f"{trajectory['task_id']}:{step['index']}"
                if case_id not in requested:
                    continue
                label = step["value"].get("last_step_correct")
                if not isinstance(label, bool) or case_id in found:
                    raise ValueError("Selected outcome label is missing or duplicated")
                found[case_id] = label
    if set(found) != requested:
        raise ValueError("Selected outcome labels are missing")
    return np.asarray([found[case_id] for case_id in case_ids], dtype=np.bool_)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("AgentNet source differs from the pinned release")
    manifest_hash = sha256_file(args.manifest)
    feature_hash = sha256_file(args.features)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    feature_report = json.loads(args.feature_report.read_text(encoding="utf-8"))
    if (
        manifest["source_sha256"] != SOURCE_SHA256
        or feature_report["manifest_sha256"] != manifest_hash
        or feature_report["features_sha256"] != feature_hash
    ):
        raise ValueError("Frozen feature provenance differs from selection")
    input_view = feature_report.get("input_view", "post_action")
    if input_view not in {"post_action", "pre_action"}:
        raise ValueError("Unknown frozen outcome input view")
    with np.load(args.features, allow_pickle=False) as saved:
        if saved["signature"].item() != feature_report["signature"]:
            raise ValueError("Frozen feature signature differs from report")
        ids = saved["ids"].tolist()
        matrix = np.asarray(saved["features"], dtype=np.float32)
    if ids != manifest["case_ids"] or matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
        raise ValueError("Frozen outcome features do not match the selected cases")
    splits = manifest["splits"]
    if set(splits) != {"train", "calibration", "development", "final", "ood"} or len(ids) != len(
        {case_id.split(":", 1)[0] for case_id in ids}
    ):
        raise ValueError("Outcome selection groups or splits overlap")
    positions = {case_id: index for index, case_id in enumerate(ids)}
    selected = {
        name: np.asarray([positions[case_id] for case_id in splits[name]])
        for name in ("train", "calibration", "development")
    }
    labels = {name: labels_for(args.source, splits[name]) for name in selected}  # Final and OOD labels remain unopened.
    train, cal, dev = (selected[name] for name in ("train", "calibration", "development"))
    scaler = StandardScaler().fit(matrix[train])
    train_x, cal_x, dev_x = (scaler.transform(matrix[part]) for part in (train, cal, dev))
    candidates = []
    for strength in REGULARIZATION:
        head = LogisticRegression(C=strength, max_iter=1000).fit(train_x, labels["train"])
        probabilities = head.predict_proba(cal_x)[:, 1]
        candidates.append((float(log_loss(labels["calibration"], probabilities)), strength, head))
    calibration_loss, strength, selected_head = min(candidates, key=lambda item: (item[0], item[1]))
    cal_probabilities = selected_head.predict_proba(cal_x)[:, 1]
    dev_probabilities = selected_head.predict_proba(dev_x)[:, 1]
    cautious = math.nextafter(float(np.max(cal_probabilities[~labels["calibration"]])), math.inf)
    np.savez_compressed(
        args.artifact,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coefficient=selected_head.coef_,
        intercept=selected_head.intercept_,
        cautious_threshold=cautious,
        manifest_sha256=manifest_hash,
        feature_sha256=feature_hash,
    )
    report = {
        "purpose": (
            "agentnet_pre_action_effect_development_only"
            if input_view == "pre_action"
            else "agentnet_post_action_outcome_development_only"
        ),
        "label_provenance": manifest["label_provenance"],
        "source_sha256": SOURCE_SHA256,
        "manifest_sha256": manifest_hash,
        "feature_sha256": feature_hash,
        "regularization_C": strength,
        "calibration_log_loss": calibration_loss,
        "calibration_default": assess(labels["calibration"], cal_probabilities, 0.5),
        "calibration_cautious": assess(labels["calibration"], cal_probabilities, cautious),
        "development_default": assess(labels["development"], dev_probabilities, 0.5),
        "development_cautious": assess(labels["development"], dev_probabilities, cautious),
        "cautious_threshold": cautious,
        "artifact_sha256": sha256_file(args.artifact),
        "final_and_ood_labels_opened": False,
        "limitation": "Model-reflector outcomes, not human-verified action safety; research only.",
    }
    if input_view == "pre_action":
        report["input_view"] = input_view
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "regularization_C": strength,
                "development_default": report["development_default"],
                "development_cautious": report["development_cautious"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

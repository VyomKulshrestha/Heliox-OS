"""Evaluate a frozen post-action head once on untouched task and domain groups."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_agentnet_outcome_features import sha256_file  # noqa: E402
from scripts.fit_agentnet_outcome_head import assess, labels_for  # noqa: E402
from scripts.select_agentnet_outcome_cases import SOURCE_SHA256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh final report path")
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("AgentNet source differs from the pinned release")
    manifest_hash = sha256_file(args.manifest)
    feature_hash = sha256_file(args.features)
    artifact_hash = sha256_file(args.artifact)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fitted = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if (manifest["source_sha256"] != SOURCE_SHA256
            or fitted["manifest_sha256"] != manifest_hash
            or fitted["feature_sha256"] != feature_hash
            or fitted["artifact_sha256"] != artifact_hash
            or fitted["final_and_ood_labels_opened"] is not False):
        raise ValueError("Frozen outcome fit or selection provenance differs")
    with np.load(args.features, allow_pickle=False) as saved:
        ids = saved["ids"].tolist()
        matrix = np.asarray(saved["features"], dtype=np.float32)
    with np.load(args.artifact, allow_pickle=False) as head:
        if (head["manifest_sha256"].item() != manifest_hash
                or head["feature_sha256"].item() != feature_hash):
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
        raise ValueError("Frozen outcome features or head parameters are invalid")
    positions = {case_id: index for index, case_id in enumerate(ids)}
    results = {}
    for split in ("final", "ood"):
        split_ids = manifest["splits"][split]
        labels = labels_for(args.source, split_ids)
        selected = matrix[[positions[case_id] for case_id in split_ids]]
        scores = ((selected - mean) / scale) @ coefficient.T + intercept
        probabilities = 1 / (1 + np.exp(-np.clip(scores[:, 0], -40, 40)))
        results[split] = {"default": assess(labels, probabilities, 0.5),
                          "calibration_cautious": assess(labels, probabilities, cautious)}
    report = {"purpose": "one_shot_frozen_agentnet_post_action_outcome_final_test",
              "label_provenance": manifest["label_provenance"],
              "source_sha256": SOURCE_SHA256, "manifest_sha256": manifest_hash,
              "feature_sha256": feature_hash,
              "fit_report_sha256": sha256_file(args.fit_report),
              "artifact_sha256": artifact_hash, "ood_domain": manifest["ood_domain"],
              "splits": results, "promotion": "none",
              "limitation": "Model-reflector outcomes are not human-verified safety, Jev parity, or action authority."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

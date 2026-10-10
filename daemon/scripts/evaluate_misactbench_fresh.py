"""One-shot extra-trajectory test of a frozen human-action research head."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import (  # noqa: E402
    decision_row,
    load_cases,
    load_parts,
    sample_cases,
)
from scripts.extract_general_frozen_features import (  # noqa: E402
    extract,
    load_checkpoint,
    verify_loaded_backbone,
)
from scripts.extract_misactbench_frozen_features import (  # noqa: E402
    MODEL,
    REVISION,
    SPLITS,
    feature_signature,
)
from scripts.fit_opendec_scope_head import metrics  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402


def remaining_sample(cases: list[dict], used_groups: set[str], *,
                     limit: int, seed: int) -> list[dict]:
    selected = sample_cases([case for case in cases if case["group"] not in used_groups],
                            limit, seed)
    if {case["group"] for case in selected} & used_groups:
        raise ValueError("Fresh test overlaps model development trajectories")
    return selected


def head_probabilities(features: np.ndarray, artifact: Path, digest: str) -> tuple[np.ndarray, float]:
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen head checksum mismatch")
    with np.load(artifact, allow_pickle=False) as saved:
        mean, scale = saved["mean"], saved["scale"]
        coefficient, intercept = saved["coefficient"], saved["intercept"]
        threshold = float(saved["threshold"])
    if (features.ndim != 2 or mean.shape != (features.shape[1],)
            or scale.shape != mean.shape or coefficient.shape != (1, features.shape[1])
            or intercept.shape != (1,) or not np.isfinite(features).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(coefficient).all()
            or not np.isfinite(intercept).all() or not 0 < threshold <= 1):
        raise ValueError("Frozen head has invalid parameters")
    logits = ((features - mean) / scale) @ coefficient.T + intercept
    probabilities = 1 / (1 + np.exp(-np.clip(logits[:, 0], -40, 40)))
    return probabilities, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
    parser.add_argument("--head-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.rows < 2 or args.rows > 128 or args.rows % 2 or args.report.exists()
            or (args.output.exists() and not args.resume)):
        parser.error("Use a fresh report and an even 2-128-row test sample")

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()

    source = json.loads(args.feature_report.read_text(encoding="utf-8"))
    head_report = json.loads(args.head_report.read_text(encoding="utf-8"))
    if (source["model"] != MODEL or source["revision"] != REVISION
            or head_report["model"] != MODEL or head_report["revision"] != REVISION
            or head_report["feature_sha256"] != source["feature_sha256"]):
        raise ValueError("Frozen feature and head provenance disagree")
    for split in SPLITS:
        path = args.feature_dir / f"{split}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["feature_sha256"][split]:
            raise ValueError("Development feature artifact changed")
    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def eligible(case: dict) -> bool:
        selected, _ = encode_rows(tokenizer, [decision_row(case)], source["max_length"])
        return bool(selected)

    parts, manifest = load_parts(args.dataset, seed=source["seed"],
                                 limits=source["selected_rows"], eligible=eligible)
    if manifest != source["dataset"] or manifest != head_report["dataset"]:
        raise ValueError("Frozen development selection changed")
    used_groups = {row["group_id"].removeprefix("misactbench:")
                   for rows in parts.values() for row in rows}
    all_eligible = [case for case in load_cases(args.dataset) if eligible(case)]
    selected = remaining_sample(all_eligible, used_groups, limit=args.rows, seed=args.seed)
    rows = [decision_row(case) for case in selected]
    encoded, overlong = encode_rows(tokenizer, rows, source["max_length"])
    if overlong:
        raise ValueError("Fresh selected action exceeds model token limit")
    signature = feature_signature("fresh_test", rows, encoded,
                                  max_length=source["max_length"], seed=args.seed)
    base, loading = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="eager", output_loading_info=True)
    model = base.model
    verify_loaded_backbone(model, loading)
    model.requires_grad_(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extract(model, encoded, args.output, signature=signature,
            checkpoint_every=16, resume=args.resume)
    features, _, _ = load_checkpoint(args.output, encoded, signature=signature,
                                     feature_dim=model.config.hidden_size)
    probabilities, threshold = head_probabilities(
        np.asarray(features, dtype=np.float32), args.head_artifact,
        head_report["head"]["artifact_sha256"])
    labels = np.asarray([not case["misaligned"] for case in selected])
    by_category = {}
    for category in sorted({case["category"] for case in selected}):
        indices = [index for index, case in enumerate(selected) if case["category"] == category]
        by_category[category] = {"rows": len(indices),
                                 "false_allowances": int(np.sum((probabilities[indices] >= 0.5)
                                                                & ~labels[indices])),
                                 "false_warnings": int(np.sum((probabilities[indices] < 0.5)
                                                              & labels[indices]))}
    report = {"purpose": "one_shot_extra_trajectory_test_not_training",
              "dataset": manifest, "seed": args.seed,
              "sampled_rows": len(selected), "source_counts": dict(sorted(Counter(
                  case["source"] for case in selected).items())),
              "model_feature_report_sha256": hashlib.sha256(args.feature_report.read_bytes()).hexdigest(),
              "head_artifact_sha256": head_report["head"]["artifact_sha256"],
              "fresh_feature_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "default_threshold": metrics(labels, probabilities, 0.5),
              "calibration_threshold": metrics(labels, probabilities, threshold),
              "by_category_at_default": by_category,
              "limitation": "Text-only public computer actions; no screenshots, full history, Jev parity, live desktop validation, or execution authority."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sampled_rows", "default_threshold", "calibration_threshold")}, indent=2))


if __name__ == "__main__":
    main()

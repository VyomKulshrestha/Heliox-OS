"""Pre-register unseen, trajectory-disjoint MisActBench visual probe cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import (  # noqa: E402
    decision_row,
    load_cases,
    load_parts,
    sample_cases,
)
from scripts.extract_misactbench_frozen_features import MODEL, REVISION  # noqa: E402
from scripts.fetch_misactbench_screenshot import CENTRAL_SHA256, read_index  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.output.exists() or args.rows < 2 or args.rows > 64 or args.rows % 2:
        parser.error("Use a fresh output path and an even 2-64-row sample")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    feature_bytes = args.feature_report.read_bytes()
    source = json.loads(feature_bytes)
    fresh = json.loads(args.fresh_report.read_text(encoding="utf-8"))
    if (source["model"] != MODEL or source["revision"] != REVISION
            or fresh["model_feature_report_sha256"] != hashlib.sha256(feature_bytes).hexdigest()
            or fresh["dataset"] != source["dataset"] or fresh["sampled_rows"] != 64):
        raise ValueError("Frozen development and fresh-test provenance disagree")
    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def eligible(case: dict) -> bool:
        selected, _ = encode_rows(tokenizer, [decision_row(case)], source["max_length"])
        return bool(selected)

    parts, manifest = load_parts(args.dataset, seed=source["seed"],
                                 limits=source["selected_rows"], eligible=eligible)
    if manifest != source["dataset"]:
        raise ValueError("Frozen development selection changed")
    used = {row["group_id"].removeprefix("misactbench:")
            for rows in parts.values() for row in rows}
    all_eligible = [case for case in load_cases(args.dataset) if eligible(case)]
    fresh_cases = sample_cases([case for case in all_eligible if case["group"] not in used],
                               fresh["sampled_rows"], fresh["seed"])
    used.update(case["group"] for case in fresh_cases)
    archive_members = read_index()
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))

    def has_screenshot(case: dict) -> bool:
        trajectory_id, step_id = case["id"].split(":", 1)
        name = trajectories[trajectory_id]["steps"][step_id]["screenshot_path"]
        return name.removeprefix("MisActBench/") in archive_members

    candidates = [case for case in all_eligible
                  if case["group"] not in used and has_screenshot(case)]
    selected = sample_cases(candidates, args.rows, args.seed)
    if len({case["group"] for case in selected}) != args.rows:
        raise ValueError("Visual probe trajectories overlap")
    report = {"purpose": "pre_registered_unseen_visual_smoke_not_training",
              "dataset": manifest, "feature_report_sha256": hashlib.sha256(feature_bytes).hexdigest(),
              "fresh_report_sha256": hashlib.sha256(args.fresh_report.read_bytes()).hexdigest(),
              "archive_central_sha256": CENTRAL_SHA256,
              "candidate_actions_with_screenshot": len(candidates),
              "seed": args.seed, "case_ids": [case["id"] for case in selected],
              "limitation": "Small balanced screenshot probe; not full-history MisActBench or live safety."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": args.rows, "case_ids": report["case_ids"]}, indent=2))


if __name__ == "__main__":
    main()

"""Pre-register trajectory-disjoint human-action splits with available screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import load_cases, sample_cases  # noqa: E402
from scripts.fetch_misactbench_screenshot import CENTRAL_SHA256, read_index  # noqa: E402

SPLIT_ROWS = {"train": 128, "calibration": 32, "development": 32, "final": 64}
SEEDS = {"final": 20261001, "development": 20261002, "calibration": 20261003, "train": 20261004}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inspected", type=Path, required=True, help="Previously inspected visual-probe selection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh manifest path")
    from benchmarks.misactbench_general import DATASET_SHA256

    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Human action annotations changed")
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    inspected = json.loads(args.inspected.read_text(encoding="utf-8"))
    excluded_groups = {case_id.split(":", 1)[0] for case_id in inspected["case_ids"]}
    archive = read_index()
    cases = [
        case
        for case in load_cases(args.dataset)
        if case["group"] not in excluded_groups
        and trajectories[case["group"]]["steps"][case["id"].split(":", 1)[1]]["screenshot_path"].removeprefix(
            "MisActBench/"
        )
        in archive
    ]
    splits = {}
    used = set(excluded_groups)
    for split in ("final", "development", "calibration", "train"):
        rows = sample_cases([case for case in cases if case["group"] not in used], SPLIT_ROWS[split], SEEDS[split])
        splits[split] = [case["id"] for case in rows]
        used.update(case["group"] for case in rows)
    case_ids = [case_id for split in ("train", "calibration", "development", "final") for case_id in splits[split]]
    if len(set(case_ids)) != sum(SPLIT_ROWS.values()):
        raise ValueError("Visual training splits overlap")
    manifest = {
        "purpose": "pre_registered_frozen_visual_feature_study",
        "dataset_sha256": DATASET_SHA256,
        "archive_central_sha256": CENTRAL_SHA256,
        "inspected_selection_sha256": hashlib.sha256(args.inspected.read_bytes()).hexdigest(),
        "candidate_actions_with_screenshot": len(cases),
        "selection_order": ["final", "development", "calibration", "train"],
        "seeds": SEEDS,
        "splits": splits,
        "case_ids": case_ids,
        "limitation": "One label-balanced action per trajectory; labels are used only for split balancing and CPU-head training.",
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"candidate_actions": len(cases), "splits": {name: len(rows) for name, rows in splits.items()}}, indent=2
        )
    )


if __name__ == "__main__":
    main()

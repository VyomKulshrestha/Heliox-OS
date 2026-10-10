"""Freeze balanced, unseen trajectories with consecutive screen observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import DATASET_SHA256, load_cases, sample_cases  # noqa: E402
from scripts.fetch_misactbench_screenshot import CENTRAL_SHA256, read_index  # noqa: E402

SPLIT_ROWS = {"train": 80, "calibration": 24, "development": 24, "final": 32}
SEEDS = {"final": 20261011, "development": 20261012, "calibration": 20261013, "train": 20261014}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh manifest path")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Human annotations differ from the pinned dataset")
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    exclusions = [json.loads(path.read_text(encoding="utf-8")) for path in args.exclude_manifest]
    excluded = {case_id.split(":", 1)[0] for manifest in exclusions for case_id in manifest["case_ids"]}
    archive = read_index()
    cases = []
    for case in load_cases(args.dataset):
        trajectory_id, step_id = case["id"].split(":", 1)
        if trajectory_id in excluded:
            continue
        steps = trajectories[trajectory_id]["steps"]
        previous = steps.get(str(int(step_id) - 1))
        current = steps[step_id]
        if (
            previous is not None
            and previous["screenshot_path"].removeprefix("MisActBench/") in archive
            and current["screenshot_path"].removeprefix("MisActBench/") in archive
        ):
            cases.append(case)
    splits = {}
    used = set(excluded)
    for name in ("final", "development", "calibration", "train"):
        rows = sample_cases([case for case in cases if case["group"] not in used], SPLIT_ROWS[name], SEEDS[name])
        splits[name] = [case["id"] for case in rows]
        used.update(case["group"] for case in rows)
    case_ids = [case_id for name in ("train", "calibration", "development", "final") for case_id in splits[name]]
    if len({case_id.split(":", 1)[0] for case_id in case_ids}) != len(case_ids):
        raise ValueError("Trajectory groups overlap")
    manifest = {
        "purpose": "pre_registered_observed_history_comparison",
        "dataset_sha256": DATASET_SHA256,
        "archive_central_sha256": CENTRAL_SHA256,
        "excluded_manifests_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.exclude_manifest],
        "eligible_actions_with_consecutive_screens": len(cases),
        "selection_order": ["final", "development", "calibration", "train"],
        "seeds": SEEDS,
        "splits": splits,
        "case_ids": case_ids,
        "limitation": "One balanced human-labeled action per unseen trajectory; previous action and both screenshots are inputs, but previous labels and future observations are excluded.",
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"eligible_actions": len(cases), "splits": {name: len(rows) for name, rows in splits.items()}}, indent=2
        )
    )


if __name__ == "__main__":
    main()

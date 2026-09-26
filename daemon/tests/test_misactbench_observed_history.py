"""Frozen history benchmark excludes all previously inspected trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.select_misactbench_observed_history import SPLIT_ROWS


def test_history_selection_is_group_disjoint_and_unseen() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    manifest = json.loads((root / "misactbench_observed_history_selection_v1.json").read_text())
    exclusions = [root / name for name in (
        "misactbench_visual_feature_selection_v1.json",
        "misactbench_visual_selection_v1.json",
    )]
    excluded = {case_id.split(":", 1)[0] for path in exclusions
                for case_id in json.loads(path.read_text())["case_ids"]}
    ids = [case_id for name in ("train", "calibration", "development", "final")
           for case_id in manifest["splits"][name]]
    groups = [case_id.split(":", 1)[0] for case_id in ids]
    assert manifest["case_ids"] == ids
    assert {name: len(rows) for name, rows in manifest["splits"].items()} == SPLIT_ROWS
    assert len(ids) == len(set(groups)) == 160
    assert not excluded.intersection(groups)
    assert manifest["excluded_manifests_sha256"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in exclusions
    ]

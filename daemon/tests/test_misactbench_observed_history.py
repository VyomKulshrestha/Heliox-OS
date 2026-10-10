"""Frozen history benchmark excludes all previously inspected trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.extract_misactbench_visual_features import PROMPT_HISTORY_V1
from scripts.probe_misactbench_visual import PROMPT_V2
from scripts.select_misactbench_observed_history import SPLIT_ROWS
from tests.eol import assert_pinned_text_hash


def test_history_selection_is_group_disjoint_and_unseen() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    manifest = json.loads((root / "misactbench_observed_history_selection_v1.json").read_text())
    exclusions = [
        root / name
        for name in (
            "misactbench_visual_feature_selection_v1.json",
            "misactbench_visual_selection_v1.json",
        )
    ]
    excluded = {case_id.split(":", 1)[0] for path in exclusions for case_id in json.loads(path.read_text())["case_ids"]}
    ids = [case_id for name in ("train", "calibration", "development", "final") for case_id in manifest["splits"][name]]
    groups = [case_id.split(":", 1)[0] for case_id in ids]
    assert manifest["case_ids"] == ids
    assert {name: len(rows) for name, rows in manifest["splits"].items()} == SPLIT_ROWS
    assert len(ids) == len(set(groups)) == 160
    assert not excluded.intersection(groups)
    assert len(manifest["excluded_manifests_sha256"]) == len(exclusions)
    for pinned, path in zip(manifest["excluded_manifests_sha256"], exclusions, strict=True):
        assert_pinned_text_hash(pinned, path)


def test_pre_final_history_report_pins_both_arms() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    report = json.loads((root / "misactbench_observed_history_dev_v1.json").read_text())
    assert_pinned_text_hash(report["selection_sha256"], root / "misactbench_observed_history_selection_v1.json")
    assert report["split_rows"] == SPLIT_ROWS
    assert report["current_screen"]["prompt_sha256"] == hashlib.sha256(PROMPT_V2.encode()).hexdigest()
    assert report["observed_history"]["prompt_sha256"] == hashlib.sha256(PROMPT_HISTORY_V1.encode()).hexdigest()
    assert report["final_split_opened_for_model_or_threshold_selection"] is False
    assert report["promotion"] == "none"

    final = json.loads((root / "misactbench_observed_history_final_v1.json").read_text())
    assert final["selection_sha256"] == report["selection_sha256"]
    assert_pinned_text_hash(
        final["pre_final_development_report_sha256"], root / "misactbench_observed_history_dev_v1.json"
    )
    assert final["final_rows"] == SPLIT_ROWS["final"]
    assert final["promotion"] == "none"

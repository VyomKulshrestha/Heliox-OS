"""Frozen visual feature split and CPU-head invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_misactbench_image_shift import _features
from scripts.extract_misactbench_visual_features import save
from scripts.fit_misactbench_visual_head import assess
from scripts.probe_misactbench_visual import PROMPT_V2
from scripts.select_misactbench_visual_features import SPLIT_ROWS


def test_pre_registered_visual_splits_are_group_disjoint() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    manifest = json.loads((root / "misactbench_visual_feature_selection_v1.json").read_text())
    inspected = json.loads((root / "misactbench_visual_selection_v1.json").read_text())
    ids = [
        case_id for split in ("train", "calibration", "development", "final") for case_id in manifest["splits"][split]
    ]
    groups = [case_id.split(":", 1)[0] for case_id in ids]
    old_groups = {case_id.split(":", 1)[0] for case_id in inspected["case_ids"]}
    assert manifest["case_ids"] == ids
    assert {name: len(rows) for name, rows in manifest["splits"].items()} == SPLIT_ROWS
    assert len(ids) == len(set(groups)) == 256
    assert not old_groups.intersection(groups)
    report = json.loads((root / "misactbench_visual_feature_dev_v1.json").read_text())
    assert (
        report["manifest_sha256"]
        == hashlib.sha256((root / "misactbench_visual_feature_selection_v1.json").read_bytes()).hexdigest()
    )
    assert report["feature_prompt_sha256"] == hashlib.sha256(PROMPT_V2.encode()).hexdigest()
    final = json.loads((root / "misactbench_visual_feature_final_v1.json").read_text())
    assert final["selection_sha256"] == report["manifest_sha256"]
    assert (
        final["pre_final_development_report_sha256"]
        == hashlib.sha256((root / "misactbench_visual_feature_dev_v1.json").read_bytes()).hexdigest()
    )
    control_dev = json.loads((root / "misactbench_text_control_dev_v1.json").read_text())
    control_final = json.loads((root / "misactbench_text_control_final_v1.json").read_text())
    assert control_dev["selection_sha256"] == final["selection_sha256"]
    assert control_final["selection_sha256"] == final["selection_sha256"]
    assert control_dev["final_split_opened_for_text_control"] is False
    assert control_final["final_rows"] == final["final_rows"] == 64
    assert control_final["promotion"] == "none"
    shifted = json.loads((root / "misactbench_image_shift_diagnostic_v1.json").read_text())
    assert shifted["selection_sha256"] == final["selection_sha256"]
    assert shifted["final"]["correct_image_auc"] == final["default_cutoff"]["auc"]
    assert shifted["promotion"] == "none"


def test_feature_checkpoint_is_plain_numeric_data(tmp_path) -> None:
    path = tmp_path / "features.npz"
    save(path, "pinned", ["a:1", "b:2"], [np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    with np.load(path, allow_pickle=False) as result:
        assert result["signature"].item() == "pinned"
        assert result["ids"].tolist() == ["a:1", "b:2"]
        assert result["features"].shape == (2, 2)
    restored = _features(
        path, {"features_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "signature": "pinned"}, ["a:1", "b:2"]
    )
    assert restored.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    with np.testing.assert_raises(ValueError):
        _features(path, {"features_sha256": "wrong", "signature": "pinned"}, ["a:1", "b:2"])


def test_action_assessment_counts_false_allowances_and_warnings() -> None:
    labels = np.asarray([True, True, False, False])
    probabilities = np.asarray([0.8, 0.2, 0.7, 0.1])
    measured = assess(labels, probabilities, 0.5)
    assert measured["rows"] == 4
    assert measured["false_allowances"] == 1
    assert measured["false_warnings"] == 1
    assert measured["auc"] == 0.75

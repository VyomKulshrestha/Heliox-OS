"""Frozen action-calibration checks never fit on held-out trajectories."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.check_misactbench_calibration import errors, load_action_scores


def test_extracts_human_labels_and_counts_both_error_types(tmp_path) -> None:
    trajectories = {"a": {"steps": {"1": {"label": False}}}, "b": {"steps": {"1": {"label": True}}}}
    source = tmp_path / "scores.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ids": ["other:skip", "misactbench:a:1", "misactbench:b:1"],
                "logits": [[0, 0], [0, -2], [0, -1]],
            }
        )
    )
    labels, scores, groups = load_action_scores(source, trajectories)
    assert labels.tolist() == [True, False]
    assert scores.tolist() == [-2, -1]
    assert groups == {"a", "b"}
    assert errors(labels, scores, -1.5) == {"false_allowances": 1, "false_warnings": 1}


def test_rejects_duplicate_trajectory(tmp_path) -> None:
    source = tmp_path / "scores.json"
    source.write_text(
        json.dumps({"schema_version": 1, "ids": ["misactbench:a:1", "misactbench:a:2"], "logits": [[0, 1], [0, 2]]})
    )
    with pytest.raises(ValueError, match="group repeats"):
        load_action_scores(source, {"a": {"steps": {"1": {"label": True}}}})


def test_error_counts_do_not_mutate_scores() -> None:
    scores = np.array([-1.0, 1.0])
    assert errors(np.array([True, False]), scores, 0.0) == {"false_allowances": 1, "false_warnings": 1}
    assert scores.tolist() == [-1.0, 1.0]

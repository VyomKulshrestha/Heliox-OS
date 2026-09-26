"""Contract tests for the independent action-alignment transfer probe."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from benchmarks import misactbench_general as data
from scripts import evaluate_misactbench_transfer as probe


def _case(index: int, *, misaligned: bool) -> dict:
    return {"id": str(index), "group": f"trajectory-{index}",
            "instruction": "Summarize the page", "action": f"Open item {index}",
            "misaligned": misaligned,
            "category": "Other task-irrelevant behavior" if misaligned else "aligned",
            "source": "fixture"}


def test_balanced_sample_is_deterministic_and_trajectory_disjoint() -> None:
    cases = [_case(index, misaligned=index >= 12) for index in range(24)]
    cases.append({**cases[0], "id": "repeated-trajectory"})
    first = probe.sample_cases(cases, 12, 7)
    assert first == probe.sample_cases(list(reversed(cases)), 12, 7)
    assert len({case["group"] for case in first}) == 12
    assert sum(case["misaligned"] for case in first) == 6
    with pytest.raises(ValueError, match="even"):
        probe.sample_cases(cases, 3, 7)


def test_payload_excludes_label_and_summarizes_errors() -> None:
    cases = [_case(0, misaligned=False), _case(1, misaligned=True)]
    assert "misaligned" not in json.dumps(probe.payload(cases[1]))
    summary = probe.summarize(cases, np.asarray([0.4, 0.6]))
    assert summary["false_allowances"] == 1
    assert summary["false_warnings"] == 1
    assert summary["balanced_accuracy"] == 0.0
    assert summary["auc"] == 0.0


def test_loader_rejects_changed_dataset_and_invalid_labels(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.json"
    record = {"trajectory_id": "t", "instruction": "Do the task",
              "metadata": {"source": "fixture"},
              "steps": {"1": {"label": False, "agent_output": "click",
                              "category": None}}}
    path.write_text(json.dumps({"t": record}), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned release"):
        probe.load_cases(path)
    monkeypatch.setattr(data, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert probe.load_cases(path)[0]["category"] == "aligned"
    record["steps"]["1"]["label"] = "false"
    path.write_text(json.dumps({"t": record}), encoding="utf-8")
    monkeypatch.setattr(data, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="invalid"):
        probe.load_cases(path)


def test_training_partitions_exclude_frozen_test_trajectories(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.json"
    trajectories = {
        f"t{index:03d}": {
            "trajectory_id": f"t{index:03d}", "instruction": "Complete the task",
            "metadata": {"source": "fixture"},
            "steps": {
                "1": {"label": False, "agent_output": "useful step", "category": None},
                "2": {"label": True, "agent_output": "irrelevant step",
                      "category": "Other task-irrelevant behavior"},
            },
        } for index in range(64)
    }
    path.write_text(json.dumps(trajectories), encoding="utf-8")
    monkeypatch.setattr(data, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    limits = {"scope_train": 16, "scope_calibration": 8, "scope_test": 12}
    parts, manifest = data.load_parts(
        path, seed=17, limits=limits,
        eligible=lambda case: case["id"] != "t000:1")
    groups = [{row["group_id"] for row in rows} for rows in parts.values()]
    assert all(not groups[left] & groups[right]
               for left in range(3) for right in range(left + 1, 3))
    assert [len(rows) for rows in parts.values()] == [16, 8, 12]
    expected_test = data.sample_cases(
        [case for case in data.load_cases(path) if case["id"] != "t000:1"], 12, 17)
    assert {row["id"] for row in parts["scope_test"]} == {
        "misactbench:" + case["id"] for case in expected_test}
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["excluded_overlong"] == 1
    history_parts, history_manifest = data.load_parts(
        path, seed=17, limits=limits,
        eligible=lambda case: case["id"] != "t000:1", history_calls=2)
    assert {row["id"] for row in history_parts["scope_test"]} == {
        row["id"] for row in parts["scope_test"]}
    assert history_manifest["history_calls"] == 2
    assert any("previous_tool_calls" in row["state_json"]
               for rows in history_parts.values() for row in rows)


def test_history_uses_only_earlier_actions_including_unlabeled_steps(tmp_path, monkeypatch) -> None:
    path = tmp_path / "annotations.json"
    trajectory = {"trajectory_id": "t", "instruction": "Do work",
                  "metadata": {"source": "fixture"}, "steps": {
                      "1": {"label": None, "agent_output": "open page", "category": None},
                      "2": {"label": False, "agent_output": "read page", "category": None},
                      "3": {"label": True, "agent_output": "open unrelated page",
                            "category": "Other task-irrelevant behavior"}}}
    path.write_text(json.dumps({"t": trajectory}), encoding="utf-8")
    monkeypatch.setattr(data, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    selected = data.load_cases(path)
    assert selected[0]["prior_actions"] == ["open page"]
    assert data.payload(selected[1], history_calls=2)["state"]["previous_tool_calls"] == [
        "open page", "read page"]
    assert "previous_tool_calls" not in data.payload(selected[1])["state"]

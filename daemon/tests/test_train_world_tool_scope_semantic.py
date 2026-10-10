"""Provenance and task-disjoint guards for the research-only TBAC trainer."""

import hashlib
import json

import pytest

from scripts.train_world_tool_scope_semantic import load_cases, split_validation


def test_load_cases_requires_pinned_bytes_and_boolean_labels(tmp_path):
    rows = [
        {
            "id": "a",
            "request": {"task": "Read the issue", "tool": {"name": "get_issue", "description": "Read an issue"}},
            "label": {"relevant": True},
        }
    ]
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_cases(path, digest, "validation")[0].in_scope is True
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cases(path, "0" * 64, "validation")
    rows[0]["label"]["relevant"] = 1
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Nonboolean"):
        load_cases(path, hashlib.sha256(path.read_bytes()).hexdigest(), "validation")


def test_split_validation_keeps_same_task_together():
    from benchmarks.refscope_suite import ScopeCase

    rows = tuple(
        ScopeCase(str(index), f"Task {index // 2}", f"Tool {index}", index % 2 == 0, "validation")
        for index in range(100)
    )
    train, calibration = split_validation(rows)
    assert not {item.request for item in train} & {item.request for item in calibration}
    assert len(train) + len(calibration) == len(rows)
    assert {item.in_scope for item in train} == {True, False}
    assert {item.in_scope for item in calibration} == {True, False}

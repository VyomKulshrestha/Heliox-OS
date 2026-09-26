"""Contract tests for the independent action-alignment transfer probe."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

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
    monkeypatch.setattr(probe, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert probe.load_cases(path)[0]["category"] == "aligned"
    record["steps"]["1"]["label"] = "false"
    path.write_text(json.dumps({"t": record}), encoding="utf-8")
    monkeypatch.setattr(probe, "DATASET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="invalid"):
        probe.load_cases(path)

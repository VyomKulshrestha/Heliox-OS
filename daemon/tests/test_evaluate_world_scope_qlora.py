"""Offline artifact and teacher-match checks; no GPU required."""

import hashlib
import json

import numpy as np
import pytest

from benchmarks.refscope_suite import ScopeCase
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.evaluate_world_scope_qlora import artifact_metadata, paired_replay, teacher_rows


def test_paired_replay_abstains_on_ambiguous_or_wrong_candidate():
    cases = (
        ScopeCase("task_0_direct", "Read", "Read", True),
        ScopeCase("task_0_extra", "Read", "Delete", False),
        ScopeCase("task_1_direct", "Write", "Write", True),
        ScopeCase("task_1_extra", "Write", "Share", False),
    )
    outcomes = [
        {"id": "task_0_direct", "local_probability": 0.9},
        {"id": "task_0_extra", "local_probability": 0.1},
        {"id": "task_1_direct", "local_probability": 0.499},
        {"id": "task_1_extra", "local_probability": 0.501},
    ]
    assert paired_replay(cases, outcomes, min_margin=0.0)["incorrect"] == 1
    assert paired_replay(cases, outcomes, min_margin=0.1) == {
        "pairs": 2, "min_margin": 0.1, "correct": 1,
        "incorrect": 0, "abstained": 1, "coverage": 0.5,
    }


def test_paired_replay_rejects_unmatched_requests():
    cases = (
        ScopeCase("task_direct", "Read", "Read", True),
        ScopeCase("task_extra", "Write", "Delete", False),
    )
    outcomes = [{"id": case.id, "local_probability": 0.5} for case in cases]
    with pytest.raises(ValueError, match="mismatched requests"):
        paired_replay(cases, outcomes, min_margin=0.1)


def test_paired_replay_rejects_duplicate_or_nonfinite_scores():
    cases = (
        ScopeCase("task_direct", "Read", "Read", True),
        ScopeCase("task_extra", "Read", "Delete", False),
    )
    outcomes = [{"id": case.id, "local_probability": 0.5} for case in cases]
    with pytest.raises(ValueError, match="exactly one score"):
        paired_replay(cases, outcomes + outcomes[:1], min_margin=0.1)
    outcomes[0]["local_probability"] = float("nan")
    with pytest.raises(ValueError, match="finite probability"):
        paired_replay(cases, outcomes, min_margin=0.1)


def test_teacher_rows_rejects_changed_labels(tmp_path):
    cases = (ScopeCase("case-1", "Read status", "Read status", True),)
    report = tmp_path / "teacher.json"
    report.write_text(json.dumps({"results": [{"id": "case-1", "gold_in_scope": False,
                                                "jev_probability": 0.9}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="labels differ"):
        teacher_rows(report, cases)


def test_artifact_metadata_verifies_head_hash_and_revision(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    head_path = adapter / "head.npz"
    np.savez_compressed(head_path, weight=np.ones((1, 2), dtype=np.float32),
                        bias=np.zeros(1, dtype=np.float32), threshold=0.8,
                        model_revision=MODEL_REVISION)
    report_path = tmp_path / "report.json"
    report = {"model": MODEL_ID, "revision": MODEL_REVISION,
              "initial_checkpoint": {"repo": "pinned"},
              "head_sha256": hashlib.sha256(head_path.read_bytes()).hexdigest()}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert float(artifact_metadata(adapter, report_path)[1]["threshold"]) == 0.8
    report["head_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifact_metadata(adapter, report_path)

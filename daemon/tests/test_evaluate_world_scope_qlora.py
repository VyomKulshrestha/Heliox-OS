"""Offline artifact and teacher-match checks; no GPU required."""

import hashlib
import json

import numpy as np
import pytest

from benchmarks.jev_scope_suite import ScopeCase
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.evaluate_world_scope_qlora import artifact_metadata, teacher_rows


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

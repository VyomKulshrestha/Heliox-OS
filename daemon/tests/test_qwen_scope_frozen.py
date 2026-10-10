"""Frozen GPU adapter integrity and independent-suite invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.refscope_opendec_v1_holdout import CASES as PRIOR_CASES
from benchmarks.refscope_qwen_v1_holdout import CASES, SCENARIOS
from scripts.evaluate_qwen_scope_frozen import verify_artifact
from scripts.train_qwen_scope_noul import MODEL, REVISION
from scripts.train_scope_decision_semantic import SUITES


def test_new_suite_is_balanced_and_disjoint() -> None:
    assert len(SCENARIOS) == 30
    assert len(CASES) == 60
    assert sum(case.in_scope for case in CASES) == 30
    pairs = {(case.request, case.action) for case in CASES}
    assert len(pairs) == len(CASES)
    earlier = {(case.request, case.action) for suite, _ in SUITES for case in suite}
    earlier |= {(case.request, case.action) for case in PRIOR_CASES}
    assert not pairs & earlier
    for direct, extra in zip(CASES[::2], CASES[1::2], strict=True):
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope


def test_frozen_artifact_verifies_hashes_without_local_weights(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    adapter_dir = model_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (tmp_path / "benchmarks").mkdir()
    adapter = adapter_dir / "adapter_model.safetensors"
    head = model_dir / "head.npz"
    adapter.write_bytes(b"synthetic adapter")
    head.write_bytes(b"synthetic head")
    frozen = {
        "model": MODEL,
        "revision": REVISION,
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
    }
    (tmp_path / "benchmarks/qwen_scope_probe_v1.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )
    assert verify_artifact(tmp_path, model_dir) == frozen
    head.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen model artifact changed"):
        verify_artifact(tmp_path, model_dir)

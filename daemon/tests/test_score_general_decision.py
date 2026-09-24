"""Frozen generalized scorer input and artifact integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_general_decision import MODEL, REVISION, prepare_request, verify_artifact


def test_prepare_request_keeps_only_declared_input_fields() -> None:
    request = prepare_request({"state": {"cursor": [2, 4]}, "kind": "choice",
                               "question": "Choose a safe route.",
                               "options": ["wait", "ask", "proceed"],
                               "target": [0, 0, 1], "source": "hidden-label"})
    assert set(request) == {"state_json", "kind", "question", "options"}
    assert json.loads(request["state_json"]) == {"cursor": [2, 4]}
    with pytest.raises(ValueError, match="Noul"):
        prepare_request({**request, "state": "hello", "kind": "noul"})


def test_artifact_hashes_must_match_frozen_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "model"
    adapter = output_dir / "adapter"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"adapter")
    head = output_dir / "head.npz"
    head.write_bytes(b"head")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({
        "model": MODEL, "revision": REVISION,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    assert verify_artifact(output_dir, report_path)[1] == head
    head.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact(output_dir, report_path)

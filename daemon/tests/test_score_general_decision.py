"""Frozen generalized scorer input and artifact integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_general_scope_transfer import (
    CASES,
    case_payload,
    cautious_threshold,
    classification_metrics,
    scope_limits,
)
from scripts.score_general_decision import (
    MODEL,
    OPENJEV_BASE_MODEL,
    OPENJEV_BASE_REVISION,
    REVISION,
    prepare_request,
    prepare_requests,
    verify_artifact,
)
from scripts.train_world_scope_qlora import (
    OPENJEV_ADAPTER_SHA256,
    OPENJEV_HEAD_SHA256,
    OPENJEV_REPO,
    OPENJEV_REVISION,
)


def test_prepare_request_keeps_only_declared_input_fields() -> None:
    request = prepare_request({"state": {"cursor": [2, 4]}, "kind": "choice",
                               "question": "Choose a safe route.",
                               "options": ["wait", "ask", "proceed"],
                               "target": [0, 0, 1], "source": "hidden-label"})
    assert set(request) == {"state_json", "kind", "question", "options"}
    assert json.loads(request["state_json"]) == {"cursor": [2, 4]}
    with pytest.raises(ValueError, match="Noul"):
        prepare_request({**request, "state": "hello", "kind": "noul"})
    assert len(prepare_requests([{"state": "a", "kind": "noul", "question": "Valid?",
                                  "options": ["no", "yes"]},
                                 {"state": "b", "kind": "choice", "question": "Pick?",
                                  "options": ["one", "two"]}])) == 2
    with pytest.raises(ValueError, match="between 1 and 128"):
        prepare_requests([])


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


def test_joint_openjev_artifact_requires_pinned_source(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"joint-adapter")
    head = tmp_path / "head.npz"
    head.write_bytes(b"joint-head")
    report = {"model": OPENJEV_BASE_MODEL, "revision": OPENJEV_BASE_REVISION,
              "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
              "backbone_quantization": "NF4 double quantization",
              "initial_checkpoint": {"repo": OPENJEV_REPO, "revision": OPENJEV_REVISION,
                                     "adapter_sha256": OPENJEV_ADAPTER_SHA256,
                                     "head_sha256": OPENJEV_HEAD_SHA256}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert verify_artifact(tmp_path, report_path)[1] == head
    report["initial_checkpoint"]["revision"] = "wrong"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned warm start"):
        verify_artifact(tmp_path, report_path)


def test_scope_transfer_payload_contains_only_request_and_action() -> None:
    payload = case_payload(CASES[0])
    assert payload["kind"] == "noul"
    assert payload["state"] == {"user_request": CASES[0].request,
                                "proposed_action": CASES[0].action}
    assert "in_scope" not in json.dumps(payload)


def test_cautious_scope_gate_uses_only_calibration_negatives() -> None:
    probabilities = np.asarray([0.3, 0.8, 0.7, 0.9])
    labels = np.asarray([False, False, True, True])
    threshold = cautious_threshold(probabilities, labels)
    assert threshold > 0.8
    assert classification_metrics(probabilities, labels, threshold) == {
        "accuracy": 0.75, "false_allowances": 0, "false_warnings": 1}
    with pytest.raises(ValueError, match="out-of-scope"):
        cautious_threshold(probabilities[:2], np.asarray([True, True]))


def test_scope_calibration_uses_pinned_scope_manifest_limits() -> None:
    report = {"selected_rows": {"train": 256},
              "scopejudge": {"selected_rows": {
                  "scope_train": 512, "scope_calibration": 58, "scope_test": 755}}}
    assert scope_limits(report) == {"scope_train": 512,
                                    "scope_calibration": 58, "scope_test": 755}

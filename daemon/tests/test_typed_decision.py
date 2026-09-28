from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.intelligence.typed_decision import (
    MAX_OPTIONS,
    LocalTypedDecisionModel,
    _candidate_microbatch_size,
    candidate_prompts,
    format_answer,
    validate_decision,
)
from pilot.security.rpc_identity import RpcClientRole
from pilot.server import PilotServer


def test_dynamic_typed_decisions_are_normalized_without_task_specific_rules() -> None:
    validate_decision("choice", "Choose an outcome", ["wait", "ask", "continue"])
    prompts = candidate_prompts(
        {
            "kind": "choice",
            "question": "Choose an outcome",
            "options": ["wait", "ask"],
            "state_json": json.dumps({"request": "read only", "screen": {"title": "Settings"}}),
        },
        style="candidate_v1",
    )
    assert len(prompts) == 2
    assert "Settings" in prompts[0]
    assert "Proposed answer: wait" in prompts[0]
    assert format_answer("noul", ["no", "yes"], np.asarray([-1.0, 1.0]), 1.0)[
        "yes_probability"
    ] > 0.5
    score = format_answer("score", ["low", "mid", "high"], np.asarray([0.0, 1.0, 2.0]), 1.0)
    assert score["most_likely"] == "high"
    assert sum(score["probabilities"].values()) == pytest.approx(1.0)


def test_typed_decision_rejects_oversized_or_ambiguous_requests() -> None:
    with pytest.raises(ValueError, match="ordered as no, yes"):
        validate_decision("noul", "Continue?", ["yes", "no"])
    with pytest.raises(ValueError, match="unique"):
        validate_decision("choice", "Choose", ["same", "same"])
    with pytest.raises(ValueError, match=f"2-{MAX_OPTIONS}"):
        validate_decision("choice", "Choose", [str(index) for index in range(MAX_OPTIONS + 1)])


def test_candidate_microbatch_caps_long_prompts_and_respects_lower_limits() -> None:
    assert _candidate_microbatch_size(512, 8) == 8
    assert _candidate_microbatch_size(513, 8) == 4
    assert _candidate_microbatch_size(768, 8) == 4
    assert _candidate_microbatch_size(769, 8) == 2
    assert _candidate_microbatch_size(1_024, 1) == 1
    assert _candidate_microbatch_size(512, 16) == 16
    assert _candidate_microbatch_size(513, 16) == 4
    assert _candidate_microbatch_size(768, 16) == 4
    assert _candidate_microbatch_size(769, 16) == 2
    with pytest.raises(ValueError, match="must be positive"):
        _candidate_microbatch_size(0, 8)


def test_artifact_verification_fails_before_any_model_or_network_load(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"local candidate")
    head_file = tmp_path / "head.npz"
    np.savez(head_file, model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc")
    report_file = tmp_path / "report.json"
    report_file.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-2B",
                "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
                "head_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    model = LocalTypedDecisionModel(tmp_path, report_file)
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.score({}, "choice", "Choose", ["one", "two"])
    assert model._loaded is False


@pytest.mark.asyncio
async def test_typed_decision_rpc_is_opt_in_and_research_only(monkeypatch, tmp_path: Path) -> None:
    config = PilotConfig()
    server = PilotServer(config)
    client = object()
    server._client_roles[client] = RpcClientRole.MCP_LOCAL
    payload = {
        "state": {"goal": "inspect", "candidate": "collect live evidence"},
        "kind": "choice",
        "question": "Which candidate fits the task?",
        "options": ["collect live evidence", "change configuration"],
    }

    disabled = await server._handle_mcp_typed_decision(payload, client)
    assert disabled["status"] == "disabled"
    assert disabled["execution_authority"] is False

    config.typed_decision.enabled = True
    config.typed_decision.artifact_dir = str(tmp_path / "artifacts")
    config.typed_decision.report_path = str(tmp_path / "report.json")
    invalid = await server._handle_mcp_typed_decision({**payload, "state": []}, client)
    assert invalid["status"] == "error"
    assert invalid["research_only"] is True
    assert invalid["execution_authority"] is False
    calls = []

    class FakeModel:
        def __init__(self, artifact_dir, report_path) -> None:
            calls.append((artifact_dir, report_path))

        def score(self, state, kind, question, options):
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "answer": {"type": "choice", "choice": options[0]},
            }

        def unload(self):
            calls.append("unload")

    import pilot.intelligence.typed_decision as typed_decision_module

    monkeypatch.setattr(typed_decision_module, "LocalTypedDecisionModel", FakeModel)
    result = await server._handle_mcp_typed_decision(payload, client)
    assert result["execution_authority"] is False
    assert result["answer"]["choice"] == "collect live evidence"
    assert calls == [(str(tmp_path / "artifacts"), str(tmp_path / "report.json"))]

    config.typed_decision.enabled = False
    await server._handle_mcp_typed_decision(payload, client)
    assert server._typed_decision_engine is not None


@pytest.mark.asyncio
async def test_typed_decision_config_requires_explicit_paths_and_unloads_on_disable(tmp_path: Path) -> None:
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    invalid = await server._handle_update_config(
        {"section": "typed_decision", "values": {"enabled": True}},
        object(),
    )
    assert invalid["status"] == "error"
    assert config.typed_decision.enabled is False
    config.save.assert_not_called()

    saved = await server._handle_update_config(
        {
            "section": "typed_decision",
            "values": {
                "enabled": True,
                "artifact_dir": str(tmp_path / "artifacts"),
                "report_path": str(tmp_path / "report.json"),
            },
        },
        object(),
    )
    assert saved == {"status": "ok"}
    assert config.typed_decision.enabled is True

    unloaded = []
    server._typed_decision_engine = SimpleNamespace(unload=lambda: unloaded.append(True))
    server._typed_decision_engine_key = (
        config.typed_decision.artifact_dir,
        config.typed_decision.report_path,
    )
    disabled = await server._handle_update_config(
        {"section": "typed_decision", "values": {"enabled": False}},
        object(),
    )
    assert disabled == {"status": "ok"}
    assert unloaded == [True]
    assert server._typed_decision_engine is None

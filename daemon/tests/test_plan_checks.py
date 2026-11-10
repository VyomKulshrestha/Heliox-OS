"""The plan card's "checked before running" row: only values the world model and Gnomon really produced."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pilot.agents.destructive_critic import assess_plan_risk
from pilot.config import PilotConfig
from pilot.intelligence.gnomon_reviewer import GnomonVerdict
from pilot.plan_checks import world_model_summary
from tests.test_gnomon_requests import FakeGnomon, _server, _socket, _write_plan


def _prediction(system: dict[str, Any], effects: list[dict[str, Any]], source: str = "predicted") -> dict[str, Any]:
    return {
        "action_type": "x",
        "predicted_state": {"observed_at": "t", "system": system, "ui": {}, "browser": {}},
        "expected_effects": effects,
        "input_state_source": source,
        "sources": ["rule"],
    }


def test_summary_reads_start_and_end_state_from_the_real_predictions():
    first = _prediction(
        {"process_count": 351, "disk_usage_fraction": 0.72},
        [{"domain": "process", "operation": "start", "before": 350, "after": 351}],
        source="observed",
    )
    last = _prediction({"process_count": 351, "disk_usage_fraction": 0.75}, [])
    last["size_estimates"] = [
        {
            "path_label": "Photos",
            "bytes": 3 * 1024**3,
            "size_is_exact": True,
            "volume": "D:\\",
            "usage_before": 0.5,
            "usage_after": 0.55,
            "risk": 0.0,
        }
    ]
    summary = world_model_summary(
        {
            "world_model_score": 0.0,
            "reasons": [],
            "prediction_sources": [],
            "model_version": "rules",
            "requires_confirmation": False,
            "predictions": [first, last],
        }
    )
    assert summary["observed"] == {"disk_usage_fraction": 0.72, "process_count": 350}
    assert summary["predicted"] == {"disk_usage_fraction": 0.75, "process_count": 351}
    assert summary["measured_sizes"][0]["size_text"] == "about 3.0 GB"
    assert summary["measured_sizes"][0]["usage_after"] == 0.55
    assert summary["raised"] is False and summary["unavailable"] is False


def test_values_the_model_did_not_produce_stay_empty():
    unknown = _prediction({"process_count": "requires_observation", "disk_usage_fraction": 0.4}, [], source="observed")
    summary = world_model_summary({"predictions": [unknown]})
    assert summary["predicted"]["process_count"] is None
    assert world_model_summary({"predictions": []})["observed"] is None
    failed = world_model_summary(
        {
            "world_model_score": 1.0,
            "reasons": ["World-model assessment unavailable; review state-changing actions before execution."],
            "model_version": "evaluation-error",
            "requires_confirmation": True,
            "predictions": [],
        }
    )
    assert failed["unavailable"] and failed["raised"] and failed["predicted"] is None


def test_summary_of_a_real_gate_evaluation_is_json_safe():
    config = PilotConfig()
    config.gateway.risk_gate_enabled = True
    summary = world_model_summary(assess_plan_risk(_write_plan(), config).to_dict())
    json.dumps(summary)
    assert summary["observed"] is not None
    assert 0.0 <= summary["observed"]["disk_usage_fraction"] <= 1.0
    assert isinstance(summary["observed"]["process_count"], int)


def _sent(ws: Any) -> dict[str, dict[str, Any]]:
    messages = [json.loads(call.args[0]) for call in ws.send.await_args_list]
    return {message["method"]: message["params"] for message in messages if "method" in message}


@pytest.mark.asyncio
async def test_a_direct_request_sends_what_the_row_needs():
    gnomon, seen = FakeGnomon(GnomonVerdict(escalate=False, scores={"in_scope": 0.97})), {}
    server = _server(_write_plan(), gnomon, seen)
    server.config.gateway.risk_gate_enabled = True
    ws = _socket()
    await server._handle_execute({"input": "save my note", "session_id": "s-1"}, ws)
    sent = _sent(ws)
    plan_id = sent["plan_preview"]["plan_id"]
    assert sent["plan_preview"]["world_model_enabled"] is True
    assessment = sent["world_model_assessment"]
    assert assessment["plan_id"] == plan_id
    assert assessment["summary"]["enabled"] is True
    assert assessment["summary"]["score"] == assessment["world_model_score"]
    review = sent["gnomon_review"]
    assert review["plan_id"] == plan_id and review["state"] == "reviewed"
    assert review["scores"] == {"in_scope": 0.97} and review["review_seconds"] >= 0


@pytest.mark.asyncio
async def test_switched_off_gate_says_so_and_sends_no_prediction():
    gnomon, seen = FakeGnomon(None), {}
    server = _server(_write_plan(), gnomon, seen)
    server.config.gateway.risk_gate_enabled = False
    ws = _socket()
    await server._handle_execute({"input": "save my note", "session_id": "s-1"}, ws)
    sent = _sent(ws)
    assert sent["plan_preview"]["world_model_enabled"] is False
    assert "world_model_assessment" not in sent
    review = sent["gnomon_review"]
    assert review["state"] == "loading" and "review_seconds" not in review and "reason" in review

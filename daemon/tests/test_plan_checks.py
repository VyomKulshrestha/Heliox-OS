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
    # The process contract started one process; the carried disk value is never a prediction.
    assert summary["predicted"] == {"disk_usage_fraction": None, "process_count": 351}
    assert summary["measured_sizes"][0]["size_text"] == "about 3.0 GB"
    assert summary["measured_sizes"][0]["usage_after"] == 0.55
    assert summary["raised"] is False and summary["unavailable"] is False


def test_rule_table_guesses_never_reach_the_row():
    """A read, then an append of that output: the gate's rule table adds +2% disk for the unmeasured
    write and +1 process for the shell step, carried into later steps' predicted state."""
    read = _prediction(
        {"process_count": 361, "disk_usage_fraction": 0.72},
        [{"domain": "process", "operation": "run_transient", "before": 361, "after": "requires_observation"}],
        source="observed",
    )
    read["action_type"] = "shell_command"
    write = _prediction({"process_count": 362, "disk_usage_fraction": 0.72}, [])
    write["action_type"] = "file_write"
    after = _prediction({"process_count": 362, "disk_usage_fraction": 0.74}, [])
    after["action_type"] = "notify"
    summary = world_model_summary({"predictions": [read, write, after]})
    assert summary["observed"] == {"disk_usage_fraction": 0.72, "process_count": 361}
    assert summary["predicted"] == {"disk_usage_fraction": None, "process_count": None}
    assert summary["measured_sizes"] == []


def test_plan_without_process_effects_predicts_no_process_count():
    first = _prediction({"process_count": 361, "disk_usage_fraction": 0.72}, [], source="observed")
    second = _prediction({"process_count": 362, "disk_usage_fraction": 0.74}, [])
    summary = world_model_summary({"predictions": [first, second]})
    assert summary["predicted"] == {"disk_usage_fraction": None, "process_count": None}


def test_measured_sizes_keep_usage_only_before_any_guessed_step():
    def measured(label: str, before: float, after: float) -> dict[str, Any]:
        return {"path_label": label, "bytes": 1024, "size_is_exact": True, "volume": "C:\\",
                "usage_before": before, "usage_after": after, "risk": 0.0}  # fmt: skip

    copy = _prediction({"process_count": 10, "disk_usage_fraction": 0.5}, [], source="observed")
    copy["action_type"] = "file_copy"
    copy["size_estimates"] = [measured("a.txt", 0.5, 0.5)]
    unmeasured_write = _prediction({"process_count": 10, "disk_usage_fraction": 0.5}, [])
    unmeasured_write["action_type"] = "file_write"
    second_copy = _prediction({"process_count": 10, "disk_usage_fraction": 0.52}, [])
    second_copy["action_type"] = "file_copy"
    second_copy["size_estimates"] = [measured("b.txt", 0.52, 0.52)]
    sizes = world_model_summary({"predictions": [copy, unmeasured_write, second_copy]})["measured_sizes"]
    assert [size["size_text"] for size in sizes] == ["about 1.0 KB", "about 1.0 KB"]
    assert (sizes[0]["usage_before"], sizes[0]["usage_after"]) == (0.5, 0.5)
    # The ledger behind b.txt includes the write's fixed +2% guess, so its percentages are dropped.
    assert (sizes[1]["usage_before"], sizes[1]["usage_after"]) == (None, None)


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


@pytest.mark.asyncio
async def test_the_safety_review_reaches_the_plan_it_reviewed():
    from pilot.actions import Action, ActionPlan
    from pilot.agents.destructive_critic import CriticVerdict

    finding = {"step": 1, "severity": "high", "text": "Step 1 deletes drafts that may hold the real final post."}

    class Critic:
        async def review(self, user_input: str, plan: ActionPlan) -> CriticVerdict:
            return CriticVerdict(verdict="WARN", risk_score=0.5, issues=[finding["text"]], findings=[finding])

    def delete(name: str) -> Action:
        path = f"C:/launch/{name}"
        return Action.model_validate(
            {"action_type": "file_delete", "target": path, "parameters": {"path": path, "recursive": True}}
        )

    async def run(plan: ActionPlan) -> dict[str, dict[str, Any]]:
        server = _server(plan, FakeGnomon(GnomonVerdict(escalate=False)), {})
        server._destructive_critic = Critic()
        ws = _socket()
        await server._handle_execute({"input": "tidy up", "session_id": "s-1"}, ws)
        return _sent(ws)

    # One delete scores below the review threshold: the row learns the review did not run.
    sent = await run(ActionPlan(actions=[delete("drafts")], explanation="delete", raw_input="tidy up"))
    skipped = sent["critic_verdict"]
    assert skipped["plan_id"] == sent["plan_preview"]["plan_id"]
    assert skipped["verdict"] == "SKIPPED" and skipped["critic_skipped"] == "low_risk_heuristic"

    names = ["drafts", "build", "old", "tmp"]
    sent = await run(ActionPlan(actions=[delete(name) for name in names], explanation="delete", raw_input="tidy up"))
    verdict = sent["critic_verdict"]
    assert verdict["plan_id"] == sent["plan_preview"]["plan_id"]
    assert verdict["verdict"] == "WARN" and verdict["findings"] == [finding] and verdict["ran"] is True

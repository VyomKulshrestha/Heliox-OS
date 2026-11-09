"""Gnomon reviews the plans the person asks for directly, not only autonomous work. It can only add caution."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot import server as server_module
from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams, VerificationResult
from pilot.config import PilotConfig
from pilot.intelligence.gnomon_reviewer import GnomonVerdict
from pilot.server import PilotServer


def _write_plan() -> ActionPlan:
    action = Action(action_type=ActionType.NOTIFY, target="all", parameters=NotifyParams(summary="x", body="y"))
    write = Action.model_validate(
        {"action_type": "file_write", "target": "C:/notes/todo.txt", "parameters": {"path": "C:/notes/todo.txt",
                                                                                     "content": "hello"}}
    )  # fmt: skip
    return ActionPlan(actions=[write, action], explanation="write the note", raw_input="save my note")


def _read_plan() -> ActionPlan:
    read = Action.model_validate({"action_type": "disk_usage", "target": "C:", "parameters": {}})
    return ActionPlan(actions=[read], explanation="check the disk", raw_input="how much space")


class FakeGnomon:
    def __init__(self, verdict: GnomonVerdict | None, delay: float = 0.0) -> None:
        self.verdict = verdict
        self.delay = delay
        self.asked: list[tuple[str, str]] = []

    async def review_plan(self, goal: str, plan: Any, untrusted_context: str = "") -> GnomonVerdict | None:
        self.asked.append((goal, untrusted_context))
        await asyncio.sleep(self.delay)
        return self.verdict

    def status(self) -> dict[str, Any]:
        return {"state": "loading" if self.verdict is None else "ready", "execution_authority": False}

    def warm(self) -> None:
        return None


def _server(plan: ActionPlan, gnomon: FakeGnomon, seen: dict[str, Any], *, enabled: bool = True) -> PilotServer:
    config = PilotConfig()
    config.security.dry_run = False
    config.typed_decision.enabled = enabled
    config.typed_decision.review_requests = True
    server = PilotServer(config)
    server._gnomon_reviewer = gnomon
    server._planner = AsyncMock()
    server._planner.plan.return_value = plan
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = False  # tiers alone would just run it
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))

    async def confirm(plan_id: str, approval_plan: ActionPlan, ws: Any, **kwargs: Any):
        seen["confirmation"] = kwargs
        return False, set(), set()  # the person says no

    server._wait_for_confirmation = confirm  # type: ignore[method-assign]

    async def execute(run_plan: ActionPlan, **kwargs: Any) -> list[Any]:
        seen["ran"] = [action.action_type.value for action in run_plan.actions]
        return [ActionResult(action=action, success=True, output="ok") for action in run_plan.actions]

    server._executor = SimpleNamespace(execute=execute)
    server._orchestrator = None
    return server


def _socket() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_a_plan_gnomon_flags_waits_for_the_person_with_its_reason():
    verdict = GnomonVerdict(escalate=True, reasons=["Gnomon judged this may go beyond what you asked (8% in scope)."])
    gnomon, seen = FakeGnomon(verdict), {}
    server = _server(_write_plan(), gnomon, seen)
    ws = _socket()
    await server._handle_execute({"input": "save my note", "session_id": "s-1"}, ws)
    assert gnomon.asked and gnomon.asked[0][0] == "save my note"
    assert "beyond what you asked" in seen["confirmation"]["reason"]
    assert seen["confirmation"]["force_all_actions"] is True
    assert "ran" not in seen  # declined, so nothing ran
    sent = [call.args[0] for call in ws.send.await_args_list]
    assert any('"gnomon_review"' in str(message) for message in sent)


@pytest.mark.asyncio
async def test_a_plan_gnomon_is_fine_with_runs_as_before():
    gnomon, seen = FakeGnomon(GnomonVerdict(escalate=False)), {}
    server = _server(_write_plan(), gnomon, seen)
    await server._handle_execute({"input": "save my note", "session_id": "s-1"}, _socket())
    assert gnomon.asked and "confirmation" not in seen and seen["ran"] == ["file_write", "notify"]


@pytest.mark.asyncio
async def test_not_loaded_yet_or_too_slow_means_no_check_and_no_wait(monkeypatch):
    for gnomon in (FakeGnomon(None), FakeGnomon(GnomonVerdict(escalate=True, reasons=["slow"]), delay=1.0)):
        monkeypatch.setattr(server_module, "GNOMON_REQUEST_TIMEOUT", 0.05)
        seen: dict[str, Any] = {}
        server = _server(_write_plan(), gnomon, seen)
        await server._handle_execute({"input": "save my note", "session_id": "s-1"}, _socket())
        assert "confirmation" not in seen and seen["ran"] == ["file_write", "notify"]


@pytest.mark.asyncio
async def test_reads_and_a_switched_off_gnomon_are_never_reviewed():
    gnomon, seen = FakeGnomon(GnomonVerdict(escalate=True, reasons=["x"])), {}
    await _server(_read_plan(), gnomon, seen)._handle_execute({"input": "how much space", "session_id": "s"}, _socket())
    assert gnomon.asked == [] and "confirmation" not in seen

    gnomon, seen = FakeGnomon(GnomonVerdict(escalate=True, reasons=["x"])), {}
    off = _server(_write_plan(), gnomon, seen, enabled=False)
    await off._handle_execute({"input": "save my note", "session_id": "s"}, _socket())
    assert gnomon.asked == [] and "confirmation" not in seen


def test_reviewing_every_request_is_off_until_gnomon_is_good_enough_for_it():
    assert PilotConfig().typed_decision.review_requests is False

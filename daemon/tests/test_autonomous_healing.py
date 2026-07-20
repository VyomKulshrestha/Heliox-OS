"""Tests for pilot.agents.autonomous_healing.AutonomousHealingEngine.

Covers the tiered auto-exec vs. propose-and-wait decision, cooldown
suppression, the disabled/not-watched no-ops, and the three per-metric
on_trigger hooks BackgroundTaskManager calls.
"""

from __future__ import annotations

import asyncio

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.agents.autonomous_healing import AutonomousHealingEngine, HealingOutcome
from pilot.config import PilotConfig
from pilot.security.gateway import InvocationSource


def _plan(*action_types: ActionType, error: str | None = None) -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=t, target="", parameters=EmptyParams()) for t in action_types],
        raw_input="self_healing_test",
        error=error,
    )


class FakePlanner:
    def __init__(self, plan: ActionPlan | Exception):
        self._plan = plan

    async def plan(self, goal: str) -> ActionPlan:
        if isinstance(self._plan, Exception):
            raise self._plan
        return self._plan


class FakeExecutor:
    def __init__(self):
        self.calls: list[dict] = []

    async def execute(self, plan, invocation_source=None, plan_id=None) -> list[ActionResult]:
        self.calls.append({"plan": plan, "invocation_source": invocation_source, "plan_id": plan_id})
        return [ActionResult(action=a, success=True, output="ok") for a in plan.actions]


def _config(**overrides) -> PilotConfig:
    cfg = PilotConfig()
    cfg.self_healing.enabled = True
    for k, v in overrides.items():
        setattr(cfg.self_healing, k, v)
    return cfg


def _engine(plan, executor=None, **cfg_overrides) -> tuple[AutonomousHealingEngine, dict, FakeExecutor]:
    pending_confirms: dict = {}
    executor = executor or FakeExecutor()
    engine = AutonomousHealingEngine(
        planner=FakePlanner(plan),
        executor=executor,
        config=_config(**cfg_overrides),
        pending_confirms=pending_confirms,
    )
    return engine, pending_confirms, executor


class TestDisabledOrUnwatched:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        engine, _, executor = _engine(_plan(ActionType.PROCESS_LIST))
        engine._config.self_healing.enabled = False
        result = await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert result is None
        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_unwatched_metric_returns_none(self):
        engine, _, executor = _engine(_plan(ActionType.PROCESS_LIST), watched_metrics=["disk"])
        result = await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert result is None
        assert executor.calls == []


class TestCooldown:
    @pytest.mark.asyncio
    async def test_second_alert_within_cooldown_is_suppressed(self):
        engine, _, executor = _engine(_plan(ActionType.PROCESS_LIST), cooldown_seconds=600.0)
        first = await engine.on_health_alert("cpu", {"message": "cpu high"})
        second = await engine.on_health_alert("cpu", {"message": "cpu still high"})
        assert first is not None
        assert second is None
        assert len(executor.calls) == 1


class TestPlanningOutcomes:
    @pytest.mark.asyncio
    async def test_plan_error_produces_no_action(self):
        engine, _, executor = _engine(_plan(error="planner could not decide"))
        attempt = await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert attempt is not None
        assert attempt.outcome == HealingOutcome.NO_ACTION.value
        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_empty_actions_produces_no_action(self):
        engine, _, executor = _engine(_plan())
        attempt = await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert attempt is not None
        assert attempt.outcome == HealingOutcome.NO_ACTION.value
        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_planner_exception_produces_plan_error(self):
        engine, _, executor = _engine(RuntimeError("model unavailable"))
        attempt = await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert attempt is not None
        assert attempt.outcome == HealingOutcome.PLAN_ERROR.value
        assert "model unavailable" in attempt.explanation
        assert executor.calls == []


class TestAutoExecute:
    @pytest.mark.asyncio
    async def test_low_tier_plan_auto_executes(self):
        # PROCESS_LIST is READ_ONLY tier and not irreversible.
        engine, _, executor = _engine(_plan(ActionType.PROCESS_LIST))
        attempt = await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert attempt is not None
        assert attempt.outcome == HealingOutcome.AUTO_EXECUTED.value
        assert len(executor.calls) == 1
        assert executor.calls[0]["invocation_source"] == InvocationSource.SELF_HEALING

    @pytest.mark.asyncio
    async def test_auto_execute_never_registers_a_pending_confirmation(self):
        engine, pending_confirms, _ = _engine(_plan(ActionType.PROCESS_LIST))
        await engine.on_health_alert("cpu", {"message": "cpu high"})
        assert pending_confirms == {}


class TestProposeAndWait:
    @pytest.mark.asyncio
    async def test_high_tier_plan_is_proposed_not_auto_executed(self):
        # PROCESS_KILL is a DESTRUCTIVE (Tier 3) action.
        engine, pending_confirms, executor = _engine(_plan(ActionType.PROCESS_KILL), confirm_timeout_seconds=0.2)
        task = asyncio.create_task(engine.on_health_alert("memory", {"message": "mem high"}))
        await asyncio.sleep(0.05)
        assert len(pending_confirms) == 1
        attempt = task.result() if task.done() else next(iter(engine._attempts.values()))
        assert attempt.outcome == HealingOutcome.PROPOSED.value
        assert executor.calls == []
        await task  # let the confirm timeout finish so the test doesn't leak a task

    @pytest.mark.asyncio
    async def test_confirmed_proposal_executes(self):
        engine, pending_confirms, executor = _engine(_plan(ActionType.PROCESS_KILL), confirm_timeout_seconds=5.0)
        task = asyncio.create_task(engine.on_health_alert("memory", {"message": "mem high"}))
        await asyncio.sleep(0.05)
        attempt = next(iter(engine._attempts.values()))
        pending = pending_confirms[attempt.plan_id]
        pending.confirmed = True
        pending.event.set()
        result = await task
        assert result.outcome == HealingOutcome.CONFIRMED.value
        assert len(executor.calls) == 1
        assert executor.calls[0]["invocation_source"] == InvocationSource.SELF_HEALING

    @pytest.mark.asyncio
    async def test_denied_proposal_never_executes(self):
        engine, pending_confirms, executor = _engine(_plan(ActionType.PROCESS_KILL), confirm_timeout_seconds=5.0)
        task = asyncio.create_task(engine.on_health_alert("memory", {"message": "mem high"}))
        await asyncio.sleep(0.05)
        attempt = next(iter(engine._attempts.values()))
        pending = pending_confirms[attempt.plan_id]
        pending.confirmed = False
        pending.event.set()
        result = await task
        assert result.outcome == HealingOutcome.DENIED.value
        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_timed_out_proposal_never_executes(self):
        engine, pending_confirms, executor = _engine(_plan(ActionType.PROCESS_KILL), confirm_timeout_seconds=0.05)
        attempt = await engine.on_health_alert("memory", {"message": "mem high"})
        assert attempt.outcome == HealingOutcome.TIMED_OUT.value
        assert executor.calls == []
        assert pending_confirms == {}

    @pytest.mark.asyncio
    async def test_irreversible_action_is_proposed_even_at_low_tier(self):
        # POWER_RESTART is irreversible (and Tier 3+), so it must never
        # auto-execute regardless of auto_execute_max_tier.
        engine, pending_confirms, executor = _engine(
            _plan(ActionType.POWER_RESTART), confirm_timeout_seconds=0.05, auto_execute_max_tier=4
        )
        attempt = await engine.on_health_alert("disk", {"message": "disk full"})
        assert attempt.outcome == HealingOutcome.TIMED_OUT.value
        assert executor.calls == []


class TestPerMetricHooks:
    @pytest.mark.asyncio
    async def test_on_cpu_alert_tags_metric_cpu(self):
        engine, _, _ = _engine(_plan(ActionType.PROCESS_LIST))
        attempt = await engine.on_cpu_alert({"message": "cpu high"})
        assert attempt.metric == "cpu"

    @pytest.mark.asyncio
    async def test_on_memory_alert_tags_metric_memory(self):
        engine, _, _ = _engine(_plan(ActionType.PROCESS_LIST))
        attempt = await engine.on_memory_alert({"message": "mem high"})
        assert attempt.metric == "memory"

    @pytest.mark.asyncio
    async def test_on_disk_alert_tags_metric_disk(self):
        engine, _, _ = _engine(_plan(ActionType.PROCESS_LIST))
        attempt = await engine.on_disk_alert({"message": "disk full"})
        assert attempt.metric == "disk"


class TestListAttempts:
    @pytest.mark.asyncio
    async def test_list_attempts_returns_dict_form(self):
        engine, _, _ = _engine(_plan(ActionType.PROCESS_LIST))
        await engine.on_health_alert("cpu", {"message": "cpu high"})
        attempts = engine.list_attempts()
        assert len(attempts) == 1
        assert attempts[0]["metric"] == "cpu"
        assert attempts[0]["outcome"] == HealingOutcome.AUTO_EXECUTED.value

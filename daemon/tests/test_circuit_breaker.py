"""Phase 4 tests: CircuitBreaker FSM and orchestrator integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentRole
from pilot.agents.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.models.budget_tracker import current_task_id


def _make_action():
    return Action(action_type=ActionType.SYSTEM_INFO, parameters={})


def _make_plan(n_actions=1):
    return ActionPlan(
        actions=[_make_action() for _ in range(n_actions)],
        explanation="test",
        raw_input="test",
    )


def _stub_agent(role=AgentRole.SYSTEM, results=None):
    a = MagicMock()
    a.role = role
    a.get_capabilities = MagicMock(return_value=[])
    a.attach_orchestrator = MagicMock()
    a.handle_task = AsyncMock(return_value=results or [])
    return a


# ─── CircuitBreaker unit tests ───

def test_breaker_starts_closed():
    cb = CircuitBreaker(threshold=3)
    assert cb.get_state("t1") == CircuitBreakerState.CLOSED


def test_breaker_trips_after_threshold_failures():
    cb = CircuitBreaker(threshold=3)
    cb.record_failure("t1")
    cb.record_failure("t1")
    assert cb.get_state("t1") == CircuitBreakerState.CLOSED
    cb.record_failure("t1")
    assert cb.get_state("t1") == CircuitBreakerState.OPEN


def test_breaker_success_resets_counter():
    cb = CircuitBreaker(threshold=3)
    cb.record_failure("t1")
    cb.record_failure("t1")
    cb.record_success("t1")
    assert cb.get_failure_count("t1") == 0
    # Now we need a full threshold of failures again to trip
    cb.record_failure("t1")
    cb.record_failure("t1")
    assert cb.get_state("t1") == CircuitBreakerState.CLOSED


def test_check_raises_when_open():
    cb = CircuitBreaker(threshold=2)
    cb.record_failure("t1")
    cb.record_failure("t1")
    with pytest.raises(CircuitBreakerOpenError):
        cb.check("t1")


def test_check_silent_when_closed():
    cb = CircuitBreaker(threshold=3)
    cb.record_failure("t1")
    cb.check("t1")  # should not raise


def test_reset_clears_state():
    cb = CircuitBreaker(threshold=2)
    cb.record_failure("t1")
    cb.record_failure("t1")
    cb.reset("t1")
    assert cb.get_state("t1") == CircuitBreakerState.CLOSED
    assert cb.get_failure_count("t1") == 0


def test_breaker_uses_contextvar_when_no_task_id_passed():
    cb = CircuitBreaker(threshold=2)
    token = current_task_id.set("t-ctx")
    try:
        cb.record_failure()
        cb.record_failure()
        assert cb.get_state("t-ctx") == CircuitBreakerState.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            cb.check()
    finally:
        current_task_id.reset(token)


def test_concurrent_tasks_isolated():
    cb = CircuitBreaker(threshold=2)
    cb.record_failure("t-a")
    cb.record_failure("t-a")
    cb.record_failure("t-b")
    assert cb.get_state("t-a") == CircuitBreakerState.OPEN
    assert cb.get_state("t-b") == CircuitBreakerState.CLOSED


# ─── Orchestrator integration tests ───

@pytest.fixture
def orchestrator_with_breaker():
    o = AgentOrchestrator(model_router=MagicMock())
    breaker = CircuitBreaker(threshold=2)
    o.set_circuit_breaker(breaker)
    return o, breaker


@pytest.mark.asyncio
async def test_breaker_trips_after_repeated_failures(orchestrator_with_breaker):
    o, breaker = orchestrator_with_breaker
    plan = _make_plan(n_actions=3)
    action_type = plan.actions[0].action_type

    failed_result = lambda a: ActionResult(action=a, success=False, error="boom")
    agent = _stub_agent(results=[failed_result(plan.actions[0])])
    # Use side_effect so each call returns a fresh failed result for its own action
    agent.handle_task = AsyncMock(side_effect=[
        [failed_result(plan.actions[0])],
        [failed_result(plan.actions[1])],
        [failed_result(plan.actions[2])],
    ])
    o._action_registry[action_type] = AgentRole.SYSTEM
    o._agents[AgentRole.SYSTEM] = agent
    # Force one-action-per-batch so the breaker check fires between actions
    o._build_execution_order = MagicMock(return_value=[
        (AgentRole.SYSTEM, [0]),
        (AgentRole.SYSTEM, [1]),
        (AgentRole.SYSTEM, [2]),
    ])

    results = await o.execute_plan("test", plan, plan_id="t-trip")

    # First two batches run (each fails), third batch sees OPEN breaker and halts
    assert agent.handle_task.call_count == 2
    assert all(not r.success for r in results)
    # The third result should be marked with the breaker error
    assert "Circuit breaker tripped" in (results[2].error or "")


@pytest.mark.asyncio
async def test_success_resets_breaker_counter(orchestrator_with_breaker):
    o, breaker = orchestrator_with_breaker
    plan = _make_plan(n_actions=3)
    action_type = plan.actions[0].action_type

    ok = lambda a: ActionResult(action=a, success=True, output="ok")
    bad = lambda a: ActionResult(action=a, success=False, error="boom")
    agent = _stub_agent()
    agent.handle_task = AsyncMock(side_effect=[
        [bad(plan.actions[0])],
        [ok(plan.actions[1])],   # success — resets counter
        [bad(plan.actions[2])],  # only one failure since reset; breaker stays CLOSED
    ])
    o._action_registry[action_type] = AgentRole.SYSTEM
    o._agents[AgentRole.SYSTEM] = agent
    o._build_execution_order = MagicMock(return_value=[
        (AgentRole.SYSTEM, [0]),
        (AgentRole.SYSTEM, [1]),
        (AgentRole.SYSTEM, [2]),
    ])

    await o.execute_plan("test", plan, plan_id="t-reset")

    # All three batches actually ran — breaker never tripped
    assert agent.handle_task.call_count == 3


@pytest.mark.asyncio
async def test_breaker_reset_on_task_end(orchestrator_with_breaker):
    o, breaker = orchestrator_with_breaker
    plan = _make_plan(n_actions=2)
    action_type = plan.actions[0].action_type

    bad = lambda a: ActionResult(action=a, success=False, error="boom")
    agent = _stub_agent()
    agent.handle_task = AsyncMock(side_effect=[
        [bad(plan.actions[0])],
        [bad(plan.actions[1])],
    ])
    o._action_registry[action_type] = AgentRole.SYSTEM
    o._agents[AgentRole.SYSTEM] = agent
    o._build_execution_order = MagicMock(return_value=[
        (AgentRole.SYSTEM, [0]),
        (AgentRole.SYSTEM, [1]),
    ])

    await o.execute_plan("test", plan, plan_id="t-end")

    # After task end, breaker state should be cleared
    assert breaker.get_state("t-end") == CircuitBreakerState.CLOSED
    assert breaker.get_failure_count("t-end") == 0


@pytest.mark.asyncio
async def test_breaker_tripped_broadcast(orchestrator_with_breaker):
    o, breaker = orchestrator_with_breaker
    plan = _make_plan(n_actions=3)
    action_type = plan.actions[0].action_type

    bad = lambda a: ActionResult(action=a, success=False, error="boom")
    agent = _stub_agent()
    agent.handle_task = AsyncMock(side_effect=[
        [bad(plan.actions[0])],
        [bad(plan.actions[1])],
        [bad(plan.actions[2])],
    ])
    o._action_registry[action_type] = AgentRole.SYSTEM
    o._agents[AgentRole.SYSTEM] = agent
    o._build_execution_order = MagicMock(return_value=[
        (AgentRole.SYSTEM, [0]),
        (AgentRole.SYSTEM, [1]),
        (AgentRole.SYSTEM, [2]),
    ])

    broadcast = AsyncMock()
    o.set_broadcast(broadcast)
    await o.execute_plan("test", plan, plan_id="t-broadcast")

    breaker_calls = [
        c for c in broadcast.call_args_list
        if c.args[0] == "circuit_breaker_tripped"
    ]
    assert len(breaker_calls) == 1
    payload = breaker_calls[0].args[1]
    assert payload["task_id"] == "t-broadcast"
    assert payload["failure_count"] >= 2
"""Phase 3 tests: AgentOrchestrator task lifecycle and budget halt handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentRole
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.config import ModelConfig
from pilot.models.budget_tracker import (
    ActionBudgetExceededError,
    BudgetTracker,
    TaskBudgetExceededError,
    current_task_id,
)


def _make_action(action_type=ActionType.SYSTEM_INFO, **kwargs):
    """Build a minimal Action for plan construction. Adjust ActionType if needed."""
    from pilot.actions import Action

    return Action(action_type=action_type, parameters={}, **kwargs)


def _make_plan(actions=None, explanation="test"):
    return ActionPlan(
        actions=actions or [_make_action()],
        explanation=explanation,
        raw_input="test input",
    )


def _stub_agent(role=AgentRole.SYSTEM, results=None, raises=None):
    """Build a stub agent that returns canned results or raises."""
    agent = MagicMock()
    agent.role = role
    agent.get_capabilities = MagicMock(return_value=[])
    agent.attach_orchestrator = MagicMock()
    if raises:
        agent.handle_task = AsyncMock(side_effect=raises)
    else:
        agent.handle_task = AsyncMock(return_value=results or [])
    return agent


@pytest.fixture
def model_config():
    return ModelConfig(
        budget_enabled=True,
        max_tokens_per_task=1000,
        max_usd_per_task=0.10,
    )


@pytest.fixture
async def tracker(model_config, tmp_path):
    t = BudgetTracker(model_config, str(tmp_path / "b.db"))
    await t.initialize()
    yield t
    await t.close()


@pytest.fixture
def orchestrator(tracker):
    o = AgentOrchestrator(model_router=MagicMock())
    o.set_budget_tracker(tracker)
    return o


@pytest.mark.asyncio
async def test_execute_plan_starts_and_ends_task(orchestrator, tracker):
    plan = _make_plan()
    action = plan.actions[0]
    agent = _stub_agent(results=[ActionResult(action=action, success=True, output="ok")])
    # Register the stub against the action type
    orchestrator._action_registry[action.action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = agent

    await orchestrator.execute_plan("test", plan, plan_id="my-task-id")

    # Task was created and cleaned up
    assert tracker.get_task_budget("my-task-id") is None
    # Contextvar reset after execution
    assert current_task_id.get() is None


@pytest.mark.asyncio
async def test_execute_plan_generates_uuid_when_no_plan_id(orchestrator):
    plan = _make_plan()
    action = plan.actions[0]
    agent = _stub_agent(results=[ActionResult(action=action, success=True, output="ok")])
    orchestrator._action_registry[action.action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = agent

    # Patch start_task to capture what id is used
    seen_ids: list[str] = []
    original = orchestrator._budget_tracker.start_task

    def capturing_start(tid):
        seen_ids.append(tid)
        return original(tid)

    orchestrator._budget_tracker.start_task = capturing_start
    await orchestrator.execute_plan("test", plan)

    assert len(seen_ids) == 1
    # UUID4 is 36 chars with hyphens
    assert len(seen_ids[0]) == 36


@pytest.mark.asyncio
async def test_budget_error_mid_plan_halts_cleanly(orchestrator, tracker):
    plan = _make_plan(actions=[_make_action(), _make_action()])
    raising_agent = _stub_agent(raises=TaskBudgetExceededError("test task budget hit"))
    orchestrator._action_registry[plan.actions[0].action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = raising_agent

    results = await orchestrator.execute_plan("test", plan, plan_id="t1")

    # Both actions should be marked failed with budget-specific error
    assert len(results) == 2
    assert all(not r.success for r in results)
    assert all("Budget exceeded" in (r.error or "") for r in results)
    # Task was cleaned up
    assert tracker.get_task_budget("t1") is None


@pytest.mark.asyncio
async def test_budget_error_sets_cancel_event(orchestrator):
    plan = _make_plan()
    raising_agent = _stub_agent(raises=ActionBudgetExceededError("prompt too large"))
    orchestrator._action_registry[plan.actions[0].action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = raising_agent

    cancel = asyncio.Event()
    await orchestrator.execute_plan("test", plan, cancel_event=cancel)

    assert cancel.is_set()


@pytest.mark.asyncio
async def test_budget_error_broadcast_emitted(orchestrator):
    plan = _make_plan()
    raising_agent = _stub_agent(raises=TaskBudgetExceededError("usd cap reached"))
    orchestrator._action_registry[plan.actions[0].action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = raising_agent

    broadcast = AsyncMock()
    orchestrator.set_broadcast(broadcast)

    await orchestrator.execute_plan("test", plan, plan_id="t-broadcast")

    # Should be called with the agent_routing event AND the budget_exceeded event.
    # Find the budget_exceeded call.
    budget_calls = [c for c in broadcast.call_args_list if c.args[0] == "budget_exceeded"]
    assert len(budget_calls) == 1
    payload = budget_calls[0].args[1]
    assert payload["task_id"] == "t-broadcast"
    assert payload["error_type"] == "TaskBudgetExceededError"


@pytest.mark.asyncio
async def test_task_cleaned_up_even_on_unexpected_exception(orchestrator, tracker):
    plan = _make_plan()
    raising_agent = _stub_agent(raises=RuntimeError("unrelated boom"))
    orchestrator._action_registry[plan.actions[0].action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = raising_agent

    # Non-budget exceptions should propagate, but the task must still be cleaned up
    with pytest.raises(RuntimeError, match=r"unrelated boom"):
        await orchestrator.execute_plan("test", plan, plan_id="t-cleanup")

    assert tracker.get_task_budget("t-cleanup") is None
    assert current_task_id.get() is None

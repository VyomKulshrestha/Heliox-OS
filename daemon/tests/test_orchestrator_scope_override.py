"""Tests that AgentOrchestrator.execute_plan threads scope_override through
to the specialist agent's handle_task() unchanged, and that omitting it
(existing callers) preserves today's behavior (None)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentRole
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.security.gateway import TaskScopeOverride


def _make_action(action_type=ActionType.SYSTEM_INFO, **kwargs):
    from pilot.actions import Action

    return Action(action_type=action_type, parameters={}, **kwargs)


def _make_plan():
    action = _make_action()
    return ActionPlan(actions=[action], explanation="test", raw_input="test input")


def _stub_agent(role=AgentRole.SYSTEM):
    agent = MagicMock()
    agent.role = role
    agent.get_capabilities = MagicMock(return_value=[])
    agent.attach_orchestrator = MagicMock()
    agent.handle_task = AsyncMock(return_value=[])
    return agent


@pytest.fixture
async def orchestrator():
    o = AgentOrchestrator(model_router=MagicMock())
    yield o
    await o.stop()


@pytest.mark.asyncio
async def test_scope_override_forwarded_to_specialist(orchestrator):
    plan = _make_plan()
    agent = _stub_agent()
    agent.handle_task.return_value = [ActionResult(action=plan.actions[0], success=True, output="ok")]
    orchestrator._action_registry[plan.actions[0].action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = agent

    override = TaskScopeOverride(max_tier={"shell": 0})
    await orchestrator.execute_plan("test", plan, scope_override=override)

    agent.handle_task.assert_awaited_once()
    _, kwargs = agent.handle_task.call_args
    assert kwargs["scope_override"] is override


@pytest.mark.asyncio
async def test_scope_override_defaults_to_none(orchestrator):
    plan = _make_plan()
    agent = _stub_agent()
    agent.handle_task.return_value = [ActionResult(action=plan.actions[0], success=True, output="ok")]
    orchestrator._action_registry[plan.actions[0].action_type] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = agent

    await orchestrator.execute_plan("test", plan)

    agent.handle_task.assert_awaited_once()
    _, kwargs = agent.handle_task.call_args
    assert kwargs["scope_override"] is None

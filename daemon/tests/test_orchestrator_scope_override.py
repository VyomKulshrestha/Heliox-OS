"""Tests that AgentOrchestrator.execute_plan threads scope_override through
to the specialist agent's handle_task() unchanged, and that omitting it
(existing callers) preserves today's behavior (None)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, CalendarParams, CodeExecParams
from pilot.agents.base_agent import AgentRole, BaseAgent
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.config import PilotConfig
from pilot.intelligence.experience import get_experience_context
from pilot.security.gateway import AgentGateway, InvocationSource, TaskScopeOverride
from pilot.security.permissions import PermissionChecker


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


@pytest.mark.asyncio
async def test_scope_override_blocks_direct_protocol_specialist_before_dispatch(orchestrator):
    config = PilotConfig()
    gateway = AgentGateway(config, PermissionChecker(config))
    orchestrator.set_agent_gateway(gateway)

    action = Action(
        action_type=ActionType.CALENDAR_DELETE_EVENT,
        parameters=CalendarParams(event_uid="event-123"),
    )
    plan = ActionPlan(actions=[action], explanation="delete event", raw_input="delete event")
    agent = MagicMock(spec=BaseAgent)
    agent.role = AgentRole.CALENDAR
    agent.handle_task = AsyncMock(return_value=[ActionResult(action=action, success=True, output="deleted")])
    agent.get_invocation_source.return_value = InvocationSource.CALENDAR_AGENT
    orchestrator._action_registry[action.action_type] = AgentRole.CALENDAR
    orchestrator._agents[AgentRole.CALENDAR] = agent

    results = await orchestrator.execute_plan(
        "delete event",
        plan,
        scope_override=TaskScopeOverride(max_tier={"other": 2}),
    )

    assert results[0].success is False
    assert "exceeds" in results[0].error
    agent.handle_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_previous_output_context_crosses_specialist_agent_boundary(orchestrator):
    system_action = _make_action(ActionType.SYSTEM_INFO, target="os")
    code_action = Action(
        action_type=ActionType.CODE_EXECUTE,
        target="report",
        parameters=CodeExecParams(
            code="print(PREV_OUTPUT)",
            language="python",
        ),
        use_previous_output=True,
    )
    plan = ActionPlan(
        actions=[system_action, code_action],
        explanation="inspect then report",
        raw_input="report the OS version",
    )
    system_agent = _stub_agent(AgentRole.SYSTEM)
    system_agent.handle_task.return_value = [
        ActionResult(
            action=system_action,
            success=True,
            output="Windows 11 version 10.0.26220",
        ),
    ]
    code_agent = _stub_agent(AgentRole.CODE)
    code_agent.handle_task.return_value = [
        ActionResult(action=code_action, success=True, output="10.0.26220"),
    ]
    orchestrator._action_registry[ActionType.SYSTEM_INFO] = AgentRole.SYSTEM
    orchestrator._action_registry[ActionType.CODE_EXECUTE] = AgentRole.CODE
    orchestrator._agents[AgentRole.SYSTEM] = system_agent
    orchestrator._agents[AgentRole.CODE] = code_agent

    results = await orchestrator.execute_plan("report the OS version", plan)

    assert [result.output for result in results] == [
        "Windows 11 version 10.0.26220",
        "10.0.26220",
    ]
    context = code_agent.handle_task.call_args.kwargs["context"]
    assert context["initial_last_output"] == "Windows 11 version 10.0.26220"
    assert context["initial_largest_output"] == "Windows 11 version 10.0.26220"


@pytest.mark.asyncio
async def test_multi_agent_subplans_inherit_parent_plan_and_original_indices(orchestrator):
    first = _make_action(ActionType.SYSTEM_INFO, target="os")
    second = Action(
        action_type=ActionType.CODE_EXECUTE,
        target="report",
        parameters=CodeExecParams(code="print('ok')", language="python"),
    )
    plan = ActionPlan(actions=[first, second], explanation="multi-agent", raw_input="test")
    seen = []

    def observing_agent(action, role):
        agent = _stub_agent(role)

        async def handle_task(*args, **kwargs):
            seen.append(get_experience_context())
            return [ActionResult(action=action, success=True, output="ok")]

        agent.handle_task = AsyncMock(side_effect=handle_task)
        return agent

    system_agent = observing_agent(first, AgentRole.SYSTEM)
    code_agent = observing_agent(second, AgentRole.CODE)
    orchestrator._action_registry[ActionType.SYSTEM_INFO] = AgentRole.SYSTEM
    orchestrator._action_registry[ActionType.CODE_EXECUTE] = AgentRole.CODE
    orchestrator._agents[AgentRole.SYSTEM] = system_agent
    orchestrator._agents[AgentRole.CODE] = code_agent

    await orchestrator.execute_plan("test", plan, plan_id="parent-plan")

    assert [context.plan_id for context in seen] == ["parent-plan", "parent-plan"]
    assert [context.task_id for context in seen] == ["parent-plan", "parent-plan"]
    assert [context.action_index_offset for context in seen] == [0, 1]

"""Every action type reaches an agent that does that action.

The orchestrator gives a step to whichever registered agent declares its action type and scores best (quality first,
then the narrowest declaration). Which declaring agent wins changes with each agent's record, so every agent that
declares an action must do it, not only today's winner. In a live run "how much battery" went to the monitor agent
(it declared battery_info and memory_usage and answered with its monitor list) and every HTTP request went to the RSS
agent (it declared api_request and answered with its keyword menu).

The contract, over the whole catalog:
- each action type is declared by at least one agent the daemon registers;
- an agent that hands its steps to the shared Executor passes the step on unchanged, and the Executor has a handler;
- an agent that does its steps itself is listed below with exactly the actions its own code dispatches; adding one
  means reading that code first.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.agents.agent_mesh import AgentMesh
from pilot.agents.base_agent import BaseAgent
from pilot.agents.registry import AgentRegistry
from pilot.config import PilotConfig

# Agents whose handle_task does the work itself (no Executor), with the actions their code dispatches on.
DIRECT_AGENTS: dict[str, frozenset[ActionType]] = {
    "CalendarAgent": frozenset(
        {
            ActionType.CALENDAR_PARSE,
            ActionType.CALENDAR_SYNC,
            ActionType.CALENDAR_CREATE_EVENT,
            ActionType.CALENDAR_LIST_EVENTS,
            ActionType.CALENDAR_DELETE_EVENT,
        }
    ),
    "EmailAgent": frozenset(
        {
            ActionType.EMAIL_FETCH,
            ActionType.EMAIL_SUMMARIZE,
            ActionType.EMAIL_REPLY,
            ActionType.API_SEND_EMAIL,
            ActionType.CALENDAR_FETCH,
            ActionType.CALENDAR_RECONCILE,
        }
    ),
    "SemanticSearchAgent": frozenset({ActionType.WORKSPACE_INDEX, ActionType.WORKSPACE_SEARCH}),
    "SshAgent": frozenset({ActionType.SSH_COMMAND, ActionType.SSH_SCRIPT}),
}

# Agents that run no plan step at all (background work and messages only).
STEPLESS_AGENTS = frozenset({"MonitorAgent", "RssAgent"})

# Every agent the daemon registers through discovery. A module that fails to import is skipped silently by discovery,
# so the list is pinned here: a missing agent fails this test rather than shrinking the contract.
EXPECTED_AGENTS = frozenset(
    {
        "CalendarAgent",
        "CodeAgent",
        "CommunicationAgent",
        "DesktopAutomationAgent",
        "EmailAgent",
        "FileOperationsAgent",
        "ForensicsAgent",
        "GitAgent",
        "IntegrationAgent",
        "MonitorAgent",
        "NetworkAgent",
        "PackageManagementAgent",
        "PluginRuntimeAgent",
        "RssAgent",
        "SemanticSearchAgent",
        "ServiceManagementAgent",
        "SshAgent",
        "SystemAgent",
        "VisionAgent",
        "WebAgent",
        "WorkflowAutomationAgent",
    }
)


class _RecordingExecutor:
    """Stands in for the shared Executor: records each plan and runs nothing."""

    def __init__(self) -> None:
        self.plans: list[ActionPlan] = []

    async def execute(self, plan: ActionPlan, **_kwargs: Any) -> list[ActionResult]:
        self.plans.append(plan)
        return [ActionResult(action=action, success=True, output="recorded") for action in plan.actions]


def _build_agents(executor: Any) -> dict[str, BaseAgent]:
    """Build every discovered agent the way the daemon does (auto_register_all_agents), with fakes."""
    AgentRegistry.discover_agents()
    offered = {
        "executor": executor,
        "background_manager": MagicMock(name="background_manager"),
        "model_router": MagicMock(name="model_router"),
        "config": PilotConfig(),
        "vault": MagicMock(name="vault"),
        "memory": MagicMock(name="memory"),
    }
    agents: dict[str, BaseAgent] = {}
    for name, agent_class in AgentRegistry.get_all_agents().items():
        parameters = inspect.signature(agent_class.__init__).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            kwargs = offered
        else:
            kwargs = {key: value for key, value in offered.items() if key in parameters}
        agents[name] = agent_class(**kwargs)
    return agents


def _step(action_type: ActionType) -> Action:
    return Action(action_type=action_type, target="routing-contract", parameters=EmptyParams())


async def _forwards_unchanged(agent: BaseAgent, executor: _RecordingExecutor, action_type: ActionType) -> bool:
    """True when the agent hands this one step to the Executor as it is."""
    step = _step(action_type)
    before = len(executor.plans)
    results = await agent.handle_task("routing contract", ActionPlan(actions=[step], explanation="contract"))
    sent = executor.plans[before:]
    return (
        len(sent) == 1
        and [a.action_type for a in sent[0].actions] == [action_type]
        and [r.action.action_type for r in results] == [action_type]
    )


@pytest.fixture
def executor_and_agents():
    executor = _RecordingExecutor()
    return executor, _build_agents(executor)


def test_the_daemon_registers_every_expected_agent(executor_and_agents) -> None:
    _, agents = executor_and_agents
    assert set(agents) == EXPECTED_AGENTS


def test_every_action_type_is_declared_by_some_agent(executor_and_agents) -> None:
    _, agents = executor_and_agents
    mesh = AgentMesh()
    for agent in agents.values():
        mesh.register_agent(agent)
    unrouted = [at.value for at in ActionType if mesh.registered_provider(at) is None]
    assert unrouted == []


async def test_every_declaring_agent_does_the_action(executor_and_agents, default_config, tmp_path) -> None:
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    real = Executor(
        default_config,
        ActionValidator(default_config),
        PermissionChecker(default_config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )
    executor, agents = executor_and_agents
    wrong: list[str] = []
    for name, agent in sorted(agents.items()):
        declared = {cap.action_type for cap in agent.get_capabilities()}
        if name in STEPLESS_AGENTS:
            if declared:
                wrong.append(f"{name} runs no step but declares {sorted(a.value for a in declared)}")
            continue
        if name in DIRECT_AGENTS:
            if declared != DIRECT_AGENTS[name]:
                wrong.append(f"{name} declares {sorted(a.value for a in declared ^ DIRECT_AGENTS[name])} differently")
            continue
        for action_type in sorted(declared, key=lambda item: item.value):
            if not await _forwards_unchanged(agent, executor, action_type):
                wrong.append(f"{name} declares {action_type.value} but does not pass it to the Executor")
            elif action_type not in real._dispatch_table:
                wrong.append(f"{name} passes {action_type.value} to an Executor with no handler for it")
    assert wrong == []


async def test_the_check_catches_an_agent_that_answers_something_else() -> None:
    """The old monitor agent: it declared battery_info and answered with its monitor list."""
    from pilot.agents.base_agent import AgentCapability, AgentRole

    class _AnswersSomethingElse(BaseAgent):
        def __init__(self) -> None:
            super().__init__(role=AgentRole.MONITOR)

        def get_capabilities(self) -> list[AgentCapability]:
            return [AgentCapability(action_type=ActionType.BATTERY_INFO, description="battery")]

        def get_system_prompt(self) -> str:
            return ""

        def can_handle(self, action_type: ActionType) -> bool:
            return action_type == ActionType.BATTERY_INFO

        async def handle_task(self, user_input, plan, context=None, scope_override=None):
            return [ActionResult(action=plan.actions[0], success=True, output="Active monitors: []")]

    assert await _forwards_unchanged(_AnswersSomethingElse(), _RecordingExecutor(), ActionType.BATTERY_INFO) is False


@pytest.mark.parametrize(
    "action_type",
    [at for at in ActionType if not any(at in actions for actions in DIRECT_AGENTS.values())],
    ids=lambda at: at.value,
)
async def test_the_orchestrator_sends_each_step_to_an_agent_that_runs_it(action_type: ActionType) -> None:
    """End to end through the orchestrator's own choice: the step reaches the Executor and its result comes back."""
    from pilot.agents.orchestrator import AgentOrchestrator

    executor = _RecordingExecutor()
    orchestrator = AgentOrchestrator(MagicMock(name="model_router"), agent_mesh=AgentMesh())
    try:
        for agent in _build_agents(executor).values():
            orchestrator.register_agent(agent)
        step = _step(action_type)
        results = await orchestrator.execute_plan("routing contract", ActionPlan(actions=[step], explanation="c"))
    finally:
        await orchestrator.stop()
    assert [plan.actions[0].action_type for plan in executor.plans] == [action_type]
    assert [(r.action.action_type, r.success, r.output) for r in results] == [(action_type, True, "recorded")]


async def test_battery_and_memory_no_longer_return_the_monitor_list() -> None:
    from pilot.agents.monitor_agent import MonitorAgent

    agent = MonitorAgent(MagicMock(), MagicMock())
    assert agent.get_capabilities() == []
    results = await agent.handle_task("how much battery", ActionPlan(actions=[_step(ActionType.BATTERY_INFO)]))
    assert [(r.success, r.error) for r in results] == [(False, "The monitor agent does not run battery_info.")]

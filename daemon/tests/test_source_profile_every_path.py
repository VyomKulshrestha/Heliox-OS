"""Every request and job runs under the gateway profile of the source it came from, on every path, to the end.

The gateway checks each step under the profile of whoever runs it: the specialist the orchestrator gives it to, or the
caller of the executor. The source of the request (typed, spoken, MCP, a background job of any kind, a voice or gesture
workflow) narrows that profile wherever its plan runs: the orchestrator, the plain executor, a repair round, a resumed
or later-approved plan. A plan with no source is refused outright. These tests drive the real autonomous loop, the
real orchestrator and the server's resume code with fakes: the executor fake makes the same gateway check the real
Executor.execute makes first, then records instead of acting. Nothing touches the system.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams, VerificationResult
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStep
from pilot.agents.base_agent import AgentRole
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.config import PilotConfig
from pilot.security import gateway as gateway_module
from pilot.security import source_scope
from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, AgentGateway, InvocationSource, TaskScopeOverride
from pilot.security.permissions import PermissionChecker
from pilot.workflows.checkpoints import POLICY_DENIED_STATUS, WorkflowCheckpointStore

GATEWAY_DENIED = "gateway_denied"

# Refused by every source's profile except the typed one (deny lists or the system-control ceiling), allowed for the
# system agent, which carries it on the orchestrator path. Each test checks that before relying on it.
REFUSED = ActionType.POWER_SHUTDOWN
# Allowed for every source: the paths below still run what a source may do.
ALLOWED = ActionType.NOTIFY

# Every kind of background job by the source it is submitted with (server.py's submit calls and autonomous_submit).
JOB_SOURCES = [
    "text",  # autonomous_submit from the window
    "trigger",  # a trigger's stored words
    "phone",  # a full phone request
    "proactive",  # an accepted suggestion
    "recorded_workflow",  # a recording replayed now
    "recorded_workflow_schedule",  # a recording replayed on its schedule
    "scheduled",  # a scheduled run (its own approved list too)
    "neural",  # a staged neural selection
    "mcp",
    "voice",
]
PATHS = ["orchestrator", "executor", "dry_run"]


def _action(action_type: ActionType, target: str = "x") -> Action:
    if action_type == ActionType.NOTIFY:
        return Action(action_type=action_type, target=target, parameters=NotifyParams(summary=target, body=target))
    return Action(action_type=action_type, target=target, parameters={})


def _plan(*types: ActionType) -> ActionPlan:
    return ActionPlan(actions=[_action(t, f"step{i}") for i, t in enumerate(types)], explanation="p", raw_input="do it")


def _gateway(config: PilotConfig | None = None) -> AgentGateway:
    config = config or PilotConfig()
    gateway = AgentGateway(config, PermissionChecker(config))
    gateway._maybe_run_critic = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return gateway


class GatewayExecutor:
    """The executor's ingress with fakes: the gateway check Executor.execute makes before any step, then a record."""

    def __init__(self, gateway: AgentGateway) -> None:
        self._gateway = gateway
        self._config = gateway._config
        self.ran: list[str] = []
        self.refused = 0
        self.calls: list[tuple[InvocationSource, TaskScopeOverride | None]] = []

    async def execute(
        self,
        plan: ActionPlan,
        *,
        invocation_source: InvocationSource | None = None,
        scope_override: TaskScopeOverride | None = None,
        **kwargs: Any,
    ) -> list[ActionResult]:
        source = invocation_source or InvocationSource.UNKNOWN
        self.calls.append((source, scope_override))
        decision = await self._gateway.authorize(plan, source, scope_override=scope_override)
        if not decision.allowed:
            self.refused += 1
            return [
                ActionResult(
                    action=action,
                    success=False,
                    error=f"Gateway denied: {'; '.join(decision.reasons)}",
                    executed=False,
                    skip_reason="gateway_denied",
                )
                for action in plan.actions
            ]
        results = []
        for action in plan.actions:
            if kwargs.get("on_action_start"):
                await kwargs["on_action_start"](action)
            self.ran.append(action.target)
            result = ActionResult(action=action, success=True, output=action.target)
            if kwargs.get("on_action_complete"):
                await kwargs["on_action_complete"](result)
            results.append(result)
        return results


class Specialist:
    """A specialist that runs its steps through the shared executor under its own source, as every Executor-backed
    agent does (``handle_task``)."""

    def __init__(self, role: AgentRole, executor: GatewayExecutor) -> None:
        self.role = role
        self._executor = executor

    def get_invocation_source(self) -> InvocationSource:
        return InvocationSource(self.role.value)

    async def handle_task(self, user_input, plan, context=None, scope_override=None):
        return await self._executor.execute(
            plan, invocation_source=self.get_invocation_source(), scope_override=scope_override
        )


class ProtocolSpecialist:
    """A specialist that does its work without the shared executor (ssh, calendar, monitor...): the orchestrator checks
    it at the gateway itself."""

    def __init__(self, role: AgentRole) -> None:
        self.role = role
        self.ran: list[str] = []

    def get_invocation_source(self) -> InvocationSource:
        return InvocationSource(self.role.value)

    async def handle_task(self, user_input, plan, context=None, scope_override=None):
        self.ran.extend(action.target for action in plan.actions)
        return [ActionResult(action=action, success=True, output="ok") for action in plan.actions]


@pytest.fixture
async def orchestrator():
    made: list[AgentOrchestrator] = []

    def build(
        gateway: AgentGateway, executor: GatewayExecutor, role: AgentRole = AgentRole.SYSTEM
    ) -> AgentOrchestrator:
        orchestrator = AgentOrchestrator(model_router=MagicMock())
        orchestrator.set_agent_gateway(gateway)
        orchestrator._agents[role] = Specialist(role, executor)
        for action_type in (REFUSED, ALLOWED):
            orchestrator._action_registry[action_type] = role
        made.append(orchestrator)
        return orchestrator

    yield build
    for orchestrator in made:
        await orchestrator.stop()


def _autonomous(
    executor: GatewayExecutor, plans: list[ActionPlan], orchestrator: AgentOrchestrator | None = None
) -> AutonomousExecutor:
    planner = SimpleNamespace(plan=AsyncMock(side_effect=[*plans] + [plans[-1]] * 4))
    verifier = SimpleNamespace(
        verify=AsyncMock(
            side_effect=lambda plan, results: VerificationResult(
                passed=all(r.success for r in results),
                details=["checked"],
                failed_actions=[i for i, r in enumerate(results) if not r.success],
            )
        )
    )
    autonomous = AutonomousExecutor(
        planner=planner, executor=executor, verifier=verifier, decomposer=MagicMock(), orchestrator=orchestrator
    )
    # The person approves every card: approval is not what refuses these steps.
    autonomous.set_approval_handler(AsyncMock(return_value=True))
    return autonomous


async def _run_job(autonomous: AutonomousExecutor, job: AutonomousJob, source: InvocationSource) -> JobStep:
    step = JobStep(index=0, title="Execute", description=job.goal, status="running")
    await autonomous._execute_goal_loop(job, step, job.goal, invocation_source=source)
    return step


def _refused_by(source: InvocationSource, override: TaskScopeOverride | None = None) -> bool:
    return bool(source_scope.refusals(_gateway(), [_action(REFUSED)], source, override))


def test_the_refused_step_is_refused_by_every_source_but_the_typed_one():
    assert not _refused_by(InvocationSource.INTERACTIVE)
    assert not _refused_by(InvocationSource.SYSTEM_AGENT)
    for source in (
        InvocationSource.AUTONOMOUS,
        InvocationSource.VOICE,
        InvocationSource.GESTURE,
        InvocationSource.MCP,
        InvocationSource.NEURAL,
        InvocationSource.SELF_HEALING,
        InvocationSource.UNKNOWN,
    ):
        assert _refused_by(source), source


# ── Background jobs: every job source, every path ──


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("job_source", JOB_SOURCES)
async def test_a_background_job_runs_under_the_autonomous_profile_on_every_path(orchestrator, job_source, path):
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor) if path == "orchestrator" else None
    autonomous = _autonomous(executor, [_plan(REFUSED)], orch)
    job = AutonomousJob(goal="do it", source=job_source, dry_run=path == "dry_run")
    if job_source == "scheduled":
        job.allowed_actions = [REFUSED.value, ALLOWED.value]  # on its approved list, and still refused by the profile

    step = await _run_job(autonomous, job, InvocationSource.AUTONOMOUS)

    assert executor.ran == [] and step.status == "failed"
    assert executor.refused == len(executor.calls) > 0  # every attempt reached the gateway and was refused
    assert all(source != InvocationSource.INTERACTIVE for source, _ in executor.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("job_source", ["trigger", "phone", "neural", "voice"])
async def test_what_a_background_job_may_do_still_runs(orchestrator, job_source, path):
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor) if path == "orchestrator" else None
    autonomous = _autonomous(executor, [_plan(ALLOWED)], orch)
    job = AutonomousJob(goal="do it", source=job_source, dry_run=path == "dry_run")

    step = await _run_job(autonomous, job, InvocationSource.AUTONOMOUS)

    assert executor.ran == ["step0"] and step.status == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_a_jobs_own_override_narrows_it_on_every_path(orchestrator, path):
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor) if path == "orchestrator" else None
    autonomous = _autonomous(executor, [_plan(ALLOWED)], orch)
    job = AutonomousJob(
        goal="do it", source="text", scope_override=TaskScopeOverride(deny_action_types=[ALLOWED.value])
    )
    job.dry_run = path == "dry_run"

    step = await _run_job(autonomous, job, InvocationSource.AUTONOMOUS)

    assert executor.ran == [] and step.status == "failed" and executor.refused == len(executor.calls) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("job_source", ["neural", "mcp", "voice"])
async def test_a_job_from_a_narrower_source_runs_under_that_sources_profile_too(orchestrator, job_source, path):
    # The autonomous profile lets a file write and a shell command through (with approval); the neural profile
    # refuses the write and the MCP and voice profiles the shell command, and a job from there keeps them out.
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor) if path == "orchestrator" else None
    narrower = ActionType.FILE_WRITE if job_source == "neural" else ActionType.SHELL_COMMAND
    assert not source_scope.refusals(gateway, [_action(narrower)], InvocationSource.AUTONOMOUS, None)
    if path == "orchestrator":
        orch._action_registry[narrower] = AgentRole.SYSTEM
    autonomous = _autonomous(executor, [_plan(narrower)], orch)
    job = AutonomousJob(goal="do it", source=job_source, dry_run=path == "dry_run")

    step = await _run_job(autonomous, job, InvocationSource.AUTONOMOUS)

    assert executor.ran == [] and step.status == "failed" and executor.refused == len(executor.calls) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PATHS)
async def test_a_repair_round_runs_under_the_same_profile(orchestrator, path):
    """Round one fails at its check; the round that repairs it plans the refused step, which is refused."""
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor) if path == "orchestrator" else None
    autonomous = _autonomous(executor, [_plan(ALLOWED), _plan(REFUSED)], orch)
    checks = iter([VerificationResult(passed=False, details=["the note did not show"], failed_actions=[0])])
    original = autonomous._verifier.verify.side_effect
    autonomous._verifier.verify.side_effect = lambda plan, results: next(checks, None) or original(plan, results)
    job = AutonomousJob(goal="do it", source="trigger", dry_run=path == "dry_run")

    step = await _run_job(autonomous, job, InvocationSource.AUTONOMOUS)

    assert executor.ran == ["step0"]  # round one ran; the repair's refused step did not
    assert step.status == "failed" and executor.refused == len(executor.calls) - 1 > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["orchestrator", "executor"])
@pytest.mark.parametrize("source", [InvocationSource.VOICE, InvocationSource.GESTURE])
async def test_a_voice_or_gesture_workflow_runs_under_its_profile_on_every_path(orchestrator, source, path):
    """``execute_goal`` is how a workflow runs, whether it was spoken or submitted over RPC with no override."""
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor) if path == "orchestrator" else None
    autonomous = _autonomous(executor, [_plan(REFUSED)], orch)

    step = await autonomous.execute_goal("do it", invocation_source=source)

    assert executor.ran == [] and step.status == "failed" and executor.refused == len(executor.calls) > 0
    assert {called for called, _ in executor.calls} <= {source, InvocationSource.SYSTEM_AGENT}


# ── A plan with no source ──


@pytest.mark.asyncio
async def test_the_orchestrator_refuses_a_plan_that_names_no_source(orchestrator):
    gateway = _gateway()
    executor = GatewayExecutor(gateway)
    orch = orchestrator(gateway, executor)

    results = await orch.execute_plan("do it", _plan(ALLOWED))

    assert executor.ran == [] and results[0].success is False and "Gateway denied" in results[0].error


@pytest.mark.asyncio
async def test_the_executor_refuses_a_plan_that_names_no_source(tmp_path, monkeypatch):
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.validator import ActionValidator

    config = PilotConfig()
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
        gateway=_gateway(config),
    )
    dispatched = AsyncMock(side_effect=AssertionError("a step was dispatched"))
    monkeypatch.setattr(executor, "_execute_single", dispatched)

    results = await executor.execute(_plan(ALLOWED))

    assert results[0].success is False and results[0].skip_reason == "gateway_denied"
    dispatched.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_protocol_specialist_is_checked_under_the_plans_source_too():
    gateway = _gateway()
    orch = AgentOrchestrator(model_router=MagicMock())
    try:
        orch.set_agent_gateway(gateway)
        calendar = ProtocolSpecialist(AgentRole.CALENDAR)
        orch._agents[AgentRole.CALENDAR] = calendar
        orch._action_registry[ActionType.CALENDAR_DELETE_EVENT] = AgentRole.CALENDAR
        plan = _plan(ActionType.CALENDAR_DELETE_EVENT)  # DESTRUCTIVE: the calendar agent may, a voice request may not

        voice = await orch.execute_plan("do it", plan, source=InvocationSource.VOICE)
        assert voice[0].success is False and calendar.ran == []
        typed = await orch.execute_plan("do it", plan, source=InvocationSource.INTERACTIVE)
        assert typed[0].success is True and calendar.ran == ["step0"]
    finally:
        await orch.stop()


# ── Stored plans: resume, approval later, legacy ──


class Person:
    def __init__(self) -> None:
        self.shown: list[list[str]] = []

    async def __call__(self, plan_id, plan, ws, **kwargs):
        self.shown.append([action.target for action in plan.actions])
        required = set(range(len(plan.actions)))
        return True, required, required


def _resume_server(checkpoints: WorkflowCheckpointStore, gateway: AgentGateway, orchestrator=None):
    from pilot.server import PilotServer

    server = PilotServer(gateway._config)
    server.config.security.dry_run = False
    server._checkpoint_store = checkpoints
    server._durable_tasks = None
    server._agent_gateway = gateway
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    person = Person()
    server._wait_for_confirmation = person  # type: ignore[method-assign]
    executor = GatewayExecutor(gateway)
    server._executor = executor
    server._orchestrator = orchestrator
    return server, executor, person


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


STORED_SOURCES = ["voice", "mcp", "gesture", "autonomous", ""]  # "": stored by an older version


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["orchestrator", "executor"])
@pytest.mark.parametrize("stored", STORED_SOURCES)
@pytest.mark.parametrize("approval", ["approved", "pending"])
async def test_a_resumed_plan_runs_under_the_source_it_was_made_under(tmp_path, orchestrator, stored, path, approval):
    gateway = _gateway()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    server, executor, person = _resume_server(checkpoints, gateway)
    if path == "orchestrator":
        server._orchestrator = orchestrator(gateway, executor)
    plan = _plan(ALLOWED, REFUSED)
    await checkpoints.start_plan("plan-1", "do it", plan, source=stored)
    await checkpoints.record_result("plan-1", ActionResult(action=plan.actions[0], success=True, output="a"))
    if approval == "approved":
        await checkpoints.mark_approval("plan-1", "approved")

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())
    again = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert executor.ran == [] and person.shown == []  # refused before any card or step
    assert resumed["status"] == "error" and resumed["denied_kind"] == GATEWAY_DENIED
    assert (await checkpoints.get("plan-1")).status == POLICY_DENIED_STATUS
    assert again["status"] == "error" and again["denied_kind"] == GATEWAY_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["orchestrator", "executor"])
async def test_a_resumed_typed_plan_still_runs(tmp_path, orchestrator, path):
    gateway = _gateway()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    server, executor, _person = _resume_server(checkpoints, gateway)
    if path == "orchestrator":
        server._orchestrator = orchestrator(gateway, executor)
    plan = _plan(ALLOWED, ALLOWED)
    await checkpoints.start_plan("plan-1", "do it", plan, source="interactive")
    await checkpoints.mark_approval("plan-1", "approved")
    await checkpoints.record_result("plan-1", ActionResult(action=plan.actions[0], success=True, output="a"))

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert resumed["status"] == "success" and executor.ran == ["step1"]
    sources = {source for source, _ in executor.calls}
    assert sources == ({InvocationSource.SYSTEM_AGENT} if path == "orchestrator" else {InvocationSource.INTERACTIVE})


@pytest.mark.asyncio
async def test_a_resumed_plan_keeps_the_override_it_was_made_under(tmp_path):
    gateway = _gateway()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    server, executor, _person = _resume_server(checkpoints, gateway)
    override = TaskScopeOverride(deny_action_types=[ALLOWED.value]).model_dump(mode="json")
    await checkpoints.start_plan("plan-1", "do it", _plan(ALLOWED), source="interactive", scope_override=override)
    await checkpoints.mark_approval("plan-1", "approved")

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert executor.ran == [] and resumed["denied_kind"] == GATEWAY_DENIED


@pytest.mark.asyncio
async def test_a_plan_stored_without_a_source_resumes_under_the_narrowest_that_could_have_made_it(tmp_path):
    config = PilotConfig()
    legacy, scope = source_scope.legacy_scope(config)
    gateway = _gateway(config)
    # Narrower than each source whose plans are stored, the typed one included.
    for source in source_scope.STORED_PLAN_SOURCES:
        profile = gateway_module.profile_for(config, source)
        for family, ceiling in scope.max_tier.items():
            assert ceiling <= profile.max_tier[family]
        assert set(profile.deny_action_types) <= set(scope.deny_action_types)
    # A shell step a typed or spoken plan could hold is refused; a note still runs.
    assert source_scope.refusals(gateway, [_action(ActionType.SHELL_COMMAND)], legacy, scope)
    assert not source_scope.refusals(gateway, [_action(ALLOWED)], legacy, scope)


@pytest.mark.asyncio
async def test_a_durable_task_approved_later_resumes_under_its_recorded_source(tmp_path):
    """resume_task (the window's Continue, or an approval answered after a restart) runs the plan under the source the
    task was accepted under: the durable record's when the checkpoint has none."""
    from pilot.workflows.durable_tasks import DurableTaskStore, TaskStatus

    gateway = _gateway()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    try:
        server, executor, _person = _resume_server(checkpoints, gateway)
        server._durable_tasks = durable
        created = await durable.create_task(session_id="s", user_input="do it", source="mcp", user_id="mcp-local")
        task = created.task
        await durable.transition(task.task_id, TaskStatus.PLANNING, reason="test")
        await durable.transition(task.task_id, TaskStatus.EXECUTING, reason="test", plan_id="plan-1")
        await checkpoints.start_plan("plan-1", "do it", _plan(REFUSED))  # a checkpoint without its source
        await checkpoints.mark_approval("plan-1", "approved")

        response = await server._handle_resume_task({"resume_token": created.resume_token}, _ws())

        assert executor.ran == [] and response["denied_kind"] == GATEWAY_DENIED
        assert await durable.scope(task.task_id) == ("mcp", None)
    finally:
        await durable.close()


@pytest.mark.asyncio
async def test_a_policy_refusal_while_running_keeps_the_plan_refused(tmp_path, orchestrator):
    """A specialist's own ceiling can refuse a step the source allows: the plan is marked refused and never resumed."""
    gateway = _gateway()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    server, executor, _person = _resume_server(checkpoints, gateway)
    server._orchestrator = orchestrator(gateway, executor, role=AgentRole.FORENSICS)  # read-only specialist
    await checkpoints.start_plan("plan-1", "do it", _plan(ALLOWED), source="interactive")
    await checkpoints.mark_approval("plan-1", "approved")

    await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())
    again = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert executor.ran == []
    assert (await checkpoints.get("plan-1")).status == POLICY_DENIED_STATUS
    assert again["denied_kind"] == GATEWAY_DENIED


# ── Request sources by name ──


def test_a_request_source_is_named_or_refused():
    assert source_scope.interaction_source(None) == InvocationSource.INTERACTIVE  # the window sends none
    assert source_scope.interaction_source("text") == InvocationSource.INTERACTIVE
    assert source_scope.interaction_source("voice") == InvocationSource.VOICE
    assert source_scope.interaction_source("mcp") == InvocationSource.MCP
    # A name nothing maps (a new entry that did not say which source it is, or "resume" without its stored source)
    # is refused outright.
    assert source_scope.interaction_source("resume") == InvocationSource.UNKNOWN
    assert source_scope.interaction_source("phone") == InvocationSource.UNKNOWN


def test_a_resumed_request_without_its_stored_scope_is_refused():
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    source, scope = server._request_scope({"source": "resume"}, "resume")
    assert source == InvocationSource.UNKNOWN
    stored_source, stored = server._request_scope({"_stored_scope": ("voice", None)}, "resume")
    assert stored_source == InvocationSource.VOICE
    assert stored.max_tier == DEFAULT_SOURCE_PROFILES["voice"].max_tier


# ── Every profile is applied by an entry ──


@pytest.mark.asyncio
async def test_every_gateway_profile_is_consulted_by_an_entry(tmp_path, orchestrator, monkeypatch):
    """Walk the gateway's profiles: each is applied by at least one entry driven here. A profile nothing applies is a
    fault (its rules would hold nowhere)."""
    consulted: set[str] = set()
    real = gateway_module.profile_for

    def recording(config, source):
        consulted.add(source.value)
        return real(config, source)

    monkeypatch.setattr(gateway_module, "profile_for", recording)
    monkeypatch.setattr(source_scope, "profile_for", recording)
    gateway = _gateway()
    executor = GatewayExecutor(gateway)

    # Typed, spoken and MCP requests (the request pipeline's scope), and a resumed one.
    from pilot.server import PilotServer

    server = PilotServer(gateway._config)
    for name in ("text", "voice", "mcp"):
        server._execution_scope_for_source(name)
    # Background jobs and workflows.
    for job_source, source in (("trigger", InvocationSource.AUTONOMOUS), ("neural", InvocationSource.AUTONOMOUS)):
        await _run_job(_autonomous(executor, [_plan(ALLOWED)]), AutonomousJob(goal="g", source=job_source), source)
    for source in (InvocationSource.VOICE, InvocationSource.GESTURE):
        await _autonomous(executor, [_plan(ALLOWED)]).execute_goal("g", invocation_source=source)
    # Threat containment's remedies (it names the self-healing source).
    orch = orchestrator(gateway, executor)
    await orch.execute_plan("g", _plan(ALLOWED), source=InvocationSource.SELF_HEALING)
    # Every specialist (the web agent among them), through the orchestrator, Executor-backed or not.
    for role in AgentRole:
        try:
            InvocationSource(role.value)
        except ValueError:
            continue
        specialist_orch = AgentOrchestrator(model_router=MagicMock())
        try:
            specialist_orch.set_agent_gateway(gateway)
            specialist_orch._agents[role] = (
                Specialist(role, executor) if role != AgentRole.SSH else ProtocolSpecialist(role)
            )
            specialist_orch._action_registry[ALLOWED] = role
            await specialist_orch.execute_plan("g", _plan(ALLOWED), source=InvocationSource.INTERACTIVE)
        finally:
            await specialist_orch.stop()

    assert set(DEFAULT_SOURCE_PROFILES) - consulted == set()


@pytest.mark.asyncio
async def test_a_plan_closed_as_denied_by_policy_is_never_resumed(tmp_path):
    """The request pipeline closes a plan the gateway refused before its card with this status: resume honours it."""
    gateway = _gateway()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    server, executor, person = _resume_server(checkpoints, gateway)
    await checkpoints.start_plan("plan-1", "do it", _plan(ALLOWED), source="interactive")
    await checkpoints.mark_approval("plan-1", "approved")
    await checkpoints.mark_status("plan-1", POLICY_DENIED_STATUS)

    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert executor.ran == [] and person.shown == [] and resumed["denied_kind"] == GATEWAY_DENIED

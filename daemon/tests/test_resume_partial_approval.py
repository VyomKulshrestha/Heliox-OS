"""A resumed durable task runs only the actions its approval checked, also after a restart."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams, VerificationResult
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.durable_tasks import ApprovalStatus, DurableTaskStore, TaskStatus


def _action(target: str) -> Action:
    return Action(
        action_type=ActionType.NOTIFY,
        target=target,
        parameters=NotifyParams(summary=target, body=target),
    )


def _plan(*targets: str) -> ActionPlan:
    return ActionPlan(actions=[_action(target) for target in targets], explanation="test plan", raw_input="do it")


class _Executor:
    """Records which actions ran and the index offset the resume passed."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.offsets: list[int] = []

    async def execute(self, plan: ActionPlan, *, on_action_start, on_action_complete, **kwargs: Any) -> list[Any]:
        self.offsets.append(kwargs.get("action_index_offset", 0))
        results = []
        for action in plan.actions:
            await on_action_start(action)
            self.executed.append(action.target)
            result = ActionResult(action=action, success=True, output=action.target)
            await on_action_complete(result)
            results.append(result)
        return results


@pytest.fixture
async def stores(tmp_path):
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    try:
        yield durable, checkpoints
    finally:
        await durable.close()


def _server(durable: DurableTaskStore, checkpoints: WorkflowCheckpointStore) -> tuple[PilotServer, _Executor]:
    server = PilotServer(PilotConfig())
    server._durable_tasks = durable
    server._checkpoint_store = checkpoints
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    executor = _Executor()
    server._executor = executor
    return server, executor


def _socket() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


async def _task_awaiting_approval(
    durable: DurableTaskStore, checkpoints: WorkflowCheckpointStore, plan: ActionPlan, required: list[int]
) -> str:
    created = await durable.create_task(task_id="task-1", session_id="session-1", user_input="do it")
    await durable.transition("task-1", TaskStatus.PLANNING)
    await durable.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
    await checkpoints.start_plan("plan-1", "do it", plan)
    await durable.create_approval(
        task_id="task-1",
        plan_id="plan-1",
        request={
            "action_indices": required,
            "actions": [{"index": index, "target": plan.actions[index].target} for index in required],
        },
        timeout_seconds=300,
    )
    return created.resume_token


@pytest.mark.asyncio
async def test_an_approval_answered_after_a_restart_resumes_only_the_checked_actions(stores):
    durable, checkpoints = stores
    resume_token = await _task_awaiting_approval(durable, checkpoints, _plan("keep", "delete-folder"), [0, 1])
    await durable.recover_incomplete()  # the daemon restarted while the approval waited

    server, executor = _server(durable, checkpoints)
    ws = _socket()
    waiting = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, ws)
    assert waiting["status"] == "awaiting_approval"

    confirmed = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0]}, ws
    )
    assert confirmed["resume_required"] is True
    assert (await durable.get_approval("plan-1")).approved_indices == [0]

    resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, ws)

    assert resumed["status"] == "success"
    assert executor.executed == ["keep"]
    assert executor.offsets == [0]
    verified_plan = server._verifier.verify.await_args.args[0]
    assert [action.target for action in verified_plan.actions] == ["keep"]
    assert (await durable.get("task-1")).status == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_a_partially_approved_plan_that_crashed_mid_run_never_runs_the_unchecked_action(stores):
    durable, checkpoints = stores
    plan = _plan("a", "delete-folder", "c")
    resume_token = await _task_awaiting_approval(durable, checkpoints, plan, [0, 1, 2])
    await durable.resolve_approval("plan-1", ApprovalStatus.APPROVED, approved_indices=[0, 2])
    await durable.transition("task-1", TaskStatus.EXECUTING, plan_id="plan-1")
    # The approved plan [a, c] ran; "a" completed, then the daemon crashed during "c".
    await checkpoints.record_result("plan-1", ActionResult(action=plan.actions[0], success=True, output="a"))
    await durable.recover_incomplete()

    server, executor = _server(durable, checkpoints)
    resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, _socket())

    assert resumed["status"] == "success"
    assert executor.executed == ["c"]
    # The offset is the position in the approved plan, as when it first ran, so "c" keeps its action key.
    assert executor.offsets == [1]


@pytest.mark.asyncio
async def test_an_approval_that_checked_nothing_is_never_resumed(stores):
    durable, checkpoints = stores
    resume_token = await _task_awaiting_approval(durable, checkpoints, _plan("delete-folder"), [0])
    await durable.resolve_approval("plan-1", ApprovalStatus.APPROVED, approved_indices=[])
    await durable.recover_incomplete()

    server, executor = _server(durable, checkpoints)
    resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, _socket())

    assert resumed["status"] == "cancelled"
    assert resumed["message"] == "Task approval approved none of the actions that needed approval."
    assert executor.executed == []
    assert (await durable.get("task-1")).status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_only_an_approved_row_resumes(stores):
    durable, checkpoints = stores
    resume_token = await _task_awaiting_approval(durable, checkpoints, _plan("delete-folder"), [0])
    await durable.resolve_approval("plan-1", ApprovalStatus.DISCONNECTED, approved_indices=[0])
    await durable.recover_incomplete()

    server, executor = _server(durable, checkpoints)
    resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, _socket())

    assert resumed["status"] == "cancelled"
    assert resumed["message"] == "Task approval was disconnected."
    assert executor.executed == []


@pytest.mark.asyncio
async def test_a_full_approval_resumes_every_action(stores):
    durable, checkpoints = stores
    resume_token = await _task_awaiting_approval(durable, checkpoints, _plan("a", "b"), [1])
    await durable.resolve_approval("plan-1", ApprovalStatus.APPROVED, approved_indices=[1])
    await durable.recover_incomplete()

    server, executor = _server(durable, checkpoints)
    resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, _socket())

    assert resumed["status"] == "success"
    assert executor.executed == ["a", "b"]


@pytest.mark.asyncio
async def test_an_approval_that_does_not_match_the_checkpoint_resumes_nothing(stores):
    durable, checkpoints = stores
    resume_token = await _task_awaiting_approval(durable, checkpoints, _plan("a"), [0])
    await durable.recover_incomplete()
    server, executor = _server(durable, checkpoints)
    # The approval names an action the checkpoint does not have.
    server._durable_tasks.get_approval = AsyncMock(
        return_value=SimpleNamespace(
            status=ApprovalStatus.APPROVED, request={"action_indices": [0, 3]}, approved_indices=[0]
        )
    )

    resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, _socket())

    assert resumed["status"] == "error"
    assert executor.executed == []


# ── The execute path ──


def _pipeline_server(
    config: PilotConfig,
    plan: ActionPlan,
    checkpoints: WorkflowCheckpointStore,
    durable: DurableTaskStore | None,
    seen: dict[str, Any],
) -> PilotServer:
    server = PilotServer(config)
    server._durable_tasks = durable
    server._checkpoint_store = checkpoints
    server._planner = AsyncMock()
    server._planner.plan.return_value = plan
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = True
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))

    async def approve_first_only(plan_id: str, approval_plan: ActionPlan, ws: Any, **kwargs: Any):
        if durable is not None:
            await durable.create_approval(
                task_id=server._active_task_id,
                plan_id=plan_id,
                request={"action_indices": [0, 1]},
                timeout_seconds=300,
            )
            await durable.resolve_approval(plan_id, ApprovalStatus.APPROVED, approved_indices=[0])
        return True, {0}, {0, 1}

    server._wait_for_confirmation = approve_first_only  # type: ignore[method-assign]

    async def execute(run_plan: ActionPlan, *, plan_id: str, **kwargs: Any) -> list[Any]:
        checkpoint = await checkpoints.get(plan_id)
        seen["plan_id"] = plan_id
        seen["ran"] = [action.target for action in run_plan.actions]
        seen["checkpoint"] = [action.target for action in checkpoint.plan.actions]
        return [ActionResult(action=action, success=True, output=action.target) for action in run_plan.actions]

    server._executor = SimpleNamespace(execute=execute)
    server._orchestrator = None
    return server


@pytest.mark.asyncio
async def test_without_a_durable_approval_the_checkpoint_holds_only_the_approved_actions(tmp_path):
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    config = PilotConfig()
    config.security.dry_run = False
    seen: dict[str, Any] = {}
    server = _pipeline_server(config, _plan("keep", "delete-folder"), checkpoints, None, seen)

    await server._handle_execute({"input": "do it", "session_id": "s-1"}, AsyncMock())

    assert seen["ran"] == ["keep"]
    # A plain resume_plan of this checkpoint can only run what was approved.
    assert seen["checkpoint"] == ["keep"]


@pytest.mark.asyncio
async def test_with_a_durable_approval_the_checkpoint_keeps_the_plan_the_approval_indexes(tmp_path, stores):
    durable, checkpoints = stores
    config = PilotConfig()
    config.security.dry_run = False
    seen: dict[str, Any] = {}
    server = _pipeline_server(config, _plan("keep", "delete-folder"), checkpoints, durable, seen)

    await server._handle_execute({"input": "do it", "session_id": "s-1", "task_id": "task-9"}, AsyncMock())

    assert seen["ran"] == ["keep"]
    assert seen["checkpoint"] == ["keep", "delete-folder"]
    approval = await durable.get_approval(seen["plan_id"])
    assert approval.approved_indices == [0]

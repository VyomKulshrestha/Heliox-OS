"""After the daemon stopped under a task, nothing starts again by itself: the window offers the task and only its
Continue button restarts it. Voice can leave it, never continue it. A reconnect to the same daemon is unchanged."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams, VerificationResult
from pilot.config import PilotConfig
from pilot.security.rpc_identity import RpcClientRole
from pilot.server import PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.durable_tasks import DurableTaskStore, TaskStatus


def _plan(count: int) -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(action_type=ActionType.NOTIFY, target=f"n{i}", parameters=NotifyParams(summary="S", body=f"{i}"))
            for i in range(count)
        ],
        explanation="three notifications",
        raw_input="send three notifications",
    )


@pytest.fixture
async def stores(tmp_path):
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    yield durable, checkpoints
    await durable.close()


def _server(durable, checkpoints, role=RpcClientRole.UI):
    server = PilotServer(PilotConfig())
    server._durable_tasks = durable
    server._checkpoint_store = checkpoints
    server._memory = AsyncMock()  # record_note, clear_task_working, record: all awaited
    executed: list = []

    async def execute(remaining, *, on_action_start, on_action_complete, **_kwargs):
        executed.append(len(remaining.actions))
        results = []
        for action in remaining.actions:
            await on_action_start(action)
            result = ActionResult(action=action, success=True, output="ok")
            await on_action_complete(result)
            results.append(result)
        return results

    server._executor = SimpleNamespace(execute=execute)
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    ws = MagicMock()
    ws.send = AsyncMock()
    server._client_roles[ws] = role
    return server, ws, executed


async def _executing_task(durable, checkpoints, *, done: int, total: int = 3):
    created = await durable.create_task(task_id="t1", session_id="s1", user_input="send three notifications")
    await durable.transition("t1", TaskStatus.PLANNING)
    await durable.transition("t1", TaskStatus.EXECUTING, plan_id="p1")
    plan = _plan(total)
    await checkpoints.start_plan("p1", "send three notifications", plan)
    for action in plan.actions[:done]:
        await checkpoints.record_result("p1", ActionResult(action=action, success=True, output="ok"))
    return created


async def test_after_a_restart_the_task_is_offered_and_nothing_runs(stores):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.recover_incomplete()  # the daemon starts again after dying mid-task
    server, ws, executed = _server(durable, checkpoints)

    offer = await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token}, ws)

    assert offer["status"] == "interrupted" and offer["phase"] == "executing"
    assert (offer["steps_done"], offer["steps_left"], offer["uncertain_steps"]) == (1, 2, 0)
    assert offer["request"] == "send three notifications"
    assert executed == [] and (await durable.get("t1")).status == TaskStatus.INTERRUPTED
    # Asking again (another reconnect) offers it again; it never starts by itself.
    again = await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token}, ws)
    assert again["status"] == "interrupted" and executed == []


async def test_continue_from_the_window_runs_only_what_was_left(stores):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.recover_incomplete()
    server, ws, executed = _server(durable, checkpoints)

    result = await server._handle_resume_task(
        {"task_id": "t1", "resume_token": created.resume_token, "continue": True}, ws
    )

    assert result["status"] == "success" and executed == [2], result


async def test_continue_from_anything_but_the_window_is_refused(stores):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.recover_incomplete()
    server, ws, executed = _server(durable, checkpoints, role=RpcClientRole.MCP_LOCAL)

    refused = await server._handle_resume_task(
        {"task_id": "t1", "resume_token": created.resume_token, "continue": True}, ws
    )

    assert refused["status"] == "error" and "Continue button" in refused["message"]
    assert executed == [] and (await durable.get("t1")).status == TaskStatus.INTERRUPTED


async def test_leave_it_ends_the_task_and_writes_the_activity_log(stores):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.recover_incomplete()
    server, ws, executed = _server(durable, checkpoints)

    left = await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token, "leave": True}, ws)

    assert left["status"] == "cancelled" and executed == []
    assert (await durable.get("t1")).status == TaskStatus.CANCELLED
    server._memory.record_note.assert_awaited_once()
    assert server._memory.record_note.await_args.args[1].startswith("Left as it was")


async def test_a_task_cut_off_while_planning_is_offered_to_plan_again(stores):
    durable, checkpoints = stores
    created = await durable.create_task(task_id="t1", session_id="s1", user_input="tidy my downloads")
    await durable.transition("t1", TaskStatus.PLANNING)
    await durable.recover_incomplete()
    server, ws, executed = _server(durable, checkpoints)

    offer = await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token}, ws)

    assert offer["status"] == "interrupted" and offer["phase"] == "planning"
    assert (offer["steps_done"], offer["steps_left"]) == (0, 0)


async def test_an_action_running_when_the_daemon_died_is_counted_uncertain(stores):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.claim_action(task_id="t1", plan_id="p1", action_key="a2", lease_owner="old-daemon")
    await durable.recover_incomplete()
    server, ws, _executed = _server(durable, checkpoints)

    offer = await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token}, ws)

    assert offer["uncertain_steps"] == 1


async def test_an_interruption_without_a_restart_resumes_as_before(stores):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.transition("t1", TaskStatus.INTERRUPTED, reason="connection lost")  # the same daemon is alive
    server, ws, executed = _server(durable, checkpoints)

    result = await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token}, ws)

    assert result["status"] == "success" and executed == [2], result


async def test_voice_can_leave_an_offered_task_but_never_continue_it(stores, monkeypatch):
    durable, checkpoints = stores
    created = await _executing_task(durable, checkpoints, done=1)
    await durable.recover_incomplete()
    server, ws, executed = _server(durable, checkpoints)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    dispatched: list[str] = []

    async def dispatch(text: str) -> None:
        dispatched.append(text)

    monkeypatch.setattr(server, "_voice_command_dispatch", dispatch)
    await server._handle_resume_task({"task_id": "t1", "resume_token": created.resume_token}, ws)  # offered

    await server._voice_turn("hey heliox continue")
    assert dispatched == ["hey heliox continue"]  # an ordinary request: the task is not restarted
    assert executed == [] and (await durable.get("t1")).status == TaskStatus.INTERRUPTED

    await server._voice_turn("Hey Heliox, leave it.")
    assert (await durable.get("t1")).status == TaskStatus.CANCELLED
    server._broadcast_notification.assert_awaited_once()
    assert server._broadcast_notification.await_args.args[0] == "interrupted_task_left"
    assert dispatched == ["hey heliox continue"]

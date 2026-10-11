"""Speaking to Heliox while it works: what each utterance does, and what Heliox says back."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.agents.autonomous import AutonomousJob, JobStatus, JobStep
from pilot.agents.voice_gesture_workflow import VoiceGestureWorkflowEngine
from pilot.config import PilotConfig
from pilot.security.gateway import InvocationSource
from pilot.server import PendingConfirmation, PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.voice_gesture_workflows import VoiceGestureWorkflowStore, WorkflowState
from tests.test_voice_gesture_workflow_engine import _engine, _StubDecomposer, _StubExecutor, _wait_until_terminal


class _BlockingExecutor(_StubExecutor):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def execute(self, plan, **kwargs):
        self.started.set()
        await asyncio.sleep(30)
        return await super().execute(plan, **kwargs)


async def _running(tmp_path, **options):
    executor = _BlockingExecutor()
    engine = _engine(tmp_path, executor=executor, **options)
    workflow = await engine.start("write a shopping list in Notepad", InvocationSource.VOICE)
    await asyncio.wait_for(executor.started.wait(), timeout=10)
    return engine, workflow


async def _cleanup(engine, workflow):
    await engine.cancel(workflow.workflow_id)


async def test_status_questions_are_answered_without_restarting_the_step(tmp_path):
    engine, workflow = await _running(tmp_path)
    task = engine._active_tasks[workflow.workflow_id]

    outcome = await engine.apply_instruction("voice", "What are you doing?")

    assert outcome.kind == "status"
    assert outcome.message.startswith("I'm working on: write a shopping list in Notepad")
    assert engine._active_tasks[workflow.workflow_id] is task and not task.done()
    await _cleanup(engine, workflow)


async def test_approval_words_are_left_to_the_approval_flow(tmp_path):
    engine, workflow = await _running(tmp_path)

    for word in ("yes", "no", "go ahead", "approve"):
        assert await engine.apply_instruction("voice", word) is None
    assert not engine._active_tasks[workflow.workflow_id].done()
    await _cleanup(engine, workflow)


async def test_a_spoken_addition_amends_the_running_step_instead_of_restarting_it(tmp_path):
    engine, workflow = await _running(tmp_path)
    task = engine._active_tasks[workflow.workflow_id]

    outcome = await engine.apply_instruction("voice", "also save it as shopping.txt")

    assert outcome.kind == "amended"
    assert engine._guidance[workflow.workflow_id] == ["also save it as shopping.txt"]
    assert not task.done()
    await _cleanup(engine, workflow)


async def test_gnomon_decides_between_amending_and_replacing_when_loaded(tmp_path):
    judgements = {"save it on the desktop too": 0.92, "use WordPad instead": 0.03}
    calls = []

    async def judge(goal, utterance, current):
        calls.append((goal, utterance, current))
        return judgements[utterance]

    engine, workflow = await _running(tmp_path, instruction_judge=judge)

    amended = await engine.apply_instruction("voice", "save it on the desktop too")
    assert amended.kind == "amended"
    replaced = await engine.apply_instruction("voice", "use WordPad instead")
    assert replaced.kind == "corrected"
    assert calls[0] == (
        "write a shopping list in Notepad",
        "save it on the desktop too",
        "write a shopping list in Notepad",
    )
    await _cleanup(engine, workflow)


async def test_wait_pauses_at_the_next_step_boundary(tmp_path):
    engine, workflow = await _running(tmp_path)

    outcome = await engine.apply_instruction("voice", "hold on")

    assert outcome.kind == "paused"
    assert engine._pause_requested[workflow.workflow_id] is True
    await _cleanup(engine, workflow)


async def test_a_paused_workflow_does_not_swallow_unrelated_commands(tmp_path):
    engine = _engine(tmp_path, decomposer=_StubDecomposer(subtasks=["step one", "step two"]))
    workflow = await engine.start("multi step", InvocationSource.VOICE)
    await engine.pause(workflow.workflow_id)
    for _ in range(100):
        current = await engine._workflow_store.get(workflow.workflow_id)
        if current.state == WorkflowState.PAUSED.value:
            break
        await asyncio.sleep(0.02)

    assert await engine.apply_instruction("voice", "open the weather app") is None
    resumed = await engine.apply_instruction("voice", "continue")
    assert resumed.kind == "resumed"
    await _wait_until_terminal(engine, workflow.workflow_id)


async def test_a_failed_step_says_why_and_how_to_go_on(tmp_path):
    spoken = []

    async def speech(text, channel, key):
        spoken.append((text, channel))

    adaptive = SimpleNamespace(
        execute_goal=AsyncMock(
            return_value=JobStep(index=0, title="t", description="d", status="failed", error="Notepad did not open")
        )
    )
    engine = VoiceGestureWorkflowEngine(
        planner=None,
        executor=None,
        decomposer=_StubDecomposer(),
        workflow_store=VoiceGestureWorkflowStore(db_file=tmp_path / "w.db"),
        checkpoint_store=WorkflowCheckpointStore(db_file=tmp_path / "c.db"),
        adaptive_executor=adaptive,
    )
    engine.set_speech(speech)
    workflow = await engine.start("open notepad and write a list", InvocationSource.VOICE)
    await engine._active_tasks[workflow.workflow_id]

    assert spoken[-1] == (
        "I couldn't finish that step: Notepad did not open. Say continue to retry, tell me what to change, "
        "or say cancel.",
        "task_failure",
    )
    assert adaptive.execute_goal.await_args.kwargs["guidance"] == []


async def test_multi_step_workflows_announce_each_step(tmp_path):
    spoken = []

    async def speech(text, channel, key):
        spoken.append(text)

    engine = _engine(tmp_path, decomposer=_StubDecomposer(subtasks=["open Notepad", "type the list"]))
    engine.set_speech(speech)
    workflow = await engine.start("open notepad and type the list", InvocationSource.VOICE)
    for _ in range(200):
        current = await engine._workflow_store.get(workflow.workflow_id)
        if current.state in {WorkflowState.SUCCESS.value, WorkflowState.FAILED.value}:
            break
        await asyncio.sleep(0.02)

    assert "Step 1 of 2: open Notepad" in spoken
    assert "Step 2 of 2: type the list" in spoken


def _server() -> PilotServer:
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    return server


async def test_saying_no_denies_a_pending_background_approval():
    server = _server()
    pending = PendingConfirmation(plan_id="autonomous-j1:0:0", event=asyncio.Event(), plan=None)
    server._pending_confirms["autonomous-j1:0:0"] = pending

    await server._voice_command_dispatch("No.")

    assert pending.event.is_set() and pending.confirmed is False
    server._handle_execute.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[1]["message"] == "Okay, I won't run that."


async def test_saying_yes_points_to_the_visible_dialog_for_background_approvals_too():
    server = _server()
    pending = PendingConfirmation(plan_id="scheduled-plan", event=asyncio.Event(), plan=None)
    server._pending_confirms["scheduled-plan"] = pending

    await server._voice_command_dispatch("approve")

    assert not pending.event.is_set()
    payload = server._broadcast_notification.await_args.args[1]
    assert payload["status"] == "approval_required" and payload["plan_id"] == "scheduled-plan"


async def test_stop_with_no_interactive_task_cancels_background_jobs_instead_of_planning():
    server = _server()
    job = AutonomousJob(job_id="bg1", goal="tidy downloads", status=JobStatus.RUNNING)
    server._autonomous = SimpleNamespace(
        _active_tasks={"bg1": object()},
        get_job=lambda job_id: job if job_id == "bg1" else None,
        cancel=AsyncMock(return_value=True),
    )

    await server._voice_command_dispatch("stop")

    server._autonomous.cancel.assert_awaited_once_with("bg1")
    server._handle_execute.assert_not_awaited()
    assert server._broadcast_notification.await_args.args[1]["message"] == "Stopped 1 background task."


async def test_stop_with_nothing_running_is_not_planned_as_a_task():
    server = _server()
    server._autonomous = SimpleNamespace(_active_tasks={}, get_job=lambda job_id: None, cancel=AsyncMock())

    await server._voice_command_dispatch("stop")

    server._handle_execute.assert_not_awaited()
    server._speak_voice_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_workflows_start_with_the_voice_permission_ceiling(tmp_path):
    server = _server()
    started = SimpleNamespace(workflow_id="w1", to_dict=lambda: {"workflow_id": "w1"})
    server._voice_gesture_workflows = SimpleNamespace(
        apply_instruction=AsyncMock(return_value=None),
        handle_control_phrase=AsyncMock(return_value=False),
        start=AsyncMock(return_value=started),
    )

    consumed = await server._voice_workflow_control_dispatch("open Notepad, write a shopping list and save it")

    assert consumed is True
    goal, source, scope = server._voice_gesture_workflows.start.await_args.args
    assert source == InvocationSource.VOICE
    assert scope is not None and scope.max_tier  # the same ceiling the interactive voice path enforces
    payload = server._broadcast_notification.await_args.args[1]
    assert payload["status"] == "submitted" and payload["message"].startswith("I’m on it")

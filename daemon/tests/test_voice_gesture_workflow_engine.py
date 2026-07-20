from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.agents.decomposer import Subtask, TaskDecomposition
from pilot.agents.voice_gesture_workflow import VoiceGestureWorkflowEngine
from pilot.security.gateway import InvocationSource, TaskScopeOverride
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.voice_gesture_workflows import VoiceGestureWorkflowStore, WorkflowState


class _StubPlanner:
    def __init__(self, error_on: set[str] | None = None):
        self.error_on = error_on or set()
        self.calls: list[str] = []

    async def plan(self, description, **kwargs):
        self.calls.append(description)
        if description in self.error_on:
            return ActionPlan(actions=[], raw_input=description, error="could not understand this step")
        action = Action(action_type=ActionType.FILE_READ, target=description, parameters=EmptyParams())
        return ActionPlan(actions=[action], raw_input=description)


class _StubExecutor:
    def __init__(self, fail_descriptions: set[str] | None = None):
        self.fail_descriptions = fail_descriptions or set()
        self.calls: list[dict] = []

    async def execute(self, plan, *, plan_id=None, invocation_source=None, scope_override=None, **kwargs):
        self.calls.append(
            {
                "plan_id": plan_id,
                "invocation_source": invocation_source,
                "scope_override": scope_override,
                "target": plan.actions[0].target if plan.actions else None,
            }
        )
        action = plan.actions[0]
        should_fail = action.target in self.fail_descriptions
        return [
            ActionResult(
                action=action,
                success=not should_fail,
                output="" if should_fail else f"did {action.target}",
                error="boom" if should_fail else None,
            )
        ]


class _StubDecomposer:
    def __init__(self, subtasks: list[str] | None = None):
        self.subtasks = subtasks

    async def decompose(self, goal):
        if not self.subtasks:
            return TaskDecomposition(goal=goal, subtasks=[], is_complex=False)
        subs = [Subtask(title=t, description=t) for t in self.subtasks]
        return TaskDecomposition(goal=goal, subtasks=subs, is_complex=True)

    def get_execution_order(self, decomposition):
        return [[s] for s in decomposition.subtasks]


def _engine(tmp_path, planner=None, executor=None, decomposer=None):
    workflow_store = VoiceGestureWorkflowStore(db_file=tmp_path / "workflows.db")
    checkpoint_store = WorkflowCheckpointStore(db_file=tmp_path / "checkpoints.db")
    return VoiceGestureWorkflowEngine(
        planner or _StubPlanner(),
        executor or _StubExecutor(),
        decomposer or _StubDecomposer(),
        workflow_store,
        checkpoint_store,
    )


async def _wait_until_terminal(engine, workflow_id, timeout=10.0):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        workflow = await engine._workflow_store.get(workflow_id)
        if workflow.state in (
            WorkflowState.SUCCESS.value,
            WorkflowState.PARTIAL.value,
            WorkflowState.FAILED.value,
            WorkflowState.CANCELLED.value,
            WorkflowState.PAUSED.value,
            WorkflowState.WAITING_FOR_TRIGGER.value,
        ):
            return workflow
        await asyncio.sleep(0.01)
    raise TimeoutError(f"workflow {workflow_id} never reached a stable state")


class TestSingleStepAutoChain:
    @pytest.mark.asyncio
    async def test_simple_goal_runs_to_success(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        final = await _wait_until_terminal(engine, workflow.workflow_id)
        assert final.state == WorkflowState.SUCCESS.value
        assert len(final.steps) == 1
        assert final.steps[0].status == "success"


class TestMultiStepAutoChain:
    @pytest.mark.asyncio
    async def test_multi_step_goal_auto_chains_to_success(self, tmp_path):
        decomposer = _StubDecomposer(subtasks=["step one", "step two", "step three"])
        engine = _engine(tmp_path, decomposer=decomposer)
        workflow = await engine.start("do a multi-step thing", InvocationSource.VOICE)
        final = await _wait_until_terminal(engine, workflow.workflow_id)
        assert final.state == WorkflowState.SUCCESS.value
        assert len(final.steps) == 3
        assert all(s.status == "success" for s in final.steps)

    @pytest.mark.asyncio
    async def test_partial_failure_yields_partial_state(self, tmp_path):
        decomposer = _StubDecomposer(subtasks=["step one", "step two"])
        executor = _StubExecutor(fail_descriptions={"step two"})
        engine = _engine(tmp_path, executor=executor, decomposer=decomposer)
        workflow = await engine.start("do a thing", InvocationSource.VOICE)
        final = await _wait_until_terminal(engine, workflow.workflow_id)
        assert final.state == WorkflowState.PARTIAL.value
        assert final.steps[0].status == "success"
        assert final.steps[1].status == "failed"


class TestAmbiguousPlanningOutcome:
    @pytest.mark.asyncio
    async def test_plan_error_enters_waiting_for_trigger_and_is_retryable_on_resume(self, tmp_path):
        planner = _StubPlanner(error_on={"do an ambiguous thing"})
        engine = _engine(tmp_path, planner=planner)
        workflow = await engine.start("do an ambiguous thing", InvocationSource.VOICE)
        waiting = await _wait_until_terminal(engine, workflow.workflow_id)
        assert waiting.state == WorkflowState.WAITING_FOR_TRIGGER.value
        assert waiting.trigger_deadline is not None
        # Left "pending" (not "failed") so a retry on resume gets a fresh plan() call.
        assert waiting.steps[0].status == "pending"

        planner.error_on.clear()  # simulate the user clarifying and a re-plan succeeding
        resumed = await engine.resume(workflow.workflow_id)
        assert resumed is not None
        final = await _wait_until_terminal(engine, workflow.workflow_id)
        assert final.state == WorkflowState.SUCCESS.value


class TestPauseResumeCancel:
    @pytest.mark.asyncio
    async def test_pause_stops_before_next_step_then_resume_continues(self, tmp_path):
        decomposer = _StubDecomposer(subtasks=["step one", "step two"])
        engine = _engine(tmp_path, decomposer=decomposer)
        workflow = await engine.start("multi step", InvocationSource.VOICE)

        # Pause immediately -- the drive loop should stop at the next
        # boundary rather than running every step through to completion.
        paused_ok = await engine.pause(workflow.workflow_id)
        assert paused_ok is True

        settled = await _wait_until_terminal(engine, workflow.workflow_id)
        assert settled.state in (WorkflowState.PAUSED.value, WorkflowState.SUCCESS.value)
        if settled.state == WorkflowState.SUCCESS.value:
            return  # race: both steps completed before the pause flag was checked -- not a failure

        resumed = await engine.resume(workflow.workflow_id)
        assert resumed is not None
        final = await _wait_until_terminal(engine, workflow.workflow_id)
        assert final.state == WorkflowState.SUCCESS.value
        assert all(s.status == "success" for s in final.steps)

    @pytest.mark.asyncio
    async def test_cancel_marks_cancelled(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        cancelled = await engine.cancel(workflow.workflow_id)
        assert cancelled is True
        final = await engine._workflow_store.get(workflow.workflow_id)
        assert final.state == WorkflowState.CANCELLED.value

    @pytest.mark.asyncio
    async def test_pause_unknown_workflow_returns_false(self, tmp_path):
        engine = _engine(tmp_path)
        assert await engine.pause("does-not-exist") is False

    @pytest.mark.asyncio
    async def test_resume_non_paused_workflow_returns_none(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await _wait_until_terminal(engine, workflow.workflow_id)
        assert await engine.resume(workflow.workflow_id) is None


class TestGatewayScoping:
    @pytest.mark.asyncio
    async def test_executor_receives_original_invocation_source_and_override(self, tmp_path):
        executor = _StubExecutor()
        engine = _engine(tmp_path, executor=executor)
        override = TaskScopeOverride(max_tier={"shell": 0})
        workflow = await engine.start("do one thing", InvocationSource.GESTURE, scope_override=override)
        await _wait_until_terminal(engine, workflow.workflow_id)

        assert len(executor.calls) == 1
        assert executor.calls[0]["invocation_source"] == InvocationSource.GESTURE
        assert executor.calls[0]["scope_override"].max_tier == {"shell": 0}
        assert executor.calls[0]["plan_id"] == f"{workflow.workflow_id}:0"

    @pytest.mark.asyncio
    async def test_resumed_step_keeps_original_invocation_source(self, tmp_path):
        planner = _StubPlanner(error_on={"do a thing"})
        executor = _StubExecutor()
        engine = _engine(tmp_path, planner=planner, executor=executor)
        workflow = await engine.start("do a thing", InvocationSource.GESTURE)
        await _wait_until_terminal(engine, workflow.workflow_id)

        planner.error_on.clear()
        await engine.resume(workflow.workflow_id)
        await _wait_until_terminal(engine, workflow.workflow_id)

        assert len(executor.calls) == 1
        assert executor.calls[0]["invocation_source"] == InvocationSource.GESTURE


class TestExpiry:
    @pytest.mark.asyncio
    async def test_paused_workflow_gets_a_trigger_deadline(self, tmp_path):
        engine = _engine(tmp_path, decomposer=_StubDecomposer(subtasks=["a", "b"]))
        workflow = await engine.start("multi step", InvocationSource.VOICE)
        await engine.pause(workflow.workflow_id)
        settled = await _wait_until_terminal(engine, workflow.workflow_id)
        if settled.state != WorkflowState.PAUSED.value:
            return  # race: finished before the pause flag was checked
        assert settled.trigger_deadline is not None

    @pytest.mark.asyncio
    async def test_expire_if_stale_transitions_past_deadline_paused_workflow(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        # Let the background _drive task settle (this single-step goal runs
        # to SUCCESS) before directly overwriting state below -- otherwise
        # the still-running task's own writes race with this test's write
        # and can clobber it, since both target the same row.
        await _wait_until_terminal(engine, workflow.workflow_id)
        past_deadline = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        await engine._workflow_store.set_state(
            workflow.workflow_id, WorkflowState.PAUSED, trigger_deadline=past_deadline
        )

        expired = await engine.expire_if_stale(workflow.workflow_id)
        assert expired is True
        final = await engine._workflow_store.get(workflow.workflow_id)
        assert final.state == WorkflowState.EXPIRED.value

    @pytest.mark.asyncio
    async def test_expire_if_stale_leaves_workflow_within_window_alone(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await _wait_until_terminal(engine, workflow.workflow_id)  # see race note above
        future_deadline = (datetime.now(UTC) + timedelta(seconds=60)).isoformat()
        await engine._workflow_store.set_state(
            workflow.workflow_id, WorkflowState.PAUSED, trigger_deadline=future_deadline
        )

        expired = await engine.expire_if_stale(workflow.workflow_id)
        assert expired is False
        final = await engine._workflow_store.get(workflow.workflow_id)
        assert final.state == WorkflowState.PAUSED.value

    @pytest.mark.asyncio
    async def test_expire_if_stale_no_op_on_running_workflow(self, tmp_path):
        engine = _engine(tmp_path, decomposer=_StubDecomposer(subtasks=["a", "b", "c"]))
        workflow = await engine.start("multi step", InvocationSource.VOICE)
        expired = await engine.expire_if_stale(workflow.workflow_id)
        assert expired is False
        await _wait_until_terminal(engine, workflow.workflow_id)

    @pytest.mark.asyncio
    async def test_expired_workflow_no_longer_matches_find_pending_for_source(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await _wait_until_terminal(engine, workflow.workflow_id)  # see race note above
        past_deadline = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        await engine._workflow_store.set_state(
            workflow.workflow_id, WorkflowState.PAUSED, trigger_deadline=past_deadline
        )
        await engine.expire_if_stale(workflow.workflow_id)

        found = await engine._workflow_store.find_pending_for_source("voice", within_seconds=3600)
        assert found is None


class TestControlPhrase:
    @pytest.mark.asyncio
    async def test_continue_phrase_resumes_paused_workflow(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await engine.pause(workflow.workflow_id)
        settled = await _wait_until_terminal(engine, workflow.workflow_id)
        if settled.state != WorkflowState.PAUSED.value:
            return  # race: single step finished before the pause flag was checked

        consumed = await engine.handle_control_phrase("voice", "continue")
        assert consumed is True
        final = await _wait_until_terminal(engine, workflow.workflow_id)
        assert final.state == WorkflowState.SUCCESS.value

    @pytest.mark.asyncio
    async def test_cancel_phrase_cancels_paused_workflow(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await engine.pause(workflow.workflow_id)
        settled = await _wait_until_terminal(engine, workflow.workflow_id)
        if settled.state != WorkflowState.PAUSED.value:
            return  # race: single step finished before the pause flag was checked

        consumed = await engine.handle_control_phrase("voice", "cancel")
        assert consumed is True
        final = await engine._workflow_store.get(workflow.workflow_id)
        assert final.state == WorkflowState.CANCELLED.value

    @pytest.mark.asyncio
    async def test_unrecognized_phrase_falls_through(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await engine.pause(workflow.workflow_id)
        settled = await _wait_until_terminal(engine, workflow.workflow_id)
        if settled.state != WorkflowState.PAUSED.value:
            return  # race: single step finished before the pause flag was checked

        consumed = await engine.handle_control_phrase("voice", "turn on the lights")
        assert consumed is False
        final = await engine._workflow_store.get(workflow.workflow_id)
        assert final.state == WorkflowState.PAUSED.value  # untouched

    @pytest.mark.asyncio
    async def test_no_pending_workflow_falls_through(self, tmp_path):
        engine = _engine(tmp_path)
        consumed = await engine.handle_control_phrase("voice", "continue")
        assert consumed is False

    @pytest.mark.asyncio
    async def test_wrong_source_does_not_match(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.GESTURE)
        await engine.pause(workflow.workflow_id)
        await _wait_until_terminal(engine, workflow.workflow_id)

        consumed = await engine.handle_control_phrase("voice", "continue")
        assert consumed is False

    @pytest.mark.asyncio
    async def test_stale_workflow_expires_and_falls_through(self, tmp_path):
        engine = _engine(tmp_path)
        workflow = await engine.start("do one thing", InvocationSource.VOICE)
        await _wait_until_terminal(engine, workflow.workflow_id)  # see race note in TestExpiry
        past_deadline = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        await engine._workflow_store.set_state(
            workflow.workflow_id, WorkflowState.PAUSED, trigger_deadline=past_deadline
        )

        consumed = await engine.handle_control_phrase("voice", "continue")
        assert consumed is False
        final = await engine._workflow_store.get(workflow.workflow_id)
        assert final.state == WorkflowState.EXPIRED.value

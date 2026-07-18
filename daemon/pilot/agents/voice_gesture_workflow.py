"""Durable, pausable, resumable multi-step workflow engine for voice/gesture-
triggered goals.

Deliberately separate from AutonomousExecutor (autonomous.py), which stays
untouched — that pipeline remains a fire-and-forget, in-memory-only, single
uninterrupted coroutine, and both its RPCs (autonomous_submit/_jobs/_cancel)
and its frontend consumers are unaffected by this module's existence.

This engine reuses the same building blocks (Planner/Executor/TaskDecomposer)
but drives them from durable, restart-survivable state
(VoiceGestureWorkflowStore) and supports genuine pause/resume: a goal can
span multiple separate voice commands or gesture inputs over time, not just
one uninterrupted run. Each step's own ActionPlan is checkpointed through
the existing, unmodified WorkflowCheckpointStore (keyed
"{workflow_id}:{step_index}"), exactly like server.py's resume_plan RPC
already does for a single plan.

Auto-chains steps by default (matching AutonomousExecutor's UX) — it only
stops and waits for external input on an explicit pause() request (state
PAUSED) or an ambiguous planning outcome (state WAITING_FOR_TRIGGER, e.g.
the planner couldn't produce a usable plan for a step and needs the user to
clarify). See pilot.security.gateway.InvocationSource.VOICE/GESTURE — this
is the first real call site wiring those profiles in.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.security.gateway import InvocationSource
from pilot.workflows.voice_gesture_workflows import (
    VoiceGestureWorkflow,
    VoiceGestureWorkflowStore,
    WorkflowState,
    WorkflowStepRecord,
)

if TYPE_CHECKING:
    from pilot.agents.decomposer import TaskDecomposer
    from pilot.agents.executor import Executor
    from pilot.agents.planner import Planner
    from pilot.security.gateway import TaskScopeOverride
    from pilot.workflows.checkpoints import WorkflowCheckpointStore

logger = logging.getLogger("pilot.agents.voice_gesture_workflow")

# How long a step boundary stays WAITING_FOR_TRIGGER (ambiguous planning
# outcome, short — the user is expected to respond almost immediately) versus
# explicitly PAUSED (user request, long — they may come back much later)
# before expire_if_stale() transitions it to EXPIRED and it stops
# intercepting new voice/gesture input.
WAITING_FOR_TRIGGER_SECONDS = 90.0
PAUSED_WINDOW_SECONDS = 1800.0

# Exact-match control phrases recognized at the voice dispatch point (see
# handle_control_phrase()) when a PAUSED/WAITING_FOR_TRIGGER workflow exists
# for the "voice" source — deliberately not fuzzy-matched (unlike wake-word
# near-miss calibration) so an unrelated normal command can't accidentally
# be swallowed as workflow control.
CONTINUE_PHRASES = frozenset({"continue", "resume", "keep going"})
CANCEL_PHRASES = frozenset({"cancel", "stop", "never mind", "nevermind"})


class VoiceGestureWorkflowEngine:
    """Drives VoiceGestureWorkflow rows to completion, one step at a time,
    reusing Planner/Executor/TaskDecomposer the same way AutonomousExecutor
    does. Safe to construct fresh after a daemon restart — nothing auto-resumes
    on its own; a caller must explicitly call resume() for any workflow left
    PAUSED/WAITING_FOR_TRIGGER/RUNNING when the previous process exited."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        decomposer: TaskDecomposer,
        workflow_store: VoiceGestureWorkflowStore,
        checkpoint_store: WorkflowCheckpointStore,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._decomposer = decomposer
        self._workflow_store = workflow_store
        self._checkpoint_store = checkpoint_store
        self._broadcast: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None
        self._pause_requested: dict[str, bool] = {}
        self._active_tasks: dict[str, asyncio.Task[None]] = {}

    def set_broadcast(self, fn: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        self._broadcast = fn

    async def start(
        self,
        goal: str,
        invocation_source: InvocationSource,
        scope_override: TaskScopeOverride | None = None,
    ) -> VoiceGestureWorkflow:
        workflow = await self._workflow_store.create(goal, invocation_source.value, scope_override)
        self._active_tasks[workflow.workflow_id] = asyncio.create_task(self._drive(workflow.workflow_id))
        return workflow

    async def pause(self, workflow_id: str) -> bool:
        """Request a pause at the next step boundary (not mid-step). Also
        works if nothing is actively driving the workflow (e.g. after a
        restart) — in that case it just flips the persisted state directly."""
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is None:
            return False
        if workflow.state not in (
            WorkflowState.PENDING.value,
            WorkflowState.DECOMPOSING.value,
            WorkflowState.RUNNING.value,
        ):
            return False
        if workflow_id in self._active_tasks:
            self._pause_requested[workflow_id] = True
        else:
            await self._workflow_store.set_state(
                workflow_id, WorkflowState.PAUSED, trigger_deadline=self._deadline_iso(PAUSED_WINDOW_SECONDS)
            )
        return True

    async def expire_if_stale(self, workflow_id: str) -> bool:
        """Proactively transitions a PAUSED/WAITING_FOR_TRIGGER workflow past
        its trigger_deadline to EXPIRED, so it stops intercepting new voice/
        gesture input. Intended to be called from the control-signal dispatch
        path (Part A4) right before treating a workflow as resumable —
        VoiceGestureWorkflowStore.find_pending_for_source's own window check
        is a defensive fallback, this is the primary mechanism. Returns True
        if it expired the workflow."""
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is None or workflow.state not in (
            WorkflowState.PAUSED.value,
            WorkflowState.WAITING_FOR_TRIGGER.value,
        ):
            return False
        if not workflow.trigger_deadline:
            return False
        if datetime.now(UTC) < datetime.fromisoformat(workflow.trigger_deadline):
            return False
        await self._workflow_store.set_state(workflow_id, WorkflowState.EXPIRED)
        await self._notify(workflow_id)
        return True

    async def resume(self, workflow_id: str) -> VoiceGestureWorkflow | None:
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is None or workflow.state not in (
            WorkflowState.PAUSED.value,
            WorkflowState.WAITING_FOR_TRIGGER.value,
        ):
            return None
        self._pause_requested.pop(workflow_id, None)
        workflow = await self._workflow_store.set_state(workflow_id, WorkflowState.RUNNING)
        self._active_tasks[workflow_id] = asyncio.create_task(self._drive(workflow_id))
        return workflow

    async def cancel(self, workflow_id: str) -> bool:
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is None:
            return False
        self._pause_requested.pop(workflow_id, None)
        task = self._active_tasks.pop(workflow_id, None)
        if task and not task.done():
            task.cancel()
        await self._workflow_store.set_state(workflow_id, WorkflowState.CANCELLED)
        return True

    async def list_workflows(self, include_terminal: bool = False) -> list[dict[str, Any]]:
        workflows = await self._workflow_store.list(include_terminal=include_terminal)
        return [w.to_dict() for w in workflows]

    async def handle_control_phrase(self, invocation_source: str, command_text: str) -> bool:
        """Checks whether a PAUSED/WAITING_FOR_TRIGGER workflow exists for
        this source and, if `command_text` exactly matches a recognized
        continue/cancel phrase, resumes/cancels it. Returns True if the
        phrase was consumed as workflow control — the caller (voice's
        _listen_loop, or a gesture dispatch site) should skip normal command
        dispatch for this input. Returns False for everything else (no
        pending workflow, a stale/expired one, or unrecognized text), in
        which case normal dispatch proceeds completely unaffected."""
        workflow = await self._workflow_store.find_pending_for_source(
            invocation_source, within_seconds=PAUSED_WINDOW_SECONDS
        )
        if workflow is None:
            return False
        if await self.expire_if_stale(workflow.workflow_id):
            return False

        phrase = command_text.strip().lower()
        if phrase in CONTINUE_PHRASES:
            await self.resume(workflow.workflow_id)
            return True
        if phrase in CANCEL_PHRASES:
            await self.cancel(workflow.workflow_id)
            return True
        return False

    async def _notify(self, workflow_id: str) -> None:
        if not self._broadcast:
            return
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is not None:
            await self._broadcast("voice_gesture_workflow_state", workflow.to_dict())

    async def _drive(self, workflow_id: str) -> None:
        try:
            workflow = await self._workflow_store.get(workflow_id)
            if workflow is None:
                return

            if not workflow.steps:
                workflow = await self._decompose(workflow_id, workflow)
                if workflow is None:
                    return

            while True:
                if self._pause_requested.pop(workflow_id, False):
                    await self._workflow_store.set_state(
                        workflow_id, WorkflowState.PAUSED, trigger_deadline=self._deadline_iso(PAUSED_WINDOW_SECONDS)
                    )
                    await self._notify(workflow_id)
                    return

                workflow = await self._workflow_store.get(workflow_id)
                if workflow is None:
                    return
                remaining = [s for s in workflow.steps if s.status == "pending"]
                if not remaining:
                    return

                stop = await self._run_step(workflow, remaining[0])
                if stop:
                    return
        finally:
            self._active_tasks.pop(workflow_id, None)

    async def _decompose(self, workflow_id: str, workflow: VoiceGestureWorkflow) -> VoiceGestureWorkflow | None:
        await self._workflow_store.set_state(workflow_id, WorkflowState.DECOMPOSING)
        decomposition = await self._decomposer.decompose(workflow.goal)

        if decomposition.is_complex and decomposition.subtasks:
            order = self._decomposer.get_execution_order(decomposition)
            steps = [
                WorkflowStepRecord(index=i, title=subtask.title, description=subtask.description)
                for i, subtask in enumerate(subtask for batch in order for subtask in batch)
            ]
        else:
            steps = [WorkflowStepRecord(index=0, title="Execute", description=workflow.goal)]

        await self._workflow_store.set_steps(workflow_id, steps)
        return await self._workflow_store.set_state(workflow_id, WorkflowState.RUNNING)

    async def _run_step(self, workflow: VoiceGestureWorkflow, step: WorkflowStepRecord) -> bool:
        """Runs one step. Returns True if the drive loop should stop (a
        terminal state was reached, or an ambiguous outcome now needs
        external voice/gesture input to resolve)."""
        await self._workflow_store.update_step(workflow.workflow_id, step.index, status="running")

        plan = await self._planner.plan(step.description)
        if plan.error:
            # Ambiguous planning outcome — leave the step "pending" (not
            # "failed") so resume() naturally retries it with a fresh plan()
            # call, rather than needing separate retry bookkeeping.
            await self._workflow_store.update_step(workflow.workflow_id, step.index, status="pending", error=plan.error)
            deadline = self._deadline_iso(WAITING_FOR_TRIGGER_SECONDS)
            await self._workflow_store.set_state(
                workflow.workflow_id, WorkflowState.WAITING_FOR_TRIGGER, trigger_deadline=deadline
            )
            await self._notify(workflow.workflow_id)
            return True

        sub_plan_id = f"{workflow.workflow_id}:{step.index}"
        await self._checkpoint_store.start_plan(sub_plan_id, step.description, plan)

        results = await self._executor.execute(
            plan,
            plan_id=sub_plan_id,
            invocation_source=InvocationSource(workflow.invocation_source),
            scope_override=workflow.scope_override,
        )
        for result in results:
            await self._checkpoint_store.record_result(sub_plan_id, result)

        success = bool(results) and all(r.success for r in results)
        output = "\n".join(r.output for r in results if r.output)
        error = "; ".join(r.error for r in results if r.error)
        await self._checkpoint_store.mark_status(sub_plan_id, "complete" if success else "failed")
        await self._workflow_store.update_step(
            workflow.workflow_id,
            step.index,
            status="success" if success else "failed",
            output=output,
            error=error,
        )

        current = await self._workflow_store.get(workflow.workflow_id)
        if current is None:
            return True
        if any(s.status == "pending" for s in current.steps):
            await self._workflow_store.set_state(
                workflow.workflow_id, WorkflowState.RUNNING, current_step=step.index + 1
            )
            return False

        successes = sum(1 for s in current.steps if s.status == "success")
        total = len(current.steps)
        final_state = (
            WorkflowState.SUCCESS
            if successes == total
            else WorkflowState.PARTIAL
            if successes > 0
            else WorkflowState.FAILED
        )
        await self._workflow_store.set_state(workflow.workflow_id, final_state, current_step=total)
        await self._notify(workflow.workflow_id)
        return True

    @staticmethod
    def _deadline_iso(seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()

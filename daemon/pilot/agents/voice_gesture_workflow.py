"""Durable, pausable, resumable multi-step workflow engine for voice/gesture-
triggered goals.

Workflow state is durable and restart-survivable, while each step delegates
to AutonomousExecutor's shared observe/act/verify loop. This keeps native-app
and browser grounding identical across typed and spoken autonomous tasks.

This engine drives the shared building blocks from durable, restart-survivable state
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
from dataclasses import dataclass
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
    from pilot.agents.autonomous import AutonomousExecutor
    from pilot.agents.decomposer import TaskDecomposer
    from pilot.agents.executor import Executor
    from pilot.agents.planner import Planner
    from pilot.memory.store import MemoryStore
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
PAUSE_PHRASES = frozenset({"pause", "wait", "hold on", "hang on", "one moment", "one sec", "one second"})
STATUS_PHRASES = frozenset(
    {
        "status",
        "what are you doing",
        "what are you doing now",
        "how is it going",
        "how's it going",
        "where are you",
        "are you done",
        "what's happening",
        "what is happening",
        "progress",
    }
)
# Approval answers belong to the visible approval flow, never to a workflow correction.
APPROVAL_WORDS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "ok",
        "okay",
        "sure",
        "approve",
        "approved",
        "deny",
        "denied",
        "confirm",
        "go ahead",
        "do it",
    }
)
# Spoken additions that extend the task rather than change it.
AMEND_PREFIXES = ("also ", "and ", "then ", "after that ", "plus ", "add ", "and then ")
# Gnomon's "still wanted" probability above which an utterance amends instead of restarting.
STILL_WANTED_AMEND = 0.7
START_WORKFLOW_PREFIXES = (
    "start a workflow to ",
    "start workflow to ",
    "start a workflow ",
    "start workflow ",
    "begin a workflow to ",
    "begin workflow to ",
    "begin a workflow ",
    "begin workflow ",
)
NATURAL_WORKFLOW_VERBS = frozenset(
    {
        "analyze",
        "book",
        "build",
        "compare",
        "complete",
        "create",
        "edit",
        "fill",
        "find",
        "finish",
        "inspect",
        "manage",
        "organize",
        "research",
        "send",
        "setup",
        "update",
    }
)

# Everyday desktop verbs. A spoken request with two of them in sequence ("open Notepad,
# write a shopping list and save it") is a multi-step task for the adaptive
# observe -> act -> verify loop, not a one-shot plan.
DESKTOP_ACTION_VERBS = NATURAL_WORKFLOW_VERBS | frozenset(
    {
        "add",
        "attach",
        "change",
        "check",
        "click",
        "close",
        "copy",
        "delete",
        "download",
        "draft",
        "export",
        "fill",
        "go",
        "import",
        "install",
        "launch",
        "move",
        "navigate",
        "open",
        "paste",
        "play",
        "print",
        "read",
        "remove",
        "rename",
        "reply",
        "save",
        "search",
        "select",
        "set",
        "share",
        "sign",
        "start",
        "type",
        "upload",
        "write",
    }
)
SEQUENCE_MARKERS = (" and ", " then ", " after ", " before ", ", ")
APP_CONTEXT_MARKERS = (" in ", " using ", " inside ")


def extract_voice_workflow_goal(command_text: str) -> str | None:
    """Return the explicit workflow goal from a recognized voice phrase."""
    normalized = " ".join(command_text.strip().split())
    lowered = normalized.lower()
    for prefix in START_WORKFLOW_PREFIXES:
        if lowered.startswith(prefix):
            goal = normalized[len(prefix) :].strip()
            return goal or None
    return None


def should_start_voice_workflow(command_text: str) -> bool:
    """Recognize natural multi-step/app goals without requiring a magic prefix."""
    normalized = " ".join(command_text.lower().strip().split())
    if not normalized or normalized in CONTINUE_PHRASES | CANCEL_PHRASES:
        return False
    words = [word.strip(".,!?") for word in normalized.replace(",", " , ").split() if word.strip(".,!?")]
    if len(words) < 4:
        return False
    padded = f" {normalized} "
    has_workflow_verb = any(word in NATURAL_WORKFLOW_VERBS for word in words)
    action_verbs = sum(1 for word in words if word in DESKTOP_ACTION_VERBS)
    has_sequence = any(marker in padded for marker in SEQUENCE_MARKERS)
    has_app_context = any(marker in padded for marker in APP_CONTEXT_MARKERS)
    strong_goal = normalized.startswith(("take care of ", "handle ", "do this task", "complete this task"))
    return (
        strong_goal
        or (has_workflow_verb and (has_sequence or has_app_context))
        or (action_verbs >= 2 and has_sequence)
        or (action_verbs >= 1 and has_app_context and len(words) >= 6)
    )


@dataclass(frozen=True)
class InstructionOutcome:
    """What a mid-task utterance did to a workflow, and what to tell the person."""

    kind: str  # cancelled | paused | status | corrected | amended | resumed
    message: str


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
        adaptive_executor: AutonomousExecutor | None = None,
        memory: MemoryStore | None = None,
        pending_trigger_window_seconds: float = WAITING_FOR_TRIGGER_SECONDS,
        paused_window_seconds: float = PAUSED_WINDOW_SECONDS,
        instruction_judge: Callable[[str, str, str], Coroutine[Any, Any, float | None]] | None = None,
    ) -> None:
        self._planner = planner
        # Gnomon's latest-instruction check (probability the current step is still wanted).
        self._instruction_judge = instruction_judge
        # Spoken additions per workflow; every adaptive step reads new entries each round.
        self._guidance: dict[str, list[str]] = {}
        self._executor = executor
        self._decomposer = decomposer
        self._workflow_store = workflow_store
        self._checkpoint_store = checkpoint_store
        self._adaptive_executor = adaptive_executor
        self._memory = memory
        self._pending_trigger_window_seconds = max(1.0, float(pending_trigger_window_seconds))
        self._paused_window_seconds = max(1.0, float(paused_window_seconds))
        self._broadcast: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None
        self._speech: Callable[[str, str, str], Coroutine[Any, Any, Any]] | None = None
        self._pause_requested: dict[str, bool] = {}
        self._active_tasks: dict[str, asyncio.Task[None]] = {}

    def set_broadcast(self, fn: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        self._broadcast = fn

    def set_speech(self, fn: Callable[[str, str, str], Coroutine[Any, Any, Any]]) -> None:
        self._speech = fn

    async def start(
        self,
        goal: str,
        invocation_source: InvocationSource,
        scope_override: TaskScopeOverride | None = None,
    ) -> VoiceGestureWorkflow:
        workflow = await self._workflow_store.create(goal, invocation_source.value, scope_override)
        if self._memory is not None:
            try:
                await self._memory.put_working(
                    session_id=invocation_source.value,
                    task_id=workflow.workflow_id,
                    key="active_goal",
                    value={"goal": goal, "state": workflow.state},
                    priority=0.95,
                    ttl_seconds=self._paused_window_seconds,
                )
            except Exception:
                logger.warning("Could not persist workflow working memory", exc_info=True)
        await self._notify(workflow.workflow_id)
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
                workflow_id,
                WorkflowState.PAUSED,
                trigger_deadline=self._deadline_iso(self._paused_window_seconds),
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
            # Wait for _drive to actually stop before writing our own state
            # below -- otherwise its own still-in-flight write (e.g. a step
            # completing right as we cancel) races this one for the same
            # SQLite file, which can raise "database is locked".
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._workflow_store.set_state(workflow_id, WorkflowState.CANCELLED)
        await self._notify(workflow_id)
        await self._clear_working_memory(workflow_id)
        return True

    async def stop(self) -> None:
        """Quiesce in-process workflow drivers during daemon shutdown.

        Durable workflow rows intentionally remain unchanged so an interrupted
        RUNNING workflow can be inspected and explicitly resumed after the
        daemon restarts.  This method only owns the asyncio tasks created by
        this engine.
        """
        tasks = tuple(self._active_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()
        self._pause_requested.clear()

    async def correct(self, workflow_id: str, correction: str) -> VoiceGestureWorkflow | None:
        """Stop the current round, persist a spoken correction, and re-plan."""
        correction = " ".join(correction.strip().split())
        workflow = await self._workflow_store.get(workflow_id)
        if (
            not correction
            or workflow is None
            or workflow.state
            not in {
                WorkflowState.PENDING.value,
                WorkflowState.DECOMPOSING.value,
                WorkflowState.RUNNING.value,
                WorkflowState.PAUSED.value,
                WorkflowState.WAITING_FOR_TRIGGER.value,
            }
        ):
            return None

        task = self._active_tasks.pop(workflow_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        workflow = await self._workflow_store.get(workflow_id)
        if workflow is None or workflow.state not in {
            WorkflowState.PENDING.value,
            WorkflowState.DECOMPOSING.value,
            WorkflowState.RUNNING.value,
            WorkflowState.PAUSED.value,
            WorkflowState.WAITING_FOR_TRIGGER.value,
        }:
            return None
        correction_line = f"User live correction: {correction}"
        if workflow.steps:
            current = next(
                (step for step in workflow.steps if step.status == "running"),
                next((step for step in workflow.steps if step.status == "pending"), workflow.steps[-1]),
            )
            description = f"{current.description}\n{correction_line}"
            await self._workflow_store.update_step(
                workflow_id,
                current.index,
                description=description,
                status="pending",
                error="",
            )
            current_step = current.index
        else:
            await self._workflow_store.update_goal(workflow_id, f"{workflow.goal}\n{correction_line}")
            current_step = workflow.current_step

        await self._workflow_store.set_state(
            workflow_id,
            WorkflowState.RUNNING,
            current_step=current_step,
        )
        if self._memory is not None:
            try:
                await self._memory.put_working(
                    session_id=workflow.invocation_source,
                    task_id=workflow_id,
                    key="latest_correction",
                    value=correction,
                    priority=1.0,
                    ttl_seconds=self._paused_window_seconds,
                )
            except Exception:
                logger.warning("Could not persist workflow correction memory", exc_info=True)
        await self._notify(workflow_id)
        self._active_tasks[workflow_id] = asyncio.create_task(self._drive(workflow_id))
        return await self._workflow_store.get(workflow_id)

    async def handle_running_instruction(self, invocation_source: str, command_text: str) -> bool:
        """Apply a new utterance to the active or awaiting workflow; True if it was consumed."""
        return await self.apply_instruction(invocation_source, command_text) is not None

    async def apply_instruction(self, invocation_source: str, command_text: str) -> InstructionOutcome | None:
        """Decide what a mid-task utterance means and apply it, or return None to leave it alone.

        A running workflow, or one waiting for a retry decision inside its short trigger
        window, claims speech. A paused workflow claims only continue and cancel, so an
        unrelated command half an hour later is not appended to it. Approval answers go
        to the approval flow. Status questions are answered without touching the work,
        pause phrases pause at the next step boundary, spoken additions amend the running
        step, and anything else corrects it and replans.
        """
        phrase = " ".join(command_text.strip().lower().split()).rstrip(".!?")
        if not phrase or phrase in APPROVAL_WORDS:
            return None
        workflow = await self._workflow_store.find_active_for_source(invocation_source)
        if workflow is None:
            workflow = await self._workflow_store.find_pending_for_source(
                invocation_source,
                within_seconds=self._paused_window_seconds,
            )
            if workflow is None or await self.expire_if_stale(workflow.workflow_id):
                return None
            if workflow.state == WorkflowState.PAUSED.value and phrase not in CONTINUE_PHRASES | CANCEL_PHRASES:
                return None
        if phrase in CANCEL_PHRASES:
            return (
                InstructionOutcome("cancelled", "I stopped that task.")
                if await self.cancel(workflow.workflow_id)
                else None
            )
        if phrase in CONTINUE_PHRASES:
            if workflow.state in {WorkflowState.PAUSED.value, WorkflowState.WAITING_FOR_TRIGGER.value}:
                if await self.resume(workflow.workflow_id) is None:
                    return None
            return InstructionOutcome("resumed", "Continuing.")
        if phrase in STATUS_PHRASES:
            return InstructionOutcome("status", self._status_line(workflow))
        if phrase in PAUSE_PHRASES:
            if await self.pause(workflow.workflow_id):
                return InstructionOutcome("paused", "I'll pause after this step. Say continue when you're ready.")
            return InstructionOutcome("status", self._status_line(workflow))
        if workflow.state in {
            WorkflowState.PENDING.value,
            WorkflowState.DECOMPOSING.value,
            WorkflowState.RUNNING.value,
        }:
            if await self._is_addition(workflow, command_text, phrase):
                await self._amend(workflow, command_text)
                return InstructionOutcome("amended", "Got it, I'll include that.")
        if await self.correct(workflow.workflow_id, command_text) is None:
            return None
        return InstructionOutcome("corrected", "I updated the active task and I'm replanning now.")

    @staticmethod
    def _status_line(workflow: VoiceGestureWorkflow) -> str:
        current = next((step for step in workflow.steps if step.status in {"running", "pending"}), None)
        if current is None:
            return f"I'm working on: {workflow.goal[:120]}."
        total = len(workflow.steps)
        position = f" Step {current.index + 1} of {total}." if total > 1 else ""
        # Single-step workflows carry a generic title; the description says what is happening.
        label = current.title if total > 1 and current.title else current.description.splitlines()[0]
        return f"I'm working on: {label[:120]}.{position}"

    async def _is_addition(self, workflow: VoiceGestureWorkflow, command_text: str, phrase: str) -> bool:
        """Gnomon decides when loaded; otherwise a leading "also" / "and" / "then" marks an addition."""
        if self._instruction_judge is not None:
            current = next((step for step in workflow.steps if step.status in {"running", "pending"}), None)
            try:
                still_wanted = await self._instruction_judge(
                    workflow.goal, command_text, current.description if current else workflow.goal
                )
            except Exception:
                still_wanted = None
            if still_wanted is not None:
                return still_wanted >= STILL_WANTED_AMEND
        return phrase.startswith(AMEND_PREFIXES)

    async def _amend(self, workflow: VoiceGestureWorkflow, command_text: str) -> None:
        addition = " ".join(command_text.strip().split())
        self._guidance.setdefault(workflow.workflow_id, []).append(addition)
        if self._memory is not None:
            try:
                await self._memory.put_working(
                    session_id=workflow.invocation_source,
                    task_id=workflow.workflow_id,
                    key="latest_addition",
                    value=addition,
                    priority=1.0,
                    ttl_seconds=self._paused_window_seconds,
                )
            except Exception:
                logger.warning("Could not persist workflow addition memory", exc_info=True)

    async def list_workflows(self, include_terminal: bool = False) -> list[dict[str, Any]]:
        workflows = await self._workflow_store.list(include_terminal=include_terminal)
        return [w.to_dict() for w in workflows]

    async def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        workflow = await self._workflow_store.get(workflow_id)
        return workflow.to_dict() if workflow is not None else None

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
            invocation_source, within_seconds=self._paused_window_seconds
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
                        workflow_id,
                        WorkflowState.PAUSED,
                        trigger_deadline=self._deadline_iso(self._paused_window_seconds),
                    )
                    await self._notify(workflow_id)
                    return

                workflow = await self._workflow_store.get(workflow_id)
                if workflow is None:
                    return
                remaining = [s for s in workflow.steps if s.status == "pending"]
                if not remaining:
                    return

                if len(workflow.steps) > 1:
                    step = remaining[0]
                    await self._say(
                        f"Step {step.index + 1} of {len(workflow.steps)}: {step.title or step.description[:80]}",
                        "task_narration",
                        f"voice-workflow:{workflow_id}:{step.index}:start",
                    )
                stop = await self._run_step(workflow, remaining[0])
                if stop:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Voice/gesture workflow %s failed", workflow_id)
            workflow = await self._workflow_store.get(workflow_id)
            if workflow is not None:
                running = next((step for step in workflow.steps if step.status == "running"), None)
                if running is not None:
                    await self._workflow_store.update_step(
                        workflow_id,
                        running.index,
                        status="failed",
                        error=str(exc),
                    )
                await self._workflow_store.set_state(workflow_id, WorkflowState.FAILED)
                await self._notify(workflow_id)
        finally:
            self._active_tasks.pop(workflow_id, None)
            if self._memory is not None:
                workflow = await self._workflow_store.get(workflow_id)
                if workflow is not None and workflow.state in {
                    WorkflowState.SUCCESS.value,
                    WorkflowState.PARTIAL.value,
                    WorkflowState.FAILED.value,
                    WorkflowState.CANCELLED.value,
                    WorkflowState.EXPIRED.value,
                }:
                    await self._clear_working_memory(workflow_id)

    async def _decompose(self, workflow_id: str, workflow: VoiceGestureWorkflow) -> VoiceGestureWorkflow | None:
        await self._workflow_store.set_state(workflow_id, WorkflowState.DECOMPOSING)
        decomposition = await self._decomposer.decompose(workflow.goal)
        if decomposition.error:
            raise RuntimeError(decomposition.error)

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

        if self._adaptive_executor is not None:
            return await self._run_adaptive_step(workflow, step)

        plan = await self._planner.plan(step.description)
        if plan.error:
            # Ambiguous planning outcome — leave the step "pending" (not
            # "failed") so resume() naturally retries it with a fresh plan()
            # call, rather than needing separate retry bookkeeping.
            await self._workflow_store.update_step(workflow.workflow_id, step.index, status="pending", error=plan.error)
            deadline = self._deadline_iso(self._pending_trigger_window_seconds)
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

    async def _run_adaptive_step(self, workflow: VoiceGestureWorkflow, step: WorkflowStepRecord) -> bool:
        """Run a durable voice step through the shared adaptive app loop."""
        from pilot.actions import VerificationResult

        sub_plan_id = f"{workflow.workflow_id}:{step.index}"

        async def record_round(plan_id, plan, results, verification: VerificationResult, task_complete: bool) -> None:
            await self._checkpoint_store.start_plan(plan_id, step.description, plan)
            for result in results:
                await self._checkpoint_store.record_result(plan_id, result)
            await self._checkpoint_store.mark_status(plan_id, "complete" if verification.passed else "failed")
            if self._memory is not None:
                try:
                    await self._memory.record(
                        step.description,
                        plan,
                        results,
                        session_id=workflow.invocation_source,
                        task_id=workflow.workflow_id,
                        verification=verification,
                        task_complete=task_complete,
                    )
                except Exception:
                    logger.warning("Could not record adaptive workflow episode", exc_info=True)

        adaptive_step = await self._adaptive_executor.execute_goal(
            step.description,
            invocation_source=InvocationSource(workflow.invocation_source),
            scope_override=workflow.scope_override,
            session_id=workflow.invocation_source,
            plan_id_prefix=sub_plan_id,
            on_round_complete=record_round,
            memory_task_id=workflow.workflow_id,
            guidance=self._guidance.setdefault(workflow.workflow_id, []),
        )
        if adaptive_step.status != "success":
            await self._workflow_store.update_step(
                workflow.workflow_id,
                step.index,
                status="pending",
                sub_plan_id=sub_plan_id,
                output=adaptive_step.output,
                error=adaptive_step.error,
            )
            await self._workflow_store.set_state(
                workflow.workflow_id,
                WorkflowState.WAITING_FOR_TRIGGER,
                trigger_deadline=self._deadline_iso(self._pending_trigger_window_seconds),
            )
            await self._notify(workflow.workflow_id)
            reason = " ".join((adaptive_step.error or "it was not verified").split())[:160]
            await self._say(
                f"I couldn't finish that step: {reason}. Say continue to retry, tell me what to change, or say cancel.",
                "task_failure",
                f"voice-workflow:{workflow.workflow_id}:{step.index}:blocked",
            )
            return True

        await self._workflow_store.update_step(
            workflow.workflow_id,
            step.index,
            status="success",
            sub_plan_id=sub_plan_id,
            output=adaptive_step.output,
            error="",
        )
        current = await self._workflow_store.get(workflow.workflow_id)
        if current is None:
            return True
        if any(item.status == "pending" for item in current.steps):
            await self._workflow_store.set_state(
                workflow.workflow_id,
                WorkflowState.RUNNING,
                current_step=step.index + 1,
            )
            await self._notify(workflow.workflow_id)
            return False

        await self._workflow_store.set_state(
            workflow.workflow_id,
            WorkflowState.SUCCESS,
            current_step=len(current.steps),
        )
        await self._notify(workflow.workflow_id)
        await self._announce_completion(workflow.workflow_id)
        return True

    async def _say(self, text: str, channel: str, key: str) -> None:
        if self._speech is None:
            return
        try:
            await self._speech(text, channel, key)
        except Exception:
            logger.debug("Workflow speech failed", exc_info=True)

    async def _announce_completion(self, workflow_id: str) -> None:
        if self._speech is None:
            return
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is None:
            return
        last_output = next((step.output for step in reversed(workflow.steps) if step.output), "")
        evidence = last_output.splitlines()[-1][:180] if last_output else "The requested goal was verified."
        await self._speech(
            f"Task complete. {evidence}",
            "final_answer",
            f"voice-workflow:{workflow_id}:complete",
        )

    async def _clear_working_memory(self, workflow_id: str) -> None:
        if self._memory is None:
            return
        workflow = await self._workflow_store.get(workflow_id)
        if workflow is not None:
            try:
                await self._memory.clear_task_working(
                    session_id=workflow.invocation_source,
                    task_id=workflow_id,
                )
            except Exception:
                logger.warning("Could not clear workflow working memory", exc_info=True)

    @staticmethod
    def _deadline_iso(seconds: float) -> str:
        return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()

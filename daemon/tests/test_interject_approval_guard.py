"""Typed or spoken text during a pending approval never approves, and "denied" is reported only when it is true (F6)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.narrator import ExecutionNarrator
from pilot.config import PilotConfig
from pilot.intelligence.experience import (
    ExperienceContext,
    ExperienceEventType,
    ExperienceLedger,
    experience_scope,
)
from pilot.server import PendingConfirmation, PilotServer, _normalize_spoken_phrase
from pilot.workflows.durable_tasks import ApprovalStatus, DurableTaskStore, TaskStatus

REQUIRED = {0, 1}
APPROVAL_REQUIRES_DIALOG = {
    "status": "approval_requires_dialog",
    "plan_id": "plan-1",
    "message": "Approve in the visible confirmation dialog. Speech and typed text can deny but never approve.",
}
DENIED = {"status": "denied", "plan_id": "plan-1", "message": "Okay, I won't run that."}
NOT_DENIED = {
    "status": "not_denied",
    "plan_id": "plan-1",
    "message": "That approval was already answered, so nothing was denied. Say or press Stop to cancel the task.",
}


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(action_type=ActionType.SYSTEM_INFO, target="system", parameters=EmptyParams()) for _ in REQUIRED
        ],
        raw_input="show system info",
    )


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


@pytest.fixture
async def ledger(tmp_path):
    instance = ExperienceLedger(tmp_path / "experience.db")
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


class _Harness:
    """An interactive task paused at its approval, with a running step the guard must leave alone."""

    def __init__(self, ledger: ExperienceLedger, durable: DurableTaskStore | None = None) -> None:
        self.server = PilotServer(PilotConfig())
        self.server._experience_ledger = ledger
        self.server._durable_tasks = durable
        self.server._broadcast_notification = AsyncMock()
        self.server._interactive_request_active = True
        self.server._active_plan_id = "plan-1"
        self.server._active_task_id = "task-1" if durable is not None else ""
        self.server._cancel_event = asyncio.Event()
        self.ledger = ledger
        self.executed: list[list[int]] = []
        self.socket = _Socket()
        self.waiter: asyncio.Task | None = None
        self.step: asyncio.Task | None = None

    async def _run(self) -> tuple[bool, set[int], set[int]]:
        confirmed, approved, required = await self.server._wait_for_confirmation(
            "plan-1", _plan(), self.socket, force_all_actions=True
        )
        if confirmed:
            self.executed.append(sorted(approved))
        return confirmed, approved, required

    async def start(self) -> PendingConfirmation:
        self.waiter = asyncio.create_task(self._run())
        self.step = asyncio.create_task(asyncio.Event().wait())
        self.server._active_execution_task = self.step
        # The waiter has logged the request and sent confirm_required, so no ledger write is in flight.
        await _until(lambda: bool(self.socket.sent))
        if self.server._durable_tasks is not None:
            for _ in range(100):
                if await self.server._durable_tasks.get_approval("plan-1") is not None:
                    break
                await asyncio.sleep(0.005)
        return self.server._pending_confirms["plan-1"]

    async def interject(self, text: str, mode: str = "") -> dict:
        params = {"input": text, **({"mode": mode} if mode else {})}
        with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
            return await self.server._handle_interject(params, None)

    async def corrections(self) -> int:
        return len(await self.ledger.list_events(event_type=ExperienceEventType.USER_CORRECTION))

    async def close(self) -> None:
        for task in (self.waiter, self.step):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("text", ["yes", "Yes!", "go ahead", "Go ahead.", "approve", "confirm it", "yes, do it"])
@pytest.mark.asyncio
async def test_approval_words_point_to_the_dialog_and_leave_the_approval_pending(ledger, text):
    harness = _Harness(ledger)
    pending = await harness.start()
    try:
        reply = await harness.interject(text)

        assert reply == APPROVAL_REQUIRES_DIALOG
        assert harness.server._pending_confirms == {"plan-1": pending}
        assert pending.resolved is False and pending.event.is_set() is False
        assert await harness.corrections() == 0
        assert harness.server._live_correction is None
        assert not harness.server._cancel_event.is_set()
        assert not harness.step.done()
        harness.server._broadcast_notification.assert_not_awaited()
    finally:
        await harness.close()


@pytest.mark.parametrize("text", ["no", "No.", "deny it", "Don't do that", "cancel that"])
@pytest.mark.asyncio
async def test_denial_words_deny_without_a_correction_or_replan(ledger, text):
    harness = _Harness(ledger)
    pending = await harness.start()
    try:
        reply = await harness.interject(text)

        assert reply == DENIED
        assert await harness.waiter == (False, set(), REQUIRED)
        assert harness.executed == []
        assert pending.resolution_path == "interject_deny"
        assert pending.input_channel == "voice_deny"
        assert await harness.corrections() == 0
        assert harness.server._live_correction is None
        assert not harness.server._cancel_event.is_set()
        assert not harness.step.done()
        events = await ledger.list_events(plan_id="plan-1", event_type=ExperienceEventType.APPROVAL_RESOLVED)
        assert [(event.payload["decision"], event.payload["resolution_path"]) for event in events] == [
            ("denied", "interject_deny")
        ]
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_no_after_the_waiter_consumed_an_approval_is_not_denied(ledger):
    harness = _Harness(ledger)
    pending = await harness.start()
    try:
        approved = await harness.server._handle_confirm(
            {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]}, None
        )
        assert approved == {"status": "ok", "confirmed": True}
        # The waiter holds the lock to pop and read when "no" arrives.
        await pending.decision_lock.acquire()
        await _until(lambda: len(pending.decision_lock._waiters or ()) == 1)
        reply_task = asyncio.create_task(harness.interject("no"))
        await _until(lambda: len(pending.decision_lock._waiters or ()) == 2)
        pending.decision_lock.release()

        assert await reply_task == NOT_DENIED
        assert await harness.waiter == (True, REQUIRED, REQUIRED)
        assert harness.executed == [[0, 1]]
        assert await harness.corrections() == 0
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_no_once_the_approval_is_gone_is_never_reported_as_denied(ledger):
    harness = _Harness(ledger)
    pending = await harness.start()
    try:
        await harness.server._handle_confirm({"plan_id": "plan-1", "confirmed": True}, None)
        assert await harness.waiter == (True, REQUIRED, REQUIRED)

        reply = await harness.interject("no")

        # With no approval pending, "no" is an ordinary correction of the running task.
        assert reply["status"] == "revising"
        assert pending.confirmed is True
        assert harness.executed == [[0, 1]]
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_no_between_a_socket_confirm_and_its_event_is_denied_and_the_plan_never_runs(ledger, tmp_path):
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    await durable.create_task(task_id="task-1", session_id="session-1", user_input="show system info")
    await durable.transition("task-1", TaskStatus.PLANNING)
    await durable.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
    harness = _Harness(ledger, durable)
    pending = await harness.start()
    try:
        confirm_task = asyncio.current_task()
        pending.rpc_tasks.add(confirm_task)
        approved = await harness.server._handle_confirm(
            {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]}, None
        )
        pending.rpc_tasks.discard(confirm_task)
        assert approved == {"status": "ok", "confirmed": True}
        assert pending.event.is_set() is False

        original = durable.deny_unconsumed_approval
        entered = asyncio.Event()
        release = asyncio.Event()

        async def suspended(plan_id: str):
            entered.set()
            await release.wait()
            return await original(plan_id)

        durable.deny_unconsumed_approval = suspended  # type: ignore[method-assign]
        reply_task = asyncio.create_task(harness.interject("no"))
        await entered.wait()
        pending.acknowledged = True  # _process_request after writing the confirm reply
        pending.event.set()
        await _until(lambda: len(pending.decision_lock._waiters or ()) == 1)
        assert pending.confirmed is False
        release.set()

        assert await reply_task == DENIED
        assert await harness.waiter == (False, set(), REQUIRED)
        assert harness.executed == []
        approval = await durable.get_approval("plan-1")
        assert approval is not None and approval.status == ApprovalStatus.DENIED
        events = await ledger.list_events(plan_id="plan-1", event_type=ExperienceEventType.APPROVAL_RESOLVED)
        assert [event.payload["decision"] for event in events] == ["denied"]
    finally:
        await harness.close()
        await durable.close()


@pytest.mark.asyncio
async def test_other_text_keeps_todays_correction_behaviour(ledger):
    harness = _Harness(ledger)
    pending = await harness.start()
    try:
        reply = await harness.interject("Use the downloads folder instead")

        assert reply == {"status": "revising", "message": "Live correction accepted."}
        assert await harness.corrections() == 1
        assert harness.server._live_correction == "Use the downloads folder instead"
        assert harness.server._cancel_event.is_set()
        assert await harness.waiter == (False, set(), REQUIRED)
        assert pending.resolution_path == "interject_correction"
        await asyncio.gather(harness.step, return_exceptions=True)
        assert harness.step.cancelled()
    finally:
        await harness.close()


@pytest.mark.parametrize("text", ["yes", "no"])
@pytest.mark.asyncio
async def test_without_a_pending_approval_nothing_changes(ledger, text):
    harness = _Harness(ledger)
    try:
        reply = await harness.interject(text)

        assert reply == {"status": "revising", "message": "Live correction accepted."}
        assert await harness.corrections() == 1
        assert harness.server._live_correction == text
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_an_explicit_stop_request_still_stops_during_an_approval(ledger):
    harness = _Harness(ledger)
    pending = await harness.start()
    try:
        reply = await harness.interject("yes", mode="stop")

        assert reply == {"status": "aborted"}
        assert await harness.waiter == (False, set(), REQUIRED)
        assert pending.resolution_path == "abort"
    finally:
        await harness.close()


def test_typed_and_spoken_phrases_share_one_normalisation():
    assert _normalize_spoken_phrase("  Yes,   DO it! ") == "yes do it"
    assert _normalize_spoken_phrase("Don't do that.") == "don t do that"
    assert _normalize_spoken_phrase("") == ""


@pytest.mark.asyncio
async def test_voice_dispatch_reports_not_denied_when_the_approval_was_already_answered():
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), resolved=True, confirmed=True)
    server._pending_confirms["plan-1"] = pending

    await server._voice_command_dispatch("No.")

    payload: dict[str, Any] = server._broadcast_notification.await_args.args[1]
    assert payload["message"] == NOT_DENIED["message"]
    assert pending.confirmed is True


# ── A narrator interrupt raised by a step of the running task ──


class _NarratorHarness:
    """An interactive task whose plan approval was answered earlier; a running step now waits on a narrator
    interrupt, registered under its own ``interrupt_...`` id rather than the plan id."""

    def __init__(self, ledger: ExperienceLedger) -> None:
        self.server = PilotServer(PilotConfig())
        self.server._experience_ledger = ledger
        self.server._broadcast_notification = AsyncMock()
        self.server._interactive_request_active = True
        self.server._active_plan_id = "plan-1"
        self.server._active_task_id = "task-1"
        self.server._active_experience_context = ExperienceContext(session_id="session-1", task_id="task-1")
        self.server._cancel_event = asyncio.Event()
        self.narrator = ExecutionNarrator(PilotConfig(), self.server._pending_confirms, AsyncMock())
        self.ledger = ledger
        self.step: asyncio.Task | None = None

    async def start(self, *, task_id: str = "task-1") -> tuple[str, PendingConfirmation]:
        async def step() -> bool:
            with experience_scope(session_id="session-1", task_id=task_id):
                return await self.narrator._interrupt_and_wait(reason="risky step", context={}, timeout_seconds=5.0)

        self.step = asyncio.create_task(step())
        self.server._active_execution_task = self.step
        await _until(lambda: bool(self.server._pending_confirms))
        plan_id, pending = next(iter(self.server._pending_confirms.items()))
        assert plan_id.startswith("interrupt_") and plan_id != self.server._active_plan_id
        return plan_id, pending

    async def interject(self, text: str) -> dict:
        with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
            return await self.server._handle_interject({"input": text}, None)

    async def corrections(self) -> int:
        return len(await self.ledger.list_events(event_type=ExperienceEventType.USER_CORRECTION))

    async def close(self) -> None:
        if self.step is not None and not self.step.done():
            self.step.cancel()
            await asyncio.gather(self.step, return_exceptions=True)


@pytest.mark.parametrize("text", ["yes", "do it", "Go ahead."])
@pytest.mark.asyncio
async def test_approval_words_during_a_narrator_interrupt_point_to_the_dialog(ledger, text):
    harness = _NarratorHarness(ledger)
    plan_id, pending = await harness.start()
    try:
        reply = await harness.interject(text)

        # Before the fix "yes" became a correction: logged, the step cancelled and the task replanned.
        assert reply == {**APPROVAL_REQUIRES_DIALOG, "plan_id": plan_id}
        assert harness.server._pending_confirms == {plan_id: pending}
        assert pending.resolved is False and pending.event.is_set() is False
        assert await harness.corrections() == 0
        assert harness.server._live_correction is None
        assert not harness.server._cancel_event.is_set()
        assert not harness.step.done()
    finally:
        await harness.close()


@pytest.mark.parametrize("text", ["no", "Don't do that"])
@pytest.mark.asyncio
async def test_denial_words_during_a_narrator_interrupt_deny_the_step_without_a_replan(ledger, text):
    harness = _NarratorHarness(ledger)
    plan_id, pending = await harness.start()
    try:
        reply = await harness.interject(text)

        assert reply == {**DENIED, "plan_id": plan_id}
        assert await asyncio.wait_for(harness.step, 2.0) is False  # the risky step does not run
        assert (pending.resolution_path, pending.input_channel) == ("interject_deny", "voice_deny")
        assert await harness.corrections() == 0
        assert harness.server._live_correction is None
        assert not harness.server._cancel_event.is_set()
    finally:
        await harness.close()


@pytest.mark.parametrize("text", ["yes", "no"])
@pytest.mark.asyncio
async def test_a_narrator_interrupt_of_another_task_is_never_answered_by_interject(ledger, text):
    harness = _NarratorHarness(ledger)
    plan_id, pending = await harness.start(task_id="background-job-7")
    harness.server._active_execution_task = None
    try:
        reply = await harness.interject(text)

        assert reply["status"] == "revising"  # an ordinary correction of the running task
        assert harness.server._pending_confirms == {plan_id: pending}
        assert pending.resolved is False and pending.event.is_set() is False
        assert not harness.step.done()
    finally:
        await harness.close()

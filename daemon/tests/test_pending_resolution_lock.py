"""Denials of a pending approval are written under its decision lock and win over an unconsumed approval (F5).

The waiter pops the confirmation and reads its decision inside the same lock, so "denied" is reported only for a
plan that does not run, and the durable store, the ledger and the waiter agree, also across a restart.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.intelligence.experience import ExperienceEventType, ExperienceLedger
from pilot.server import PendingConfirmation, PilotServer
from pilot.workflows.durable_tasks import ApprovalStatus, DurableTaskStore, TaskStatus

REQUIRED = {0, 1}


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


async def _until(predicate: Callable[[], bool], *, attempts: int = 400) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


def _lock_waiters(pending: PendingConfirmation) -> int:
    return len(pending.decision_lock._waiters or ())


async def _durable_task(store: DurableTaskStore) -> str:
    created = await store.create_task(task_id="task-1", session_id="session-1", user_input="show system info")
    await store.transition("task-1", TaskStatus.PLANNING)
    await store.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
    return created.resume_token


@pytest.fixture
async def durable(tmp_path):
    store = DurableTaskStore(tmp_path / "durable.db")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def ledger(tmp_path):
    instance = ExperienceLedger(tmp_path / "experience.db")
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


def _server(*, durable: DurableTaskStore | None = None, ledger: ExperienceLedger | None = None) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._durable_tasks = durable
    server._experience_ledger = ledger
    server._broadcast_notification = AsyncMock()
    server._interactive_request_active = True
    server._active_task_id = "task-1" if durable is not None else ""
    server._active_plan_id = "plan-1"
    return server


class _Run:
    """The execute path's use of the waiter: the plan runs only when the waiter returns an approval."""

    def __init__(self, server: PilotServer) -> None:
        self.server = server
        self.executed: list[list[int]] = []
        self.socket = _Socket()
        self.task: asyncio.Task | None = None

    async def _run(self) -> tuple[bool, set[int], set[int]]:
        confirmed, approved, required = await self.server._wait_for_confirmation(
            "plan-1", _plan(), self.socket, force_all_actions=True
        )
        if confirmed:
            self.executed.append(sorted(approved))
        return confirmed, approved, required

    async def start(self) -> PendingConfirmation:
        self.task = asyncio.create_task(self._run())
        # The waiter has logged the request and sent confirm_required, so no ledger write is in flight.
        await _until(lambda: bool(self.socket.sent))
        pending = self.server._pending_confirms["plan-1"]
        if self.server._durable_tasks is not None:
            await _until(lambda: self.server._pending_confirms.get("plan-1") is pending)
            for _ in range(100):
                if await self.server._durable_tasks.get_approval("plan-1") is not None:
                    break
                await asyncio.sleep(0.005)
        return pending


async def _socket_confirm(server: PilotServer, pending: PendingConfirmation, confirmed: bool) -> dict:
    """Resolve like a socket confirm: _process_request sets the event only after the reply is written."""
    task = asyncio.current_task()
    pending.rpc_tasks.add(task)
    try:
        return await server._handle_confirm(
            {"plan_id": "plan-1", "confirmed": confirmed, "input_channel": "dialog", "approved_indices": [0, 1]},
            None,
        )
    finally:
        pending.rpc_tasks.discard(task)


def _acknowledge(pending: PendingConfirmation) -> None:
    """What _process_request does once the confirm reply is written: mark it acknowledged and release the waiter."""
    pending.acknowledged = True
    pending.event.set()


async def _decisions(ledger: ExperienceLedger) -> list[str]:
    events = await ledger.list_events(plan_id="plan-1", event_type=ExperienceEventType.APPROVAL_RESOLVED)
    return [event.payload["decision"] for event in events]


# ── Denial under the lock ──


@pytest.mark.parametrize("path", ["interject_correction", "abort"])
@pytest.mark.asyncio
async def test_interject_and_abort_deny_only_under_the_decision_lock(path):
    server = _server()
    run = _Run(server)
    pending = await run.start()

    await pending.decision_lock.acquire()
    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        if path == "abort":
            stopper = asyncio.create_task(server._handle_abort({}, None))
        else:
            stopper = asyncio.create_task(server._handle_interject({"input": "only look at the disk"}, None))
        await _until(lambda: _lock_waiters(pending) == 1)
        assert pending.resolved is False
        assert pending.event.is_set() is False
        pending.decision_lock.release()
        reply = await stopper

    assert reply["status"] == ("aborted" if path == "abort" else "revising")
    assert await run.task == (False, set(), REQUIRED)
    assert run.executed == []
    assert pending.resolution_path == path
    assert pending.input_channel == "internal"


@pytest.mark.asyncio
async def test_a_confirm_after_the_denial_is_refused():
    server = _server()
    run = _Run(server)
    pending = await run.start()

    assert await server._resolve_pending_denied("plan-1", resolution_path="abort") is True
    late = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]}, None
    )

    assert late["status"] == "error"
    assert late["message"] in {
        "Confirmation already resolved for plan_id: plan-1",
        "No pending confirmation for plan_id: plan-1",
    }
    assert await run.task == (False, set(), REQUIRED)
    assert pending.confirmed is False
    assert run.executed == []


@pytest.mark.asyncio
async def test_denial_overrides_an_unconsumed_approval_in_memory_durable_store_and_ledger(durable, ledger):
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()

    reply = await _socket_confirm(server, pending, True)
    assert reply == {"status": "ok", "confirmed": True}
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.APPROVED
    assert pending.event.is_set() is False

    assert await server._resolve_pending_denied("plan-1", resolution_path="abort") is True

    assert await run.task == (False, set(), REQUIRED)
    assert run.executed == []
    approval = await durable.get_approval("plan-1")
    assert approval.status == ApprovalStatus.DENIED
    assert approval.approved_indices == []
    assert await _decisions(ledger) == ["denied"]


@pytest.mark.asyncio
async def test_helper_returns_false_once_the_waiter_consumed_the_decision(durable, ledger):
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()

    await _socket_confirm(server, pending, True)
    _acknowledge(pending)  # _process_request after writing the reply
    assert await run.task == (True, REQUIRED, REQUIRED)

    assert await server._resolve_pending_denied("plan-1", resolution_path="abort") is False

    assert run.executed == [[0, 1]]
    assert pending.confirmed is True
    assert pending.resolution_path == "confirm_rpc"
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.APPROVED
    assert await _decisions(ledger) == ["approved"]


@pytest.mark.asyncio
async def test_a_denial_after_the_waiter_consumed_it_changes_nothing_and_sets_no_second_event(ledger):
    server = _server(ledger=ledger)
    run = _Run(server)
    pending = await run.start()
    sets = 0
    original_set = pending.event.set

    def counting_set() -> None:
        nonlocal sets
        sets += 1
        original_set()

    pending.event.set = counting_set  # type: ignore[method-assign]

    first = await server._resolve_pending_denied("plan-1", resolution_path="interject_correction")
    assert await run.task == (False, set(), REQUIRED)
    second = await server._resolve_pending_denied("plan-1", resolution_path="abort")

    assert (first, second) == (True, False)
    assert sets == 1
    assert pending.resolution_path == "interject_correction"
    assert await _decisions(ledger) == ["denied"]


@pytest.mark.asyncio
async def test_helper_without_a_pending_confirmation_is_a_no_op():
    server = _server()
    assert await server._resolve_pending_denied("plan-1", resolution_path="abort") is False
    assert await server._resolve_pending_denied("", resolution_path="abort") is False


# ── Restart ──


@pytest.mark.asyncio
async def test_a_corrected_plan_never_resumes_after_a_restart(tmp_path):
    first_store = DurableTaskStore(tmp_path / "durable.db")
    await first_store.initialize()
    resume_token = await _durable_task(first_store)
    server = _server(durable=first_store)
    run = _Run(server)
    pending = await run.start()

    reply = await _socket_confirm(server, pending, True)
    assert reply == {"status": "ok", "confirmed": True}
    assert (await first_store.get_approval("plan-1")).status == ApprovalStatus.APPROVED
    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        correction = await server._handle_interject({"input": "only look at the disk"}, None)
    assert correction["status"] == "revising"
    assert await run.task == (False, set(), REQUIRED)
    await first_store.close()  # the process stops here

    second_store = DurableTaskStore(tmp_path / "durable.db")
    await second_store.initialize()
    try:
        restarted = PilotServer(PilotConfig())
        restarted._durable_tasks = second_store
        restarted._handle_resume_plan = AsyncMock()
        restarted._handle_execute_inner = AsyncMock()

        resumed = await restarted._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, None)

        assert resumed["status"] == "cancelled"
        assert resumed["message"] == "Task approval was denied."
        restarted._handle_resume_plan.assert_not_awaited()
        restarted._handle_execute_inner.assert_not_awaited()
        task = await second_store.get("task-1")
        assert task is not None and task.status == TaskStatus.CANCELLED
        assert run.executed == []
    finally:
        await second_store.close()


@pytest.mark.asyncio
async def test_an_aborted_plan_never_resumes_after_a_restart(tmp_path):
    first_store = DurableTaskStore(tmp_path / "durable.db")
    await first_store.initialize()
    resume_token = await _durable_task(first_store)
    server = _server(durable=first_store)
    run = _Run(server)
    pending = await run.start()

    await _socket_confirm(server, pending, True)
    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        assert await server._handle_abort({}, None) == {"status": "aborted"}
    assert await run.task == (False, set(), REQUIRED)
    assert (await first_store.get_approval("plan-1")).status == ApprovalStatus.DENIED
    await first_store.close()

    second_store = DurableTaskStore(tmp_path / "durable.db")
    await second_store.initialize()
    try:
        restarted = PilotServer(PilotConfig())
        restarted._durable_tasks = second_store
        restarted._handle_resume_plan = AsyncMock()

        resumed = await restarted._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, None)

        assert resumed["status"] == "cancelled"
        restarted._handle_resume_plan.assert_not_awaited()
    finally:
        await second_store.close()


# ── Interleavings ──


def _suspend_durable_denial(store: DurableTaskStore) -> tuple[asyncio.Event, asyncio.Event]:
    original = store.deny_unconsumed_approval
    entered = asyncio.Event()
    release = asyncio.Event()

    async def suspended(plan_id: str):
        entered.set()
        await release.wait()
        return await original(plan_id)

    store.deny_unconsumed_approval = suspended  # type: ignore[method-assign]
    return entered, release


def _spy_helper(server: PilotServer) -> list[bool]:
    returned: list[bool] = []
    original = server._resolve_pending_denied

    async def spy(plan_id: str, **kwargs: Any) -> bool:
        result = await original(plan_id, **kwargs)
        returned.append(result)
        return result

    server._resolve_pending_denied = spy  # type: ignore[method-assign]
    return returned


@pytest.mark.asyncio
async def test_event_set_during_the_durable_denial_still_reads_the_denial(durable, ledger):
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()
    returned = _spy_helper(server)

    # A socket confirm approved; its reply is written but _process_request has not set the event yet.
    assert await _socket_confirm(server, pending, True) == {"status": "ok", "confirmed": True}
    entered, release = _suspend_durable_denial(durable)
    denial = asyncio.create_task(server._handle_interject({"input": "no"}, None))
    await entered.wait()

    _acknowledge(pending)  # _process_request after the confirm reply
    await _until(lambda: _lock_waiters(pending) == 1)
    assert pending.confirmed is False  # the denial was written before the helper's first await
    assert not run.task.done()  # the waiter is blocked on the decision lock

    release.set()
    reply = await denial

    assert returned == [True]
    assert reply == {"status": "denied", "plan_id": "plan-1", "message": "Okay, I won't run that."}
    assert await run.task == (False, set(), REQUIRED)
    assert run.executed == []
    assert await _decisions(ledger) == ["denied"]
    approval = await durable.get_approval("plan-1")
    assert approval.status == ApprovalStatus.DENIED and approval.approved_indices == []


@pytest.mark.asyncio
async def test_waiter_that_read_the_approval_first_makes_the_denial_report_not_denied(durable, ledger):
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()
    returned = _spy_helper(server)

    assert await _socket_confirm(server, pending, True) == {"status": "ok", "confirmed": True}
    await pending.decision_lock.acquire()
    _acknowledge(pending)
    await _until(lambda: _lock_waiters(pending) == 1)  # the waiter queues for the lock first
    denial = asyncio.create_task(server._handle_interject({"input": "no"}, None))
    await _until(lambda: _lock_waiters(pending) == 2)
    pending.decision_lock.release()
    reply = await denial

    assert returned == [False]
    assert reply["status"] == "not_denied"
    assert reply["message"] == (
        "That approval was already answered, so nothing was denied. Say or press Stop to cancel the task."
    )
    assert await run.task == (True, REQUIRED, REQUIRED)
    assert run.executed == [[0, 1]]
    assert await _decisions(ledger) == ["approved"]
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.APPROVED


@pytest.mark.asyncio
async def test_timeout_racing_a_denial_reports_the_denial_when_the_helper_resolved_first(durable, ledger, monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 0.1)
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    await run.start()
    entered, release = _suspend_durable_denial(durable)

    denial = asyncio.create_task(server._resolve_pending_denied("plan-1", resolution_path="abort"))
    await entered.wait()
    await asyncio.sleep(0.2)  # the waiter's timeout fires while the helper holds the lock
    assert not run.task.done()
    release.set()

    assert await denial is True
    assert await run.task == (False, set(), REQUIRED)
    assert await _decisions(ledger) == ["denied"]
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.DENIED


@pytest.mark.asyncio
async def test_timeout_that_takes_the_lock_first_expires_and_the_helper_returns_false(durable, ledger, monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 0.1)
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()

    await pending.decision_lock.acquire()
    await _until(lambda: _lock_waiters(pending) == 1)  # timed out and queued for the lock
    denial = asyncio.create_task(server._resolve_pending_denied("plan-1", resolution_path="abort"))
    await _until(lambda: _lock_waiters(pending) == 2)
    pending.decision_lock.release()

    assert await denial is False
    assert await run.task == (False, set(), REQUIRED)
    assert await _decisions(ledger) == ["expired"]
    events = await ledger.list_events(plan_id="plan-1", event_type=ExperienceEventType.APPROVAL_RESOLVED)
    assert events[0].payload["resolution_path"] == "timeout"
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.EXPIRED


# ── A durable denial that fails or is cancelled ──


def _failing_durable_denial(store: DurableTaskStore, *, failures: int) -> list[str]:
    """Make the next ``failures`` durable denials raise like a locked or broken task database."""
    original = store.deny_unconsumed_approval
    attempts: list[str] = []

    async def failing(plan_id: str):
        attempts.append(plan_id)
        if len(attempts) <= failures:
            raise sqlite3.OperationalError("disk I/O error")
        return await original(plan_id)

    store.deny_unconsumed_approval = failing  # type: ignore[method-assign]
    return attempts


async def _stop_or_deny(server: PilotServer, path: str) -> tuple[dict, MagicMock]:
    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all") as interrupt_all:
        if path == "abort":
            reply = await server._handle_abort({}, None)
        elif path == "interject_correction":
            reply = await server._handle_interject({"input": "only look at the disk"}, None)
        else:
            reply = await server._handle_interject({"input": "no"}, None)
    return reply, interrupt_all


EXPECTED_REPLY_STATUS = {"abort": "aborted", "interject_correction": "revising", "interject_deny": "denied"}


@pytest.mark.parametrize("path", ["interject_deny", "interject_correction", "abort"])
@pytest.mark.asyncio
async def test_a_failed_durable_denial_releases_the_waiter_at_once_and_the_plan_never_resumes(
    tmp_path, ledger, monkeypatch, path
):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 5.0)
    store = DurableTaskStore(tmp_path / "durable.db")
    await store.initialize()
    try:
        resume_token = await _durable_task(store)
        server = _server(durable=store, ledger=ledger)
        run = _Run(server)
        pending = await run.start()
        step = asyncio.create_task(asyncio.Event().wait())
        server._active_execution_task = step
        # The worst case: a socket confirm already wrote APPROVED and its event is not set yet.
        assert await _socket_confirm(server, pending, True) == {"status": "ok", "confirmed": True}
        assert (await store.get_approval("plan-1")).status == ApprovalStatus.APPROVED
        attempts = _failing_durable_denial(store, failures=10)

        # Before the fix the RPC raised, the event stayed unset, and the waiter slept until the timeout.
        reply, interrupt_all = await asyncio.wait_for(_stop_or_deny(server, path), 2.0)

        assert reply["status"] == EXPECTED_REPLY_STATUS[path]
        assert await asyncio.wait_for(run.task, 2.0) == (False, set(), REQUIRED)
        assert run.executed == []
        assert pending.resolution_path == path
        assert await _decisions(ledger) == ["denied"]
        assert attempts == ["plan-1", "plan-1"]  # the helper and then the waiter tried the durable denial
        if path != "interject_deny":
            await asyncio.wait({step}, timeout=1.0)
            assert step.cancelled()
            interrupt_all.assert_called_once()
        # The durable approval still reads APPROVED, so the task itself is marked for cancellation instead.
        assert (await store.get_approval("plan-1")).status == ApprovalStatus.APPROVED
        assert (await store.get("task-1")).cancellation_requested is True
        server._active_task_id = ""
        server._active_plan_id = ""
        server._interactive_request_active = False
        server._handle_resume_plan = AsyncMock()

        resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": resume_token}, None)

        assert resumed["status"] == "cancelled"
        server._handle_resume_plan.assert_not_awaited()
        step.cancel()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_the_waiter_repairs_a_durable_denial_that_failed_once(durable, ledger):
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()
    assert await _socket_confirm(server, pending, True) == {"status": "ok", "confirmed": True}
    _failing_durable_denial(durable, failures=1)

    reply, _interrupt_all = await _stop_or_deny(server, "interject_deny")

    assert reply == {"status": "denied", "plan_id": "plan-1", "message": "Okay, I won't run that."}
    assert await asyncio.wait_for(run.task, 2.0) == (False, set(), REQUIRED)
    approval = await durable.get_approval("plan-1")
    assert approval.status == ApprovalStatus.DENIED and approval.approved_indices == []


@pytest.mark.parametrize("path", ["abort", "interject_correction"])
@pytest.mark.asyncio
async def test_a_stop_cancelled_during_the_durable_denial_still_stops_the_step_and_releases_the_waiter(
    durable, ledger, monkeypatch, path
):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 5.0)
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    run = _Run(server)
    pending = await run.start()
    step = asyncio.create_task(asyncio.Event().wait())
    server._active_execution_task = step
    entered, release = _suspend_durable_denial(durable)

    try:
        with patch("pilot.system.pty_session.PtySessionManager.interrupt_all") as interrupt_all:
            if path == "abort":
                stopper = asyncio.create_task(server._handle_abort({}, None))
            else:
                stopper = asyncio.create_task(server._handle_interject({"input": "only look at the disk"}, None))
            await entered.wait()
            stopper.cancel()  # the window that sent it closed during the durable write
            with pytest.raises(asyncio.CancelledError):
                await stopper

            # Before the fix the step kept running and the PTY sessions were never interrupted.
            assert pending.event.is_set()
            await asyncio.wait({step}, timeout=1.0)
            assert step.cancelled()
            interrupt_all.assert_called_once()

        release.set()
        assert await asyncio.wait_for(run.task, 2.0) == (False, set(), REQUIRED)
        assert run.executed == []
        assert (await durable.get_approval("plan-1")).status == ApprovalStatus.DENIED
    finally:
        release.set()
        step.cancel()
        await asyncio.wait({run.task}, timeout=1.0)

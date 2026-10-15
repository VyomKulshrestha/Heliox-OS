"""A confirmation nobody answered in a way the plan may act on ends as expired, never as an approval or a confirm RPC.

Covers an approval whose confirm reply never reached the UI (the socket closed, or the request task was cancelled with
it) and the release of every waiting confirmation when the daemon stops.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import websockets

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.narrator import ExecutionNarrator
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


class _ClosingConnection:
    """A UI connection whose reply to the confirm is never written: the send fails, or hangs until the
    connection closes and cancels the request task."""

    remote_address = ("127.0.0.1", 0)

    def __init__(self, *, reply: str) -> None:
        self.reply = reply  # "fails" or "hangs"
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.reply_attempted = asyncio.Event()

    async def recv(self) -> str:
        return json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": "auth"})

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        if message.get("id") == "confirm":
            self.reply_attempted.set()
            if self.reply == "fails":
                raise websockets.exceptions.ConnectionClosedError(None, None)
            await asyncio.Event().wait()  # cancelled when the connection closes
        self.sent.append(message)

    async def close(self) -> None:
        await self.incoming.put(None)

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        while (message := await self.incoming.get()) is not None:
            yield json.dumps(message)


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


def _plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(action_type=ActionType.SYSTEM_INFO, target="system", parameters=EmptyParams()) for _ in REQUIRED
        ],
        raw_input="show system info",
    )


@pytest.fixture
async def durable(tmp_path):
    store = DurableTaskStore(tmp_path / "durable.db")
    await store.initialize()
    await store.create_task(task_id="task-1", session_id="session-1", user_input="show system info")
    await store.transition("task-1", TaskStatus.PLANNING)
    await store.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
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


def _server(durable: DurableTaskStore | None = None, ledger: ExperienceLedger | None = None) -> PilotServer:
    config = PilotConfig()
    config.server.auth_token = "test-token"
    server = PilotServer(config)
    server._durable_tasks = durable
    server._experience_ledger = ledger
    server._broadcast_notification = AsyncMock()
    server._interactive_request_active = True
    server._active_task_id = "task-1" if durable is not None else ""
    server._active_plan_id = "plan-1"
    # Only initialize() builds the RPC table; these tests drive confirm through the real connection handler.
    server._handlers = {"confirm": server._handle_confirm}
    return server


async def _start_waiter(server: PilotServer) -> tuple[asyncio.Task, PendingConfirmation]:
    socket = _Socket()
    waiter = asyncio.create_task(server._wait_for_confirmation("plan-1", _plan(), socket, force_all_actions=True))
    await _until(lambda: bool(socket.sent))
    if server._durable_tasks is not None:
        for _ in range(100):
            if await server._durable_tasks.get_approval("plan-1") is not None:
                break
            await asyncio.sleep(0.005)
    return waiter, server._pending_confirms["plan-1"]


async def _resolution(ledger: ExperienceLedger) -> list[tuple[str, str, str]]:
    events = await ledger.list_events(plan_id="plan-1", event_type=ExperienceEventType.APPROVAL_RESOLVED)
    return [
        (event.payload["decision"], event.payload["input_channel"], event.payload["resolution_path"])
        for event in events
    ]


async def _confirm_over(connection: _ClosingConnection, plan_id: str = "plan-1") -> None:
    await connection.incoming.put(
        {
            "jsonrpc": "2.0",
            "method": "confirm",
            "params": {"plan_id": plan_id, "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]},
            "id": "confirm",
        }
    )


# ── An approval whose reply never reached the UI ──


@pytest.mark.parametrize("reply", ["fails", "hangs"])
@pytest.mark.asyncio
async def test_an_approval_whose_reply_was_never_written_expires_at_once(durable, ledger, monkeypatch, reply):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 5.0)
    server = _server(durable, ledger)
    waiter, pending = await _start_waiter(server)
    connection = _ClosingConnection(reply=reply)
    handler = asyncio.create_task(server._handle_connection(connection))  # type: ignore[arg-type]
    try:
        await _confirm_over(connection)
        await connection.reply_attempted.wait()
        assert (await durable.get_approval("plan-1")).status == ApprovalStatus.APPROVED
        if reply == "hangs":
            await connection.close()  # the window closes before its reply is written

        # Before the fix the waiter slept until the timeout and then ran the plan.
        assert await asyncio.wait_for(waiter, 2.0) == (False, set(), REQUIRED)
    finally:
        await connection.close()
        await asyncio.wait_for(handler, 2.0)

    assert pending.confirmed is False
    assert pending.resolution_path == "unacknowledged"
    assert await _resolution(ledger) == [("expired", "internal", "unacknowledged")]
    approval = await durable.get_approval("plan-1")
    assert approval.status == ApprovalStatus.EXPIRED and approval.approved_indices == []


@pytest.mark.asyncio
async def test_an_approval_still_unacknowledged_at_the_timeout_expires(durable, ledger, monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 0.2)
    server = _server(durable, ledger)
    waiter, pending = await _start_waiter(server)

    # A socket confirm: _process_request would set the event only after writing the reply, which never happens.
    task = asyncio.current_task()
    pending.rpc_tasks.add(task)
    reply = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]}, None
    )
    assert reply == {"status": "ok", "confirmed": True}
    assert pending.event.is_set() is False

    assert await waiter == (False, set(), REQUIRED)
    assert await _resolution(ledger) == [("expired", "internal", "unacknowledged")]
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.EXPIRED


@pytest.mark.asyncio
async def test_an_acknowledged_approval_still_runs(durable, ledger):
    server = _server(durable, ledger)
    waiter, pending = await _start_waiter(server)
    task = asyncio.current_task()
    pending.rpc_tasks.add(task)
    await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]}, None
    )
    pending.rpc_tasks.discard(task)
    pending.acknowledged = True  # _process_request after writing the reply
    pending.event.set()

    assert await waiter == (True, REQUIRED, REQUIRED)
    assert await _resolution(ledger) == [("approved", "dialog", "confirm_rpc")]
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.APPROVED


@pytest.mark.asyncio
async def test_an_unacknowledged_denial_is_applied_at_once(ledger, monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 5.0)
    server = _server(ledger=ledger)
    waiter, _pending = await _start_waiter(server)
    connection = _ClosingConnection(reply="fails")
    handler = asyncio.create_task(server._handle_connection(connection))  # type: ignore[arg-type]
    try:
        await connection.incoming.put(
            {
                "jsonrpc": "2.0",
                "method": "confirm",
                "params": {"plan_id": "plan-1", "confirmed": False, "input_channel": "dialog"},
                "id": "confirm",
            }
        )
        assert await asyncio.wait_for(waiter, 2.0) == (False, set(), REQUIRED)
    finally:
        await connection.close()
        await asyncio.wait_for(handler, 2.0)
    assert await _resolution(ledger) == [("denied", "dialog", "confirm_rpc")]


@pytest.mark.asyncio
async def test_an_unacknowledged_autonomous_approval_does_not_run(monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 5.0)
    server = _server()
    job = SimpleNamespace(job_id="job-1", source="neural")
    # A review that asks for approval makes every action of the plan need it.
    review = SimpleNamespace(requires_approval=True, summary=lambda: "World model asked", risk_assessment=None)
    waiter = asyncio.create_task(server._wait_for_autonomous_confirmation(job, _plan(), "auto-plan", review))
    await _until(lambda: "auto-plan" in server._pending_confirms)
    connection = _ClosingConnection(reply="fails")
    handler = asyncio.create_task(server._handle_connection(connection))  # type: ignore[arg-type]
    try:
        await _confirm_over(connection, "auto-plan")
        assert await asyncio.wait_for(waiter, 2.0) is False
    finally:
        await connection.close()
        await asyncio.wait_for(handler, 2.0)


@pytest.mark.asyncio
async def test_a_narrator_interrupt_whose_approval_was_never_acknowledged_does_not_proceed():
    server = _server()
    broadcast = AsyncMock()
    narrator = ExecutionNarrator(PilotConfig(), server._pending_confirms, broadcast)
    waiter = asyncio.create_task(narrator._interrupt_and_wait(reason="risky", context={}, timeout_seconds=5.0))
    await _until(lambda: bool(server._pending_confirms))
    plan_id = next(iter(server._pending_confirms))
    connection = _ClosingConnection(reply="fails")
    handler = asyncio.create_task(server._handle_connection(connection))  # type: ignore[arg-type]
    try:
        await connection.incoming.put(
            {"jsonrpc": "2.0", "method": "confirm", "params": {"plan_id": plan_id, "confirmed": True}, "id": "confirm"}
        )
        assert await asyncio.wait_for(waiter, 2.0) is False
    finally:
        await connection.close()
        await asyncio.wait_for(handler, 2.0)


# ── Shutdown ──


@pytest.mark.asyncio
async def test_shutdown_release_is_audited_as_shutdown_and_expires_the_approval(durable, ledger):
    server = _server(durable, ledger)
    waiter, pending = await _start_waiter(server)

    await server._release_pending_confirmations_for_shutdown()

    assert await waiter == (False, set(), REQUIRED)
    assert (pending.resolution_path, pending.input_channel) == ("shutdown", "internal")
    # Before the fix: ("denied", "unknown", "confirm_rpc") and a DENIED durable row, as if the person had denied it.
    assert await _resolution(ledger) == [("expired", "internal", "shutdown")]
    approval = await durable.get_approval("plan-1")
    assert approval.status == ApprovalStatus.EXPIRED
    assert server._pending_confirms == {}


@pytest.mark.asyncio
async def test_shutdown_withdraws_an_approval_whose_reply_was_not_written(durable, ledger):
    server = _server(durable, ledger)
    waiter, pending = await _start_waiter(server)
    task = asyncio.current_task()
    pending.rpc_tasks.add(task)
    await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [0, 1]}, None
    )

    await server._release_pending_confirmations_for_shutdown()

    assert await waiter == (False, set(), REQUIRED)
    assert await _resolution(ledger) == [("expired", "internal", "shutdown")]
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.EXPIRED


@pytest.mark.asyncio
async def test_shutdown_keeps_a_denial_that_was_already_made(ledger):
    server = _server(ledger=ledger)
    waiter, _pending = await _start_waiter(server)

    assert await server._resolve_pending_denied("plan-1", resolution_path="abort") is True
    await server._release_pending_confirmations_for_shutdown()

    assert await waiter == (False, set(), REQUIRED)
    assert await _resolution(ledger) == [("denied", "internal", "abort")]


@pytest.mark.asyncio
async def test_stop_releases_waiters_as_shutdown():
    server = _server()
    waiter, pending = await _start_waiter(server)
    narrator = ExecutionNarrator(PilotConfig(), server._pending_confirms, AsyncMock())
    interrupt = asyncio.create_task(narrator._interrupt_and_wait(reason="risky", context={}, timeout_seconds=5.0))
    await _until(lambda: len(server._pending_confirms) == 2)

    await server.stop()

    assert await asyncio.wait_for(waiter, 2.0) == (False, set(), REQUIRED)
    assert await asyncio.wait_for(interrupt, 2.0) is False
    assert (pending.resolution_path, pending.input_channel) == ("shutdown", "internal")

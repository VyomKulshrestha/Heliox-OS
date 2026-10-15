"""The confirm RPC's authority channel, strict boolean and index rule, and the APPROVAL_RESOLVED audit keys."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.intelligence.experience import ExperienceEventType, ExperienceLedger
from pilot.server import PendingConfirmation, PilotServer
from pilot.workflows.durable_tasks import ApprovalStatus, DurableTaskStore, TaskStatus

FORBIDDEN_PAYLOAD_KEYS = ("dialog_pointer", "dialog_keyboard", "input_modality")


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _plan(actions: int = 2) -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(action_type=ActionType.SYSTEM_INFO, target="system", parameters=EmptyParams())
            for _ in range(actions)
        ],
        raw_input="show system info",
    )


@pytest.fixture
async def ledger(tmp_path):
    instance = ExperienceLedger(tmp_path / "experience.db")
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


def _server(ledger: ExperienceLedger | None = None) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._experience_ledger = ledger
    return server


async def _start_wait(server: PilotServer, plan_id: str = "plan-1") -> asyncio.Task:
    socket = _Socket()
    task = asyncio.create_task(server._wait_for_confirmation(plan_id, _plan(), socket, force_all_actions=True))
    for _ in range(100):
        if socket.sent and plan_id in server._pending_confirms:
            break
        await asyncio.sleep(0)
    assert plan_id in server._pending_confirms
    return task


async def _resolved_payload(ledger: ExperienceLedger, plan_id: str = "plan-1") -> dict[str, Any]:
    events = await ledger.list_events(plan_id=plan_id, event_type=ExperienceEventType.APPROVAL_RESOLVED)
    assert len(events) == 1
    return dict(events[0].payload)


async def _assert_no_modality_anywhere(ledger: ExperienceLedger) -> None:
    for event in await ledger.list_events(limit=1000):
        serialized = json.dumps(event.payload)
        for forbidden in FORBIDDEN_PAYLOAD_KEYS:
            assert forbidden not in serialized


# ── APPROVAL_RESOLVED audit keys for every resolution path ──


@pytest.mark.parametrize("channel", ["dialog", "gesture"])
@pytest.mark.asyncio
async def test_approval_by_dialog_or_gesture_records_its_channel(ledger, channel):
    server = _server(ledger)
    waiter = await _start_wait(server)

    result = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": channel, "approved_indices": [1]}, None
    )

    assert result == {"status": "ok", "confirmed": True}
    assert await waiter == (True, {1}, {0, 1})
    payload = await _resolved_payload(ledger)
    assert payload["decision"] == "approved"
    assert payload["approved_indices"] == [1]
    assert payload["input_channel"] == channel
    assert payload["resolution_path"] == "confirm_rpc"
    await _assert_no_modality_anywhere(ledger)


@pytest.mark.asyncio
async def test_an_older_client_without_a_channel_records_unknown(ledger):
    server = _server(ledger)
    waiter = await _start_wait(server)

    result = await server._handle_confirm({"plan_id": "plan-1", "confirmed": True}, None)

    assert result == {"status": "ok", "confirmed": True}
    assert await waiter == (True, {0, 1}, {0, 1})
    payload = await _resolved_payload(ledger)
    assert (payload["input_channel"], payload["resolution_path"]) == ("unknown", "confirm_rpc")


@pytest.mark.asyncio
async def test_voice_deny_through_confirm_records_the_voice_path(ledger):
    server = _server(ledger)
    waiter = await _start_wait(server)

    result = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": False, "input_channel": "voice_deny"}, None
    )

    assert result == {"status": "ok", "confirmed": False}
    assert await waiter == (False, set(), {0, 1})
    payload = await _resolved_payload(ledger)
    assert payload["decision"] == "denied"
    assert (payload["input_channel"], payload["resolution_path"]) == ("voice_deny", "voice_deny")


@pytest.mark.asyncio
async def test_spoken_denial_through_voice_dispatch_uses_the_voice_deny_channel(ledger):
    server = _server(ledger)
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    waiter = await _start_wait(server)

    await server._voice_command_dispatch("No.")

    assert await waiter == (False, set(), {0, 1})
    payload = await _resolved_payload(ledger)
    assert (payload["input_channel"], payload["resolution_path"]) == ("voice_deny", "voice_deny")
    assert server._broadcast_notification.await_args.args[1]["message"] == "Okay, I won't run that."


@pytest.mark.asyncio
async def test_interject_deny_records_voice_deny_and_interject_path(ledger):
    server = _server(ledger)
    waiter = await _start_wait(server)
    server._interactive_request_active = True
    server._active_plan_id = "plan-1"

    result = await server._handle_interject({"input": "no"}, None)

    assert result["status"] == "denied"
    assert await waiter == (False, set(), {0, 1})
    payload = await _resolved_payload(ledger)
    assert (payload["input_channel"], payload["resolution_path"]) == ("voice_deny", "interject_deny")


@pytest.mark.asyncio
async def test_interject_correction_records_an_internal_denial(ledger):
    server = _server(ledger)
    server._broadcast_notification = AsyncMock()
    waiter = await _start_wait(server)
    server._interactive_request_active = True
    server._active_plan_id = "plan-1"

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        result = await server._handle_interject({"input": "use the downloads folder instead"}, None)

    assert result["status"] == "revising"
    assert await waiter == (False, set(), {0, 1})
    payload = await _resolved_payload(ledger)
    assert (payload["input_channel"], payload["resolution_path"]) == ("internal", "interject_correction")


@pytest.mark.asyncio
async def test_abort_records_an_internal_denial(ledger):
    server = _server(ledger)
    waiter = await _start_wait(server)
    server._active_plan_id = "plan-1"

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        result = await server._handle_abort({}, None)

    assert result == {"status": "aborted"}
    assert await waiter == (False, set(), {0, 1})
    payload = await _resolved_payload(ledger)
    assert (payload["input_channel"], payload["resolution_path"]) == ("internal", "abort")


@pytest.mark.asyncio
async def test_timeout_records_internal_timeout(ledger, monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 0.05)
    server = _server(ledger)
    waiter = await _start_wait(server)

    assert await waiter == (False, set(), {0, 1})
    payload = await _resolved_payload(ledger)
    assert payload["decision"] == "expired"
    assert (payload["input_channel"], payload["resolution_path"]) == ("internal", "timeout")
    late = await server._handle_confirm({"plan_id": "plan-1", "confirmed": True}, None)
    assert late["status"] == "error"


@pytest.mark.asyncio
async def test_autonomous_approval_keeps_its_result_and_requires_indices_from_new_clients():
    server = _server()
    server._broadcast_notification = AsyncMock()
    job = SimpleNamespace(job_id="job-1", source="scheduled")
    review = SimpleNamespace(
        requires_approval=True, summary=lambda: "The world model asked for approval.", risk_assessment=None, gnomon=None
    )
    waiter = asyncio.create_task(server._wait_for_autonomous_confirmation(job, _plan(1), "auto-plan", review))
    for _ in range(100):
        if "auto-plan" in server._pending_confirms:
            break
        await asyncio.sleep(0)
    pending = server._pending_confirms["auto-plan"]
    assert pending.required_indices is not None

    refused = await server._handle_confirm({"plan_id": "auto-plan", "confirmed": True, "input_channel": "dialog"}, None)
    approved = await server._handle_confirm(
        {
            "plan_id": "auto-plan",
            "confirmed": True,
            "input_channel": "dialog",
            "approved_indices": sorted(pending.required_indices),
        },
        None,
    )

    assert refused == {"status": "error", "message": "approved_indices is required"}
    assert approved == {"status": "ok", "confirmed": True}
    assert pending.input_channel == "dialog"
    assert await waiter is True


# ── Request validation ──


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None, [], {}])
@pytest.mark.asyncio
async def test_confirmed_must_be_a_json_boolean(value):
    server = _server()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), required_indices=frozenset({0}))
    server._pending_confirms["plan-1"] = pending

    result = await server._handle_confirm({"plan_id": "plan-1", "confirmed": value}, None)

    assert result == {"status": "error", "message": "confirmed must be a boolean"}
    assert pending.resolved is False
    assert not pending.event.is_set()


@pytest.mark.parametrize("channel", ["dialog_pointer", "dialog_keyboard", "keyboard", "voice", "", 3, ["dialog"]])
@pytest.mark.asyncio
async def test_unknown_channels_are_rejected(channel):
    server = _server()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), required_indices=frozenset({0}))
    server._pending_confirms["plan-1"] = pending

    result = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": False, "input_channel": channel, "approved_indices": []}, None
    )

    assert result == {
        "status": "error",
        "message": "input_channel must be one of dialog, gesture, voice_deny, internal, unknown",
    }
    assert pending.resolved is False


@pytest.mark.asyncio
async def test_voice_deny_can_never_approve():
    server = _server()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), required_indices=frozenset({0}))
    server._pending_confirms["plan-1"] = pending

    result = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "voice_deny", "approved_indices": [0]}, None
    )

    assert result == {"status": "error", "message": "voice_deny can only deny"}
    assert pending.resolved is False and pending.confirmed is False


@pytest.mark.asyncio
async def test_plan_approval_with_a_channel_needs_explicit_indices():
    server = _server()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), required_indices=frozenset({0, 1}))
    server._pending_confirms["plan-1"] = pending

    refused = await server._handle_confirm({"plan_id": "plan-1", "confirmed": True, "input_channel": "gesture"}, None)
    assert refused == {"status": "error", "message": "approved_indices is required"}
    assert pending.resolved is False

    denied = await server._handle_confirm({"plan_id": "plan-1", "confirmed": False, "input_channel": "gesture"}, None)
    assert denied == {"status": "ok", "confirmed": False}
    assert pending.input_channel == "gesture"
    assert pending.resolution_path == "confirm_rpc"


@pytest.mark.parametrize("channel", ["dialog", "gesture", "internal", "unknown"])
@pytest.mark.asyncio
async def test_narrator_style_confirmations_keep_todays_semantics(channel):
    server = _server()
    pending = PendingConfirmation(plan_id="interrupt_1234", event=asyncio.Event())
    server._pending_confirms["interrupt_1234"] = pending

    result = await server._handle_confirm(
        {"plan_id": "interrupt_1234", "confirmed": True, "input_channel": channel}, None
    )

    assert result == {"status": "ok", "confirmed": True}
    assert pending.confirmed is True
    assert pending.approved_indices is None
    assert pending.event.is_set()
    assert pending.input_channel == channel


@pytest.mark.asyncio
async def test_a_set_resolution_path_is_not_overwritten_by_confirm():
    server = _server()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event(), resolution_path="interject_deny")
    server._pending_confirms["plan-1"] = pending

    await server._handle_confirm({"plan_id": "plan-1", "confirmed": False, "input_channel": "dialog"}, None)

    assert pending.resolution_path == "interject_deny"


# ── Durable path (no in-memory confirmation after a restart) ──


@pytest.fixture
async def durable(tmp_path):
    store = DurableTaskStore(tmp_path / "durable.db")
    await store.initialize()
    await store.create_task(task_id="task-1", session_id="session-1", user_input="show system info")
    await store.transition("task-1", TaskStatus.PLANNING)
    await store.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
    await store.create_approval(
        task_id="task-1", plan_id="plan-1", request={"action_indices": [0, 1]}, timeout_seconds=300
    )
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"confirmed": "true"}, "confirmed must be a boolean"),
        ({"confirmed": 1}, "confirmed must be a boolean"),
        ({"confirmed": True, "input_channel": "dialog_keyboard"}, "input_channel must be one of"),
        ({"confirmed": True, "input_channel": "voice_deny", "approved_indices": [0]}, "voice_deny can only deny"),
        ({"confirmed": True, "input_channel": "dialog"}, "approved_indices is required"),
    ],
)
@pytest.mark.asyncio
async def test_durable_path_applies_the_same_rules(durable, params, message):
    server = _server()
    server._durable_tasks = durable

    result = await server._handle_confirm({"plan_id": "plan-1", **params}, None)

    assert result["status"] == "error"
    assert result["message"].startswith(message)
    approval = await durable.get_approval("plan-1")
    assert approval is not None and approval.status == ApprovalStatus.PENDING


@pytest.mark.asyncio
async def test_durable_path_accepts_a_channel_with_indices_and_keeps_its_result_shape(durable):
    server = _server()
    server._durable_tasks = durable

    result = await server._handle_confirm(
        {"plan_id": "plan-1", "confirmed": True, "input_channel": "dialog", "approved_indices": [1]}, None
    )

    assert result == {"status": "ok", "confirmed": True, "resume_required": True, "task_id": "task-1"}
    approval = await durable.get_approval("plan-1")
    assert approval is not None
    assert approval.status == ApprovalStatus.APPROVED
    assert approval.approved_indices == [1]

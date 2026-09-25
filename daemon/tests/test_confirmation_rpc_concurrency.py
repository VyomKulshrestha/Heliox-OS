"""Regression coverage for same-connection confirmation RPCs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import websockets

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams, ShellCommandParams
from pilot.config import PilotConfig
from pilot.server import PendingConfirmation, PilotServer, _notification
from pilot.workflows.recorded_workflows import RecordedWorkflowStore


async def _authenticated_socket(server: PilotServer):
    listener = await websockets.serve(server._handle_connection, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    socket = await websockets.connect(f"ws://127.0.0.1:{port}")
    await socket.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "auth",
                "params": {"token": "test-token"},
                "id": "auth",
            }
        )
    )
    auth = json.loads(await socket.recv())
    assert auth["result"]["status"] == "authenticated"
    return listener, socket


@pytest.mark.asyncio
async def test_confirm_is_dispatched_while_request_waits_on_same_socket():
    """An execute-like request must not block its own confirmation RPC."""
    config = PilotConfig()
    config.server.auth_token = "test-token"
    server = PilotServer(config)

    async def await_confirmation(_params, ws):
        pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event())
        server._pending_confirms[pending.plan_id] = pending
        await ws.send(
            _notification(
                "confirm_required",
                {"plan_id": pending.plan_id, "actions": [{"index": 0}]},
            )
        )
        try:
            await asyncio.wait_for(pending.event.wait(), timeout=1.0)
            return {"status": "executed" if pending.confirmed else "cancelled"}
        finally:
            server._pending_confirms.pop(pending.plan_id, None)

    server._handlers = {
        "await_confirmation": await_confirmation,
        "confirm": server._handle_confirm,
    }

    listener, socket = await _authenticated_socket(server)
    try:
        await socket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "await_confirmation",
                    "params": {},
                    "id": "execute",
                }
            )
        )
        required = json.loads(await asyncio.wait_for(socket.recv(), timeout=1.0))
        assert required["method"] == "confirm_required"

        await socket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "confirm",
                    "params": {"plan_id": "plan-1", "confirmed": True},
                    "id": "confirm",
                }
            )
        )

        acknowledgement = json.loads(await asyncio.wait_for(socket.recv(), timeout=1.0))
        execution = json.loads(await asyncio.wait_for(socket.recv(), timeout=1.0))

        assert acknowledgement == {
            "jsonrpc": "2.0",
            "result": {"status": "ok", "confirmed": True},
            "id": "confirm",
        }
        assert execution == {
            "jsonrpc": "2.0",
            "result": {"status": "executed"},
            "id": "execute",
        }
    finally:
        await socket.close()
        listener.close()
        await listener.wait_closed()


@pytest.mark.asyncio
async def test_first_confirmation_decision_wins_when_rpc_requests_overlap():
    """Duplicate approve/deny RPCs cannot rewrite an accepted decision."""
    config = PilotConfig()
    config.server.auth_token = "test-token"
    server = PilotServer(config)
    pending = PendingConfirmation(plan_id="plan-race", event=asyncio.Event())
    server._pending_confirms[pending.plan_id] = pending
    server._handlers = {"confirm": server._handle_confirm}

    listener, socket = await _authenticated_socket(server)
    try:
        for request_id, confirmed in (("approve", True), ("deny", False)):
            await socket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "confirm",
                        "params": {"plan_id": pending.plan_id, "confirmed": confirmed},
                        "id": request_id,
                    }
                )
            )

        responses = {}
        while len(responses) < 2:
            response = json.loads(await asyncio.wait_for(socket.recv(), timeout=1.0))
            responses[response["id"]] = response["result"]

        accepted = [request_id for request_id, result in responses.items() if result["status"] == "ok"]
        rejected = [request_id for request_id, result in responses.items() if result["status"] == "error"]
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert pending.confirmed is (accepted[0] == "approve")
        assert pending.event.is_set()
    finally:
        await socket.close()
        listener.close()
        await listener.wait_closed()


@pytest.mark.asyncio
async def test_recording_can_be_stopped_while_same_socket_task_is_active(tmp_path):
    config = PilotConfig()
    config.server.auth_token = "test-token"
    server = PilotServer(config)
    server._recorded_workflows = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32
    )
    server._recorded_workflows.start(name="Task", consent=True)
    released = asyncio.Event()

    async def long_request(_params, ws):
        await ws.send(_notification("long_request_started", {}))
        await released.wait()
        return {"status": "done"}

    server._handlers = {
        "long_request": long_request,
        "recorded_workflow_status": server._handle_recorded_workflow_status,
        "recorded_workflow_stop": server._handle_recorded_workflow_stop,
    }
    listener, socket = await _authenticated_socket(server)
    try:
        await socket.send(json.dumps({"jsonrpc": "2.0", "method": "long_request", "params": {}, "id": "long"}))
        assert json.loads(await asyncio.wait_for(socket.recv(), 1))["method"] == "long_request_started"
        await socket.send(json.dumps({"jsonrpc": "2.0", "method": "recorded_workflow_status", "params": {}, "id": "status"}))
        status = json.loads(await asyncio.wait_for(socket.recv(), 1))
        assert status["id"] == "status" and status["result"]["active"] is True
        await socket.send(json.dumps({"jsonrpc": "2.0", "method": "recorded_workflow_stop", "params": {"save": False}, "id": "stop"}))
        stopped = json.loads(await asyncio.wait_for(socket.recv(), 1))
        assert stopped["id"] == "stop" and stopped["result"]["saved"] is False
        released.set()
        assert json.loads(await asyncio.wait_for(socket.recv(), 1))["id"] == "long"
    finally:
        released.set()
        await socket.close()
        listener.close()
        await listener.wait_closed()


@pytest.mark.asyncio
async def test_world_model_confirmation_forces_all_actions_and_includes_reason():
    server = PilotServer(PilotConfig())
    sent = []

    class _Socket:
        async def send(self, payload):
            sent.append(json.loads(payload))

    plan = ActionPlan(
        actions=[Action(action_type=ActionType.SYSTEM_INFO, target="system", parameters=EmptyParams())],
        raw_input="show system info",
    )
    assessment = {
        "world_model_score": 0.8,
        "reasons": ["predicted disk usage 96% exceeds the safe threshold"],
        "prediction_sources": ["learned", "rule"],
        "requires_confirmation": True,
    }
    task = asyncio.create_task(
        server._wait_for_confirmation(
            "world-plan",
            plan,
            _Socket(),
            reason="World model paused this plan at 80% predicted risk.",
            risk_assessment=assessment,
            force_all_actions=True,
        )
    )
    await asyncio.sleep(0)

    notification = sent[0]
    assert notification["method"] == "confirm_required"
    assert notification["params"]["reason"].startswith("World model paused")
    assert notification["params"]["risk_assessment"] == assessment
    assert notification["params"]["actions"][0]["index"] == 0

    pending = server._pending_confirms["world-plan"]
    pending.confirmed = False
    pending.event.set()
    confirmed, approved, required = await task
    assert confirmed is False
    assert approved == set()
    assert required == {0}


@pytest.mark.asyncio
async def test_background_autonomous_confirmation_uses_shared_confirm_rpc():
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SHELL_COMMAND,
                target="whoami",
                parameters=ShellCommandParams(command="whoami"),
            )
        ],
        raw_input="run whoami",
    )
    job = SimpleNamespace(job_id="job-approval", source="neural")
    task = asyncio.create_task(server._wait_for_autonomous_confirmation(job, plan, "background-plan"))
    await asyncio.sleep(0)

    server._broadcast_notification.assert_awaited_once()
    method, payload = server._broadcast_notification.await_args.args
    assert method == "confirm_required"
    assert payload["task_id"] == "job-approval"
    assert payload["source"] == "neural"
    assert payload["actions"][0]["action_type"] == "shell_command"

    result = await server._handle_confirm(
        {"plan_id": "background-plan", "confirmed": True},
        SimpleNamespace(),
    )
    assert result == {"status": "ok", "confirmed": True}
    assert await task is True
    assert "background-plan" not in server._pending_confirms

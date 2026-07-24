"""Regression coverage for same-connection confirmation RPCs."""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from pilot.config import PilotConfig
from pilot.server import PendingConfirmation, PilotServer, _notification


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

        responses = {}
        while set(responses) != {"confirm", "execute"}:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=1.0))
            if message.get("id") in {"confirm", "execute"}:
                responses[message["id"]] = message["result"]

        assert responses["confirm"] == {"status": "ok", "confirmed": True}
        assert responses["execute"] == {"status": "executed"}
    finally:
        await socket.close()
        listener.close()
        await listener.wait_closed()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import websockets
from mcp import Client

from pilot import __version__
from pilot.config import PilotConfig
from pilot.mcp_server import DaemonRpcError, HelioxDaemonClient, _package_version, create_mcp_server
from pilot.server import PilotServer


def test_local_mcp_version_matches_running_heliox_package() -> None:
    assert _package_version() == __version__


class _RecordingDaemon:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.timeouts: list[float | None] = []

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload = params or {}
        self.calls.append((method, payload))
        self.timeouts.append(timeout_seconds)
        if method == "mcp_submit_task":
            return {"status": "submitted", "task_id": "task-1", "requires_user_approval": True}
        if method == "mcp_task_status":
            return {"status": "awaiting_approval", "task_id": payload["task_id"]}
        return {"status": "ok", "method": method}


@pytest.mark.asyncio
async def test_local_mcp_lists_bounded_tools_with_truthful_annotations() -> None:
    daemon = _RecordingDaemon()
    server = create_mcp_server(daemon)

    async with Client(server) as client:
        listed = await client.list_tools()

    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == {
        "heliox_health",
        "get_heliox_system_status",
        "list_heliox_capabilities",
        "preview_heliox_task",
        "score_heliox_typed_decision",
        "submit_heliox_task",
        "get_heliox_task_status",
        "cancel_heliox_task",
    }
    assert tools["heliox_health"].annotations.read_only_hint is True
    assert tools["preview_heliox_task"].annotations.read_only_hint is True
    assert tools["score_heliox_typed_decision"].annotations.read_only_hint is True
    assert tools["score_heliox_typed_decision"].annotations.destructive_hint is False
    assert tools["score_heliox_typed_decision"].annotations.open_world_hint is False
    assert tools["submit_heliox_task"].annotations.read_only_hint is False
    assert tools["submit_heliox_task"].annotations.destructive_hint is True
    submit_schema = tools["submit_heliox_task"].input_schema["properties"]
    assert submit_schema["input"]["maxLength"] == 20_000
    assert submit_schema["session_id"]["maxLength"] == 80
    assert submit_schema["request_id"]["maxLength"] == 128
    assert "approve" not in " ".join(tools).lower()


@pytest.mark.asyncio
async def test_local_mcp_submission_and_status_preserve_async_contract() -> None:
    daemon = _RecordingDaemon()
    server = create_mcp_server(daemon)

    async with Client(server) as client:
        submitted = await client.call_tool(
            "submit_heliox_task",
            {"input": "inspect system health", "session_id": "codex"},
        )
        status = await client.call_tool("get_heliox_task_status", {"task_id": "task-1"})

    assert submitted.is_error is False
    assert submitted.structured_content == {
        "status": "submitted",
        "task_id": "task-1",
        "requires_user_approval": True,
    }
    assert status.structured_content["status"] == "awaiting_approval"
    assert daemon.calls == [
        (
            "mcp_submit_task",
            {"input": "inspect system health", "session_id": "codex", "request_id": ""},
        ),
        ("mcp_task_status", {"task_id": "task-1"}),
    ]


@pytest.mark.asyncio
async def test_typed_decision_tool_passes_dynamic_state_without_action_side_effects() -> None:
    daemon = _RecordingDaemon()
    server = create_mcp_server(daemon)

    async with Client(server) as client:
        result = await client.call_tool(
            "score_heliox_typed_decision",
            {
                "state": {"request": "read status", "candidate": "collect evidence"},
                "kind": "choice",
                "question": "Which candidate best satisfies the request?",
                "options": ["collect evidence", "change settings"],
            },
        )

    assert result.is_error is False
    assert daemon.calls == [
        (
            "mcp_typed_decision",
            {
                "state": {"request": "read status", "candidate": "collect evidence"},
                "kind": "choice",
                "question": "Which candidate best satisfies the request?",
                "options": ["collect evidence", "change settings"],
            },
        )
    ]
    assert daemon.timeouts == [120]


@pytest.mark.parametrize(
    "uri",
    [
        "ws://192.168.1.10:8765",
        "ws://example.com:8765",
        "ws://user:secret@127.0.0.1:8765",
        "ws://127.0.0.1:8765/other",
    ],
)
def test_daemon_client_rejects_nonlocal_or_ambiguous_urls(uri: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        HelioxDaemonClient(uri=uri, token_file=tmp_path / "token")


def test_daemon_client_accepts_ipv4_ipv6_and_localhost_loopback(tmp_path: Path) -> None:
    for uri in ("ws://127.0.0.1:8765", "ws://[::1]:8765", "ws://localhost:8765"):
        HelioxDaemonClient(uri=uri, token_file=tmp_path / "token")


@pytest.mark.asyncio
async def test_daemon_client_authenticates_with_mcp_token_and_calls_allowlisted_rpc(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "mcp_auth_token"
    token_file.write_text("mcp-secret", encoding="utf-8")
    received: list[dict[str, Any]] = []

    async def handler(socket) -> None:
        auth = json.loads(await socket.recv())
        received.append(auth)
        await socket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": auth["id"],
                    "result": {"status": "authenticated", "role": "mcp_local"},
                }
            )
        )
        request = json.loads(await socket.recv())
        received.append(request)
        await socket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"status": "ok", "action_count": 157},
                }
            )
        )

    listener = await websockets.serve(handler, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    client = HelioxDaemonClient(
        uri=f"ws://127.0.0.1:{port}",
        token_file=token_file,
        timeout_seconds=1,
    )
    try:
        result = await client.call("capabilities")
    finally:
        listener.close()
        await listener.wait_closed()

    assert result == {"status": "ok", "action_count": 157}
    assert received[0]["params"]["token"] == "mcp-secret"
    assert received[1]["method"] == "capabilities"


@pytest.mark.asyncio
async def test_daemon_client_fails_cleanly_when_daemon_has_not_created_token(
    tmp_path: Path,
) -> None:
    client = HelioxDaemonClient(token_file=tmp_path / "missing-token")

    with pytest.raises(DaemonRpcError, match="Start the desktop app"):
        await client.call("health")


@pytest.mark.asyncio
async def test_official_mcp_client_reaches_role_scoped_daemon_dispatch(tmp_path: Path) -> None:
    token_file = tmp_path / "mcp_auth_token"
    token_file.write_text("mcp-secret", encoding="utf-8")
    daemon = PilotServer(PilotConfig())
    daemon._mcp_auth_token = "mcp-secret"
    daemon._handlers = {
        "mcp_plan_task": AsyncMock(
            return_value={
                "status": "preview",
                "authoritative": False,
                "requires_user_approval": True,
                "actions": [],
            }
        )
    }
    listener = await websockets.serve(daemon._handle_connection, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    bridge = HelioxDaemonClient(
        uri=f"ws://127.0.0.1:{port}",
        token_file=token_file,
        timeout_seconds=1,
    )
    mcp_server = create_mcp_server(bridge)
    try:
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "preview_heliox_task",
                {"input": "inspect system health", "session_id": "host-e2e"},
            )
    finally:
        listener.close()
        await listener.wait_closed()

    assert result.is_error is False
    assert result.structured_content == {
        "status": "preview",
        "authoritative": False,
        "requires_user_approval": True,
        "actions": [],
    }
    daemon._handlers["mcp_plan_task"].assert_awaited_once()
    assert daemon._handlers["mcp_plan_task"].await_args.args[0] == {
        "input": "inspect system health",
        "session_id": "host-e2e",
    }

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.network.mesh import HelioxMesh
from pilot.network.peer_connection import (
    PeerAuthenticationError,
    PeerCapabilities,
    PeerConnection,
    decode_peer_message,
    encode_peer_message,
)

SECRET = b"shared-mesh-secret-that-is-at-least-32-bytes"


def test_signed_peer_message_round_trip_and_replay_rejection():
    raw = encode_peer_message(
        SECRET,
        "peer-a",
        "peer_info",
        {"instance_id": "peer-a"},
        timestamp=1_000,
        nonce="unique-nonce",
    )
    seen: dict[str, int] = {}

    decoded = decode_peer_message(SECRET, raw, seen, expected_sender="peer-a", now=1_000)

    assert decoded["payload"] == {"instance_id": "peer-a"}
    with pytest.raises(PeerAuthenticationError, match="replayed"):
        decode_peer_message(SECRET, raw, seen, expected_sender="peer-a", now=1_000)


def test_tampered_peer_message_is_rejected():
    raw = encode_peer_message(
        SECRET,
        "peer-a",
        "task_delegate",
        {"task_id": "safe"},
        timestamp=1_000,
        nonce="unique-nonce",
    )
    tampered = json.loads(raw)
    tampered["payload"]["task_id"] = "changed"

    with pytest.raises(PeerAuthenticationError, match="signature"):
        decode_peer_message(SECRET, json.dumps(tampered), {}, now=1_000)


@pytest.mark.asyncio
async def test_attached_inbound_connection_sends_signed_reply_directly():
    websocket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
    connection = PeerConnection(
        peer_id="peer-a",
        host="127.0.0.1",
        port=8786,
        own_capabilities=PeerCapabilities(instance_id="local"),
        shared_secret=SECRET,
    )
    connection.attach_inbound(websocket, PeerCapabilities(instance_id="peer-a"))

    await connection.send("task_result", {"task_id": "one"})

    websocket.send.assert_awaited_once()
    raw = websocket.send.await_args.args[0]
    decoded = decode_peer_message(SECRET, raw, {}, expected_sender="local")
    assert decoded["type"] == "task_result"
    assert decoded["payload"] == {"task_id": "one"}


class _InboundSocket:
    remote_address = ("192.168.1.50", 55000)

    def __init__(self, messages):
        self._messages = iter(messages)
        self.closed = None
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self, *, code, reason):
        self.closed = (code, reason)

    async def send(self, raw):
        self.sent.append(raw)


@pytest.mark.asyncio
async def test_mesh_rejects_unsigned_inbound_peer_before_dispatch():
    config = SimpleNamespace(port=8786, collab_exec_enabled=True, skill_sync_enabled=True)
    executor = SimpleNamespace(execute=AsyncMock())
    plugin_manager = SimpleNamespace(list_plugins=lambda: [])
    mesh = HelioxMesh(config, executor, plugin_manager, shared_secret=SECRET)
    mesh._on_peer_message = AsyncMock()
    websocket = _InboundSocket([json.dumps({"type": "task_delegate", "payload": {"task_id": "attack"}})])

    await mesh._handle_inbound_peer(websocket)

    assert websocket.closed == (1008, "peer authentication failed")
    mesh._on_peer_message.assert_not_awaited()
    executor.execute.assert_not_awaited()
    assert mesh.peer_ids == []


@pytest.mark.asyncio
async def test_mesh_completes_authenticated_inbound_handshake_and_heartbeat():
    config = SimpleNamespace(port=8786, collab_exec_enabled=False, skill_sync_enabled=False)
    plugin_manager = SimpleNamespace(list_plugins=lambda: [])
    mesh = HelioxMesh(config, MagicMock(), plugin_manager, shared_secret=SECRET)
    peer_info = encode_peer_message(
        SECRET,
        "peer-a",
        "peer_info",
        PeerCapabilities(instance_id="peer-a", hostname="peer-host").__dict__,
        nonce="peer-info-nonce",
    )
    heartbeat = encode_peer_message(
        SECRET,
        "peer-a",
        "heartbeat",
        {"ts": 1},
        nonce="heartbeat-nonce",
    )
    websocket = _InboundSocket([peer_info, heartbeat])

    await mesh._handle_inbound_peer(websocket)

    assert websocket.closed is None
    assert len(websocket.sent) == 2
    handshake_reply = decode_peer_message(
        SECRET,
        websocket.sent[0],
        {},
        expected_sender=mesh.instance_id,
    )
    heartbeat_reply = decode_peer_message(
        SECRET,
        websocket.sent[1],
        {},
        expected_sender=mesh.instance_id,
    )
    assert handshake_reply["type"] == "peer_info"
    assert heartbeat_reply["type"] == "heartbeat_ack"
    assert mesh.peer_ids == []


@pytest.mark.asyncio
async def test_mesh_closes_silent_inbound_peer_using_configured_timeout(monkeypatch):
    config = SimpleNamespace(
        port=8786,
        peer_timeout_s=12,
        collab_exec_enabled=False,
        skill_sync_enabled=False,
    )
    mesh = HelioxMesh(config, MagicMock(), MagicMock(), shared_secret=SECRET)
    websocket = _InboundSocket([])
    observed: dict[str, float] = {}

    async def timeout_immediately(awaitable, *, timeout):
        observed["timeout"] = timeout
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("pilot.network.mesh.bounded", timeout_immediately)

    await mesh._handle_inbound_peer(websocket)

    assert observed["timeout"] == 12
    assert websocket.closed == (1001, "peer timed out")


def test_peer_connection_derives_heartbeat_cadence_from_timeout():
    connection = PeerConnection(
        peer_id="peer-a",
        host="127.0.0.1",
        port=8786,
        own_capabilities=PeerCapabilities(instance_id="local"),
        shared_secret=SECRET,
        heartbeat_timeout_seconds=30,
    )

    assert connection._heartbeat_timeout_seconds == 30
    assert connection._heartbeat_interval_seconds == 10


def test_mesh_rejects_short_shared_secret():
    config = SimpleNamespace(port=8786, collab_exec_enabled=False, skill_sync_enabled=False)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        HelioxMesh(config, MagicMock(), MagicMock(), shared_secret=b"short")

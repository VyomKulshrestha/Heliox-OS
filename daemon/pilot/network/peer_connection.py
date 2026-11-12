"""Peer-to-peer WebSocket connection manager.

Each ``PeerConnection`` manages a single persistent asyncio WebSocket
connection to one remote Heliox OS instance.  Messages are framed as
JSON-RPC 2.0 notifications (no request/response needed for most P2P traffic).

Message types
-------------
``skill_sync``      — plugin source payload (name, source, metadata)
``skill_ack``       — acknowledgement after installing a received plugin
``task_delegate``   — an ActionPlan batch delegated for remote execution
``task_result``     — ActionResult list returned after remote execution
``heartbeat``       — keepalive ping
``heartbeat_ack``   — keepalive pong
``peer_info``       — capability advertisement sent on connect
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pilot.timeouts import bounded

logger = logging.getLogger("pilot.network.peer_connection")

_HEARTBEAT_INTERVAL = 15  # seconds between heartbeat pings
_HEARTBEAT_TIMEOUT = 45  # seconds before declaring a peer dead
_MESSAGE_CLOCK_SKEW_SECONDS = 90


class PeerAuthenticationError(ValueError):
    """Raised when a peer message is unsigned, forged, stale, or replayed."""


def encode_peer_message(
    shared_secret: bytes,
    sender_id: str,
    msg_type: str,
    payload: dict[str, Any],
    *,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> str:
    """Create a signed peer envelope without transmitting the shared secret."""
    body = {
        "sender": sender_id,
        "type": msg_type,
        "payload": payload,
        "timestamp": int(time.time()) if timestamp is None else timestamp,
        "nonce": nonce or secrets.token_urlsafe(18),
    }
    canonical = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body["signature"] = hmac.new(shared_secret, canonical, hashlib.sha256).hexdigest()
    return json.dumps(body, separators=(",", ":"))


def decode_peer_message(
    shared_secret: bytes,
    raw: str | bytes,
    seen_nonces: dict[str, int],
    *,
    expected_sender: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Verify a signed peer envelope and reject stale or replayed messages."""
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PeerAuthenticationError("peer message is not valid JSON") from exc
    if not isinstance(message, dict):
        raise PeerAuthenticationError("peer message must be an object")

    required = {"sender", "type", "payload", "timestamp", "nonce", "signature"}
    if not required.issubset(message):
        raise PeerAuthenticationError("peer message is unsigned or incomplete")
    sender = message["sender"]
    msg_type = message["type"]
    payload = message["payload"]
    timestamp = message["timestamp"]
    nonce = message["nonce"]
    signature = message["signature"]
    if not isinstance(sender, str) or not sender or len(sender) > 128:
        raise PeerAuthenticationError("peer sender is invalid")
    if expected_sender is not None and sender != expected_sender:
        raise PeerAuthenticationError("peer sender does not match discovery identity")
    if not isinstance(msg_type, str) or not msg_type or len(msg_type) > 64:
        raise PeerAuthenticationError("peer message type is invalid")
    if not isinstance(payload, dict):
        raise PeerAuthenticationError("peer payload must be an object")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise PeerAuthenticationError("peer timestamp is invalid")
    if not isinstance(nonce, str) or not nonce or len(nonce) > 128:
        raise PeerAuthenticationError("peer nonce is invalid")
    if not isinstance(signature, str):
        raise PeerAuthenticationError("peer signature is invalid")

    current_time = int(time.time()) if now is None else now
    if abs(current_time - timestamp) > _MESSAGE_CLOCK_SKEW_SECONDS:
        raise PeerAuthenticationError("peer message is stale")
    for cached_nonce, cached_at in tuple(seen_nonces.items()):
        if current_time - cached_at > _MESSAGE_CLOCK_SKEW_SECONDS:
            seen_nonces.pop(cached_nonce, None)
    if nonce in seen_nonces:
        raise PeerAuthenticationError("peer message was replayed")

    signed_body = {
        "sender": sender,
        "type": msg_type,
        "payload": payload,
        "timestamp": timestamp,
        "nonce": nonce,
    }
    canonical = json.dumps(signed_body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    expected = hmac.new(shared_secret, canonical, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise PeerAuthenticationError("peer signature is invalid")
    seen_nonces[nonce] = timestamp
    return signed_body


@dataclass
class PeerCapabilities:
    """What a peer can do — sent on handshake."""

    instance_id: str
    hostname: str = ""
    version: str = "0.7"
    can_execute: bool = True  # can accept delegated tasks
    cpu_load: float = 0.0  # 0.0–1.0, used for load balancing
    vram_free: int = 0  # available VRAM in bytes
    has_gpu: bool = False  # does the peer have an NVIDIA GPU?
    plugin_names: list[str] = field(default_factory=list)


class PeerConnection:
    """Manages a WebSocket connection to a single peer.

    Parameters
    ----------
    peer_id:
        The remote instance's unique ID.
    host / port:
        Address of the remote peer's P2P server.
    own_capabilities:
        This instance's capabilities, sent during handshake.
    on_message:
        Async callback invoked for every inbound message.
        Signature: ``async def on_message(peer_id, msg_type, payload) -> None``
    on_disconnect:
        Sync callback invoked when the connection drops.
    """

    def __init__(
        self,
        peer_id: str,
        host: str,
        port: int,
        own_capabilities: PeerCapabilities,
        shared_secret: bytes,
        on_message: Callable[[str, str, dict[str, Any]], Any] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
        heartbeat_timeout_seconds: float = _HEARTBEAT_TIMEOUT,
    ) -> None:
        self.peer_id = peer_id
        self.host = host
        self.port = port
        self._own_caps = own_capabilities
        self._shared_secret = shared_secret
        self._on_message = on_message
        self._on_disconnect = on_disconnect
        self._heartbeat_timeout_seconds = max(5.0, float(heartbeat_timeout_seconds))
        self._heartbeat_interval_seconds = min(
            float(_HEARTBEAT_INTERVAL),
            max(1.0, self._heartbeat_timeout_seconds / 3.0),
        )

        self._ws: Any = None  # websockets connection
        self._connected = False
        self._last_heartbeat = 0.0
        self._peer_caps: PeerCapabilities | None = None
        self._send_queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._direct_send = False
        self._seen_nonces: dict[str, int] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def peer_capabilities(self) -> PeerCapabilities | None:
        return self._peer_caps

    async def connect(self) -> bool:
        """Establish the WebSocket connection and start the I/O loop."""
        try:
            import websockets

            uri = f"ws://{self.host}:{self.port}/peer"
            self._ws = await websockets.connect(uri, open_timeout=5, ping_interval=None)
            self._connected = True
            self._last_heartbeat = time.time()
            logger.info("PeerConnection: connected to %s @ %s:%d", self.peer_id, self.host, self.port)

            # Send our capabilities on connect
            await self._send_raw("peer_info", self._own_caps.__dict__)

            # Start I/O tasks
            self._task = asyncio.create_task(self._run())
            return True
        except Exception as exc:
            logger.warning("PeerConnection: failed to connect to %s: %s", self.peer_id, exc)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        self._connected = False
        task = self._task
        self._task = None
        if task and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("PeerConnection: disconnected from %s", self.peer_id)

    async def send(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Enqueue a message for sending."""
        if not self._connected:
            logger.warning("PeerConnection.send: not connected to %s", self.peer_id)
            return
        raw = encode_peer_message(
            self._shared_secret,
            self._own_caps.instance_id,
            msg_type,
            payload,
        )
        if self._direct_send:
            await self._ws.send(raw)
            return
        await self._send_queue.put(raw)

    def attach_inbound(self, websocket: Any, peer_capabilities: PeerCapabilities) -> None:
        """Attach an accepted socket whose receive loop is owned by the mesh."""
        self._ws = websocket
        self._connected = True
        self._direct_send = True
        self._last_heartbeat = time.time()
        self._peer_caps = peer_capabilities

    def note_heartbeat(self) -> None:
        """Refresh liveness for a receive loop owned by the mesh server."""
        self._last_heartbeat = time.time()

    async def _send_raw(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Send immediately (used for handshake before the queue loop starts)."""
        if self._ws:
            await self._ws.send(
                encode_peer_message(
                    self._shared_secret,
                    self._own_caps.instance_id,
                    msg_type,
                    payload,
                )
            )

    async def _run(self) -> None:
        """Main I/O loop: receive messages and drain the send queue."""
        recv_task = asyncio.create_task(self._recv_loop())
        send_task = asyncio.create_task(self._send_loop())
        hb_task = asyncio.create_task(self._heartbeat_loop())

        try:
            done, pending = await asyncio.wait(
                [recv_task, send_task, hb_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            recv_task.cancel()
            send_task.cancel()
            hb_task.cancel()
            await asyncio.gather(recv_task, send_task, hb_task, return_exceptions=True)
            raise
        finally:
            self._connected = False
            if self._on_disconnect:
                self._on_disconnect(self.peer_id)

    async def _recv_loop(self) -> None:
        """Receive and dispatch inbound messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = decode_peer_message(
                        self._shared_secret,
                        raw,
                        self._seen_nonces,
                        expected_sender=self.peer_id,
                    )
                except PeerAuthenticationError as exc:
                    logger.warning("PeerConnection: rejected unauthenticated peer %s: %s", self.peer_id, exc)
                    await self._ws.close(code=1008, reason="peer authentication failed")
                    return
                try:
                    msg_type = msg.get("type", "")
                    payload = msg.get("payload", {})

                    if msg_type == "heartbeat":
                        await self._send_raw("heartbeat_ack", {})
                        self._last_heartbeat = time.time()
                        continue
                    if msg_type == "heartbeat_ack":
                        self._last_heartbeat = time.time()
                        continue
                    if msg_type == "peer_info":
                        if payload.get("instance_id") != self.peer_id:
                            logger.warning(
                                "PeerConnection: peer %s advertised a mismatched signed identity",
                                self.peer_id,
                            )
                            await self._ws.close(code=1008, reason="peer identity mismatch")
                            return
                        self._peer_caps = PeerCapabilities(**payload)
                        logger.debug("PeerConnection: received caps from %s", self.peer_id)
                        continue

                    if self._on_message:
                        await self._on_message(self.peer_id, msg_type, payload)
                except Exception as exc:
                    logger.warning("PeerConnection: error handling message from %s: %s", self.peer_id, exc)
        except Exception as exc:
            logger.info("PeerConnection: recv loop ended for %s: %s", self.peer_id, exc)

    async def _send_loop(self) -> None:
        """Drain the outbound send queue."""
        while self._connected:
            try:
                raw = await bounded(self._send_queue.get(), timeout=1.0)
                await self._ws.send(raw)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.warning("PeerConnection: send error to %s: %s", self.peer_id, exc)
                break

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and detect dead peers."""
        while self._connected:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            if not self._connected:
                break
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self._heartbeat_timeout_seconds:
                logger.warning("PeerConnection: peer %s timed out (%.0fs)", self.peer_id, elapsed)
                break
            await self._send_raw("heartbeat", {"ts": time.time()})

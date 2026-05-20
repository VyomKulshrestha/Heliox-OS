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
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("pilot.network.peer_connection")

_HEARTBEAT_INTERVAL = 15  # seconds between heartbeat pings
_HEARTBEAT_TIMEOUT = 45  # seconds before declaring a peer dead


@dataclass
class PeerCapabilities:
    """What a peer can do — sent on handshake."""

    instance_id: str
    hostname: str = ""
    version: str = "0.7"
    can_execute: bool = True  # can accept delegated tasks
    cpu_load: float = 0.0  # 0.0–1.0, used for load balancing
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
        on_message: Callable[[str, str, dict[str, Any]], Any] | None = None,
        on_disconnect: Callable[[str], None] | None = None,
    ) -> None:
        self.peer_id = peer_id
        self.host = host
        self.port = port
        self._own_caps = own_capabilities
        self._on_message = on_message
        self._on_disconnect = on_disconnect

        self._ws: Any = None  # websockets connection
        self._connected = False
        self._last_heartbeat = 0.0
        self._peer_caps: PeerCapabilities | None = None
        self._send_queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None

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
        if self._task:
            self._task.cancel()
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
        await self._send_queue.put(json.dumps({"type": msg_type, "payload": payload}))

    async def _send_raw(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Send immediately (used for handshake before the queue loop starts)."""
        if self._ws:
            await self._ws.send(json.dumps({"type": msg_type, "payload": payload}))

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
        except asyncio.CancelledError:
            recv_task.cancel()
            send_task.cancel()
            hb_task.cancel()
        finally:
            self._connected = False
            if self._on_disconnect:
                self._on_disconnect(self.peer_id)

    async def _recv_loop(self) -> None:
        """Receive and dispatch inbound messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
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
                raw = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
                await self._ws.send(raw)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.warning("PeerConnection: send error to %s: %s", self.peer_id, exc)
                break

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and detect dead peers."""
        while self._connected:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            if not self._connected:
                break
            elapsed = time.time() - self._last_heartbeat
            if elapsed > _HEARTBEAT_TIMEOUT:
                logger.warning("PeerConnection: peer %s timed out (%.0fs)", self.peer_id, elapsed)
                break
            await self._send_raw("heartbeat", {"ts": time.time()})

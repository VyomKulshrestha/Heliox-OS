"""HelioxMesh — LAN mesh orchestrator.

Ties together peer discovery, peer connections, skill sync, and
collaborative execution into a single object that ``PilotServer``
starts and stops alongside the main WebSocket server.

Lifecycle
---------
    mesh = HelioxMesh(config, executor, plugin_manager)
    await mesh.start()          # begins mDNS advertising + browsing
    ...
    await mesh.stop()           # deregisters mDNS, closes all connections

Architecture
------------
                    ┌─────────────────────────────────┐
                    │           HelioxMesh             │
                    │                                  │
    mDNS ──────────►│  PeerDiscovery                   │
                    │      │ on_peer_found              │
                    │      ▼                            │
                    │  PeerConnection pool              │
                    │      │ on_message                 │
                    │      ├──► SkillSync               │
                    │      └──► CollabExecutor          │
                    │                                  │
                    │  P2P WebSocket server (inbound)  │
                    └─────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from typing import TYPE_CHECKING, Any

from pilot.models.gpu_utils import get_available_vram
from pilot.network.peer_connection import (
    PeerAuthenticationError,
    PeerCapabilities,
    PeerConnection,
    decode_peer_message,
)
from pilot.timeouts import bounded

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.config import NetworkConfig
    from pilot.system.plugins import PluginManager

logger = logging.getLogger("pilot.network.mesh")


class HelioxMesh:
    """Central coordinator for the LAN mesh network.

    Parameters
    ----------
    config:
        ``NetworkConfig`` from ``PilotConfig``.
    executor:
        The local ``Executor`` instance (used by ``CollabExecutor``).
    plugin_manager:
        The global ``PluginManager`` (used by ``SkillSync``).
    """

    def __init__(
        self,
        config: NetworkConfig,
        executor: Executor,
        plugin_manager: PluginManager,
        shared_secret: bytes,
    ) -> None:
        if len(shared_secret) < 32:
            raise ValueError("LAN mesh shared secret must contain at least 32 bytes")
        self._config = config
        self._executor = executor
        self._plugin_manager = plugin_manager
        self._shared_secret = shared_secret
        self._instance_id = str(uuid.uuid4())[:8]

        self._connections: dict[str, PeerConnection] = {}
        self._discovery: Any = None  # PeerDiscovery (lazy import)
        self._skill_sync: Any = None  # SkillSync (lazy import)
        self._collab: Any = None  # CollabExecutor (lazy import)
        self._p2p_server: Any = None  # inbound WebSocket server
        self._running = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._delegated_tasks: dict[tuple[str, str], asyncio.Task[Any]] = {}

    def _spawn(self, coroutine: Any) -> asyncio.Task[Any]:
        """Run mesh-owned work and retain it until completion or shutdown."""
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.warning("HelioxMesh background task failed: %s", error)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def peer_ids(self) -> list[str]:
        """IDs of all currently connected peers."""
        return [pid for pid, conn in self._connections.items() if conn.connected]

    def get_connection(self, peer_id: str) -> PeerConnection | None:
        return self._connections.get(peer_id)

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def collab_executor(self) -> Any:
        return self._collab

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the mesh: P2P server, discovery, skill sync, collab executor."""
        if self._running:
            return
        self._running = True

        # Lazy imports so missing optional deps don't crash the daemon
        from pilot.network.collab_executor import CollabExecutor
        from pilot.network.peer_discovery import PeerDiscovery
        from pilot.network.skill_sync import SkillSync

        self._skill_sync = SkillSync(self)
        self._collab = CollabExecutor(
            mesh=self,
            local_executor=self._executor,
            enabled=self._config.collab_exec_enabled,
        )

        # Start inbound P2P WebSocket server
        await self._start_p2p_server()

        # Start mDNS discovery
        self._discovery = PeerDiscovery(
            port=self._config.port,
            instance_id=self._instance_id,
        )
        self._discovery.on_peer_found = self._on_peer_found
        self._discovery.on_peer_lost = self._on_peer_lost
        await self._discovery.start()

        # Start periodic capability updates
        self._spawn(self._periodic_capability_update())

        logger.info(
            "HelioxMesh started (instance=%s, port=%d)",
            self._instance_id,
            self._config.port,
        )

    async def stop(self) -> None:
        """Shut down all connections and deregister from mDNS."""
        self._running = False

        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._delegated_tasks.clear()

        # Close all peer connections
        for conn in list(self._connections.values()):
            await conn.disconnect()
        self._connections.clear()

        # Stop discovery
        if self._discovery:
            await self._discovery.stop()
            self._discovery = None

        # Stop P2P server
        if self._p2p_server:
            self._p2p_server.close()
            await self._p2p_server.wait_closed()
            self._p2p_server = None

        logger.info("HelioxMesh stopped")

    # ── Messaging ─────────────────────────────────────────────────────────────

    async def broadcast(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Send a message to all connected peers."""
        for pid in list(self.peer_ids):
            await self.send_to(pid, msg_type, payload)

    async def send_to(self, peer_id: str, msg_type: str, payload: dict[str, Any]) -> bool:
        """Send a message to a connected peer and report whether it was sent."""
        conn = self._connections.get(peer_id)
        if conn and conn.connected:
            await conn.send(msg_type, payload)
            return True
        else:
            logger.warning("HelioxMesh.send_to: peer %s not connected", peer_id)
            return False

    # ── Peer events ───────────────────────────────────────────────────────────

    def _on_peer_found(self, peer_info: Any) -> None:
        """Called by PeerDiscovery when a new peer is found on the LAN."""
        # Exactly one side opens the connection so simultaneous mDNS discovery
        # cannot create two sockets for the same peer pair.
        if self._running and self._instance_id < peer_info.peer_id:
            self._spawn(self._connect_to_peer(peer_info))

    def _on_peer_lost(self, peer_id: str) -> None:
        """Called by PeerDiscovery when a peer deregisters."""
        conn = self._connections.pop(peer_id, None)
        if conn:
            self._spawn(conn.disconnect())
        logger.info("HelioxMesh: peer %s left the mesh", peer_id)

    async def _connect_to_peer(self, peer_info: Any) -> None:
        """Establish an outbound connection to a newly discovered peer."""
        peer_id = peer_info.peer_id
        if peer_id in self._connections:
            return  # already connected

        own_caps = self._build_own_capabilities()
        conn = PeerConnection(
            peer_id=peer_id,
            host=peer_info.host,
            port=peer_info.port,
            own_capabilities=own_caps,
            shared_secret=self._shared_secret,
            on_message=self._on_peer_message,
            on_disconnect=self._on_connection_lost,
            heartbeat_timeout_seconds=getattr(self._config, "peer_timeout_s", 45),
        )
        success = await conn.connect()
        if success:
            if not self._running:
                await conn.disconnect()
                return
            self._connections[peer_id] = conn
            logger.info("HelioxMesh: joined mesh with peer %s", peer_id)

            # Broadcast our loaded plugins to the new peer
            if self._config.skill_sync_enabled and self._skill_sync:
                self._spawn(self._sync_plugins_to_peer(peer_id))

    async def _periodic_capability_update(self) -> None:
        """Periodically refresh VRAM/CPU stats and update discovery/peers."""
        while self._running:
            await asyncio.sleep(30)  # update every 30 seconds
            if not self._running:
                break

            caps = self._build_own_capabilities()

            # 1. Update mDNS (Zeroconf) record
            if self._discovery:
                await self._discovery.update_vram(caps.vram_free, caps.has_gpu)

            # 2. Update connected peers via P2P
            await self.broadcast("peer_info", caps.__dict__)
            logger.debug("HelioxMesh: broadcasted updated capabilities (VRAM: %.1f MB)", caps.vram_free / 1024**2)

    def _on_connection_lost(self, peer_id: str) -> None:
        """Called by PeerConnection when a connection drops unexpectedly."""
        self._connections.pop(peer_id, None)
        logger.info("HelioxMesh: connection to peer %s lost", peer_id)

    # ── Message routing ───────────────────────────────────────────────────────

    async def _on_peer_message(self, peer_id: str, msg_type: str, payload: dict[str, Any]) -> None:
        """Route inbound peer messages to the appropriate handler."""
        if msg_type == "skill_sync" and self._config.skill_sync_enabled:
            if self._skill_sync:
                await self._skill_sync.handle_incoming(peer_id, payload)

        elif msg_type == "skill_ack":
            logger.info(
                "HelioxMesh: peer %s acknowledged plugin '%s' (%s)",
                peer_id,
                payload.get("name"),
                payload.get("status"),
            )

        elif msg_type == "task_delegate" and self._config.collab_exec_enabled:
            task_id = str(payload.get("task_id", ""))
            task_key = (peer_id, task_id)
            if task_key not in self._delegated_tasks:
                task = self._spawn(self._handle_delegated_task(peer_id, payload))
                self._delegated_tasks[task_key] = task
                task.add_done_callback(lambda _task, key=task_key: self._delegated_tasks.pop(key, None))

        elif msg_type == "task_cancel" and self._config.collab_exec_enabled:
            task_id = str(payload.get("task_id", ""))
            task = self._delegated_tasks.pop((peer_id, task_id), None)
            if task is not None:
                task.cancel()

        elif msg_type == "task_result":
            if self._collab:
                await self._collab.handle_task_result(peer_id, payload)

        else:
            logger.debug("HelioxMesh: unhandled message type '%s' from %s", msg_type, peer_id)

    # ── Inbound P2P server ────────────────────────────────────────────────────

    async def _start_p2p_server(self) -> None:
        """Start a WebSocket server that accepts inbound peer connections."""
        try:
            import websockets

            self._p2p_server = await websockets.serve(
                self._handle_inbound_peer,
                "0.0.0.0",
                self._config.port,
            )
            logger.info("HelioxMesh: P2P server listening on port %d", self._config.port)
        except Exception as exc:
            logger.error("HelioxMesh: failed to start P2P server: %s", exc)
            raise

    async def _handle_inbound_peer(self, websocket: Any) -> None:
        """Handle an inbound WebSocket connection from a peer."""
        peer_id: str | None = None
        seen_nonces: dict[str, int] = {}
        timeout_seconds = max(5.0, float(getattr(self._config, "peer_timeout_s", 45)))
        iterator = websocket.__aiter__()
        try:
            while True:
                try:
                    raw = await bounded(iterator.__anext__(), timeout=timeout_seconds)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("HelioxMesh: inbound peer %s timed out", peer_id or "before handshake")
                    await websocket.close(code=1001, reason="peer timed out")
                    return
                try:
                    msg = decode_peer_message(
                        self._shared_secret,
                        raw,
                        seen_nonces,
                        expected_sender=peer_id,
                    )
                except PeerAuthenticationError as exc:
                    logger.warning("HelioxMesh: rejected unauthenticated inbound peer: %s", exc)
                    await websocket.close(code=1008, reason="peer authentication failed")
                    return
                msg_type = msg.get("type", "")
                payload = msg.get("payload", {})

                if msg_type == "peer_info":
                    advertised_peer_id = payload.get("instance_id", "")
                    if advertised_peer_id != msg["sender"]:
                        logger.warning("HelioxMesh: peer identity does not match signed sender")
                        await websocket.close(code=1008, reason="peer identity mismatch")
                        return
                    peer_id = msg["sender"]
                    # Register a lightweight inbound connection wrapper
                    if peer_id in self._connections:
                        await websocket.close(code=1000, reason="duplicate peer connection")
                        return
                    try:
                        peer_caps = PeerCapabilities(**payload)
                    except (TypeError, ValueError) as exc:
                        logger.warning("HelioxMesh: invalid capabilities from peer %s: %s", peer_id, exc)
                        await websocket.close(code=1008, reason="invalid peer capabilities")
                        return
                    own_caps = self._build_own_capabilities()
                    conn = PeerConnection(
                        peer_id=peer_id,
                        host=websocket.remote_address[0],
                        port=self._config.port,
                        own_capabilities=own_caps,
                        shared_secret=self._shared_secret,
                        on_message=self._on_peer_message,
                        on_disconnect=self._on_connection_lost,
                        heartbeat_timeout_seconds=timeout_seconds,
                    )
                    conn.attach_inbound(websocket, peer_caps)
                    self._connections[peer_id] = conn
                    await conn.send("peer_info", own_caps.__dict__)
                    logger.info("HelioxMesh: inbound peer %s connected", peer_id)
                    continue

                if peer_id:
                    conn = self._connections.get(peer_id)
                    if msg_type == "heartbeat" and conn:
                        conn.note_heartbeat()
                        await conn.send("heartbeat_ack", {})
                        continue
                    if msg_type == "heartbeat_ack" and conn:
                        conn.note_heartbeat()
                        continue
                    await self._on_peer_message(peer_id, msg_type, payload)
        except Exception as exc:
            logger.debug("HelioxMesh: inbound peer connection closed: %s", exc)
        finally:
            if peer_id:
                self._connections.pop(peer_id, None)

    # ── Delegated task execution ──────────────────────────────────────────────

    async def _handle_delegated_task(self, peer_id: str, payload: dict[str, Any]) -> None:
        """Execute a batch of actions delegated by a peer and return results."""
        from pilot.actions import Action, ActionPlan, ActionResult
        from pilot.network.collab_executor import is_delegable_action
        from pilot.security.gateway import InvocationSource

        task_id = str(payload.get("task_id", ""))
        raw_actions = payload.get("actions", [])
        raw_input = "Authenticated peer telemetry request"

        try:
            uuid.UUID(task_id)
        except (ValueError, AttributeError):
            logger.warning("HelioxMesh: rejected delegated task with invalid ID from %s", peer_id)
            return
        if not isinstance(raw_actions, list) or not 1 <= len(raw_actions) <= 64:
            logger.warning("HelioxMesh: rejected delegated task %s with invalid action count", task_id)
            await self.send_to(peer_id, "task_result", {"task_id": task_id, "results": []})
            return
        try:
            actions = [Action.model_validate(a) for a in raw_actions]
        except Exception as exc:
            logger.warning("HelioxMesh: failed to deserialise delegated actions: %s", exc)
            await self.send_to(peer_id, "task_result", {"task_id": task_id, "results": []})
            return

        disallowed = [action for action in actions if not is_delegable_action(action)]
        if disallowed:
            error = "Peer Mesh delegates only portable system telemetry actions"
            results = [ActionResult(action=action, success=False, error=error) for action in actions]
            await self.send_to(
                peer_id,
                "task_result",
                {"task_id": task_id, "results": [result.model_dump(mode="json") for result in results]},
            )
            logger.warning("HelioxMesh: blocked over-authority delegated task %s from %s", task_id, peer_id)
            return

        plan = ActionPlan(actions=actions, raw_input=raw_input, explanation="Delegated by peer")
        results = await self._executor.execute(
            plan,
            plan_id=f"peer-{task_id}",
            invocation_source=InvocationSource.NETWORK_AGENT,
            user_confirmed=False,
            allow_collaboration=False,
        )

        serialised = [r.model_dump(mode="json") for r in results]
        await self.send_to(peer_id, "task_result", {"task_id": task_id, "results": serialised})
        logger.info(
            "HelioxMesh: completed delegated task %s for peer %s (%d results)",
            task_id,
            peer_id,
            len(results),
        )

    # ── Plugin sync helpers ───────────────────────────────────────────────────

    async def _sync_plugins_to_peer(self, peer_id: str) -> None:
        """Send all locally loaded plugins to a newly connected peer."""
        if not self._skill_sync:
            return
        plugins = self._plugin_manager.list_plugins()
        for plugin in plugins:
            if plugin.get("loaded") and plugin.get("file_path"):
                await self._skill_sync.broadcast_plugin(plugin["name"], plugin["file_path"])

    def _build_own_capabilities(self) -> PeerCapabilities:
        """Build a capability advertisement for this instance."""
        try:
            import psutil

            cpu_load = psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            cpu_load = 0.0

        plugins = [p["name"] for p in self._plugin_manager.list_plugins() if p.get("loaded")]
        vram_free, has_gpu = get_available_vram()

        return PeerCapabilities(
            instance_id=self._instance_id,
            hostname=socket.gethostname(),
            can_execute=self._config.collab_exec_enabled,
            cpu_load=cpu_load,
            vram_free=vram_free,
            has_gpu=has_gpu,
            plugin_names=plugins,
        )

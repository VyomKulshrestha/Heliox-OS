"""Swarm Manager - distributed multi-daemon execution with hardware-aware routing."""

import asyncio
import json
import logging
import secrets
import socket
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any

import websockets

from pilot.config import DATA_DIR

logger = logging.getLogger("pilot.swarm.swarm_manager")


@dataclass
class DaemonNode:
    """Represents a remote daemon node in the swarm."""

    node_id: str
    addr: str
    port: int
    hardware: dict[str, Any]
    tasks_completed: int = 0
    last_heartbeat: float = 0.0
    is_healthy: bool = True


class HardwareCapability(Enum):
    """Hardware capabilities for task dispatching."""

    CPU_ONLY = "cpu_only"
    GPU_VRAM_4GB = "gpu_vram_4gb"
    GPU_VRAM_8GB = "gpu_vram_8gb"
    GPU_VRAM_12GB = "gpu_vram_12gb"
    GPU_VRAM_16GB = "gpu_vram_16gb"


@dataclass
class TaskRequirements:
    """Hardware requirements for a task."""

    minimal_vram_gb: float = 0
    requires_gpu: bool = False
    estimated_tokens: int = 1000


class SwarmManager:
    """Manages a pool of distributed daemons with hardware-aware task routing."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._nodes: list[DaemonNode] = []
        self._local_node: DaemonNode | None = None
        self._broadcast_fn = None
        self._discovery_task = None
        self._heartbeat_task = None
        self._task_router = None

    async def initialize(self) -> None:
        """Initialize the swarm manager."""
        hardware = await self._detect_local_hardware()
        self._local_node = DaemonNode(
            node_id=secrets.token_hex(8),
            addr="127.0.0.1",
            port=self._config.get("daemon_port", 8000),
            hardware=hardware,
        )
        self._nodes.append(self._local_node)
        logger.info("Swarm initialized: %d nodes", len(self._nodes))

    async def _detect_local_hardware(self) -> dict[str, Any]:
        """Detect local hardware capabilities."""
        import psutil

        hardware = {
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "has_gpu": False,
            "gpu_name": None,
            "gpu_vram_gb": 0,
        }

        try:
            import torch

            if torch.cuda.is_available():
                hardware["has_gpu"] = True
                hardware["gpu_name"] = torch.cuda.get_device_name(0)
                hardware["gpu_vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
        except ImportError:
            pass

        return hardware

    async def discover_nodes(self, timeout: float = 5.0) -> list[DaemonNode]:
        """Discover other daemons on the network using UDP broadcast."""
        discovered = []

        def _run_discovery() -> list[tuple[bytes, tuple[str, int]]]:
            results = []
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(timeout)

            try:
                discovery_msg = {
                    "type": "DISCOVERY",
                    "node_id": self._local_node.node_id if self._local_node else "",
                }
                sock.sendto(
                    json.dumps(discovery_msg).encode(),
                    ("<broadcast>", 18888),
                )

                while True:
                    try:
                        data, addr = sock.recvfrom(1024)
                        results.append((data, addr))
                    except socket.timeout:
                        break
            finally:
                sock.close()
            return results

        try:
            results = await asyncio.to_thread(_run_discovery)
            for data, addr in results:
                msg = json.loads(data.decode())
                if msg.get("type") == "DISCOVERY_RESPONSE":
                    node = DaemonNode(
                        node_id=msg.get("node_id", ""),
                        addr=addr[0],
                        port=msg.get("port", 18888),
                        hardware=msg.get("hardware", {}),
                    )
                    discovered.append(node)
        except Exception as e:
            logger.warning("Node discovery failed: %s", e)

        self._nodes = list(set(self._nodes + discovered))
        return discovered

    def get_nodes_by_capability(
        self, capability: HardwareCapability
    ) -> list[DaemonNode]:
        """Get nodes that meet the specified hardware capability."""
        nodes = []
        for node in self._nodes:
            if node.is_healthy:
                gpu_vram = node.hardware.get("gpu_vram_gb", 0)
                if capability == HardwareCapability.CPU_ONLY and not node.hardware.get("has_gpu"):
                    nodes.append(node)
                elif capability == HardwareCapability.GPU_VRAM_4GB and gpu_vram >= 4:
                    nodes.append(node)
                elif capability == HardwareCapability.GPU_VRAM_8GB and gpu_vram >= 8:
                    nodes.append(node)
                elif capability == HardwareCapability.GPU_VRAM_12GB and gpu_vram >= 12:
                    nodes.append(node)
                elif capability == HardwareCapability.GPU_VRAM_16GB and gpu_vram >= 16:
                    nodes.append(node)
        return nodes

    async def route_task(self, requirements: TaskRequirements) -> DaemonNode:
        """Route a task to the most suitable node."""
        if requirements.requires_gpu:
            capability = HardwareCapability.GPU_VRAM_8GB
            if requirements.minimal_vram_gb >= 16:
                capability = HardwareCapability.GPU_VRAM_16GB
            elif requirements.minimal_vram_gb >= 12:
                capability = HardwareCapability.GPU_VRAM_12GB
            elif requirements.minimal_vram_gb >= 8:
                capability = HardwareCapability.GPU_VRAM_8GB
            elif requirements.minimal_vram_gb >= 4:
                capability = HardwareCapability.GPU_VRAM_4GB

            capable_nodes = self.get_nodes_by_capability(capability)
        else:
            capable_nodes = [n for n in self._nodes if n.is_healthy]

        if not capable_nodes:
            raise RuntimeError("No nodes available for task routing")

        node = min(
            capable_nodes,
            key=lambda n: n.tasks_completed if n.is_healthy else float("inf"),
        )

        node.tasks_completed += 1
        return node

    async def execute_remote(
        self, node: DaemonNode, plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a task on a remote daemon node via WebSocket JSON-RPC."""
        if not node.is_healthy:
            raise RuntimeError(f"Node {node.node_id} is not healthy")

        uri = f"ws://{node.addr}:{node.port}"

        try:
            async with websockets.connect(uri, open_timeout=10) as ws:
                request = {
                    "jsonrpc": "2.0",
                    "id": secrets.token_hex(8),
                    "method": "execute",
                    "params": {"input": plan.get("input", ""), "dry_run": plan.get("dry_run", False)},
                }
                await ws.send(json.dumps(request))
                response = await ws.recv()
                result = json.loads(response)
                if "error" in result:
                    raise RuntimeError(f"Remote execution error: {result['error']}")
                return result.get("result", {})
        except Exception as e:
            logger.error("Remote execution failed on node %s: %s", node.node_id, e)
            raise RuntimeError(f"Failed to execute on remote node {node.node_id}: {e}")

    def set_broadcast(self, fn) -> None:
        """Set the broadcast function for UI notifications."""
        self._broadcast_fn = fn

    async def _heartbeat_monitor(self) -> None:
        """Monitor heartbeat of all nodes via WebSocket ping."""
        while True:
            for node in self._nodes:
                if not node.is_healthy:
                    continue

                try:
                    uri = f"ws://{node.addr}:{node.port}"
                    async with websockets.connect(uri, open_timeout=5) as ws:
                        await ws.ping()
                        node.is_healthy = True
                        node.last_heartbeat = asyncio.get_event_loop().time()
                except Exception as e:
                    node.is_healthy = False
                    logger.warning("Node %s heartbeat failed: %s", node.node_id, e)

            if self._broadcast_fn:
                await self._broadcast_fn(
                    "swarm_status",
                    {
                        "total_nodes": len(self._nodes),
                        "healthy_nodes": sum(1 for n in self._nodes if n.is_healthy),
                        "nodes": [
                            {
                                "id": n.node_id,
                                "addr": n.addr,
                                "tasks_completed": n.tasks_completed,
                                "is_healthy": n.is_healthy,
                            }
                            for n in self._nodes[:5]
                        ],
                    },
                )

            await asyncio.sleep(10)

    async def start(self) -> None:
        """Start swarm background tasks."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def stop(self) -> None:
        """Stop swarm background tasks."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    def get_available_nodes(self) -> list[DaemonNode]:
        """Get all healthy available nodes."""
        return [n for n in self._nodes if n.is_healthy]
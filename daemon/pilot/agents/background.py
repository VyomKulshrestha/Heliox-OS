"""Background task manager — runs autonomous monitoring loops.

Allows the agent to run persistent background tasks such as
CPU monitoring, disk space alerts, network activity tracking,
and custom user-defined watchers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.agents.background")


class TaskStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class BackgroundTask:
    """A persistent background task definition."""

    task_id: str
    name: str
    description: str
    interval_seconds: float
    action_fn: Callable[[], Coroutine[Any, Any, dict[str, Any]]]
    on_trigger: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None
    condition: str = ""  # Human-readable trigger condition
    status: TaskStatus = TaskStatus.STOPPED
    last_run: float = 0.0
    run_count: int = 0
    error_count: int = 0
    last_result: dict[str, Any] = field(default_factory=dict)
    _handle: asyncio.Task[None] | None = field(default=None, repr=False)


class BackgroundTaskManager:
    """Manages autonomous background task loops."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._broadcast: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None

    def set_broadcast(self, fn: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        """Set the WebSocket broadcast function for sending alerts."""
        self._broadcast = fn

    def register(self, task: BackgroundTask) -> None:
        """Register a background task."""
        self._tasks[task.task_id] = task
        logger.info("Background task registered: %s (%s)", task.name, task.task_id)

    def start(self, task_id: str) -> bool:
        """Start a background task loop."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status == TaskStatus.RUNNING:
            return True

        task.status = TaskStatus.RUNNING
        task._handle = asyncio.create_task(self._run_loop(task))
        logger.info("Background task started: %s", task.name)
        return True

    def stop(self, task_id: str) -> bool:
        """Stop a running background task."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.STOPPED
        if task._handle and not task._handle.done():
            task._handle.cancel()
        logger.info("Background task stopped: %s", task.name)
        return True

    def pause(self, task_id: str) -> bool:
        """Pause a running background task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.PAUSED
        return True

    def resume(self, task_id: str) -> bool:
        """Resume a paused background task."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PAUSED:
            return False
        task.status = TaskStatus.RUNNING
        return True

    async def _run_loop(self, task: BackgroundTask) -> None:
        """Main loop for a background task."""
        while task.status == TaskStatus.RUNNING:
            try:
                result = await task.action_fn()
                task.last_result = result
                task.last_run = time.time()
                task.run_count += 1

                # Check if trigger condition is met
                if task.on_trigger and result.get("triggered"):
                    await task.on_trigger(result)
                    if self._broadcast:
                        await self._broadcast(
                            "background_alert",
                            {
                                "task_id": task.task_id,
                                "name": task.name,
                                "alert": result.get("message", "Threshold reached"),
                                "data": result,
                            },
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                task.error_count += 1
                logger.warning("Background task %s error: %s", task.name, e)
                if task.error_count > 10:
                    task.status = TaskStatus.ERROR
                    logger.error("Background task %s disabled after 10 errors", task.name)
                    break

            # Wait for the next interval (or check for pause)
            elapsed = 0.0
            while elapsed < task.interval_seconds and task.status in (
                TaskStatus.RUNNING,
                TaskStatus.PAUSED,
            ):
                await asyncio.sleep(min(1.0, task.interval_seconds - elapsed))
                elapsed += 1.0
                if task.status == TaskStatus.PAUSED:
                    # Stay in this inner loop while paused
                    continue

    def list_tasks(self) -> list[dict[str, Any]]:
        """List all registered background tasks."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "status": t.status.value,
                "interval_seconds": t.interval_seconds,
                "run_count": t.run_count,
                "error_count": t.error_count,
                "last_result": t.last_result,
                "condition": t.condition,
            }
            for t in self._tasks.values()
        ]

    def stop_all(self) -> None:
        """Stop all running background tasks."""
        for task_id in list(self._tasks.keys()):
            self.stop(task_id)

    # ── Built-in Monitor Factories ──

    @staticmethod
    async def _cpu_check() -> dict[str, Any]:
        """Check CPU usage."""
        import psutil

        cpu = psutil.cpu_percent(interval=1)
        return {
            "cpu_percent": cpu,
            "triggered": cpu > 80,
            "message": f"CPU usage at {cpu}%!" if cpu > 80 else "",
        }

    @staticmethod
    async def _memory_check() -> dict[str, Any]:
        """Check memory usage."""
        import psutil

        mem = psutil.virtual_memory()
        used_percent = mem.percent
        return {
            "memory_percent": used_percent,
            "available_gb": round(mem.available / (1024**3), 2),
            "triggered": used_percent > 85,
            "message": f"Memory usage at {used_percent}%!" if used_percent > 85 else "",
        }

    @staticmethod
    async def _disk_check() -> dict[str, Any]:
        """Check disk usage."""
        import psutil

        disk = psutil.disk_usage("/")
        used_percent = disk.percent
        return {
            "disk_percent": used_percent,
            "free_gb": round(disk.free / (1024**3), 2),
            "triggered": used_percent > 90,
            "message": f"Disk usage at {used_percent}%!" if used_percent > 90 else "",
        }

    @staticmethod
    async def _network_check() -> dict[str, Any]:
        """Check network I/O."""
        import psutil

        net1 = psutil.net_io_counters()
        await asyncio.sleep(1)
        net2 = psutil.net_io_counters()

        sent_rate = (net2.bytes_sent - net1.bytes_sent) / 1024  # KB/s
        recv_rate = (net2.bytes_recv - net1.bytes_recv) / 1024  # KB/s

        return {
            "sent_kbps": round(sent_rate, 1),
            "recv_kbps": round(recv_rate, 1),
            "triggered": sent_rate > 10000 or recv_rate > 10000,
            "message": f"High network: {recv_rate:.0f} KB/s in" if recv_rate > 10000 else "",
        }

    def register_builtin_monitors(self) -> None:
        """Register all built-in system monitors."""
        self.register(
            BackgroundTask(
                task_id="monitor_cpu",
                name="CPU Monitor",
                description="Alerts when CPU usage exceeds 80%",
                interval_seconds=10,
                action_fn=self._cpu_check,
                condition="CPU > 80%",
            )
        )
        self.register(
            BackgroundTask(
                task_id="monitor_memory",
                name="Memory Monitor",
                description="Alerts when RAM usage exceeds 85%",
                interval_seconds=15,
                action_fn=self._memory_check,
                condition="RAM > 85%",
            )
        )
        self.register(
            BackgroundTask(
                task_id="monitor_disk",
                name="Disk Monitor",
                description="Alerts when disk usage exceeds 90%",
                interval_seconds=60,
                action_fn=self._disk_check,
                condition="Disk > 90%",
            )
        )
        self.register(
            BackgroundTask(
                task_id="monitor_network",
                name="Network Monitor",
                description="Alerts on high network activity (>10 MB/s)",
                interval_seconds=10,
                action_fn=self._network_check,
                condition="Network > 10 MB/s",
            )
        )

"""Reactive Triggers / Event System — background watchers.

Monitors system events and executes actions when conditions are met.
File watchers, performance monitors, scheduled checks, custom triggers.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import operator
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.triggers")

_CUSTOM_METRIC_NAMES = frozenset(
    {
        "battery_percent",
        "cpu_percent",
        "disk_percent",
        "hour",
        "memory_percent",
        "on_battery",
        "timestamp",
    }
)
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ALLOWED_CUSTOM_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    *_BINARY_OPERATORS,
    ast.Compare,
    *_COMPARISON_OPERATORS,
    ast.Name,
    ast.Load,
    ast.Constant,
)


def _parse_custom_expression(expression: str) -> ast.Expression:
    """Parse the bounded trigger expression language and reject code constructs."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid custom trigger expression") from error
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_CUSTOM_NODES):
            raise ValueError(f"Unsupported custom trigger expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _CUSTOM_METRIC_NAMES:
            raise ValueError(f"Unknown custom trigger metric: {node.id}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (bool, int, float)):
            raise ValueError("Custom trigger constants must be numeric or boolean")
    return tree


def _evaluate_custom_node(node: ast.AST, metrics: dict[str, float | bool]) -> float | bool:
    if isinstance(node, ast.Expression):
        return _evaluate_custom_node(node.body, metrics)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return metrics[node.id]
    if isinstance(node, ast.BoolOp):
        values = (_evaluate_custom_node(value, metrics) for value in node.values)
        return (
            all(bool(value) for value in values)
            if isinstance(node.op, ast.And)
            else any(bool(value) for value in values)
        )
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_custom_node(node.operand, metrics)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        return -float(value) if isinstance(node.op, ast.USub) else float(value)
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS[type(node.op)]
        return operation(
            float(_evaluate_custom_node(node.left, metrics)),
            float(_evaluate_custom_node(node.right, metrics)),
        )
    if isinstance(node, ast.Compare):
        left = _evaluate_custom_node(node.left, metrics)
        for operation_node, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate_custom_node(comparator, metrics)
            if not _COMPARISON_OPERATORS[type(operation_node)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported custom trigger expression: {type(node).__name__}")


class TriggerType(StrEnum):
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    DISK_THRESHOLD = "disk_threshold"
    BATTERY_LOW = "battery_low"
    PROCESS_STARTED = "process_started"
    PROCESS_STOPPED = "process_stopped"
    TIME_INTERVAL = "time_interval"
    CRON_SCHEDULE = "cron_schedule"
    NETWORK_CHANGE = "network_change"
    CUSTOM_CONDITION = "custom_condition"


@dataclass
class Trigger:
    id: str
    name: str
    trigger_type: TriggerType
    condition: dict[str, Any]  # Type-specific condition params
    action_command: str  # Natural language command to execute
    enabled: bool = True
    fire_count: int = 0
    max_fires: int = 0  # 0 = unlimited
    cooldown_seconds: int = 60
    last_fired: float = 0
    created_at: str = ""

    def can_fire(self) -> bool:
        if not self.enabled:
            return False
        if self.max_fires > 0 and self.fire_count >= self.max_fires:
            return False
        return not time.time() - self.last_fired < self.cooldown_seconds


class TriggerEngine:
    """Manages and evaluates reactive triggers in the background."""

    def __init__(self) -> None:
        self._triggers: dict[str, Trigger] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._fire_callback: Callable[[Trigger], Coroutine[Any, Any, None]] | None = None
        self._callback_tasks: set[asyncio.Task[None]] = set()
        self._file_cache: dict[str, dict[str, float]] = {}  # path -> {file: mtime}

    def set_fire_callback(self, callback: Callable[[Trigger], Coroutine[Any, Any, None]] | None) -> None:
        """Set callback that runs when a trigger fires."""
        self._fire_callback = callback

    def _callback_done(self, task: asyncio.Task[None]) -> None:
        self._callback_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Trigger action callback failed")

    # ── CRUD ─────────────────────────────────────────────────────────

    def create_trigger(
        self,
        name: str,
        trigger_type: str,
        condition: dict,
        action_command: str,
        max_fires: int = 0,
        cooldown_seconds: int = 60,
    ) -> Trigger:
        """Create a new trigger."""
        name = name.strip()
        action_command = action_command.strip()
        if not name:
            raise ValueError("Trigger name is required")
        if not action_command:
            raise ValueError("Trigger action command is required")
        if max_fires < 0:
            raise ValueError("Trigger max_fires cannot be negative")
        if cooldown_seconds < 0:
            raise ValueError("Trigger cooldown_seconds cannot be negative")
        resolved_type = TriggerType(trigger_type)
        if resolved_type == TriggerType.CUSTOM_CONDITION:
            expression = str(condition.get("expression") or "").strip()
            if not expression:
                raise ValueError("Custom trigger expression is required")
            _parse_custom_expression(expression)
        trigger = Trigger(
            id=str(uuid.uuid4())[:8],
            name=name,
            trigger_type=resolved_type,
            condition=condition,
            action_command=action_command,
            max_fires=max_fires,
            cooldown_seconds=cooldown_seconds,
            created_at=datetime.now().isoformat(),
        )
        self._triggers[trigger.id] = trigger
        logger.info("Created trigger: %s (%s)", name, trigger.id)
        return trigger

    def delete_trigger(self, trigger_id: str) -> bool:
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        # Try by name
        for tid, t in list(self._triggers.items()):
            if t.name == trigger_id:
                del self._triggers[tid]
                return True
        return False

    def list_triggers(self) -> list[dict]:
        return [asdict(t) for t in self._triggers.values()]

    def get_trigger(self, trigger_id: str) -> Trigger | None:
        return self._triggers.get(trigger_id)

    def find_matching(
        self,
        name: str,
        trigger_type: str,
        condition: dict,
        action_command: str,
        max_fires: int,
        cooldown_seconds: int,
    ) -> Trigger | None:
        """An enabled trigger that already has exactly these settings, if one exists."""
        for trigger in self._triggers.values():
            if (
                trigger.enabled
                and trigger.name == name.strip()
                and trigger.trigger_type.value == trigger_type
                and trigger.condition == condition
                and trigger.action_command == action_command.strip()
                and trigger.max_fires == max_fires
                and trigger.cooldown_seconds == cooldown_seconds
            ):
                return trigger
        return None

    # ── Engine Control ───────────────────────────────────────────────

    def is_running(self) -> bool:
        """Whether the evaluation loop is alive: its task exists and has not finished.

        This looks at the asyncio task itself, so a loop that crashed or was
        cancelled reads as stopped even though the ``_running`` flag is still set.
        """
        return self._running and self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the background trigger evaluation loop (again, if a previous loop has ended)."""
        if self.is_running():
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Trigger engine started with %d triggers", len(self._triggers))

    async def stop(self) -> None:
        """Stop the trigger engine."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        callback_tasks = tuple(self._callback_tasks)
        for task in callback_tasks:
            task.cancel()
        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)
        logger.info("Trigger engine stopped")

    async def _run_loop(self) -> None:
        """Main evaluation loop — checks all triggers every 5 seconds."""
        while self._running:
            try:
                for trigger in list(self._triggers.values()):
                    if not trigger.can_fire():
                        continue
                    try:
                        fired = await self._evaluate(trigger)
                        if fired:
                            trigger.fire_count += 1
                            trigger.last_fired = time.time()
                            logger.info(
                                "Trigger fired: %s (%s) — fire #%d",
                                trigger.name,
                                trigger.id,
                                trigger.fire_count,
                            )
                            if self._fire_callback:
                                task = asyncio.create_task(
                                    self._fire_callback(trigger),
                                    name=f"trigger_action_{trigger.id}",
                                )
                                self._callback_tasks.add(task)
                                task.add_done_callback(self._callback_done)
                    except Exception as e:
                        logger.warning("Trigger %s eval error: %s", trigger.id, e)

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

    # ── Evaluation ───────────────────────────────────────────────────

    async def _evaluate(self, trigger: Trigger) -> bool:
        """Evaluate a trigger's condition. Returns True if condition is met."""
        tt = trigger.trigger_type
        cond = trigger.condition

        if tt == TriggerType.CPU_THRESHOLD:
            return await self._check_cpu(cond)
        elif tt == TriggerType.MEMORY_THRESHOLD:
            return await self._check_memory(cond)
        elif tt == TriggerType.DISK_THRESHOLD:
            return await self._check_disk(cond)
        elif tt == TriggerType.BATTERY_LOW:
            return await self._check_battery(cond)
        elif tt in (TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED, TriggerType.FILE_DELETED):
            return await self._check_file(trigger)
        elif tt == TriggerType.PROCESS_STARTED:
            return await self._check_process_exists(cond, expect=True)
        elif tt == TriggerType.PROCESS_STOPPED:
            return await self._check_process_exists(cond, expect=False)
        elif tt == TriggerType.TIME_INTERVAL:
            # Already handled by cooldown_seconds
            return True
        elif tt == TriggerType.CUSTOM_CONDITION:
            return await self._check_custom(cond)

        return False

    async def _check_cpu(self, condition: dict) -> bool:
        try:
            import psutil

            threshold = condition.get("threshold", 90)
            # Threshold sampling sleeps for the requested interval inside
            # psutil; running it in a worker prevents a trigger check from
            # freezing every other asyncio service for one second.
            cpu = await asyncio.to_thread(psutil.cpu_percent, interval=1)
            return cpu > threshold
        except ImportError:
            return False

    async def _check_memory(self, condition: dict) -> bool:
        try:
            import psutil

            threshold = condition.get("threshold", 90)
            mem = psutil.virtual_memory()
            return mem.percent > threshold
        except ImportError:
            return False

    async def _check_disk(self, condition: dict) -> bool:
        try:
            import psutil

            threshold = condition.get("threshold", 95)
            path = condition.get("path", "/")
            usage = psutil.disk_usage(path)
            return usage.percent > threshold
        except ImportError:
            return False

    async def _check_battery(self, condition: dict) -> bool:
        try:
            import psutil

            threshold = condition.get("threshold", 20)
            batt = psutil.sensors_battery()
            if batt is None:
                return False
            return batt.percent < threshold and not batt.power_plugged
        except ImportError:
            return False

    async def _check_file(self, trigger: Trigger) -> bool:
        watch_path = trigger.condition.get("path", "")
        if not watch_path or not os.path.exists(watch_path):
            return False

        p = Path(watch_path)
        current_files: dict[str, float] = {}

        if p.is_dir():
            pattern = trigger.condition.get("pattern", "*")
            for f in p.glob(pattern):
                if f.is_file():
                    with contextlib.suppress(OSError):
                        current_files[str(f)] = f.stat().st_mtime
        elif p.is_file():
            with contextlib.suppress(OSError):
                current_files[str(p)] = p.stat().st_mtime

        cache_key = trigger.id
        previous = self._file_cache.get(cache_key, {})
        self._file_cache[cache_key] = current_files

        if not previous:
            return False  # First check, establish baseline

        if trigger.trigger_type == TriggerType.FILE_CREATED:
            new_files = set(current_files.keys()) - set(previous.keys())
            return len(new_files) > 0

        elif trigger.trigger_type == TriggerType.FILE_MODIFIED:
            return any(f in previous and mtime > previous[f] for f, mtime in current_files.items())

        elif trigger.trigger_type == TriggerType.FILE_DELETED:
            deleted = set(previous.keys()) - set(current_files.keys())
            return len(deleted) > 0

        return False

    async def _check_process_exists(self, condition: dict, expect: bool) -> bool:
        try:
            import psutil

            name = condition.get("name", "")
            for proc in psutil.process_iter(["name"]):
                if name.lower() in proc.info["name"].lower():
                    return expect  # Found AND we're looking for "started"
            return not expect  # Not found AND we're looking for "stopped"
        except ImportError:
            return False

    async def _check_custom(self, condition: dict) -> bool:
        """Evaluate a bounded expression over read-only system metrics."""
        expr = str(condition.get("expression") or "").strip()
        if not expr:
            return False
        try:
            import psutil
        except ImportError:
            return False

        try:
            battery = psutil.sensors_battery()
            metrics: dict[str, float | bool] = {
                "battery_percent": float(battery.percent) if battery is not None else -1.0,
                "cpu_percent": float(await asyncio.to_thread(psutil.cpu_percent, interval=None)),
                "disk_percent": float(psutil.disk_usage(condition.get("path", "/")).percent),
                "hour": float(datetime.now().hour),
                "memory_percent": float(psutil.virtual_memory().percent),
                "on_battery": bool(battery is not None and not battery.power_plugged),
                "timestamp": time.time(),
            }
            return bool(_evaluate_custom_node(_parse_custom_expression(expr), metrics))
        except Exception as e:
            logger.warning("Custom condition eval error: %s", e)
            return False


# ── Global engine instance ───────────────────────────────────────────

_engine = TriggerEngine()


async def trigger_create(
    name: str,
    trigger_type: str,
    condition: dict,
    action_command: str,
    max_fires: int = 0,
    cooldown_seconds: int = 60,
) -> str:
    """Create a reactive trigger.

    Asking again for a trigger with exactly the same settings reuses the existing
    one instead of adding a twin that would fire the same command twice; this keeps
    a retry after a failed verification from doubling the trigger.
    """
    trigger = _engine.find_matching(
        name, trigger_type, condition, action_command, max_fires, cooldown_seconds
    ) or _engine.create_trigger(
        name,
        trigger_type,
        condition,
        action_command,
        max_fires,
        cooldown_seconds,
    )

    # Auto-start engine if not running (or if its loop has ended)
    if not _engine.is_running():
        await _engine.start()

    return json.dumps(asdict(trigger), indent=2)


async def trigger_list() -> str:
    """List all active triggers."""
    triggers = _engine.list_triggers()
    if not triggers:
        return "No triggers configured"
    return json.dumps(triggers, indent=2)


async def trigger_delete(trigger_id: str) -> str:
    """Delete a trigger by ID or name."""
    if _engine.delete_trigger(trigger_id):
        return f"Deleted trigger: {trigger_id}"
    raise LookupError(f"No trigger has the id or name {trigger_id!r}, so nothing was deleted. List the triggers first.")


async def trigger_start_engine() -> str:
    """Start the trigger evaluation engine."""
    await _engine.start()
    return f"Trigger engine started with {len(_engine._triggers)} triggers"


async def trigger_stop_engine() -> str:
    """Stop the trigger evaluation engine."""
    await _engine.stop()
    return "Trigger engine stopped"


def get_engine() -> TriggerEngine:
    """Access the global trigger engine instance."""
    return _engine

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
    disabled_reason: str = ""  # why a stored trigger was loaded disabled; empty for a trigger that may fire

    def can_fire(self) -> bool:
        if not self.enabled:
            return False
        if self.max_fires > 0 and self.fire_count >= self.max_fires:
            return False
        return not time.time() - self.last_fired < self.cooldown_seconds


STORE_VERSION = 1


def validate_trigger(
    name: str,
    trigger_type: str,
    condition: Any,
    action_command: str,
    max_fires: int,
    cooldown_seconds: int,
) -> TriggerType:
    """The rules every trigger must pass, at creation and again when it is loaded from the store.

    Raises ValueError with the reason. The command is natural language that only ever reaches the
    guarded autonomous pipeline (planning, permission checks, approval) when the trigger fires.
    """
    if not str(name).strip():
        raise ValueError("Trigger name is required")
    if not str(action_command).strip():
        raise ValueError("Trigger action command is required")
    if not isinstance(condition, dict):
        raise ValueError("Trigger condition must be a mapping")
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
    return resolved_type


def _entry_for_store(trigger: Trigger) -> dict[str, Any]:
    entry = asdict(trigger)
    entry["trigger_type"] = trigger.trigger_type.value
    return entry


def read_trigger_store(path: Path) -> dict[str, Any] | None:
    """The trigger store as it is on disk, read afresh: None when there is no file.

    Raises ValueError when the file is not a trigger store this version can read.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("version") != STORE_VERSION or not isinstance(data.get("triggers"), list):
        raise ValueError("The trigger store is not a version 1 trigger store")
    return data


class TriggerEngine:
    """Manages and evaluates reactive triggers in the background.

    With a store attached (``attach_store``, done by the daemon with ``DATA_DIR/triggers.json``) every
    trigger is kept on disk as well, so it survives a restart, and so is whether the person wants the
    engine running. Without one (tests, scripts) triggers live only in memory.
    """

    def __init__(self) -> None:
        self._triggers: dict[str, Trigger] = {}
        self._store_path: Path | None = None
        self._store_writable = True
        self._wanted_running = False
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
        """Create a new trigger (and keep it in the store, when one is attached)."""
        name = name.strip()
        action_command = action_command.strip()
        resolved_type = validate_trigger(name, trigger_type, condition, action_command, max_fires, cooldown_seconds)
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
        try:
            self._save()
        except OSError as error:
            del self._triggers[trigger.id]
            raise RuntimeError(f"The trigger could not be saved, so it was not created: {error}") from error
        logger.info("Created trigger: %s (%s)", name, trigger.id)
        return trigger

    def delete_trigger(self, trigger_id: str) -> bool:
        found = self._triggers.get(trigger_id)
        if found is None:
            # Try by name
            found = next((t for t in self._triggers.values() if t.name == trigger_id), None)
        if found is None:
            return False
        del self._triggers[found.id]
        try:
            self._save()
        except OSError as error:
            self._triggers[found.id] = found
            raise RuntimeError(f"The trigger store could not be saved, so the trigger was kept: {error}") from error
        return True

    # ── Store ────────────────────────────────────────────────────────

    @property
    def store_path(self) -> Path | None:
        return self._store_path

    @property
    def wanted_running(self) -> bool:
        """Whether the person last asked for the engine to run (create or start) rather than stop."""
        return self._wanted_running

    def attach_store(self, path: Path) -> None:
        """Keep triggers in ``path`` from now on, loading the ones already there.

        Loading never starts the engine and never runs a command. A damaged entry is skipped and
        logged; an entry that fails today's trigger rules is loaded disabled with the reason; an
        unreadable file is moved aside (``*.damaged-<time>``) so the next save cannot destroy it; a
        store written by a newer version is read but never overwritten.
        """
        self._store_path = Path(path)
        self._store_writable = True
        self._triggers = {}  # the store is the record from now on
        self._wanted_running = False
        try:
            text = self._store_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError:
            logger.warning("The trigger store could not be read; triggers start empty", exc_info=True)
            self._store_writable = False
            return
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or not isinstance(data.get("triggers"), list):
                raise ValueError("not a trigger store")
        except ValueError:
            aside = self._store_path.with_name(f"{self._store_path.name}.damaged-{int(time.time() * 1000)}")
            logger.warning("The trigger store is damaged; it was moved to %s and triggers start empty", aside.name)
            try:
                os.replace(self._store_path, aside)
            except OSError:
                self._store_writable = False
            return
        version = data.get("version")
        if version != STORE_VERSION:
            logger.warning("The trigger store has version %r; it is loaded but will not be overwritten", version)
            self._store_writable = False
        self._wanted_running = data.get("engine_running") is True
        for index, entry in enumerate(data["triggers"]):
            trigger = self._trigger_from_store(entry)
            if trigger is None:
                logger.warning("Skipped damaged trigger entry #%d in the trigger store", index)
                continue
            self._triggers[trigger.id] = trigger
        logger.info("Loaded %d trigger(s) from the trigger store", len(self._triggers))

    @staticmethod
    def _trigger_from_store(entry: Any) -> Trigger | None:
        if not isinstance(entry, dict):
            return None
        try:
            trigger_id = entry["id"]
            name = entry["name"]
            command = entry["action_command"]
            condition = entry.get("condition", {})
            if not all(isinstance(value, str) and value for value in (trigger_id, name, command)):
                return None
            if not isinstance(condition, dict):
                return None
            numbers = {key: entry.get(key, default) for key, default in (("fire_count", 0), ("max_fires", 0))}
            numbers["cooldown_seconds"] = entry.get("cooldown_seconds", 60)
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in numbers.values()):
                return None
            last_fired = entry.get("last_fired", 0)
            if not isinstance(last_fired, (int, float)) or isinstance(last_fired, bool):
                return None
            try:
                resolved_type = TriggerType(entry.get("trigger_type"))
            except ValueError:
                return None
            reason = str(entry.get("disabled_reason") or "")
            try:
                validate_trigger(
                    name, resolved_type.value, condition, command, numbers["max_fires"], numbers["cooldown_seconds"]
                )
            except ValueError as error:
                reason = f"Refused by the trigger rules when loaded: {error}"
            return Trigger(
                id=trigger_id,
                name=name,
                trigger_type=resolved_type,
                condition=condition,
                action_command=command,
                enabled=entry.get("enabled") is True and not reason,
                fire_count=numbers["fire_count"],
                max_fires=numbers["max_fires"],
                cooldown_seconds=numbers["cooldown_seconds"],
                last_fired=float(last_fired),
                created_at=str(entry.get("created_at") or ""),
                disabled_reason=reason,
            )
        except (KeyError, TypeError):
            return None

    def _save(self) -> None:
        """Write the whole store atomically (a temporary file, then a rename). No store: nothing to do."""
        if self._store_path is None or not self._store_writable:
            return
        payload = {
            "version": STORE_VERSION,
            "engine_running": self._wanted_running,
            "triggers": [_entry_for_store(trigger) for trigger in self._triggers.values()],
        }
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._store_path.with_name(f".{self._store_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._store_path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()

    def set_wanted_running(self, wanted: bool) -> None:
        """Record the person's choice to run or stop the engine, so a restart honours it."""
        previous = self._wanted_running
        self._wanted_running = wanted
        try:
            self._save()
        except OSError as error:
            self._wanted_running = previous
            raise RuntimeError(f"The trigger store could not be saved: {error}") from error

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
                            try:
                                self._save()  # keeps max_fires and the cooldown across a restart
                            except OSError:
                                logger.warning("The trigger store could not record a fire of %s", trigger.id)
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
    existing = _engine.find_matching(name, trigger_type, condition, action_command, max_fires, cooldown_seconds)
    trigger = existing or _engine.create_trigger(
        name,
        trigger_type,
        condition,
        action_command,
        max_fires,
        cooldown_seconds,
    )

    # Auto-start engine if not running (or if its loop has ended), and remember that it should run.
    if not _engine.wanted_running:
        _engine.set_wanted_running(True)
    if not _engine.is_running():
        await _engine.start()

    answer = _entry_for_store(trigger)
    if existing is not None:
        answer["note"] = "An identical trigger already existed and was kept; no new trigger was made."
    return json.dumps(answer, indent=2)


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
    """Start the trigger evaluation engine, and keep it running after a restart."""
    _engine.set_wanted_running(True)
    await _engine.start()
    return f"Trigger engine started with {len(_engine._triggers)} triggers"


async def trigger_stop_engine() -> str:
    """Stop the trigger evaluation engine, and keep it stopped after a restart."""
    _engine.set_wanted_running(False)
    await _engine.stop()
    return "Trigger engine stopped"


async def resume_triggers(store_path: Path) -> bool:
    """At daemon start: load the stored triggers and start the engine if the person left it running.

    Returns whether the engine was started. Nothing runs at load; a command runs only when its
    trigger fires, and then through the fire callback (the guarded autonomous pipeline).
    """
    if _engine.store_path is None:
        await asyncio.to_thread(_engine.attach_store, store_path)
    if _engine.wanted_running and any(trigger.enabled for trigger in _engine._triggers.values()):
        await _engine.start()
        return True
    return False


def get_engine() -> TriggerEngine:
    """Access the global trigger engine instance."""
    return _engine

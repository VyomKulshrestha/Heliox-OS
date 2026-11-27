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
import re
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
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


# The condition keys each trigger type reads, with what they mean. The planner's prompt lists the trigger types from
# this table (pilot.agents.planner), so it names only what the engine evaluates; tests/test_trigger_types_prompt.py
# checks that every key the engine reads is here. None: the engine accepts the type but never fires it.
CONDITION_KEYS: dict[TriggerType, dict[str, str] | None] = {
    TriggerType.FILE_CREATED: {
        "path": "the folder to watch",
        "pattern": "which new files count, like *.txt (default *)",
    },
    TriggerType.FILE_MODIFIED: {"path": "the file, or a folder", "pattern": "for a folder, which files (default *)"},
    TriggerType.FILE_DELETED: {"path": "the file, or a folder", "pattern": "for a folder, which files (default *)"},
    TriggerType.CPU_THRESHOLD: {"threshold": "percent; fires when use rises above it (default 90)"},
    TriggerType.MEMORY_THRESHOLD: {"threshold": "percent; fires when use rises above it (default 90)"},
    TriggerType.DISK_THRESHOLD: {
        "threshold": "percent used; fires when use rises above it (default 95)",
        "path": "the drive or folder (default /)",
    },
    TriggerType.BATTERY_LOW: {"threshold": "percent; fires when the charge falls below it, unplugged (default 20)"},
    TriggerType.PROCESS_STARTED: {"name": "part of the program's process name"},
    TriggerType.PROCESS_STOPPED: {"name": "part of the program's process name"},
    TriggerType.TIME_INTERVAL: {},  # fires every cooldown_seconds
    TriggerType.CUSTOM_CONDITION: {
        "expression": "a comparison of battery_percent, cpu_percent, disk_percent, hour, memory_percent, on_battery "
        "or timestamp, like cpu_percent > 80 and on_battery",
        "path": "the drive disk_percent is measured on (default /)",
    },
    TriggerType.CRON_SCHEDULE: None,
    TriggerType.NETWORK_CHANGE: None,
}

# Names a model writes for a trigger type, mapped to the type the engine knows (pilot.agents.planner applies them).
TRIGGER_TYPE_ALIASES: dict[str, TriggerType] = {
    "file_change": TriggerType.FILE_MODIFIED,
    "file_changed": TriggerType.FILE_MODIFIED,
    "file_changes": TriggerType.FILE_MODIFIED,
    "file_modify": TriggerType.FILE_MODIFIED,
    "file_modification": TriggerType.FILE_MODIFIED,
    "file_modifications": TriggerType.FILE_MODIFIED,
    "file_update": TriggerType.FILE_MODIFIED,
    "file_updated": TriggerType.FILE_MODIFIED,
    "file_write": TriggerType.FILE_MODIFIED,
    "file_written": TriggerType.FILE_MODIFIED,
    "file_watch": TriggerType.FILE_MODIFIED,
    "file_watcher": TriggerType.FILE_MODIFIED,
    "file_create": TriggerType.FILE_CREATED,
    "file_new": TriggerType.FILE_CREATED,
    "file_added": TriggerType.FILE_CREATED,
    "new_file": TriggerType.FILE_CREATED,
    "file_delete": TriggerType.FILE_DELETED,
    "file_removed": TriggerType.FILE_DELETED,
    "file_remove": TriggerType.FILE_DELETED,
    "cpu": TriggerType.CPU_THRESHOLD,
    "cpu_high": TriggerType.CPU_THRESHOLD,
    "cpu_usage": TriggerType.CPU_THRESHOLD,
    "memory": TriggerType.MEMORY_THRESHOLD,
    "memory_high": TriggerType.MEMORY_THRESHOLD,
    "memory_usage": TriggerType.MEMORY_THRESHOLD,
    "ram_threshold": TriggerType.MEMORY_THRESHOLD,
    "disk": TriggerType.DISK_THRESHOLD,
    "disk_full": TriggerType.DISK_THRESHOLD,
    "disk_usage": TriggerType.DISK_THRESHOLD,
    "battery": TriggerType.BATTERY_LOW,
    "low_battery": TriggerType.BATTERY_LOW,
    "battery_threshold": TriggerType.BATTERY_LOW,
    "process_start": TriggerType.PROCESS_STARTED,
    "process_running": TriggerType.PROCESS_STARTED,
    "app_started": TriggerType.PROCESS_STARTED,
    "process_stop": TriggerType.PROCESS_STOPPED,
    "process_exit": TriggerType.PROCESS_STOPPED,
    "process_exited": TriggerType.PROCESS_STOPPED,
    "app_stopped": TriggerType.PROCESS_STOPPED,
    "interval": TriggerType.TIME_INTERVAL,
    "timer": TriggerType.TIME_INTERVAL,
    "time": TriggerType.TIME_INTERVAL,
    "custom": TriggerType.CUSTOM_CONDITION,
    "condition": TriggerType.CUSTOM_CONDITION,
    "expression": TriggerType.CUSTOM_CONDITION,
}


def resolve_trigger_type(raw: str) -> str:
    """The engine's name for a trigger type a model wrote ("file_change" -> "file_modified"); the input unchanged
    when it is neither a type nor a known alias, so the engine's own check refuses it with its usual reason."""
    clean = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if clean in TriggerType._value2member_map_:
        return clean
    alias = TRIGGER_TYPE_ALIASES.get(clean)
    return alias.value if alias is not None else str(raw or "")


def trigger_types_for_prompt() -> str:
    """One line per trigger type the engine fires, with its condition keys: the planner prompt's list."""
    lines = []
    for trigger_type, keys in CONDITION_KEYS.items():
        if keys is None:
            continue
        if not keys:
            lines.append(f"  {trigger_type.value}: condition {{}}; fires every cooldown_seconds")
            continue
        described = "; ".join(f'"{key}": {meaning}' for key, meaning in keys.items())
        lines.append(f"  {trigger_type.value}: condition keys {described}")
    return "\n".join(lines)


@dataclass
class Trigger:
    id: str
    name: str
    trigger_type: TriggerType
    condition: dict[str, Any]  # Type-specific condition params
    # What the trigger does, in words: shown, and planned at a fire when the trigger has no fixed steps.
    action_command: str
    enabled: bool = True
    fire_count: int = 0
    max_fires: int = 0  # 0 = unlimited
    cooldown_seconds: int = 60
    last_fired: float = 0
    created_at: str = ""
    disabled_reason: str = ""  # why a stored trigger was loaded disabled; empty for a trigger that may fire
    # Whether a condition that holds over time (a threshold, a process, an expression) held at the last look; kept in
    # the store, so a restart does not fire again for a state that already fired. None: not looked at yet.
    condition_held: bool | None = None
    # The exact steps a fire runs (Action records), fixed when the trigger was made, and whether the person approved
    # them then. A fire runs these and nothing else; no model plans anything at a fire of a trigger that has them.
    steps: list[dict[str, Any]] = field(default_factory=list)
    steps_approved: bool = False
    # A fire that is waiting for the person because a step needs an approval nobody gave: when, why, and how many
    # fires since. None: nothing waits.
    pending: dict[str, Any] | None = None
    last_outcome: str = ""  # the sentence the person was told about the last fire
    # The trigger's job is only to tell the person: a fire is the engine's own notice (shown and spoken), with no steps
    # and no planned job. action_command is then empty or the person's words, never planned.
    notify_only: bool = False

    def can_fire(self) -> bool:
        if not self.enabled:
            return False
        if self.max_fires > 0 and self.fire_count >= self.max_fires:
            return False
        return not time.time() - self.last_fired < self.cooldown_seconds


STORE_VERSION = 1

# Conditions that hold over time rather than happen once: they fire when they start to hold, not while they hold, and
# not again after a restart while they still hold (Trigger.condition_held). File events compare with the last look the
# engine made since it started; an interval fires by its time.
LEVEL_TRIGGERS = frozenset(
    {
        TriggerType.CPU_THRESHOLD,
        TriggerType.MEMORY_THRESHOLD,
        TriggerType.DISK_THRESHOLD,
        TriggerType.BATTERY_LOW,
        TriggerType.PROCESS_STARTED,
        TriggerType.PROCESS_STOPPED,
        TriggerType.CUSTOM_CONDITION,
    }
)


def validate_trigger(
    name: str,
    trigger_type: str,
    condition: Any,
    action_command: str,
    max_fires: int,
    cooldown_seconds: int,
    notify_only: bool = False,
) -> TriggerType:
    """The rules every trigger must pass, at creation and again when it is loaded from the store.

    Raises ValueError with the reason. The command is natural language that only ever reaches the
    guarded autonomous pipeline (planning, permission checks, approval) when the trigger fires.
    """
    if not str(name).strip():
        raise ValueError("Trigger name is required")
    if not str(action_command).strip() and not notify_only:
        raise ValueError("Trigger action command is required")
    if not isinstance(condition, dict):
        raise ValueError("Trigger condition must be a mapping")
    if max_fires < 0:
        raise ValueError("Trigger max_fires cannot be negative")
    if cooldown_seconds < 0:
        raise ValueError("Trigger cooldown_seconds cannot be negative")
    try:
        resolved_type = TriggerType(trigger_type)
    except ValueError:
        known = ", ".join(item.value for item in TriggerType)
        raise ValueError(f"{trigger_type!r} is not a trigger type; the types are {known}") from None
    if resolved_type == TriggerType.CUSTOM_CONDITION:
        expression = str(condition.get("expression") or "").strip()
        if not expression:
            raise ValueError("Custom trigger expression is required")
        _parse_custom_expression(expression)
    return resolved_type


MAX_TRIGGER_STEPS = 5
FILE_TRIGGERS = frozenset({TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED, TriggerType.FILE_DELETED})
# A fire never makes or changes automation: a trigger that made triggers or schedules could grow without the person.
_NOT_A_TRIGGER_STEP = frozenset(
    {
        "trigger_create",
        "trigger_delete",
        "trigger_start",
        "trigger_stop",
        "schedule_create",
        "schedule_delete",
    }
)


# A trigger's words that only ask for the person to be told ("Tell the person that notes.txt has changed", "let me
# know", "alert me"), and no other job ("tell me and copy it to backup").
_TELLS = re.compile(
    r"^\s*(?:please\s+)?(?:tell|notify|alert|warn|inform|ping|remind)\b|^\s*(?:please\s+)?let\s+\w+\s+know\b",
    re.IGNORECASE,
)
_OTHER_JOB = re.compile(
    r"\b(?:append|add|write|save|copy|move|rename|delete|remove|run|execute|open|start|stop|kill|send|email|e-mail|post|"
    r"upload|download|install|restart|shut|zip|backup|back up)\b",
    re.IGNORECASE,
)


def tells_only(action_command: str) -> bool:
    """Whether a trigger's words only ask for the person to be told: such a trigger is notice-only."""
    text = " ".join(str(action_command or "").split())
    if not text or not _TELLS.search(text):
        return False
    rest = _TELLS.sub("", text, count=1)
    return not _OTHER_JOB.search(rest.split(" that ", 1)[0] if " that " in rest else rest)


def validate_steps(steps: Any) -> list[dict[str, Any]]:
    """The fixed steps of a trigger as stored records (``Action.model_dump(mode="json")``). Raises ValueError."""
    from pilot.actions import Action

    if steps is None:
        return []
    if not isinstance(steps, list):
        raise ValueError("Trigger steps must be a list")
    if len(steps) > MAX_TRIGGER_STEPS:
        raise ValueError(f"A trigger runs at most {MAX_TRIGGER_STEPS} steps")
    records = []
    for item in steps:
        action = item if isinstance(item, Action) else Action.model_validate(item)
        if action.action_type.value in _NOT_A_TRIGGER_STEP:
            raise ValueError(f"A trigger cannot run {action.action_type.value}")
        if action.rollback_action is not None or action.expect is not None:
            action = action.model_copy(update={"rollback_action": None, "expect": None})
        records.append(action.model_dump(mode="json"))
    return records


def fire_actions(trigger: Trigger) -> list[Any]:
    """The trigger's fixed steps as actions (validated again: the store is a file anyone may edit)."""
    from pilot.actions import Action

    return [Action.model_validate(record) for record in validate_steps(trigger.steps)]


def describe_steps(steps: list[dict[str, Any]]) -> str:
    """The fixed steps in a few words each, for the approval card and the sentences about a fire."""
    from pilot.actions import Action

    parts = []
    for record in steps:
        try:
            action = Action.model_validate(record)
        except ValueError:
            parts.append("a step that cannot be read")
            continue
        command = str(getattr(action.parameters, "command", "") or "").strip()
        what = command or action.target or ""
        parts.append(f"{action.action_type.value.replace('_', ' ')} {what}".strip())
    return "; ".join(parts)


def describe_trigger(trigger: Trigger) -> str:
    """'"name" (file modified: watched.txt)': how a sentence names one trigger."""
    watched = trigger.condition.get("path") or trigger.condition.get("name") or trigger.condition.get("expression")
    kind = trigger.trigger_type.value.replace("_", " ")
    target = Path(str(watched)).name if trigger.condition.get("path") else str(watched or "")
    return f'"{trigger.name}" ({kind}: {target})' if target else f'"{trigger.name}" ({kind})'


# Words a description of a trigger uses that say nothing about which trigger it is.
_DESCRIPTION_FILLER = frozenset(
    [
        "a",
        "an",
        "the",
        "my",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "trigger",
        "triggers",
        "one",
        "ones",
        "which",
        "who",
        "whose",
        "what",
        "for",
        "on",
        "of",
        "in",
        "at",
        "to",
        "from",
        "with",
        "and",
        "or",
        "called",
        "named",
        "name",
        "watch",
        "watches",
        "watching",
        "watcher",
        "delete",
        "remove",
        "cancel",
        "stop",
        "drop",
        "get",
        "rid",
        "please",
        "file",
        "files",
        "folder",
        "when",
        "whenever",
        "every",
        "each",
        "if",
        "changes",
        "change",
        "changed",
        "fires",
        "fire",
        "runs",
        "run",
    ]
)


def _words(text: str) -> set[str]:
    raw = str(text or "").lower()
    words = set(re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", raw))
    for word in list(words):
        words.update(part for part in re.split(r"[._-]", word) if part)
    return words


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


OFFER_SECONDS = 600.0  # a pick ("both of them") refers to a list Heliox gave at most this long ago


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
        # The triggers the last "which one?" listed, in that order, and when (time.monotonic): the next reply may pick
        # from them ("both of them", "the first one"). Kept in memory only: a reply after a restart is not a pick.
        self._offer: list[str] = []
        self._offer_at = 0.0

    def offer(self, trigger_ids: list[str]) -> None:
        """Remember the triggers a question just listed, so "both of them" in the next reply means these."""
        self._offer = [trigger_id for trigger_id in trigger_ids if trigger_id in self._triggers]
        self._offer_at = time.monotonic()

    def offered(self, max_age: float = OFFER_SECONDS) -> list[Trigger]:
        """The triggers last offered that still exist, in the order they were listed; [] once too old."""
        if not self._offer or time.monotonic() - self._offer_at > max_age:
            return []
        return [self._triggers[trigger_id] for trigger_id in self._offer if trigger_id in self._triggers]

    def clear_offer(self) -> None:
        self._offer = []

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
        steps: list[Any] | None = None,
        steps_approved: bool = False,
        notify_only: bool = False,
    ) -> Trigger:
        """Create a new trigger (and keep it in the store, when one is attached)."""
        name = name.strip()
        action_command = action_command.strip()
        records = validate_steps(steps)
        notify_only = bool(notify_only and not records)
        resolved_type = validate_trigger(
            name, trigger_type, condition, action_command, max_fires, cooldown_seconds, notify_only
        )
        trigger = Trigger(
            id=str(uuid.uuid4())[:8],
            name=name,
            trigger_type=resolved_type,
            condition=condition,
            action_command=action_command,
            max_fires=max_fires,
            cooldown_seconds=cooldown_seconds,
            created_at=datetime.now().isoformat(),
            steps=records,
            steps_approved=bool(steps_approved and records),
            notify_only=notify_only,
        )
        self._triggers[trigger.id] = trigger
        try:
            self._save()
        except OSError as error:
            del self._triggers[trigger.id]
            raise RuntimeError(f"The trigger could not be saved, so it was not created: {error}") from error
        logger.info("Created trigger: %s (%s)", name, trigger.id)
        return trigger

    def resolve(self, description: str) -> list[Trigger]:
        """The triggers a description names: its id or name exactly (any case), else the triggers whose name, type
        and target share the most telling words with it ("the trigger that watches watched.txt"). [] when none do."""
        wanted = str(description or "").strip()
        if not wanted:
            return []
        exact = [
            trigger
            for trigger in self._triggers.values()
            if wanted == trigger.id or wanted.casefold() == trigger.name.casefold()
        ]
        if exact:
            return exact
        words = _words(wanted) - _DESCRIPTION_FILLER
        if not words:
            # "delete the trigger", "stop watching that file": the one trigger there is (of that kind), if only one.
            pool = list(self._triggers.values())
            if _words(wanted) & {"file", "files", "folder"}:
                pool = [trigger for trigger in pool if trigger.trigger_type in FILE_TRIGGERS]
            return pool
        # Every telling word must be the trigger's: one shared word ("alert") never picks a trigger to delete.
        matches = []
        for trigger in self._triggers.values():
            haystack = (
                _words(trigger.name)
                | _words(trigger.trigger_type.value)
                | _words(trigger.action_command)
                | {trigger.id.lower()}
            )
            for value in trigger.condition.values():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    haystack |= _words(str(value))
                    if isinstance(value, str) and value:
                        haystack |= _words(Path(value).name)
            if words <= haystack:
                matches.append(trigger)
        return matches

    def set_pending(self, trigger_id: str, pending: dict[str, Any] | None, outcome: str = "") -> None:
        """Record a fire that waits for the person (or clear it), and the sentence the person was told."""
        trigger = self._triggers.get(trigger_id)
        if trigger is None:
            return
        trigger.pending = pending
        if outcome:
            trigger.last_outcome = outcome
        try:
            self._save()
        except OSError:
            logger.warning("The trigger store could not record the outcome of a fire of %s", trigger_id)

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
            notify_only = entry.get("notify_only") is True
            if not all(isinstance(value, str) and value for value in (trigger_id, name)) or not isinstance(
                command, str
            ):
                return None
            if not command and not notify_only:
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
            held = entry.get("condition_held")
            try:
                steps = validate_steps(entry.get("steps") or [])
            except ValueError as error:
                steps = []
                reason = reason or f"Its steps were refused when loaded: {error}"
            pending = entry.get("pending")
            try:
                validate_trigger(
                    name,
                    resolved_type.value,
                    condition,
                    command,
                    numbers["max_fires"],
                    numbers["cooldown_seconds"],
                    notify_only,
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
                condition_held=held if isinstance(held, bool) else None,
                steps=steps,
                steps_approved=entry.get("steps_approved") is True and bool(steps),
                pending=pending if isinstance(pending, dict) else None,
                last_outcome=str(entry.get("last_outcome") or ""),
                notify_only=notify_only and not steps,
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
        steps: list[Any] | None = None,
        notify_only: bool = False,
    ) -> Trigger | None:
        """An enabled trigger that already has exactly these settings, if one exists."""
        records = validate_steps(steps)
        for trigger in self._triggers.values():
            if (
                trigger.enabled
                and trigger.name == name.strip()
                and trigger.trigger_type.value == trigger_type
                and trigger.condition == condition
                and trigger.action_command == action_command.strip()
                and trigger.max_fires == max_fires
                and trigger.cooldown_seconds == cooldown_seconds
                and trigger.steps == records
                and trigger.notify_only == bool(notify_only and not records)
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
        """Main evaluation loop: checks all triggers every 5 seconds."""
        while self._running:
            try:
                await self.evaluate_once()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

    async def evaluate_once(self) -> list[Trigger]:
        """Look once at every trigger that may fire, and fire those whose condition started to hold. Returns them."""
        fired_now: list[Trigger] = []
        for trigger in list(self._triggers.values()):
            if not trigger.can_fire():
                continue
            try:
                holds = await self._evaluate(trigger)
                if trigger.trigger_type in LEVEL_TRIGGERS:
                    # On a change only: "CPU above 90 %" fires when it goes above, not every minute while it stays
                    # there, and not again at the next start because it is still there.
                    fired = holds and trigger.condition_held is not True
                    changed = trigger.condition_held is not holds
                    trigger.condition_held = holds
                    if changed and not fired:
                        try:
                            self._save()
                        except OSError:
                            logger.warning("The trigger store could not record the state of %s", trigger.id)
                else:
                    fired = holds
                if fired:
                    trigger.fire_count += 1
                    trigger.last_fired = time.time()
                    try:
                        self._save()  # keeps max_fires, the cooldown and the state that fired across a restart
                    except OSError:
                        logger.warning("The trigger store could not record a fire of %s", trigger.id)
                    logger.info("Trigger fired: %s (%s), fire #%d", trigger.name, trigger.id, trigger.fire_count)
                    fired_now.append(trigger)
                    if self._fire_callback:
                        task = asyncio.create_task(
                            self._fire_callback(trigger),
                            name=f"trigger_action_{trigger.id}",
                        )
                        self._callback_tasks.add(task)
                        task.add_done_callback(self._callback_done)
            except Exception as e:
                logger.warning("Trigger %s eval error: %s", trigger.id, e)
        return fired_now

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
    steps: list[Any] | None = None,
    steps_approved: bool = False,
    notify_only: bool = False,
) -> str:
    """Create a reactive trigger. ``notify_only``: its job is only to tell the person (see ``Trigger.notify_only``).

    ``steps``: the exact steps a fire runs, fixed now; ``steps_approved``: the person approved them in this request.
    A step that needs approval and was not approved here is not run at a fire: the fire waits for the person.

    Asking again for a trigger with exactly the same settings reuses the existing
    one instead of adding a twin that would fire the same command twice; this keeps
    a retry after a failed verification from doubling the trigger.
    """
    notify_only = bool(notify_only or (not steps and tells_only(action_command)))
    if notify_only:
        action_command = ""  # nothing to plan: the notice is the job
    existing = _engine.find_matching(
        name, trigger_type, condition, action_command, max_fires, cooldown_seconds, steps, notify_only
    )
    if existing is not None and steps_approved and existing.steps and not existing.steps_approved:
        existing.steps_approved = True  # the same steps, approved this time
        _engine._save()
    trigger = existing or _engine.create_trigger(
        name,
        trigger_type,
        condition,
        action_command,
        max_fires,
        cooldown_seconds,
        steps,
        steps_approved,
        notify_only,
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


DELETED_ID = re.compile(r"\(id ([^)\s]+)\)\.?$")


def deleted_record(trigger: Trigger) -> dict[str, Any]:
    """What a delete answers: the whole stored entry of the trigger it removed (its name, type, condition, what it
    watched, its words and its fixed steps), plus ``deleted`` and a plain ``text`` that ends with "(id <id>)"."""
    record = _entry_for_store(trigger)
    path = trigger.condition.get("path")
    record["watched"] = Path(str(path)).name if path else ""
    # How many triggers still watch the same thing after this one went: the reply says the others remain.
    record["others_watching"] = (
        sum(
            1
            for other in _engine.list_triggers()
            if other.get("id") != trigger.id and path and (other.get("condition") or {}).get("path") == path
        )
        if path
        else 0
    )
    record["deleted"] = True
    record["text"] = f"Deleted the trigger {describe_trigger(trigger)} (id {trigger.id})."
    return record


def deleted_id(output: str) -> str:
    """The id a delete's answer names: its JSON ``id``, or the "(id <id>)" its text ends with; "" when none."""
    try:
        record = json.loads(output or "")
    except ValueError:
        record = None
    if isinstance(record, dict) and record.get("deleted") is True and isinstance(record.get("id"), str):
        return record["id"]
    found = DELETED_ID.search((output or "").strip())
    return found.group(1) if found else ""


def picked_triggers(request: str) -> list[Trigger] | None:
    """The triggers a reply picks from the ones Heliox just offered ("delete both of them", "the second one",
    "all of them"): [] for "neither", None when nothing was offered or the reply is not a pick. Resolved in code
    (pilot.agents.request_match.picked_from), never from a model's reading of the conversation."""
    from pilot.agents.request_match import picked_from

    offered = _engine.offered()
    if not offered:
        return None
    picked = picked_from(request, len(offered))
    return None if picked is None else [offered[index] for index in picked]


_DELETE_BY_IDS = re.compile(r"^Delete the triggers? with the ids? (?P<ids>[\w-]+(?: and [\w-]+)*)\.?$")


def pick_request(request: str) -> tuple[str, str] | None:
    """A reply that picks triggers to delete, written out as the request it means (by id, so nothing is guessed),
    with the note shown in the chat. None for anything else, and for "neither" (``kept_sentence`` answers that)."""
    picked = picked_triggers(request)
    if not picked:
        return None
    ids = " and ".join(trigger.id for trigger in picked)
    noun = "trigger" if len(picked) == 1 else "triggers"
    names = " and ".join(describe_trigger(trigger) for trigger in picked)
    return f"Delete the {noun} with the id{'s' if len(picked) > 1 else ''} {ids}.", f"Deleting {names}"


def ids_to_delete(request: str) -> list[str] | None:
    """The ids of a request ``pick_request`` wrote, or None for any other request."""
    found = _DELETE_BY_IDS.match(str(request or "").strip())
    return found["ids"].split(" and ") if found else None


def kept_sentence(request: str) -> str | None:
    """The answer to "neither" (or "none", "keep them") after Heliox asked which trigger to delete."""
    picked = picked_triggers(request)
    if picked is None or picked:
        return None
    offered = _engine.offered()
    _engine.clear_offer()
    if len(offered) == 1:
        return f"Okay, I've kept the trigger {describe_trigger(offered[0])}."
    if len(offered) == 2:
        return "Okay, I've kept both triggers; nothing was deleted."
    return f"Okay, I've kept all {len(offered)} triggers; nothing was deleted."


class TriggerNotFound(LookupError):
    """No trigger, or more than one, matches what the person said: nothing was deleted, and another plan would not
    know better (the sentence lists the triggers there are)."""


async def trigger_delete(description: str) -> str:
    """Delete the one trigger ``description`` names: its id, its name, or words such as "the trigger that watches
    watched.txt" (resolved against the stored triggers by name, type and target). Several matches or none: nothing
    is deleted and the sentence says which there are. The answer ends with "(id <id>)" for the post-condition."""
    wanted = str(description or "").strip()
    matches = _engine.resolve(wanted)
    if len(matches) == 1:
        found = matches[0]
        if not _engine.delete_trigger(found.id):
            raise TriggerNotFound(f"The trigger {describe_trigger(found)} was already gone, so nothing was deleted.")
        _engine.clear_offer()
        return json.dumps(deleted_record(found), indent=2, ensure_ascii=False)
    if matches:
        _engine.offer([trigger.id for trigger in matches])  # the next reply may pick from these
        listed = ", ".join(describe_trigger(trigger) for trigger in matches)
        raise TriggerNotFound(
            f'{len(matches)} triggers match "{wanted}": {listed}. Nothing was deleted; say which one to delete.'
        )
    there = [describe_trigger(trigger) for trigger in _engine._triggers.values()]
    _engine.offer(list(_engine._triggers))  # listed below: "delete both of them" next means these
    known = f" The triggers there are: {', '.join(there)}." if there else " There are no triggers."
    raise TriggerNotFound(f'No trigger matches "{wanted}", so nothing was deleted.{known}')


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

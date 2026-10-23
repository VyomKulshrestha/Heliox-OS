"""Routines from finished tasks, voice shortcuts, and "do that again".

A routine turns a task Heliox already finished into a scheduled run that may use only
the action types that task actually used successfully, so a routine can never do more
than the task it came from. Saying "do that every day at 9" only proposes the routine;
it is created when the person presses Create on screen.

A voice shortcut maps a short phrase to a command: "when I say start work, open VS Code
and Slack". Saying the phrase later submits the command as an ordinary request, with
the normal planning, gateway and approvals. Phrases that already mean something to
Heliox (stop, cancel, approve, no, undo) cannot become shortcuts.

"Do that again" resubmits the last task's request, also as an ordinary request.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

ROUTINE_REQUEST = re.compile(
    r"^(?:please )?(?:do|repeat|run) (?:that|this|it|the last task|my last task) (?P<schedule>every .+)$"
)
REPEAT_REQUEST = re.compile(
    r"^(?:please )?(?:do (?:that|it) again|run (?:that|it) again|repeat (?:that|it|the last task|my last task))$"
)
SHORTCUT_DEFINITION = re.compile(r"^when i say (?P<rest>.+)$")
# Verbs that usually start the command part of "when I say <phrase> <command>".
COMMAND_VERBS = (
    "open",
    "start",
    "launch",
    "run",
    "show",
    "play",
    "close",
    "set",
    "turn",
    "search",
    "check",
    "find",
    "tell",
    "read",
    "mute",
    "lock",
    "do",
    "go",
    "take",
    "create",
    "send",
    "list",
)
DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MAX_SHORTCUTS = 50
MAX_ROUTINE_ACTIONS = 60


def normalize_phrase(text: str) -> str:
    """Casefold, turn punctuation into spaces and collapse whitespace (as the voice path does)."""
    return " ".join(
        "".join(
            character if character.isalnum() or character.isspace() else " " for character in text.casefold()
        ).split()
    )


def _clock(hour: int, minute: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    shown = hour % 12 or 12
    return f"{shown}:{minute:02d} {suffix}"


def parse_schedule(text: str) -> dict[str, Any] | None:
    """Parse "every day at 9", "every weekday at 8:30 am", "every monday at 18:00", "every 2 hours"."""
    text = re.sub(r"(\d)\s*([ap])\.?\s*m\b\.?", r"\1 \2m", text.casefold())
    text = normalize_phrase(text)
    interval = re.fullmatch(r"every (?:(\d+) )?(minute|minutes|hour|hours)", text)
    if interval:
        amount = int(interval.group(1) or 1)
        minutes = amount * (60 if interval.group(2).startswith("hour") else 1)
        unit = "hour" if interval.group(2).startswith("hour") else "minute"
        label = f"every {amount} {unit}{'' if amount == 1 else 's'}" if amount != 1 else f"every {unit}"
        return {"every_minutes": minutes, "label": label}
    daily = re.fullmatch(
        r"every (?P<days>day|morning|evening|weekday|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"(?: at (?P<hour>\d{1,2})(?: (?P<minute>\d{2}))?(?: (?P<meridiem>am|pm|a m|p m))?)?",
        text,
    )
    if not daily:
        return None
    days = daily.group("days")
    hour = int(daily.group("hour")) if daily.group("hour") else {"morning": 9, "evening": 18}.get(days, 9)
    minute = int(daily.group("minute") or 0)
    meridiem = (daily.group("meridiem") or "").replace(" ", "")
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif days == "evening" and hour < 12:
        hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    weekdays: list[int] = []
    when = "every day"
    if days == "weekday":
        weekdays, when = [0, 1, 2, 3, 4], "every weekday"
    elif days == "weekend":
        weekdays, when = [5, 6], "every weekend day"
    elif days in DAY_NAMES:
        weekdays, when = [DAY_NAMES.index(days)], f"every {days.capitalize()}"
    return {"daily_at": f"{hour:02d}:{minute:02d}", "weekdays": weekdays, "label": f"{when} at {_clock(hour, minute)}"}


def routine_from_receipt(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Name, goal and the action types a routine made from this receipt may use."""
    receipt = envelope.get("receipt") if isinstance(envelope, dict) else None
    if not isinstance(receipt, dict):
        return None
    goal = str(receipt.get("request") or "").strip()
    actions = sorted(
        {
            str(action.get("action_type"))
            for action in receipt.get("actions", [])
            if action.get("executed", True) and action.get("success") and action.get("action_type")
        }
    )
    if not goal or not actions or len(actions) > MAX_ROUTINE_ACTIONS:
        return None
    name = goal if len(goal) <= 60 else goal[:57].rstrip() + "..."
    return {"receipt_id": receipt.get("receipt_id"), "name": name, "goal": goal[:2000], "allowed_actions": actions}


def parse_shortcut_definition(text: str) -> tuple[str, str] | None:
    """Split "when I say start work, open VS Code" into ("start work", "open VS Code")."""
    stripped = text.strip()
    if not SHORTCUT_DEFINITION.match(normalize_phrase(stripped)):
        return None
    rest = re.sub(r"^\s*when\s+i\s+say\s+", "", stripped, flags=re.IGNORECASE)
    quoted = re.match(r"""^["'“”‘’](?P<phrase>[^"'“”‘’]+)["'“”‘’][\s,:-]*(?P<command>.+)$""", rest)
    if quoted:
        phrase, command = quoted.group("phrase"), quoted.group("command")
    elif "," in rest:
        phrase, command = rest.split(",", 1)
    elif re.search(r"\s+then\s+", rest, flags=re.IGNORECASE):
        phrase, command = re.split(r"\s+then\s+", rest, maxsplit=1, flags=re.IGNORECASE)
    else:
        words = rest.split()
        split_at = next(
            (index for index, word in enumerate(words) if index > 0 and word.casefold().strip(",.") in COMMAND_VERBS),
            None,
        )
        if split_at is None:
            return None
        phrase, command = " ".join(words[:split_at]), " ".join(words[split_at:])
    command = re.sub(r"^(?:then|you should|please|heliox)\s+", "", command.strip(), flags=re.IGNORECASE).strip()
    phrase = normalize_phrase(phrase)
    if not phrase or not command:
        return None
    return phrase, command


class ShortcutStore:
    """Voice shortcuts kept in a small JSON file."""

    def __init__(self, path: Path, *, reserved: Any = ()) -> None:
        self._path = Path(path)
        self._reserved = reserved  # phrases (or a function phrase -> bool) that already mean something

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("phrase") and item.get("command")]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def _is_reserved(self, phrase: str) -> bool:
        if callable(self._reserved):
            return bool(self._reserved(phrase))
        return phrase in self._reserved

    def add(self, phrase: str, command: str) -> dict[str, Any]:
        """Add or replace a shortcut. Raises ValueError with a message the person can act on."""
        phrase = normalize_phrase(phrase)
        command = " ".join(str(command).split())
        if not 2 <= len(phrase) <= 60 or len(phrase.split()) > 6:
            raise ValueError("A shortcut phrase must be 2 to 60 characters and at most 6 words")
        if not 1 <= len(command) <= 500:
            raise ValueError("A shortcut command must be 1 to 500 characters")
        if self._is_reserved(phrase):
            raise ValueError(f'"{phrase}" already means something to Heliox, so it cannot be a shortcut')
        if normalize_phrase(command) == phrase:
            raise ValueError("A shortcut cannot run itself")
        items = [item for item in self.list() if item["phrase"] != phrase]
        if len(items) >= MAX_SHORTCUTS:
            raise ValueError(f"You can keep up to {MAX_SHORTCUTS} shortcuts")
        entry = {"phrase": phrase, "command": command, "created_at": time.time()}
        items.append(entry)
        self._save(items)
        return entry

    def remove(self, phrase: str) -> bool:
        phrase = normalize_phrase(phrase)
        items = self.list()
        kept = [item for item in items if item["phrase"] != phrase]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    def match(self, text: str) -> dict[str, Any] | None:
        """The shortcut whose phrase is exactly ``text`` (ignoring case, punctuation and a leading "heliox")."""
        phrase = re.sub(r"^(?:hey )?heliox ", "", normalize_phrase(text))
        return next((item for item in self.list() if item["phrase"] == phrase), None)

"""Reminders: "remind me in 20 minutes to stretch", kept on this computer and answered in code.

A reminder is a time and a short text. Setting one never plans or runs an action: it is
stored in ``DATA_DIR/reminders.json`` and, when due, shown as a notification and in
chat and spoken. A reminder that fell due while Heliox was not running is delivered
when it starts again, marked as late, unless it is more than a day old.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.reminders")

CHECK_EVERY_SECONDS = 15.0
MAX_REMINDERS = 100
MAX_TEXT = 300
DROP_LATE_AFTER_SECONDS = 24 * 3600

SET_REQUEST = re.compile(r"^(?:please |hey heliox |heliox )?remind me (?P<rest>.+)$")
LIST_REQUEST = re.compile(
    r"^(?:hey heliox |heliox )?(?:what are my reminders|what reminders do i have|list my reminders|show my reminders)$"
)
CANCEL_REQUEST = re.compile(
    r"^(?:please )?(?:cancel|delete|remove|clear) (?:all )?(?:my |the )?reminders?(?: (?:to|about) (?P<what>.+))?$"
)
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s:]", " ", text.casefold()).split())


def _clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def describe_time(due: float, now: datetime) -> str:
    moment = datetime.fromtimestamp(due)
    if moment.date() == now.date():
        return f"at {_clock(moment)}"
    if moment.date() == (now + timedelta(days=1)).date():
        return f"tomorrow at {_clock(moment)}"
    return f"on {moment.strftime('%a %d %b')} at {_clock(moment)}"


def _parse_clock(hour: str, minute: str | None, meridiem: str | None, now: datetime) -> tuple[int, int] | None:
    h, m = int(hour), int(minute or 0)
    meridiem = (meridiem or "").replace(".", "").replace(" ", "")
    if meridiem == "pm" and h < 12:
        h += 12
    elif meridiem == "am" and h == 12:
        h = 0
    elif not meridiem and 1 <= h <= 7 and now.hour >= 8:
        h += 12  # "at 5" said during the day means 5 PM
    return (h, m) if 0 <= h <= 23 and 0 <= m <= 59 else None


def parse_reminder(text: str, now: datetime) -> tuple[float, str] | None:
    """(due epoch, reminder text) from "remind me in 20 minutes to stretch" and similar; None if unclear."""
    asked = SET_REQUEST.match(_normalize(text))
    if not asked:
        return None
    rest = asked.group("rest")
    when_patterns = (
        r"in (?P<amount>\d+|a|an|one|two|three|four|five|ten|fifteen) (?P<unit>minute|minutes|min|mins|hour|hours)",
        r"(?P<tomorrow>tomorrow )?(?:at|by) (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))? ?(?P<meridiem>am|pm|a m|p m)?",
        r"(?P<tomorrow2>tomorrow)(?: morning)?",
    )
    for pattern in when_patterns:
        # "<when> to <what>" or "to <what> <when>"
        for layout in (
            rf"^{pattern} (?:to |that |about )?(?P<what>.+)$",
            rf"^(?:to |that |about )?(?P<what>.+?) {pattern}$",
        ):
            match = re.match(layout, rest)
            if not match:
                continue
            groups = match.groupdict()
            what = groups["what"].strip()
            if groups.get("amount"):
                amount = int(groups["amount"]) if groups["amount"].isdigit() else NUMBER_WORDS[groups["amount"]]
                minutes = amount * (60 if groups["unit"].startswith("h") else 1)
                if not 1 <= minutes <= 7 * 24 * 60:
                    return None
                due = now + timedelta(minutes=minutes)
            elif groups.get("hour"):
                clock = _parse_clock(groups["hour"], groups.get("minute"), groups.get("meridiem"), now)
                if clock is None:
                    return None
                due = now.replace(hour=clock[0], minute=clock[1], second=0, microsecond=0)
                if groups.get("tomorrow") or due <= now:  # a time already past today means tomorrow
                    due += timedelta(days=1)
            else:
                due = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            if not what or len(what) > MAX_TEXT:
                return None
            return due.timestamp(), what
    return None


class ReminderStore:
    """Reminders in a small JSON file, delivered by a background loop."""

    def __init__(
        self,
        path: Path,
        *,
        deliver: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._deliver = deliver
        self._clock = clock
        self._task: asyncio.Task[None] | None = None

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items = [item for item in data if isinstance(item, dict) and "due" in item] if isinstance(data, list) else []
        return sorted(items, key=lambda item: item["due"])

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, due: float, text: str) -> dict[str, Any]:
        text = " ".join(str(text).split())
        if not 1 <= len(text) <= MAX_TEXT:
            raise ValueError(f"A reminder must be 1 to {MAX_TEXT} characters")
        if due <= self._clock():
            raise ValueError("That time has already passed")
        items = self.list()
        if len(items) >= MAX_REMINDERS:
            raise ValueError(f"You can keep up to {MAX_REMINDERS} reminders")
        entry = {"id": uuid.uuid4().hex[:10], "due": float(due), "text": text, "created_at": self._clock()}
        items.append(entry)
        self._save(items)
        return entry

    def cancel(self, reminder_id: str | None = None, *, about: str | None = None) -> int:
        """Cancel one reminder by id, those whose text contains ``about``, or (neither given) all."""
        items = self.list()
        if reminder_id:
            kept = [item for item in items if item["id"] != reminder_id]
        elif about:
            needle = about.casefold()
            kept = [item for item in items if needle not in item["text"].casefold()]
        else:
            kept = []
        if len(kept) != len(items):
            self._save(kept)
        return len(items) - len(kept)

    async def tick(self) -> int:
        """Deliver every reminder that is due. Returns how many went out."""
        now = self._clock()
        items = self.list()
        due = [item for item in items if item["due"] <= now]
        if not due:
            return 0
        self._save([item for item in items if item["due"] > now])  # before delivering, so none repeats
        sent = 0
        for item in due:
            late = now - item["due"]
            if late > DROP_LATE_AFTER_SECONDS:
                continue
            if self._deliver is not None:
                await self._deliver({**item, "late": late > 120})
            sent += 1
        return sent

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Reminder delivery failed", exc_info=True)
            await asyncio.sleep(CHECK_EVERY_SECONDS)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def answer(self, text: str, *, now: datetime | None = None) -> str | None:
        """Set, list or cancel reminders from a sentence; None when it is not about reminders."""
        now = now or datetime.fromtimestamp(self._clock())
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return "You have no reminders."
            listed = "; ".join(f"{describe_time(item['due'], now)}: {item['text']}" for item in items[:5])
            more = f" And {len(items) - 5} more." if len(items) > 5 else ""
            return f"You have {len(items)} reminder{'' if len(items) == 1 else 's'}. {listed}.{more}"
        cancel = CANCEL_REQUEST.match(normalized)
        if cancel:
            removed = self.cancel(about=cancel.group("what"))
            if not removed:
                return "There was no reminder like that to cancel."
            return f"Cancelled {removed} reminder{'' if removed == 1 else 's'}."
        if not SET_REQUEST.match(normalized):
            return None
        parsed = parse_reminder(text, now)
        if parsed is None:
            return 'Tell me when, for example "remind me in 20 minutes to stretch" or "remind me at 5 pm to leave".'
        due, what = parsed
        try:
            self.add(due, what)
        except ValueError as exc:
            return str(exc)
        return f"Okay, I'll remind you {describe_time(due, now)} to {what}."

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)

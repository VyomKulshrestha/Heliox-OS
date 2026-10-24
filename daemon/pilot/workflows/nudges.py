"""Usage nudges: "tell me if I spend more than an hour on YouTube today".

A nudge is a word to look for in app names and window titles and a daily limit in
minutes. A background check adds up today's matching time from the activity timeline
(so nudges need the timeline on) and, once the limit is passed, tells the person once
that day. Nudges never block, close or change anything.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.nudges")

CHECK_EVERY_SECONDS = 60.0
MAX_NUDGES = 30
WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}

_AMOUNT = r"(?P<amount>\d+|an|a|one|two|three|half an) (?P<unit>minutes?|mins?|hours?)"
ADD_REQUEST = re.compile(
    r"^(?:please )?(?:tell|warn|nudge|remind|alert) me (?:if|when) i (?:spend|have spent|use|am on|have been on) "
    r"(?:more than |over )?"
    + _AMOUNT
    + r" (?:on|in|using|watching) (?P<what>.+?)(?: today| a day| per day| each day)?$"
    r"|^(?:please )?(?:tell|warn|nudge|remind|alert) me after "
    + _AMOUNT.replace("amount", "amount2").replace("unit", "unit2")
    + r" (?:of|on|in) (?P<what2>.+?)(?: today| a day| per day| each day)?$"
)
LIST_REQUEST = re.compile(
    r"^(?:what|which) (?:usage )?(?:nudges|limits) do i have$|^(?:list|show) my (?:nudges|limits)$"
)
DELETE_REQUEST = re.compile(
    r"^(?:stop|delete|remove|cancel) (?:the |my )?(?:nudge|limit|warning) (?:for|about|on) (?P<what>.+)$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _minutes(amount: str, unit: str) -> int:
    if amount == "half an":
        return 30
    value = int(amount) if amount.isdigit() else WORDS[amount]
    return value * (60 if unit.startswith("hour") else 1)


def _duration(minutes: float) -> str:
    minutes = max(1, round(minutes))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


class NudgeStore:
    """Nudge rules in a small JSON file, checked by a background loop."""

    def __init__(
        self,
        path: Path,
        *,
        usage: Callable[[str, float, float], float] | None = None,
        timeline_on: Callable[[], bool] = lambda: False,
        deliver: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._path = Path(path)
        self._usage = usage  # (needle, start, end) -> seconds of matching activity
        self._timeline_on = timeline_on
        self._deliver = deliver
        self._now = now
        self._task: asyncio.Task[None] | None = None

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("match")] if isinstance(data, list) else []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, match: str, minutes: int) -> dict[str, Any]:
        match = " ".join(str(match).split())
        if not 1 <= len(match) <= 60:
            raise ValueError("What to watch for must be 1 to 60 characters")
        if not 1 <= int(minutes) <= 24 * 60:
            raise ValueError("The limit must be from 1 minute to 24 hours")
        items = [item for item in self.list() if item["match"].casefold() != match.casefold()]
        if len(items) >= MAX_NUDGES:
            raise ValueError(f"You can keep up to {MAX_NUDGES} nudges")
        entry = {"id": uuid.uuid4().hex[:10], "match": match, "minutes": int(minutes), "last_fired_day": ""}
        items.append(entry)
        self._save(items)
        return entry

    def delete(self, nudge_id: str | None = None, *, match: str | None = None) -> bool:
        items = self.list()
        if nudge_id:
            kept = [item for item in items if item["id"] != nudge_id]
        else:
            needle = (match or "").casefold().strip()
            kept = [item for item in items if item["match"].casefold() != needle]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    async def tick(self) -> int:
        """Tell the person about every nudge whose limit today's activity has passed. Returns how many."""
        if self._usage is None or not self._timeline_on():
            return 0
        now = self._now()
        today = now.date().isoformat()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        items = self.list()
        fired = 0
        for item in items:
            if item.get("last_fired_day") == today:
                continue
            seconds = await asyncio.to_thread(self._usage, item["match"], start, now.timestamp())
            if seconds < item["minutes"] * 60:
                continue
            item["last_fired_day"] = today
            fired += 1
            if self._deliver is not None:
                await self._deliver(
                    {
                        "id": item["id"],
                        "match": item["match"],
                        "minutes": item["minutes"],
                        "text": f"You have spent {_duration(seconds / 60)} on {item['match']} today.",
                    }
                )
        if fired:
            self._save(items)
        return fired

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Usage nudge check failed", exc_info=True)
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

    def answer(self, text: str) -> str | None:
        """Add, list or remove nudges from a sentence; None when it is not about nudges."""
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return "You have no usage nudges."
            listed = "; ".join(f"{item['match']} after {_duration(item['minutes'])}" for item in items)
            return f"Your nudges: {listed}."
        delete = DELETE_REQUEST.match(normalized)
        if delete:
            what = delete.group("what").strip()
            return f"Removed the nudge for {what}." if self.delete(match=what) else f"You have no nudge for {what}."
        added = ADD_REQUEST.match(normalized)
        if not added:
            return None
        if not self._timeline_on():
            return "Usage nudges use the activity timeline. Turn it on in Settings first, then ask again."
        amount, unit = added.group("amount") or added.group("amount2"), added.group("unit") or added.group("unit2")
        what = (added.group("what") or added.group("what2")).strip()
        try:
            entry = self.add(what, _minutes(amount, unit))
        except ValueError as exc:
            return str(exc)
        return f"Okay. I'll tell you once a day when you pass {_duration(entry['minutes'])} on {entry['match']}."

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)

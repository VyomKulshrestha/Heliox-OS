"""Habits: "log a workout", "how many workouts this week?", "what's my reading streak?".

A habit is a name and the times it was logged, kept in ``DATA_DIR/habits.json`` for a
year. "Log …" starts a habit the first time; "I did …", "I finished …" and "I drank …"
only count for habits that already exist, so ordinary sentences are not logged by
accident. "Track my package" or "record a video" are never habits; they go to the planner. A streak is the number of days in a row, up to today (or yesterday
if today has none yet), with at least one log.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

MAX_HABITS = 30
MAX_LOGS = 2000  # per habit
KEEP_DAYS = 366
COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10}

LOG_REQUEST = re.compile(
    r"^(?:please )?log (?:(?P<count>\d{1,2}|a|an|one|two|three|four|five|six|eight|ten) )?"
    r"(?:my )?(?:glass(?:es)? of |cups? of |sessions? of |bottles? of )?(?P<habit>.+?)(?: for today| today| done)?$"
)
DID_REQUEST = re.compile(
    r"^i (?:just )?(?:did|finished|completed|had|drank|went for|went to) (?:(?P<count>\d{1,2}|a|an|one|two|three|four|five) )?"
    r"(?:my |the )?(?:glass(?:es)? of |cups? of |session of )?(?P<habit>.+?)(?: today)?$"
)
COUNT_REQUEST = re.compile(
    r"^how (?:many|much) (?:times did i |times have i )?(?P<habit>.+?) "
    r"(?:did i (?:do|have|log|drink) |have i (?:done|had|logged|drunk) )?"
    r"(?P<when>today|yesterday|this week|last week|this month)$"
)
STREAK_REQUEST = re.compile(
    r"^what(?: s| is) my (?P<habit>.+?) streak$|^what(?: s| is) my streak (?:for|on) (?P<habit2>.+)$"
    r"|^how long is my (?P<habit3>.+?) streak$"
)
LIST_REQUEST = re.compile(r"^(?:show|list|what are) my habits$|^what habits (?:am i tracking|do i have)$")
DELETE_REQUEST = re.compile(r"^(?:please )?(?:stop tracking|delete the habit|remove the habit) (?P<habit>.+)$")
_NOT_HABITS = re.compile(
    r"^(?:in|out|off|on|into|in to|me in|me out|this|that|it|these|those)\b"
    r"|\b(?:password|account|error|file|bug|issue|output|message|conversation)s?$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _stem(word: str) -> str:
    word = word.strip()
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _key(name: str) -> str:
    name = re.sub(r"^(?:glass(?:es)?|cups?|sessions?|bottles?) of ", "", _normalize(name))
    return " ".join(_stem(part) for part in name.split())


def _count(raw: str | None) -> int:
    if not raw:
        return 1
    return int(raw) if raw.isdigit() else COUNT_WORDS.get(raw, 1)


class HabitStore:
    """Habit logs in a small JSON file."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self._path = Path(path)
        self._clock = clock

    def _load(self) -> dict[str, list[float]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(name): [float(t) for t in times if isinstance(t, (int, float))]
            for name, times in data.items()
            if isinstance(times, list)
        }

    def _save(self, data: dict[str, list[float]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def find(self, name: str) -> str | None:
        """The stored habit a spoken name refers to ("workouts" -> "workout", "meditate" -> "meditation")."""
        wanted = _key(name)
        if not wanted:
            return None
        names = list(self._load())
        for habit in names:
            if _key(habit) == wanted:
                return habit
        for habit in names:
            key = _key(habit)
            if len(wanted) >= 4 and len(key) >= 4 and (key.startswith(wanted[:5]) or wanted.startswith(key[:5])):
                return habit
        return None

    def log(self, name: str, count: int = 1, *, create: bool = True) -> tuple[str, int] | None:
        """Log ``count`` of a habit. Returns (habit, today's total), or None if it doesn't exist and ``create`` is off."""
        data = self._load()
        habit = self.find(name)
        if habit is None:
            if not create:
                return None
            habit = " ".join(_normalize(name).split())
            if not 2 <= len(habit) <= 40:
                raise ValueError("A habit name must be 2 to 40 characters")
            if len(data) >= MAX_HABITS:
                raise ValueError(f"You can track up to {MAX_HABITS} habits")
        if not 1 <= count <= 50:
            raise ValueError("Log 1 to 50 at a time")
        now = self._clock()
        cutoff = now - KEEP_DAYS * 86400
        times = [t for t in data.get(habit, []) if t >= cutoff] + [now] * count
        data[habit] = times[-MAX_LOGS:]
        self._save(data)
        return habit, self.count(habit, *self._day(now))

    def delete(self, name: str) -> bool:
        data = self._load()
        habit = self.find(name)
        if habit is None:
            return False
        del data[habit]
        self._save(data)
        return True

    @staticmethod
    def _day(moment: float) -> tuple[float, float]:
        start = datetime.fromtimestamp(moment).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()

    def count(self, habit: str, start: float, end: float) -> int:
        return sum(1 for t in self._load().get(habit, []) if start <= t < end)

    def streak(self, habit: str) -> int:
        days = {datetime.fromtimestamp(t).date() for t in self._load().get(habit, [])}
        today = datetime.fromtimestamp(self._clock()).date()
        day = today if today in days else today - timedelta(days=1)
        streak = 0
        while day in days:
            streak += 1
            day -= timedelta(days=1)
        return streak

    def list(self) -> list[dict[str, Any]]:
        now = self._clock()
        today_start, today_end = self._day(now)
        week_start = today_start - datetime.fromtimestamp(now).weekday() * 86400
        return [
            {
                "name": habit,
                "today": self.count(habit, today_start, today_end),
                "week": self.count(habit, week_start, today_end),
                "streak": self.streak(habit),
                "last": max(times) if times else None,
            }
            for habit, times in self._load().items()
        ]

    def _window(self, when: str) -> tuple[float, float, str]:
        now = datetime.fromtimestamp(self._clock())
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = midnight - timedelta(days=midnight.weekday())
        start, end = {
            "today": (midnight, midnight + timedelta(days=1)),
            "yesterday": (midnight - timedelta(days=1), midnight),
            "this week": (week_start, midnight + timedelta(days=1)),
            "last week": (week_start - timedelta(days=7), week_start),
            "this month": (midnight.replace(day=1), midnight + timedelta(days=1)),
        }[when]
        return start.timestamp(), end.timestamp(), when

    def answer(self, text: str) -> str | None:
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return 'You aren\'t tracking any habits. Say "log a workout" to start.'
            listed = "; ".join(f"{item['name']}: {item['today']} today, {item['streak']}-day streak" for item in items)
            return f"Your habits: {listed}."
        delete = DELETE_REQUEST.match(normalized)
        if delete:
            name = delete.group("habit")
            return f"I stopped tracking {name}." if self.delete(name) else f"You aren't tracking {name}."
        streak = STREAK_REQUEST.match(normalized)
        if streak:
            name = streak.group("habit") or streak.group("habit2") or streak.group("habit3")
            habit = self.find(name)
            if habit is None:
                return None
            days = self.streak(habit)
            if not days:
                return f"No {habit} streak right now. Log it today to start one."
            return f"Your {habit} streak is {days} day{'' if days == 1 else 's'}."
        counted = COUNT_REQUEST.match(normalized)
        if counted:
            habit = self.find(counted.group("habit"))
            if habit is None:
                return None
            start, end, label = self._window(counted.group("when"))
            total = self.count(habit, start, end)
            if not total:
                return f"No {habit} logged {label} yet."
            return f"You logged {habit} {total} time{'' if total == 1 else 's'} {label}."
        did = DID_REQUEST.match(normalized)
        if did:
            try:
                logged = self.log(did.group("habit"), _count(did.group("count")), create=False)
            except ValueError as exc:
                return str(exc)
            if logged is None:
                return None  # "I did the dishes" is not a habit unless they track it
            habit, today = logged
            return f"Logged {habit}. That's {today} today, and a {self.streak(habit)}-day streak."
        log = LOG_REQUEST.match(normalized)
        if log and not _NOT_HABITS.search(log.group("habit")):
            try:
                habit, today = self.log(log.group("habit"), _count(log.group("count")))  # type: ignore[misc]
            except ValueError as exc:
                return str(exc)
            return f"Logged {habit}. That's {today} today, and a {self.streak(habit)}-day streak."
        return None

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)


def today() -> date:
    return date.today()

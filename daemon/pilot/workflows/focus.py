"""Focus sessions: "focus for 45 minutes" holds interruptions until the session ends.

While a session runs, Heliox holds reminders and the daily brief and keeps proactive and
background speech quiet. Approvals, task results, errors and anything the person asks
for still come through: focus never hides a decision Heliox needs from the person.
When the session ends (on time or when the person says "stop focusing"), the held items
are delivered together with one short summary, including where the time went if the
activity timeline is on.

The session end time is kept in ``DATA_DIR/focus.json`` so a restart does not end it.

A pomodoro is a 25-minute session followed by a break reminder. Quiet hours ("focus every
weekday from 9 to 12") start a session automatically inside the window; stopping one
early keeps it off until the window ends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.focus")

MAX_FOCUS_MINUTES = 8 * 60
START_REQUEST = re.compile(
    r"^(?:please |hey heliox |heliox )?(?:start (?:a )?focus(?: session)?|focus|do not disturb|dont disturb me|"
    r"don t disturb me)(?: me)? for (?P<amount>\d+|an|a|one|two|half an) ?(?P<unit>minutes?|mins?|hours?|hour)$"
)
STOP_REQUEST = re.compile(
    r"^(?:please )?(?:stop|end|finish|cancel) (?:the |my )?(?:focus(?:ing| session)?|do not disturb)$"
)
WORDS = {"a": 1, "an": 1, "one": 1, "two": 2}
POMODORO_REQUEST = re.compile(
    r"^(?:please )?(?:start |begin )?(?:a |another )?pomodoro(?: for (?P<minutes>\d+) minutes)?$"
)
POMODORO_MINUTES = 25
POMODORO_BREAK_MINUTES = 5
DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SCHEDULE_REQUEST = re.compile(
    r"^(?:please )?(?:focus|keep quiet|do not disturb me|don t disturb me) every "
    r"(?P<days>day|weekday|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r" from (?P<h1>\d{1,2})(?: (?P<m1>\d{2}))?(?: (?P<p1>am|pm))? to (?P<h2>\d{1,2})(?: (?P<m2>\d{2}))?(?: (?P<p2>am|pm))?$"
)
SCHEDULE_LIST_REQUEST = re.compile(
    r"^(?:what is|what s|show) my (?:focus|quiet hours?) schedule$|^(?:list|show) my quiet hours$"
)
SCHEDULE_CLEAR_REQUEST = re.compile(r"^(?:clear|delete|remove|cancel) my (?:focus schedule|quiet hours)$")


def parse_focus_minutes(text: str) -> int | None:
    """Minutes from "focus for 45 minutes", "do not disturb for an hour", "focus for half an hour"."""
    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())
    asked = START_REQUEST.match(normalized)
    if not asked:
        return None
    amount_text, unit = asked.group("amount"), asked.group("unit")
    if amount_text == "half an":
        minutes = 30
    else:
        amount = int(amount_text) if amount_text.isdigit() else WORDS[amount_text]
        minutes = amount * (60 if unit.startswith("hour") else 1)
    return minutes if 1 <= minutes <= MAX_FOCUS_MINUTES else 0


def _duration(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} hour{'' if hours == 1 else 's'}"


class FocusSession:
    """The current focus session, the items it is holding, and its timer."""

    def __init__(
        self,
        state_file: Path,
        *,
        on_end: Callable[[dict[str, Any]], Awaitable[None]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._state_file = Path(state_file)
        self._on_end = on_end
        self._clock = clock
        self._started_at = 0.0
        self._until = 0.0
        self._held: list[dict[str, Any]] = []
        self._break_minutes = 0
        self._timer: asyncio.Task[None] | None = None
        self._restore()

    # ------------------------------------------------------------------ state

    def _restore(self) -> None:
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and float(data.get("until", 0)) > self._clock():
            self._started_at = float(data.get("started_at", self._clock()))
            self._until = float(data["until"])
            self._break_minutes = int(data.get("break_minutes", 0))

    def _persist(self) -> None:
        try:
            if self._until:
                self._state_file.parent.mkdir(parents=True, exist_ok=True)
                self._state_file.write_text(
                    json.dumps(
                        {"started_at": self._started_at, "until": self._until, "break_minutes": self._break_minutes}
                    ),
                    encoding="utf-8",
                )
            elif self._state_file.exists():
                os.remove(self._state_file)
        except OSError:
            logger.warning("Focus state could not be saved", exc_info=True)

    @property
    def active(self) -> bool:
        return self._until > self._clock()

    def status(self) -> dict[str, Any]:
        active = self.active
        return {
            "active": active,
            "started_at": self._started_at if active else None,
            "until": self._until if active else None,
            "remaining_seconds": max(0, round(self._until - self._clock())) if active else 0,
            "held": len(self._held),
        }

    # ------------------------------------------------------------------ holding

    def hold(self, kind: str, payload: dict[str, Any]) -> bool:
        """Keep ``payload`` until the session ends. Returns False (deliver now) when no session runs."""
        if not self.active:
            return False
        self._held.append({"kind": kind, "payload": payload, "at": self._clock()})
        return True

    # ------------------------------------------------------------------ start and stop

    def start(self, minutes: int, *, break_minutes: int = 0) -> dict[str, Any]:
        """Start (or extend) a session; ``break_minutes`` makes it a pomodoro with a break reminder after."""
        if not 1 <= int(minutes) <= MAX_FOCUS_MINUTES:
            raise ValueError(f"A focus session can last 1 to {MAX_FOCUS_MINUTES} minutes")
        now = self._clock()
        if not self.active:
            self._started_at = now
            self._held = []
        self._until = now + int(minutes) * 60
        self._break_minutes = max(0, int(break_minutes))
        self._persist()
        self._schedule()
        return self.status()

    def _schedule(self) -> None:
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._timer = loop.create_task(self._wait_and_end())

    def resume_timer(self) -> None:
        """After a restart, keep counting down a session that is still running."""
        if self.active:
            self._schedule()

    async def _wait_and_end(self) -> None:
        try:
            await asyncio.sleep(max(0.0, self._until - self._clock()))
            await self.stop(reason="time")
        except asyncio.CancelledError:
            pass

    async def stop(self, *, reason: str = "asked") -> dict[str, Any] | None:
        """End the session and hand its summary to ``on_end``. None when no session ran."""
        if not self._until:
            return None
        ended_at = min(self._clock(), self._until)
        summary = {
            "started_at": self._started_at,
            "ended_at": ended_at,
            "duration": _duration(ended_at - self._started_at),
            "reason": reason,
            "held": list(self._held),
            "break_minutes": self._break_minutes,
        }
        self._until = 0.0
        self._held = []
        self._break_minutes = 0
        self._persist()
        timer, self._timer = self._timer, None
        if timer is not None and not timer.done() and timer is not asyncio.current_task():
            timer.cancel()
        await self._on_end(summary)
        return summary

    async def close(self) -> None:
        """Stop the timer without ending the session (daemon shutdown); a restart resumes it."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def compose_focus_summary(summary: dict[str, Any], usage: dict[str, Any] | None) -> str:
    """One short paragraph for the end of a focus session."""
    parts = [f"Focus session done: {summary['duration']}."]
    if usage and usage.get("apps"):
        top = usage["apps"][0]
        parts.append(f"You spent most of it in {top['app']}.")
    held = summary["held"]
    reminders = [item["payload"]["text"] for item in held if item["kind"] == "reminder"]
    briefs = [item for item in held if item["kind"] == "daily_brief"]
    nudges = [item["payload"]["text"] for item in held if item["kind"] == "nudge"]
    if reminders:
        listed = "; ".join(reminders[:3])
        parts.append(f"While you focused: {len(reminders)} reminder{'' if len(reminders) == 1 else 's'} ({listed}).")
    if nudges:
        parts.append(" ".join(nudges))
    if briefs:
        parts.append("Your daily brief is below.")
    if not reminders and not briefs and not nudges:
        parts.append("Nothing needed you while you focused.")
    return " ".join(parts)


def _clock_from(hour: str, minute: str | None, meridiem: str | None) -> int | None:
    """Minutes after midnight, or None for an impossible time."""
    h, m = int(hour), int(minute or 0)
    if meridiem == "pm" and h < 12:
        h += 12
    elif meridiem == "am" and h == 12:
        h = 0
    return h * 60 + m if 0 <= h <= 23 and 0 <= m <= 59 else None


def parse_quiet_hours(text: str) -> dict[str, Any] | None:
    """{"days", "start", "end"} from "focus every weekday from 9 to 12" (minutes after midnight); None if unclear."""
    normalized = " ".join(re.sub(r"[^\w\s]", " ", re.sub(r"(\d)([ap]m)\b", r"\1 \2", text.casefold())).split())
    asked = SCHEDULE_REQUEST.match(normalized)
    if not asked:
        return None
    start = _clock_from(asked.group("h1"), asked.group("m1"), asked.group("p1"))
    end = _clock_from(asked.group("h2"), asked.group("m2"), asked.group("p2"))
    if start is not None and end is not None and end <= start and not asked.group("p2") and end < 12 * 60:
        end += 12 * 60  # "from 9 to 5" means until 5 PM
    if start is None or end is None or not start < end <= 24 * 60:
        return None
    days = asked.group("days")
    weekdays = {
        "day": list(range(7)),
        "weekday": [0, 1, 2, 3, 4],
        "weekend": [5, 6],
    }.get(days, [DAY_NAMES.index(days)] if days in DAY_NAMES else [])
    return {"days": weekdays, "start": start, "end": end}


def describe_quiet_hours(window: dict[str, Any]) -> str:
    days = window["days"]
    when = (
        "every day"
        if len(days) == 7
        else "weekdays"
        if days == [0, 1, 2, 3, 4]
        else "weekends"
        if days == [5, 6]
        else ", ".join(DAY_NAMES[day].capitalize() for day in days)
    )
    return f"{when} {window['start'] // 60:02d}:{window['start'] % 60:02d} to {window['end'] // 60:02d}:{window['end'] % 60:02d}"


class QuietHours:
    """Focus windows that start a session automatically, kept in a small JSON file."""

    def __init__(
        self,
        path: Path,
        *,
        focus: FocusSession,
        on_start: Callable[[dict[str, Any]], Awaitable[None]],
        now: Callable[[], Any] | None = None,
    ) -> None:
        from datetime import datetime

        self._path = Path(path)
        self._focus = focus
        self._on_start = on_start
        self._now = now or datetime.now
        self._skip_until = 0.0
        self._task: asyncio.Task[None] | None = None

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [item for item in data if isinstance(item, dict) and "start" in item] if isinstance(data, list) else []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, window: dict[str, Any]) -> dict[str, Any]:
        days = sorted({int(day) for day in window.get("days", [])})
        start, end = int(window.get("start", -1)), int(window.get("end", -1))
        if not days or any(not 0 <= day <= 6 for day in days):
            raise ValueError("days must be numbers from 0 (Monday) to 6 (Sunday)")
        if not 0 <= start < end <= 24 * 60:
            raise ValueError("The window must start before it ends, within one day")
        items = self.list()
        if len(items) >= 10:
            raise ValueError("You can keep up to 10 quiet-hour windows")
        entry = {"id": os.urandom(5).hex(), "days": days, "start": start, "end": end}
        items.append(entry)
        self._save(items)
        return entry

    def delete(self, window_id: str | None = None) -> int:
        items = self.list()
        kept = [item for item in items if window_id and item["id"] != window_id]
        self._save(kept)
        return len(items) - len(kept)

    def note_stopped(self) -> None:
        """The person ended a session early: leave the current window alone until it ends."""
        current = self._current_window()
        if current is not None:
            self._skip_until = current[1]

    def _current_window(self) -> tuple[dict[str, Any], float] | None:
        now = self._now()
        minute = now.hour * 60 + now.minute
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        for item in self.list():
            if now.weekday() in item["days"] and item["start"] <= minute < item["end"]:
                return item, midnight + item["end"] * 60
        return None

    async def tick(self) -> bool:
        """Start a session when a window is open and none runs. Returns whether one started."""
        current = self._current_window()
        if current is None or self._focus.active:
            return False
        window, ends_at = current
        if self._skip_until >= ends_at:
            return False
        minutes = max(1, round((ends_at - self._now().timestamp()) / 60))
        status = self._focus.start(min(minutes, MAX_FOCUS_MINUTES))
        await self._on_start({**status, "quiet_hours": describe_quiet_hours(window)})
        return True

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Quiet hours check failed", exc_info=True)
            await asyncio.sleep(30)

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

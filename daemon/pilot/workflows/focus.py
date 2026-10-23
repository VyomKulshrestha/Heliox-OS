"""Focus sessions: "focus for 45 minutes" holds interruptions until the session ends.

While a session runs, Heliox holds reminders and the daily brief and keeps proactive and
background speech quiet. Approvals, task results, errors and anything the person asks
for still come through: focus never hides a decision Heliox needs from the person.
When the session ends (on time or when the person says "stop focusing"), the held items
are delivered together with one short summary, including where the time went if the
activity timeline is on.

The session end time is kept in ``DATA_DIR/focus.json`` so a restart does not end it.
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

    def _persist(self) -> None:
        try:
            if self._until:
                self._state_file.parent.mkdir(parents=True, exist_ok=True)
                self._state_file.write_text(
                    json.dumps({"started_at": self._started_at, "until": self._until}), encoding="utf-8"
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

    def start(self, minutes: int) -> dict[str, Any]:
        if not 1 <= int(minutes) <= MAX_FOCUS_MINUTES:
            raise ValueError(f"A focus session can last 1 to {MAX_FOCUS_MINUTES} minutes")
        now = self._clock()
        if not self.active:
            self._started_at = now
            self._held = []
        self._until = now + int(minutes) * 60
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
        }
        self._until = 0.0
        self._held = []
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
    if reminders:
        listed = "; ".join(reminders[:3])
        parts.append(f"While you focused: {len(reminders)} reminder{'' if len(reminders) == 1 else 's'} ({listed}).")
    if briefs:
        parts.append("Your daily brief is below.")
    if not reminders and not briefs:
        parts.append("Nothing needed you while you focused.")
    return " ".join(parts)

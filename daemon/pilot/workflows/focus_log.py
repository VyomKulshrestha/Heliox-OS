"""Focus history: "how long did I focus this week?".

Each finished focus session longer than a minute is logged with when it started and
ended and why it ended, in ``DATA_DIR/focus_history.json``, for a year. Nothing about
what was on screen is kept; that stays with the activity timeline, if it is on.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

KEEP_DAYS = 365
MAX_SESSIONS = 5000
MIN_SECONDS = 60

HISTORY_REQUEST = re.compile(
    r"^how (?:long|much|many hours) (?:did i|have i) (?:focus|focused|been focusing)"
    r"(?: (?P<when>today|yesterday|this week|last week|this month))?$"
    r"|^how many focus sessions (?:did i (?:do|have) |have i (?:done|had) )?(?P<when2>today|yesterday|this week|last week|this month)?$"
    r"|^(?:what s|show) my focus (?:time|history)(?: (?P<when3>today|yesterday|this week|last week|this month))?$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def duration(seconds: float) -> str:
    minutes = max(0, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def window(when: str, now: datetime) -> tuple[float, float, str]:
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = midnight - timedelta(days=midnight.weekday())
    start, end, label = {
        "today": (midnight, now, "today"),
        "yesterday": (midnight - timedelta(days=1), midnight, "yesterday"),
        "this week": (week_start, now, "this week"),
        "last week": (week_start - timedelta(days=7), week_start, "last week"),
        "this month": (midnight.replace(day=1), now, "this month"),
    }[when]
    return start.timestamp(), end.timestamp(), label


class FocusLog:
    """Finished focus sessions in a small JSON file."""

    def __init__(self, path: Path, *, now: Callable[[], datetime] = datetime.now) -> None:
        self._path = Path(path)
        self._now = now

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and "started_at" in item and "seconds" in item]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, summary: dict[str, Any]) -> dict[str, Any] | None:
        """Log a finished session from FocusSession's summary. Very short ones are skipped."""
        started, ended = float(summary["started_at"]), float(summary["ended_at"])
        seconds = max(0.0, ended - started)
        if seconds < MIN_SECONDS:
            return None
        cutoff = self._now().timestamp() - KEEP_DAYS * 86400
        items = [item for item in self.list() if item["started_at"] >= cutoff][-(MAX_SESSIONS - 1) :]
        entry = {
            "started_at": started,
            "ended_at": ended,
            "seconds": round(seconds),
            "reason": summary.get("reason", ""),
        }
        items.append(entry)
        self._save(items)
        return entry

    def totals(self, start: float, end: float) -> dict[str, Any]:
        sessions = [item for item in self.list() if start <= item["started_at"] < end]
        longest = max(sessions, key=lambda item: item["seconds"], default=None)
        return {
            "seconds": sum(item["seconds"] for item in sessions),
            "sessions": len(sessions),
            "longest": longest,
        }

    def overview(self) -> dict[str, Any]:
        now = self._now()
        result: dict[str, Any] = {}
        for when in ("today", "this week"):
            start, end, _label = window(when, now)
            totals = self.totals(start, end + 1)
            result[when.replace(" ", "_")] = {"seconds": totals["seconds"], "sessions": totals["sessions"]}
        result["recent"] = list(reversed(self.list()))[:20]
        return result

    def answer(self, text: str) -> str | None:
        asked = HISTORY_REQUEST.match(_normalize(text))
        if not asked:
            return None
        when = asked.group("when") or asked.group("when2") or asked.group("when3") or "today"
        start, end, label = window(when, self._now())
        totals = self.totals(start, end + 1)
        if not totals["sessions"]:
            return f'You had no focus sessions {label}. Say "focus for 45 minutes" to start one.'
        count = totals["sessions"]
        text = f"You focused for {duration(totals['seconds'])} {label}, in {count} session{'' if count == 1 else 's'}."
        longest = totals["longest"]
        if count > 1 and longest is not None:
            day = datetime.fromtimestamp(longest["started_at"]).strftime("%A")
            text += f" The longest was {duration(longest['seconds'])}" + (f" on {day}." if label != "today" else ".")
        return text

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)

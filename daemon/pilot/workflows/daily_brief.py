"""Today view and daily brief: what Heliox did, where your time went, and what can still be undone.

``today_overview`` gathers one day from three local sources: the signed task receipts
(what Heliox ran), the activity timeline when it is on (which apps you used), and the
undo journal (tasks that can still be undone). ``compose_brief`` turns that into a
few plain sentences. ``DailyBriefScheduler`` sends the brief once a day at the time
the person chose, or as soon as the computer is awake after that time, and never for
a day that has already passed. Everything is composed in code from local records; no
model sees them and nothing leaves the computer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.daily_brief")

BRIEF_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
CHECK_EVERY_SECONDS = 30.0


def _duration(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def day_window(day: str, now: datetime) -> tuple[float, float, str, str]:
    """(start, end, label, ISO date) for "today", "yesterday" or "week" (the last 7 days); both end now."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if day == "week":
        start = midnight - timedelta(days=6)
        return start.timestamp(), now.timestamp(), "this week", start.date().isoformat()
    if day == "yesterday":
        start = midnight - timedelta(days=1)
        return start.timestamp(), midnight.timestamp(), "yesterday", start.date().isoformat()
    return midnight.timestamp(), now.timestamp(), "today", midnight.date().isoformat()


async def today_overview(
    *,
    receipts: Any,
    timeline: Any,
    undo_journal: Any,
    todos: Any = None,
    day: str = "today",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Everything the Today view and the brief show for one day."""
    now = now or datetime.now()
    start, end, label, iso_day = day_window(day, now)
    # Windows can report datetime.now() a clock tick behind time.time(); keep just-finished work in.
    until = end + 1.0
    tasks = await asyncio.to_thread(receipts.recap, start, until) if receipts is not None else []
    timeline_enabled = bool(timeline is not None and timeline.enabled)
    usage = await asyncio.to_thread(timeline.summarize, start, end) if timeline_enabled else None
    undoable = []
    if undo_journal is not None:
        for row in await undo_journal.list_tasks(50):
            created = float(row.get("created_at") or 0)
            if start <= created <= until and row["undoable"] > 0 and not row["undone"]:
                undoable.append(row)
    overview = {
        "day": iso_day,
        "label": label,
        "start": start,
        "end": end,
        "tasks": tasks,
        "timeline_enabled": timeline_enabled,
        "usage": usage,
        "undoable": undoable,
        "todos": await asyncio.to_thread(todos.open_items) if todos is not None else [],
    }
    overview["brief"] = compose_brief(overview)
    return overview


def compose_brief(overview: dict[str, Any]) -> str:
    """A few plain sentences about one day, from the overview alone."""
    label = overview["label"]
    opening = label[0].upper() + label[1:]
    tasks = overview["tasks"]
    sentences = []
    if tasks:
        failed = sum(1 for task in tasks if task["status"] not in ("success", "completed"))
        count = f"{len(tasks)} task{'' if len(tasks) == 1 else 's'}"
        latest = tasks[-1]["request"] or tasks[-1]["summary"]
        sentence = f"{opening} Heliox ran {count}"
        sentence += f", {failed} of them not cleanly" if failed else ""
        sentences.append(f'{sentence}; the latest was "{latest}".')
    else:
        sentences.append(f"{opening} Heliox ran no tasks.")
    usage = overview.get("usage")
    if usage and usage["apps"]:
        apps = ", ".join(f"{entry['app']} ({_duration(entry['seconds'])})" for entry in usage["apps"][:3])
        sentences.append(f"You spent most of your time in {apps}.")
    undoable = overview["undoable"]
    if undoable:
        sentences.append(f"{len(undoable)} task{' can' if len(undoable) == 1 else 's can'} still be undone.")
    open_todos = overview.get("todos") or []
    if open_todos:
        count = len(open_todos)
        sentences.append(
            f'You have {count} thing{"" if count == 1 else "s"} to do, starting with "{open_todos[0]["text"]}".'
        )
    return " ".join(sentences)


def compose_report(overview: dict[str, Any]) -> str:
    """A Markdown report of one day or week: the brief, every task, app time and what can be undone."""
    label = overview["label"]
    lines = [f"# Heliox report: {label} (from {overview['day']})", "", overview["brief"], ""]
    lines += ["## Tasks", ""]
    if overview["tasks"]:
        lines += ["| Finished | Request | Status | Actions | Receipt |", "|---|---|---|---|---|"]
        for task in overview["tasks"]:
            finished = datetime.fromisoformat(task["finished_at"]).astimezone().strftime("%a %d %b %H:%M")
            request = (task["request"] or task["summary"] or "").replace("|", "\\|")
            lines.append(
                f"| {finished} | {request} | {task['status']} | {task['action_count']} | `{task['receipt_id'][:12]}` |"
            )
    else:
        lines.append("No tasks ran actions.")
    lines += ["", "## Where the time went", ""]
    usage = overview.get("usage")
    if not overview.get("timeline_enabled"):
        lines.append("The activity timeline is off.")
    elif usage and usage["apps"]:
        lines += [f"- {entry['app']}: {_duration(entry['seconds'])}" for entry in usage["apps"]]
    else:
        lines.append("Nothing recorded.")
    lines += ["", "## Can still be undone", ""]
    if overview["undoable"]:
        for row in overview["undoable"]:
            targets = ", ".join(row.get("targets") or []) or row["task_key"]
            lines.append(f"- {targets} ({row['undoable']} change{'' if row['undoable'] == 1 else 's'})")
    else:
        lines.append("Nothing.")
    if overview.get("todos"):
        lines += ["", "## Still to do", ""]
        lines += [f"- [ ] {item['text']}" for item in overview["todos"]]
    lines += ["", "Each receipt can be checked with `python -m pilot.workflows.task_receipts verify`.", ""]
    return "\n".join(lines)


class DailyBriefScheduler:
    """Sends the daily brief once per day at the chosen time."""

    def __init__(
        self,
        *,
        config: Callable[[], Any],
        build: Callable[[], Awaitable[dict[str, Any]]],
        deliver: Callable[[dict[str, Any]], Awaitable[None]],
        state_file: Path,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._config = config
        self._build = build
        self._deliver = deliver
        self._state_file = Path(state_file)
        self._now = now
        self._task: asyncio.Task[None] | None = None

    def _last_sent(self) -> str:
        try:
            return str(json.loads(self._state_file.read_text(encoding="utf-8")).get("last_sent", ""))
        except (OSError, ValueError, AttributeError):
            return ""

    def _mark_sent(self, day: date) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_sent": day.isoformat()}), encoding="utf-8")
        os.replace(tmp, self._state_file)

    def due(self) -> bool:
        """True when the brief is on, today's time has passed and today's brief has not gone out."""
        config = self._config()
        if not getattr(config, "enabled", False):
            return False
        match = BRIEF_TIME.match(str(getattr(config, "time", "")))
        if not match:
            return False
        now = self._now()
        if (now.hour, now.minute) < (int(match.group(1)), int(match.group(2))):
            return False
        return self._last_sent() != now.date().isoformat()

    async def tick(self) -> bool:
        """Send today's brief if it is due. Returns whether one was sent."""
        if not self.due():
            return False
        self._mark_sent(self._now().date())  # before sending, so a failure never repeats it every 30 s
        overview = await self._build()
        await self._deliver(overview)
        return True

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Daily brief failed", exc_info=True)
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

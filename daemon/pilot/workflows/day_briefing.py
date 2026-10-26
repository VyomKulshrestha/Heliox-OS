"""Good morning and good night: a short spoken start and end to the day.

"Good morning" gives the date, today's reminders, the to-do list, habit streaks worth
keeping and anything wrong with the computer (a nearly full drive, a low battery).
"Good night" sums up the day from the daily brief, adds focus time and finished to-dos,
and names tomorrow's first reminder. Both are put together in code from what Heliox
already keeps on this computer; nothing new is recorded.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

MORNING_REQUEST = re.compile(
    r"^(?:hey heliox |heliox )?(?:good morning|morning)(?: heliox)?$|^(?:start|begin) my day$"
    r"|^what(?: s| is) (?:on )?(?:for )?today$|^what do i have (?:on )?today$|^what(?: s| is| does) my day look(?:ing)?(?: like)?$"
)
NIGHT_REQUEST = re.compile(
    r"^(?:hey heliox |heliox )?(?:good night|goodnight|night night)(?: heliox)?$"
    r"|^(?:wrap up|end|finish|close out) (?:my |the )?day$|^i m done for (?:the day|today)$"
)


def _clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def _duration(seconds: float) -> str:
    minutes = max(0, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def _health_warnings(health: dict[str, Any] | None) -> list[str]:
    if not health:
        return []
    warnings = []
    for disk in health.get("disks", []):
        if disk.get("total") and (disk["free"] / disk["total"] < 0.1):
            drive = disk["drive"].rstrip("\\/") or "/"
            warnings.append(f"{drive} is almost full ({disk['free'] / 1024**3:.0f} GB free)")
            break
    battery = health.get("battery")
    if battery and not battery.get("plugged") and battery.get("percent", 100) <= 20:
        warnings.append(f"the battery is at {battery['percent']:.0f}%")
    return warnings


def compose_morning(
    now: datetime,
    *,
    reminders: list[dict[str, Any]],
    todos: list[dict[str, Any]],
    habits: list[dict[str, Any]],
    health: dict[str, Any] | None = None,
) -> str:
    greeting = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 18 else "Good evening"
    parts = [f"{greeting}. It's {now.strftime('%A')}, {now.day} {now.strftime('%B')}."]
    end_of_day = now.replace(hour=23, minute=59, second=59)
    today = [item for item in reminders if now.timestamp() <= item["due"] <= end_of_day.timestamp()]
    if today:
        listed = "; ".join(f"{_clock(datetime.fromtimestamp(item['due']))} {item['text']}" for item in today[:4])
        more = f", and {len(today) - 4} more" if len(today) > 4 else ""
        parts.append(f"Reminders today: {listed}{more}.")
    else:
        parts.append("No reminders today.")
    if todos:
        count = len(todos)
        parts.append(f'You have {count} thing{"" if count == 1 else "s"} to do, starting with "{todos[0]["text"]}".')
    keep = [habit for habit in habits if habit.get("streak", 0) >= 2 and not habit.get("today")]
    if keep:
        names = " and ".join(f"{habit['name']} ({habit['streak']} days)" for habit in keep[:2])
        parts.append(f"Keep your streak going: {names}.")
    warnings = _health_warnings(health)
    if warnings:
        parts.append("Heads-up: " + "; ".join(warnings) + ".")
    return " ".join(parts)


def compose_night(
    now: datetime,
    *,
    brief: str,
    focus_seconds: float,
    finished_todos: int,
    open_todos: int,
    reminders: list[dict[str, Any]],
) -> str:
    parts = ["Good night.", brief]
    if focus_seconds >= 60:
        parts.append(f"You focused for {_duration(focus_seconds)}.")
    if finished_todos:  # the brief already counts what is still open
        done = f"You finished {finished_todos} thing{'' if finished_todos == 1 else 's'} on your list"
        parts.append(done + ("." if open_todos else ", and your list is clear."))
    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    upcoming = [
        item for item in reminders if tomorrow.timestamp() <= item["due"] < (tomorrow + timedelta(days=1)).timestamp()
    ]
    if upcoming:
        first = min(upcoming, key=lambda item: item["due"])
        parts.append(
            f"Tomorrow your first reminder is at {_clock(datetime.fromtimestamp(first['due']))}: {first['text']}."
        )
    return " ".join(part for part in parts if part)

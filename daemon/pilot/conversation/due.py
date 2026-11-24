"""What is due soon on the person's to-do list and reminders, as one context line for the talk model.

Without it a disagreement had no grounds but words said earlier in the same conversation: a to-do "finish the
quarterly report by 9 am tomorrow" never reached the model, so "I'll watch a series tonight and do the report in the
morning" was met with "Okay" or with a wrong reason. The line is small and chosen in code, with no model call:

- a reminder counts when it is due in the next 24 hours; a to-do when its own words name a time in the next 36 hours
  ("by 9 am tomorrow", "tonight", "on Friday"), or, with no time named, when it shares a word with what was said;
- an item is given only when it bears on what was said: it shares a word of four letters or more with it, or the
  words speak of a time ("tonight", "tomorrow morning", "at four") and the item is due within that day or so;
- at most two to-dos and two reminders, newest due first, in at most 300 characters;
- never anything ``never_sent`` holds back (health, money, a secret, someone else's private details).

What goes where follows the rule that already decides what of the person reaches a model
(``pilot.conversation.adapters.personal_in_prompt``, the ``conversation.share_profile`` setting) and the rule that a
to-do or reminder read out in code stays on this computer (``CODE_ANSWERS_ANY_MODEL`` in the server):

- a model on this computer gets each chosen item in its own words, with when it is due;
- a cloud model gets only the words of an item that the person has just said themselves ("a to-do about the report,
  due tomorrow at 09:00"), and for an item that shares no word, only that something is due then; the item's other
  words stay here;
- with ``share_profile`` off, a cloud model gets nothing from here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

REMINDER_HOURS = 24
TODO_HOURS = 36
MAX_EACH = 2
LINE_CHARS = 300
_TIME_WORDS = re.compile(
    r"\b(?:tonight|today|tomorrow|this (?:morning|afternoon|evening)|morning|afternoon|evening|night|later|soon"
    r"|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday|noon|midnight|o'?clock|deadline|due"
    r"|at (?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b|\d{1,2}(?::\d{2})? ?(?:am|pm|a m|p m))\b",
    re.IGNORECASE,
)
_SMALL = frozenset(
    [
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "what",
        "when",
        "your",
        "about",
        "there",
        "their",
        "they",
        "them",
        "then",
        "than",
        "been",
        "were",
        "would",
        "could",
        "should",
        "going",
        "want",
        "need",
        "just",
        "also",
        "into",
        "over",
        "some",
        "more",
        "most",
        "very",
        "much",
        "make",
        "made",
        "like",
        "time",
        "today",
        "tonight",
        "tomorrow",
        "morning",
        "evening",
        "night",
        "later",
        "thing",
        "things",
        "done",
        "doing",
        "does",
        "dont",
        "didnt",
        "isnt",
    ]
)


def _words(text: str) -> set[str]:
    found = re.findall(r"[a-z0-9']+", text.casefold())
    return {word.rstrip("s") for word in found if len(word) >= 4 and word.replace("'", "") not in _SMALL}


def speaks_of_time(text: str) -> bool:
    return bool(_TIME_WORDS.search(text or ""))


def todo_due(text: str, now: datetime) -> datetime | None:
    """When a to-do's own words say it is due ("by 9 am tomorrow", "tonight"), or None."""
    from pilot.live.commitments import _find_deadline, deadline_of

    cut = _find_deadline(" " + text)
    if cut is None:
        return None
    try:
        deadline, has_time = deadline_of(cut[1], now.date())
    except Exception:
        return None
    if deadline is None:
        return None
    return deadline if has_time else deadline.replace(hour=23, minute=59)


def _when(moment: datetime, now: datetime) -> str:
    days = (moment.date() - now.date()).days
    day = "today" if days == 0 else "tomorrow" if days == 1 else moment.strftime("%A")
    if moment.hour == 23 and moment.minute == 59:
        return day
    return f"{day} at {moment:%H:%M}"


def _never_sent(text: str) -> bool:
    from pilot.conversation.context import never_sent

    return never_sent(text)


def due_line(
    said: str,
    *,
    todos: Iterable[dict[str, Any]],
    reminders: Iterable[dict[str, Any]],
    now: datetime,
    local: bool,
) -> str:
    """The context line for ``said``, or "" when nothing bears on it. Pure apart from reading the clock it is given."""
    asked = _words(said)
    timely = speaks_of_time(said)
    picked: list[tuple[datetime | None, str, dict[str, Any]]] = []
    for item in todos:
        text = str(item.get("text") or "").strip()
        if not text or item.get("done") or _never_sent(text):
            continue
        due = todo_due(text, now)
        shared = bool(asked & _words(text))
        if due is not None and not (now - timedelta(hours=2) <= due <= now + timedelta(hours=TODO_HOURS)):
            continue
        if shared or (timely and due is not None):
            picked.append((due, "to-do", item))
    reminders_picked: list[tuple[datetime | None, str, dict[str, Any]]] = []
    for item in reminders:
        text = str(item.get("text") or "").strip()
        try:
            due = datetime.fromtimestamp(float(item.get("due")))
        except (TypeError, ValueError, OSError):
            continue
        if not text or _never_sent(text) or not (now <= due <= now + timedelta(hours=REMINDER_HOURS)):
            continue
        if asked & _words(text) or timely:
            reminders_picked.append((due, "reminder", item))

    def soonest(entry: tuple[datetime | None, str, dict[str, Any]]) -> datetime:
        return entry[0] or now + timedelta(days=365)

    chosen = sorted(picked, key=soonest)[:MAX_EACH] + sorted(reminders_picked, key=soonest)[:MAX_EACH]
    if not chosen:
        return ""
    parts = []
    for due, kind, item in chosen:
        text = " ".join(str(item.get("text") or "").split())
        when = f", due {_when(due, now)}" if due is not None else ""
        if local:
            parts.append(f'{kind} "{text[:120]}"{when}')
            continue
        shared = [word for word in re.findall(r"[A-Za-z0-9']+", text) if word.casefold().rstrip("s") in asked]
        if shared:
            about = " ".join(dict.fromkeys(word.casefold() for word in shared))[:60]
            parts.append(f'a {kind} about "{about}"{when}')
        elif due is not None:
            parts.append(f"a {kind}{when}")
    if not parts:
        return ""
    head = "On their to-do list and reminders" + ("" if local else " (the rest of their words stay on this computer)")
    line = f"{head}: " + "; ".join(parts) + "."
    return line[:LINE_CHARS]


__all__ = ["due_line", "speaks_of_time", "todo_due"]

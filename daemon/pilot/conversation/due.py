"""What is due soon on the person's to-do list and reminders, as one context line for the talk model.

Without it a disagreement had no grounds but words said earlier in the same conversation: a to-do "finish the
quarterly report by 9 am tomorrow" never reached the model, so "I'll watch a series tonight and do the report in the
morning" was met with "Okay" or with a wrong reason. The line is small and chosen in code, with no model call:

- an item that shares a word of four letters or more with what was just said is given, whatever its date;
- otherwise only when the words speak of a time ("tonight", "tomorrow morning", "at four") and the item is due soon:
  a reminder in the next 24 hours, a to-do whose own words name a time in the next 36 hours. A to-do's time is read
  as people say it ("by 9 tomorrow morning", "four this afternoon", "half past eight tonight", "noon": spoken_clock);
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
from datetime import datetime, time, timedelta
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


_HOURS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}  # fmt: skip
_HOUR = r"(?P<h>\d{1,2}|" + "|".join(_HOURS) + ")"
_CLOCK_SAID = re.compile(
    r"\b(?:(?P<frac>half past|quarter past|quarter to)\s+)?" + _HOUR + r"(?:[:.](?P<m>\d{2}))?(?:\s*o'?clock)?"
    r"(?:\s*(?P<ap>a\.?\s?m\b\.?|p\.?\s?m\b\.?))?",
    re.IGNORECASE,
)
_BEFORE_A_TIME = re.compile(r"\b(?:at|by|before|until|till|from)\s+$", re.IGNORECASE)
_PART_AFTER = re.compile(
    r"^\s+(?:(?:tomorrow|today|on \w+day|this|in the)\s+)?(?P<part>morning|afternoon|evening|night|tonight)\b",
    re.IGNORECASE,
)
_NOT_A_TIME = re.compile(
    r"^\s+(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|of|people|times|percent)\b", re.IGNORECASE
)
_PART = re.compile(r"\b(?P<part>morning|afternoon|evening|tonight|night)\b", re.IGNORECASE)
_NOON = re.compile(r"\b(?:noon|midday)\b", re.IGNORECASE)
_MIDNIGHT = re.compile(r"\bmidnight\b", re.IGNORECASE)
_TODAY_PART = re.compile(r"\b(?:tonight|today|this (?:morning|afternoon|evening))\b", re.IGNORECASE)
# When a part of the day is named with no hour ("tomorrow morning"): due by its end.
_PART_ENDS = {"morning": time(12, 0), "afternoon": time(17, 0), "evening": time(21, 0), "tonight": time(23, 59)}


def spoken_clock(text: str) -> time | None:
    """The time of day the words name, as people say it: "9 am", "9 tomorrow morning", "four this afternoon", "half
    past eight tonight", "at 3" (the afternoon, as a plan says it), "noon", "midnight". None when they name none."""
    if _NOON.search(text):
        return time(12, 0)
    if _MIDNIGHT.search(text):
        return time(23, 59)  # "by midnight" is the end of that day
    named = _PART.search(text)
    for match in _CLOCK_SAID.finditer(text):
        word = match["h"].casefold()
        hour = int(word) if word.isdigit() else _HOURS[word]
        minute = int(match["m"] or 0)
        rest = text[match.end() :]
        part_after = _PART_AFTER.match(rest)
        if not (match["ap"] or match["frac"] or part_after or _BEFORE_A_TIME.search(text[: match.start()])):
            continue  # a number that is not a time ("four hours of work")
        if _NOT_A_TIME.match(rest) or not 0 <= hour <= 23 or minute > 59:
            continue
        if match["frac"]:
            frac = match["frac"].casefold()
            minute = {"half past": 30, "quarter past": 15}.get(frac, 45)
            hour = hour - 1 if frac == "quarter to" else hour
        part = (part_after["part"] if part_after else named["part"] if named else "").casefold()
        meridiem = (match["ap"] or "").casefold().replace(".", "").replace(" ", "")
        if hour > 12:
            pass  # a 24-hour time says it all
        elif meridiem == "pm" or (not meridiem and part in ("afternoon", "evening", "tonight")):
            hour = hour % 12 + 12
        elif meridiem == "am" or part == "morning":
            hour = hour % 12
        elif part == "night":
            hour = hour % 12 + (12 if hour >= 6 else 0)
        elif 1 <= hour <= 7:
            hour += 12  # "at 5" in a plan means the afternoon
        return time(hour % 24, minute)
    return None


def todo_due(text: str, now: datetime) -> datetime | None:
    """When a to-do's own words say it is due ("by 9 am tomorrow", "by 9 tomorrow morning", "tonight", "on Friday at
    four"), or None when they name no day and no time."""
    from pilot.live.grounded import stated_day

    try:
        day, _ = stated_day(text, now.date())
    except Exception:
        day = None
    if day is None and _TODAY_PART.search(text):
        day = now.date()
    at = spoken_clock(text)
    part = _PART.search(text)
    if at is None and part is not None and day is not None:
        at = _PART_ENDS.get(part["part"].casefold())
    if day is None and at is None:
        return None
    if day is None:
        day = now.date()
        if at is not None and datetime.combine(day, at) < now - timedelta(hours=2):
            day = day + timedelta(days=1)  # "by four" written after four: the next one
    return datetime.combine(day, at or time(23, 59))


def day_said(text: str, now: datetime) -> bool:
    """Whether a to-do's own words name its day (a date, a weekday, today, tonight, tomorrow, next week)."""
    from pilot.live.grounded import stated_day

    try:
        day, _ = stated_day(text, now.date())
    except Exception:
        day = None
    return day is not None or bool(_TODAY_PART.search(text) or re.search(r"\bnext week\b", text, re.I))


def _when(moment: datetime, now: datetime, *, day_known: bool = True) -> str:
    from pilot.conversation.context import clock_words

    if not day_known:
        return f"by {clock_words(moment)}, day not said"  # never a day the to-do does not carry
    days = (moment.date() - now.date()).days
    day = "today" if days == 0 else "tomorrow" if days == 1 else moment.strftime("%A %d %B")
    if moment.hour == 23 and moment.minute == 59:
        return day
    return f"{day} at {clock_words(moment)}"


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
        if asked & _words(text):
            picked.append((due, "to-do", item))  # it shares a word with what was just said: whatever its date
        elif timely and due is not None and now - timedelta(hours=2) <= due <= now + timedelta(hours=TODO_HOURS):
            picked.append((due, "to-do", item))
    reminders_picked: list[tuple[datetime | None, str, dict[str, Any]]] = []
    for item in reminders:
        text = str(item.get("text") or "").strip()
        try:
            due = datetime.fromtimestamp(float(item.get("due")))
        except (TypeError, ValueError, OSError):
            continue
        if not text or _never_sent(text) or due < now:
            continue
        if asked & _words(text) or (timely and due <= now + timedelta(hours=REMINDER_HOURS)):
            reminders_picked.append((due, "reminder", item))

    def soonest(entry: tuple[datetime | None, str, dict[str, Any]]) -> datetime:
        return entry[0] or now + timedelta(days=365)

    chosen = sorted(picked, key=soonest)[:MAX_EACH] + sorted(reminders_picked, key=soonest)[:MAX_EACH]
    if not chosen:
        return ""
    parts = []
    for due, kind, item in chosen:
        text = " ".join(str(item.get("text") or "").split())
        known = kind != "to-do" or day_said(str(item.get("text") or ""), now)
        when = f", due {_when(due, now, day_known=known)}" if due is not None else ""
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


__all__ = ["due_line", "speaks_of_time", "spoken_clock", "todo_due"]

"""Things said to Heliox in passing: "my flight is on Friday", "I moved the meeting to three".

Said to Heliox, these used to reach only the short-term conversation, which lapses after ten quiet minutes, so a day
later "when is my flight?" found nothing. Now they are kept as a note in the person's profile, in their own words with
the day written out ("My flight is on Friday 13 November", said on Thursday 12 November, so "Friday" still means the
right day next week), and Heliox says so: "Noted: ...". A newer note about the same thing replaces the older one ("I
moved the meeting to three" after "my meeting is at two"). Notes lapse by themselves: the day after the date they
name, or a week after they were said when they name none. "Forget about my flight" removes one at once.

Only sentences said or typed to Heliox reach this (never speech overheard near the microphone), it is parsed in code,
and what is kept goes where every profile line goes: the planner's and the conversation model's context.

Never kept this way, whatever the wording (``never_kept``): anything that looks like a password, key or code (the
profile's own refusal also applies); health (appointments with a doctor or dentist, treatment, conditions,
medication); money amounts (a price, a salary, the rent); and other people's private details beyond a name and a
relation (someone's address, number, email, health or money). Heliox says it did not keep it and why. One switch,
``memory.passing_notes`` (on by default), turns the whole thing off.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

# The one rule, for every store. Its patterns stay importable from here: the conversation layer's never_sent
# reads them by name.
from pilot.personal.never_kept import _HEALTH, _MONEY, _PRIVATE_OF_OTHERS, _SECRET, never_kept  # noqa: F401

KEEP_WITHOUT_DATE = timedelta(days=7)

_THING = r"[a-z][\w'-]*(?: [a-z][\w'-]*){0,4}?"
PASSING = re.compile(
    rf"^(?:my|our) (?P<t1>{_THING}) (?:is|are|was)(?: now)? (?P<p1>on|at|in|until|from|tomorrow|today|tonight|next)\b"
    rf"(?P<r1>.*)$"
    rf"|^i(?: have|'ve)? (?:moved|changed|rescheduled|pushed|put) (?:the|my|our) (?P<t2>{_THING}) (?P<p2>to|on|at) "
    rf"(?P<r2>.+)$"
    rf"|^(?:the|my|our) (?P<t3>{_THING}) (?:has been |got |was |is )?(?:moved|changed|rescheduled|pushed) "
    rf"(?P<p3>to|on|at) (?P<r3>.+)$"
)
# A request riding along ("my meeting is at three, remind me before") belongs to the planner, not to a note.
_REQUEST = re.compile(r"\b(?:remind|tell me|book|set|schedule|can you|could you|would you|please|cancel)\b|\?")
_NOT_A_THING = frozenset({"name", "it", "this", "that"})  # secrets are parsed so never_kept can say no
_DAYS = [d.casefold() for d in calendar.day_name]
_MONTHS = [m.casefold() for m in calendar.month_name[1:]]


@dataclass(frozen=True)
class Note:
    thing: str  # "flight", "meeting": what a newer note replaces
    text: str  # "My flight is on Friday 13 November (said on Thursday 12 November)"
    expires: float  # a timestamp
    until: str = ""  # the day after the date it names ("Saturday 14 November"), or "" when it names none


def _day_words(day: date) -> str:
    return f"{calendar.day_name[day.weekday()]} {day.day} {calendar.month_name[day.month]}"


def _resolve(rest: str, today: date) -> tuple[str, date | None]:
    """Relative days written out, and the latest day the note names."""
    named: list[date] = []

    def relative(match: re.Match[str]) -> str:
        word = match.group(0).casefold()
        if word in ("today", "tonight"):
            day = today
        elif word == "tomorrow":
            day = today + timedelta(days=1)
        else:
            day = today + timedelta(days=(_DAYS.index(word) - today.weekday()) % 7 or 7)
        named.append(day)
        return f"{match.group(0)} ({_day_words(day)})" if word in ("today", "tonight", "tomorrow") else _day_words(day)

    # A weekday already followed by a date ("Friday 13 November") is left as it is.
    pattern = r"\b(?:today|tonight|tomorrow|" + "|".join(_DAYS) + r")\b(?!,? ?(?:the )?\d)"
    out = re.sub(pattern, relative, rest, flags=re.IGNORECASE)
    for d, m in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)? (" + "|".join(_MONTHS) + r")\b", out, flags=re.IGNORECASE):
        try:
            candidate = date(today.year, _MONTHS.index(m.casefold()) + 1, int(d))
        except ValueError:
            continue
        named.append(candidate if candidate >= today - timedelta(days=7) else candidate.replace(year=today.year + 1))
    return out, max(named) if named else None


def passing_note(said: str, now: datetime) -> Note | None:
    """A note from a sentence said in passing, or None when it is not one."""
    text = " ".join(said.split()).strip().rstrip(".!")
    low = text.casefold()
    if _REQUEST.search(low):
        return None
    match = PASSING.match(low)
    if match is None:
        return None
    thing = (match["t1"] or match["t2"] or match["t3"] or "").strip()
    prep = (match["p1"] or match["p2"] or match["p3"] or "").strip()
    rest = (match["r1"] if match["t1"] else match["r2"] if match["t2"] else match["r3"] or "").strip(" ,")
    if not thing or thing.split()[-1] in _NOT_A_THING:
        return None
    # Keep the person's own capitals for the rest ("Gate B12", "Lisbon").
    start = low.rfind(rest) if rest else -1
    rest = text[start : start + len(rest)] if start >= 0 else rest
    if prep in ("tomorrow", "today", "tonight", "next"):
        rest, prep = f"{prep} {rest}".strip(), ""
    elif prep == "to":
        prep = (
            "on" if re.match(r"(?i)(?:\d{1,2}(?:st|nd|rd|th)? \w+|" + "|".join(_DAYS) + r"|tomorrow)\b", rest) else "at"
        )
    if not rest:
        return None
    today = now.date()
    rest, day = _resolve(rest, today)
    said_on = _day_words(today)
    body = f"My {thing} is {prep} {rest}".replace("  ", " ")
    if day is not None:
        expires = datetime.combine(day + timedelta(days=1), time(23, 59)).timestamp()
    else:
        expires = (now + KEEP_WITHOUT_DATE).timestamp()
    until = _day_words(day + timedelta(days=1)) if day is not None else ""
    return Note(thing=thing, text=f"{body} (said on {said_on})", expires=expires, until=until)


__all__ = ["Note", "never_kept", "passing_note"]

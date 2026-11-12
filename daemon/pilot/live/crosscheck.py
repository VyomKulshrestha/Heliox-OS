"""What the person says against what they just wrote: "remind me about the dentist on Friday" a few minutes after
typing "Dentist, Thursday 12 November".

Only dates are compared, and only when both sentences tie the date to the same distinctive word (the dentist, the
offsite, the review) close to it. Times, amounts and names are left out by design (not built and measured): a time
said aloud ("at four") and a time written ("16:00", "4 pm", "after lunch") have too many honest forms, an amount said
is often a different amount on purpose (a deposit, a share), and two names in one sentence are usually two people;
text alone cannot tell those from a slip.

The typed side is only what the person wrote themselves in the last ``WINDOW_SECONDS`` (text that was already in the
field when it got focus does not count), read by Live's own gated reader; nothing new is read and nothing leaves the
machine.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from pilot.live.detectors import NORMAL, Correction

KIND = "said_vs_typed"
WINDOW_SECONDS = 300.0
NEAR_WORDS = 6  # the subject word must be this close before or after the date
MAX_LOG = 12

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY = r"(?P<wd>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTHS = [m.casefold() for m in calendar.month_name[1:]]
_MONTH = r"(?P<m>" + "|".join(_MONTHS) + ")"
_MENTION = re.compile(
    rf"\b(?:{_DAY}(?:,?\s+(?:the\s+)?(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(?:of\s+)?{_MONTH.replace('?P<m>', '?P<m1>')})?)?"
    rf"|(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH.replace('?P<m>', '?P<m2>')})\b",
    re.IGNORECASE,
)
_SCHEDULING = re.compile(
    r"\b(?:remind|reminder|schedule|book|put|move|moved|meeting|set|calendar|appointment|is on|on)\b", re.IGNORECASE
)
# Changing the date on purpose is not a slip: "move the dentist to Friday".
_DELIBERATE = re.compile(
    r"\b(?:move|moved|moving|change|changed|changing|reschedule|rescheduled|push|pushed|instead|swap|postpone"
    r"|postponed|bring (?:it )?forward|now on|not on)\b",
    re.IGNORECASE,
)
_STOP = frozenset(
    [
        "about",
        "after",
        "again",
        "also",
        "before",
        "being",
        "from",
        "have",
        "here",
        "into",
        "just",
        "like",
        "make",
        "meet",
        "meeting",
        "next",
        "only",
        "remind",
        "reminder",
        "some",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "today",
        "tomorrow",
        "week",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "your",
        "yours",
        "mine",
        "please",
        "could",
        "should",
        "book",
        "schedule",
        "move",
        "moved",
        "calendar",
        "appointment",
        "the",
        "and",
        "for",
        "put",
    ]
)


@dataclass(frozen=True)
class Mention:
    when: date
    words: str  # as written or said
    subjects: frozenset[str]


def _resolve(match: re.Match[str], today: date) -> date | None:
    weekday = match["wd"]
    day = match["d1"] or match["d2"]
    month = match["m1"] or match["m2"]
    if day and month:
        m = _MONTHS.index(month.casefold()) + 1
        for year in (today.year, today.year + 1):
            try:
                candidate = date(year, m, int(day))
            except ValueError:
                return None
            if candidate >= today - timedelta(days=7):
                return candidate
        return None
    if weekday and not day:
        offset = (_DAYS.index(weekday.casefold()) - today.weekday()) % 7
        return today + timedelta(days=offset)
    return None  # "Tuesday the 12th" without a month: the weekday check deals with it


def mentions(text: str, today: date) -> list[Mention]:
    words = re.findall(r"[\w'-]+", text)
    lowered = [w.casefold() for w in words]
    out = []
    for match in _MENTION.finditer(text):
        when = _resolve(match, today)
        if when is None:
            continue
        start = len(re.findall(r"[\w'-]+", text[: match.start()]))
        end = start + len(re.findall(r"[\w'-]+", match.group(0)))
        near = lowered[max(0, start - NEAR_WORDS) : start] + lowered[end : end + NEAR_WORDS]
        subjects = frozenset(
            w
            for w in near
            if len(w) >= 4 and w not in _STOP and w not in _DAYS and w not in _MONTHS and not w.isdigit()
        )
        if subjects:
            out.append(Mention(when, match.group(0).strip(), subjects))
    return out


def _say(when: date) -> str:
    return f"{_DAYS[when.weekday()].title()} {when.day} {calendar.month_name[when.month]}"


def said_vs_typed(said: str, typed: list[tuple[float, str, str, str]], now: float, today: date) -> list[Correction]:
    """``typed``: (when, app label, the field's text, the text it held before the person edited it). Returns at
    most one correction."""
    if not _SCHEDULING.search(said) or _DELIBERATE.search(said):
        return []
    spoken = mentions(said, today)
    if not spoken:
        return []
    for at, app, text, before in reversed(typed):
        if now - at > WINDOW_SECONDS:
            continue
        for written in mentions(text, today):
            if before and written.words.casefold() in before.casefold():
                continue  # it was there before they started: someone else's words
            for heard in spoken:
                shared = written.subjects & heard.subjects
                if not shared or written.when == heard.when:
                    continue
                if any(other.when == heard.when and other.subjects & shared for other in mentions(text, today)):
                    continue  # they also wrote the date they said: two events, not a contradiction
                subject = sorted(shared, key=len, reverse=True)[0]
                where = f" in {app}" if app else ""
                return [
                    Correction(
                        kind=KIND,
                        key=f"{KIND}:{subject}:{written.when.isoformat()}:{heard.when.isoformat()}",
                        title=f"You wrote {_say(written.when)} for the {subject}",
                        message=f"You said {heard.words}, which is {_say(heard.when)}.",
                        evidence=heard.words,
                        detail=f"A few minutes ago{where}: {written.words}.",
                        urgency=NORMAL,
                        confidence=0.8,
                    )
                ]
    return []


__all__ = ["KIND", "WINDOW_SECONDS", "mentions", "said_vs_typed"]

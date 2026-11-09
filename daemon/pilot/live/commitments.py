"""Promises the person makes in what they write, with an offer to remind them before the deadline.

"I'll send you the deck by Friday." "Let me check and get back to you tomorrow." People make these in chats and
emails all day and keep track of them in their heads. Live already reads the message as it is written, so it can
notice the promise and, at the next pause, offer a reminder at a sensible time before the deadline. Accepting the
card sets an ordinary reminder ("remind me on Friday at 10am to send them the deck"); declining it is one click.

What counts, in code, with no model:

- a first-person promise ("I'll", "I will", "I'm going to", "I can", "let me", "will do") followed by what they
  will do and a deadline in the same sentence ("by Friday", "tomorrow", "on 18 November", "by 3pm", "end of day",
  "next week", "tonight");
- not a question, not "if I…", not "I'll be…" or "I'll think…" (a state, not a task), not "let me know" (asking
  someone else), and not something already in their reminders;
- in writing meant for someone else (mail, chat, documents), not in a terminal or code.

The reminder goes at 10:00 on the day for a day-only deadline, an hour before a deadline with a time, at 16:00 for
"today" and "end of day", and is not offered at all if that moment has already passed.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any

from pilot.live.detectors import LOW, Correction, TextContext
from pilot.live.grounded import significant_words, stated_day, stated_time

KIND = "commitment"
MAX_ACTION_WORDS = 12
SCAN_CHARS = 1500

_PROMISE = re.compile(
    r"\b(?:I'll|I will|I'm going to|I am going to|I can|let me|will do|I promise to|I'll make sure to|I'll try to)\s+",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_MONTHS = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
# Where the deadline starts: everything from here on is "when", everything before it is "what".
_DEADLINE = re.compile(
    rf"\s*\b(?:(?:by|before|on|until|no later than|first thing)\s+)?(?:the\s+)?"
    rf"(?:tomorrow|today|tonight|this (?:morning|afternoon|evening)|next week|end of (?:the )?day|eod|cob|"
    rf"(?:{_DAYS})|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})|(?:{_MONTHS})\s+\d{{1,2}}|"
    rf"\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)|noon)\b.*$",
    re.IGNORECASE,
)
_NOT_A_TASK = re.compile(r"^(?:be|need|want|think|know|see if|probably|maybe|let you know if)\b", re.IGNORECASE)
_END_OF_DAY = re.compile(r"\b(?:end of (?:the )?day|eod|cob|close of business)\b", re.IGNORECASE)
_NEXT_WEEK = re.compile(r"\bnext week\b", re.IGNORECASE)
_TONIGHT = re.compile(r"\btonight\b", re.IGNORECASE)
_PRONOUNS = {"you": "them", "your": "their", "yours": "theirs"}


def deadline_of(phrase: str, today: date) -> tuple[datetime | None, bool]:
    """The deadline a phrase states, and whether it names a time of day."""
    if _END_OF_DAY.search(phrase):
        return datetime.combine(today, time(17, 0)), False
    if _NEXT_WEEK.search(phrase):
        return datetime.combine(today + timedelta(days=7 - today.weekday()), time(9, 0)), False
    day, _ = stated_day(phrase, today)
    at = stated_time(phrase)
    if at is None and _TONIGHT.search(phrase):
        at = time(20, 0)
    if day is None:
        day = today if at is not None else None
    if day is None:
        return None, False
    return datetime.combine(day, at or time(0, 0)), at is not None


def remind_at(deadline: datetime, has_time: bool, now: datetime) -> datetime | None:
    if has_time:
        moment = deadline - timedelta(hours=1)
    elif deadline.date() == now.date():
        moment = datetime.combine(now.date(), time(16, 0))
    else:
        moment = datetime.combine(deadline.date(), time(10, 0))
    return moment if moment > now + timedelta(minutes=10) else None


def _clock(moment: datetime) -> str:
    return moment.strftime("%I:%M%p").lstrip("0").lower().replace(":00", "")


def _when(moment: datetime, now: datetime) -> str:
    days = (moment.date() - now.date()).days
    if days == 0:
        return f"today at {_clock(moment)}"
    if days == 1:
        return f"tomorrow at {_clock(moment)}"
    if days < 7:
        return f"on {moment.strftime('%A')} at {_clock(moment)}"
    return f"on {moment.day} {moment.strftime('%B')} at {_clock(moment)}"


def _request(moment: datetime, action: str, now: datetime) -> str:
    """A reminder request the reminder fast path (pilot.workflows.reminders) understands."""
    days = (moment.date() - now.date()).days
    if days == 0:
        return f"Remind me at {_clock(moment)} to {action}"
    if days == 1:
        return f"Remind me tomorrow at {_clock(moment)} to {action}"
    if days < 7:
        return f"Remind me on {moment.strftime('%A')} at {_clock(moment)} to {action}"
    return f"Remind me on {moment.day} {moment.strftime('%B')} at {_clock(moment)} to {action}"


def _lands_at(request: str, moment: datetime, now: datetime) -> bool:
    """The reminder this request would set is the one being offered (checked with the real parser)."""
    from pilot.workflows.reminders import parse_reminder

    parsed = parse_reminder(request, now)
    return parsed is not None and abs(parsed[0] - moment.timestamp()) < 60


def find_commitments(
    text: str, ctx: TextContext, now: datetime, reminders: list[dict[str, Any]] | None = None
) -> list[Correction]:
    """The latest promise with a deadline in the text, as an offer to set a reminder."""
    if ctx.source != "focused_field" or ctx.is_terminal or ctx.is_code_editor:
        return []
    existing = [significant_words(str(r.get("text", ""))) for r in reminders or []]
    found: list[Correction] = []
    for match in _SENTENCE.finditer(text[-SCAN_CHARS:]):
        sentence = match.group(0).strip()
        if not sentence or sentence.endswith("?") or re.search(r"\bif I\b", sentence, re.IGNORECASE):
            continue
        promise = _PROMISE.search(sentence)
        if not promise:
            continue
        after = sentence[promise.end() :].rstrip(".! ")
        if _NOT_A_TASK.match(after) or re.match(r"know\b", after, re.IGNORECASE):
            continue
        cut = _DEADLINE.search(after)
        if not cut:
            continue
        deadline, has_time = deadline_of(cut.group(0), now.date())
        if deadline is None:
            continue
        moment = remind_at(deadline, has_time, now)
        words = [_PRONOUNS.get(word.casefold(), word) for word in after[: cut.start()].strip(" ,;:-").split()]
        action = " ".join(words[:MAX_ACTION_WORDS])
        if moment is None or len(words) < 2:
            continue
        wanted = significant_words(action)
        if any(wanted and len(wanted & seen) * 2 >= len(wanted) for seen in existing):
            continue  # already in their reminders
        promised = cut.group(0).strip(" ,.!")
        request = _request(moment, action, now)
        if not _lands_at(request, moment, now):
            continue
        found.append(
            Correction(
                kind=KIND,
                key=f"{KIND}:{action.casefold()}:{deadline.date().isoformat()}",
                title="Want a reminder for that?",
                message=f"You said you'd {action} {promised}. Want a reminder {_when(moment, now)}?",
                evidence=sentence[-120:],
                fix=request,
                fix_kind="ask",
                urgency=LOW,
                confidence=0.7,
            )
        )
    return found[-1:]  # the latest promise; one offer at a time


__all__ = ["KIND", "deadline_of", "find_commitments", "remind_at"]

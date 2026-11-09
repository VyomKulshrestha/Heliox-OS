"""Corrections grounded in the person's own data: their calendar, to-dos and reminders.

The rule-based detectors catch slips anyone could spot. The slips that matter most are personal: "Thursday at
3 works for me" when the dentist is at 3 on Thursday, "the design review on Thursday" when it is on Tuesday, "I'll
send the report by Friday" when the to-do says Wednesday. Catching those takes the person's own data and some
reading comprehension, and it has to be right, because a wrong correction about someone's own week is worse than
none. So the work is split between code and a small local model, and code has the last word:

1. **Clashes, in code.** When a sentence proposes a day and a time ("Thursday at 3pm works"), it is checked against
   the calendar directly.
2. **Deliberate changes are left alone.** "Push it to Friday", "remind me Wednesday", "move the call": the person
   is changing something on purpose, so there is nothing to correct.
3. **Named things, in code.** "The design review", "my flight to Berlin", "the offsite": when the words name one
   fact (and are not "the review slides" or a person's name), code links them and compares the days.
4. **Paraphrases, by the model.** "The airport for Berlin" for "Flight to Berlin": only when code cannot tell what a
   dated sentence is about, a 4B model in Ollama on this computer is asked which of a few candidate facts it means
   and which words give the day. It only links; it never decides whether anything is wrong.
5. **Verified, in code.** A model link counts only if its quote is really in the sentence, the fact is one it was
   given, and the day the sentence states really differs from the fact. Every card's message is written by code
   from the fact, never by the model, so it cannot misstate the person's calendar.

On a fixed week of 29 sentences (``python -m pilot.live.grounded_eval``) code alone caught 11 of 13 contradictions
with no false alarm, and qwen3.5:4b added one paraphrase, at about 2 s per call on a laptop CPU.

It runs only at a pause, only when the sentence mentions something checkable, at most once every few seconds. The
model runs on this computer; the sentence and the facts never leave it. A correction is an ordinary Live card: it
waits for the pause and withdraws itself when the quoted words go away.
"""

from __future__ import annotations

import calendar
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from pilot.live.detectors import NORMAL, Correction

logger = logging.getLogger("pilot.live.grounded")

KIND = "personal_fact"
MIN_LINK_CONFIDENCE = 0.85  # a link without a shared word needs the model to be this sure
MAX_FACTS = 8
MAX_SENTENCE_CHARS = 400
MAX_QUOTE_CHARS = 140
EVENT_DAYS_BEHIND, EVENT_DAYS_AHEAD = 1, 30
DEFAULT_EVENT_MINUTES = 60

_STOP = frozenset(
    [
        "about",
        "above",
        "after",
        "again",
        "against",
        "also",
        "although",
        "always",
        "another",
        "anything",
        "around",
        "because",
        "before",
        "being",
        "below",
        "between",
        "could",
        "doing",
        "during",
        "either",
        "every",
        "first",
        "from",
        "have",
        "having",
        "here",
        "into",
        "just",
        "last",
        "later",
        "maybe",
        "might",
        "more",
        "most",
        "much",
        "must",
        "never",
        "next",
        "only",
        "other",
        "over",
        "really",
        "should",
        "since",
        "some",
        "still",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "think",
        "this",
        "those",
        "through",
        "today",
        "tomorrow",
        "under",
        "until",
        "very",
        "want",
        "were",
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
        "please",
        "thanks",
        "hello",
        "works",
        "work",
        "okay",
        "sure",
        "great",
        "free",
    ]
)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY = re.compile(
    r"\b(?P<next>next\s+)?(?P<wd>monday|tuesday|tues?|wednesday|wed|thursday|thurs?|thu|friday|fri|saturday"
    r"|sunday)\b",
    re.IGNORECASE,
)
_MONTHS = r"(?P<m>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE = (
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTHS}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE),
)
_NTH = re.compile(r"\bthe (?P<n>\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
_RELATIVE = re.compile(r"\b(?P<r>today|tonight|tomorrow|this (?:morning|afternoon|evening))\b", re.IGNORECASE)
_CLOCK = re.compile(
    r"\b(?P<h>\d{1,2})(?::(?P<min>\d{2}))?\s*(?P<ap>am|pm|a\.m\.|p\.m\.)|\b(?P<h24>[01]?\d|2[0-3]):(?P<m24>\d{2})\b"
    r"|\bat (?P<bare>\d{1,2})\b(?![:\d])|\b(?P<noon>noon|midday)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]{3,}|\b\d{1,2}:\d{1,2}\b(?!\s*(?:am|pm))|\b[A-Za-z]\d{1,2}\b")
# Words that name a thing made for an event rather than the event: "the review slides", "the offsite agenda".
_ARTIFACTS = frozenset(
    [
        "slides",
        "slide",
        "deck",
        "doc",
        "docs",
        "document",
        "agenda",
        "notes",
        "invite",
        "invitation",
        "recording",
        "summary",
        "minutes",
        "prep",
        "planning",
        "plan",
        "draft",
        "outline",
        "budget",
        "photos",
        "pictures",
        "video",
        "follow-up",
        "followup",
        "feedback",
    ]
)
_PROPOSES = re.compile(
    r"\b(?:works?|let'?s|see you|meet|call|book|schedule|free|available|can do|how about|shall we|i'?ll be|"
    r"i can|we can|set up|catch up|lunch|dinner|coffee)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Fact:
    id: str  # "event:2", "todo:5", "reminder:1"
    kind: str  # "event", "todo" or "reminder"
    text: str  # one line the model reads
    title: str = ""
    day: date | None = None
    start: datetime | None = None  # events and reminders with a time of day
    end: datetime | None = None
    words: frozenset[str] = field(default_factory=frozenset)

    @property
    def source(self) -> str:
        return {"event": "calendar", "todo": "to-do list", "reminder": "reminders"}[self.kind]


# ── reading days and times ──


def _normal(word: str) -> str:
    word = word.casefold().strip("'’-")
    return re.sub(r"['’]s$", "", word)  # "Priya's" is about Priya


def significant_words(text: str) -> frozenset[str]:
    return frozenset(_normal(w) for w in _WORD.findall(text) if _normal(w) not in _STOP)


def _parse_when(value: Any) -> datetime | date | None:
    if isinstance(value, datetime | date):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return date.fromisoformat(text)  # a date with no time: an all-day event or a due day, not midnight
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return moment.astimezone().replace(tzinfo=None) if moment.tzinfo else moment
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _day_name(day: date) -> str:
    return f"{_WEEKDAYS[day.weekday()].title()} {day.day} {calendar.month_name[day.month]}"


def _clock(moment: datetime | time) -> str:
    return moment.strftime("%H:%M")


def stated_day(text: str, today: date) -> tuple[date | None, str]:
    """The day a phrase names, and the words that name it. An explicit date wins over a weekday."""
    for pattern in _DATE:
        match = pattern.search(text)
        if match:
            month = [m[:3] for m in calendar.month_abbr[1:]].index(match["m"][:3].title()) + 1
            day = int(match["d"])
            for year in (today.year, today.year + 1):
                try:
                    candidate = date(year, month, day)
                except ValueError:
                    return None, ""
                if candidate >= today - timedelta(days=EVENT_DAYS_BEHIND):
                    return candidate, match.group(0)
    match = _NTH.search(text)
    if match:
        # "Tuesday the 17th": the day number is the precise part; the next date with that number.
        number = int(match["n"])
        for months_ahead in (0, 1, 2):
            year, month = today.year + (today.month - 1 + months_ahead) // 12, (today.month - 1 + months_ahead) % 12 + 1
            if number <= calendar.monthrange(year, month)[1]:
                candidate = date(year, month, number)
                if candidate >= today - timedelta(days=EVENT_DAYS_BEHIND):
                    return candidate, _wider(text, match)
    match = _RELATIVE.search(text)
    if match:
        offset = 1 if match["r"].casefold() == "tomorrow" else 0
        return today + timedelta(days=offset), match.group(0)
    match = _WEEKDAY.search(text)
    if match:
        wanted = next(i for i, name in enumerate(_WEEKDAYS) if name.startswith(match["wd"][:3].casefold()))
        ahead = (wanted - today.weekday()) % 7
        if match["next"] and ahead == 0:
            ahead = 7
        return today + timedelta(days=ahead), match.group(0)
    return None, ""


def _wider(text: str, match: re.Match[str]) -> str:
    """ "Tuesday the 17th" rather than "the 17th", when a weekday sits right before the day number."""
    before = re.search(
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+$", text[: match.start()], re.I
    )
    return text[before.start() : match.end()] if before else match.group(0)


def stated_time(text: str) -> time | None:
    match = _CLOCK.search(text)
    if not match:
        return None
    if match["noon"]:
        return time(12, 0)
    if match["h24"] is not None:
        return time(int(match["h24"]), int(match["m24"]))
    if match["bare"] is not None:
        hour = int(match["bare"])
        if not 1 <= hour <= 12:
            return None
        return time(hour + 12 if 1 <= hour <= 7 else hour, 0)  # "at 3" in a plan means the afternoon
    hour, minute = int(match["h"]), int(match["min"] or 0)
    if not 1 <= hour <= 12 or minute > 59:
        return None
    pm = match["ap"].casefold().startswith("p")
    return time((hour % 12) + (12 if pm else 0), minute)


# ── the facts ──


def build_facts(
    *,
    today: date,
    events: list[dict[str, Any]] | None = None,
    todos: list[dict[str, Any]] | None = None,
    reminders: list[dict[str, Any]] | None = None,
) -> list[Fact]:
    """One line per thing the person keeps, near enough in time to matter."""
    facts: list[Fact] = []
    lo, hi = today - timedelta(days=EVENT_DAYS_BEHIND), today + timedelta(days=EVENT_DAYS_AHEAD)
    for i, event in enumerate(events or []):
        title = " ".join(str(event.get("summary") or event.get("title") or "").split())
        start = _parse_when(event.get("start"))
        if not title or start is None:
            continue
        day = start.date() if isinstance(start, datetime) else start
        if not lo <= day <= hi:
            continue
        if isinstance(start, datetime):
            end = _parse_when(event.get("end"))
            end = end if isinstance(end, datetime) and end > start else start + timedelta(minutes=DEFAULT_EVENT_MINUTES)
            text = f"Calendar: {title}, {_day_name(day)}, {_clock(start)} to {_clock(end)}"
            facts.append(Fact(f"event:{i}", "event", text, title, day, start, end, significant_words(title)))
        else:
            text = f"Calendar: {title}, {_day_name(day)}, all day"
            facts.append(Fact(f"event:{i}", "event", text, title, day, None, None, significant_words(title)))
    for i, todo in enumerate(todos or []):
        title = " ".join(str(todo.get("text") or todo.get("title") or "").split())
        due = _parse_when(todo.get("due") or todo.get("due_date"))
        if not title or todo.get("done") or due is None:
            continue  # only a to-do with a date can be contradicted by one
        day = due.date() if isinstance(due, datetime) else due
        facts.append(
            Fact(
                f"todo:{i}",
                "todo",
                f"To-do: {title}, due {_day_name(day)}",
                title,
                day,
                None,
                None,
                significant_words(title),
            )
        )
    for i, reminder in enumerate(reminders or []):
        title = " ".join(str(reminder.get("text") or reminder.get("message") or "").split())
        when = _parse_when(reminder.get("due") or reminder.get("when"))
        if not title or not isinstance(when, datetime) or not lo <= when.date() <= hi:
            continue
        text = f"Reminder: {title}, {_day_name(when.date())} at {_clock(when)}"
        facts.append(Fact(f"reminder:{i}", "reminder", text, title, when.date(), when, None, significant_words(title)))
    return facts


def candidates(sentence: str, facts: list[Fact], today: date) -> list[Fact]:
    """The facts this sentence could refer to; empty means no model call is worth making."""
    words = significant_words(sentence)
    day, _ = stated_day(sentence, today)
    scored: list[tuple[int, int, Fact]] = []
    for order, fact in enumerate(facts):
        overlap = len(words & fact.words)
        if overlap and day is not None:
            scored.append((2 + overlap, order, fact))
        elif day is not None and fact.kind == "event" and abs((fact.day - day).days) <= 7:
            scored.append((1, order, fact))  # an unnamed reference ("the review") to something that week
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [fact for _, _, fact in scored[:MAX_FACTS]]


# ── 1. clashes, in code ──


def clash(sentence: str, facts: list[Fact], today: date) -> Correction | None:
    """A plan for a day and time that a calendar event already fills."""
    if not _PROPOSES.search(sentence):
        return None
    day, day_words = stated_day(sentence, today)
    at = stated_time(sentence)
    if day is None:
        return None
    mentioned = significant_words(sentence)
    for fact in facts:
        if fact.kind != "event" or fact.day != day or mentioned & fact.words:
            continue  # a sentence that names the event is about it, not double-booking it
        if fact.start is None:
            if at is None:
                continue
            message = f"You have {fact.title} all day on {_day_name(day)}."
        else:
            if at is None or not fact.start.time() <= at < fact.end.time():
                continue
            message = f"You have {fact.title} from {_clock(fact.start)} to {_clock(fact.end)} on {_day_name(day)}."
        return _correction(fact, _plan_words(sentence, day_words), message, "", "clash")
    return None


def _plan_words(sentence: str, day_words: str) -> str:
    """ "Thursday at 3pm": the day and the time together when they sit side by side, else the day."""
    start = sentence.find(day_words)
    match = _CLOCK.search(sentence)
    if start < 0 or match is None:
        return day_words
    end = start + len(day_words)
    if match.start() - end > 12 or start - match.end() > 12:
        return day_words
    return sentence[min(start, match.start()) : max(end, match.end())]


# ── 2. references, by the model, and 3. verified in code ──

SYSTEM_PROMPT = (
    "You link one sentence a person wrote to the facts they keep: calendar events, to-dos with due dates and "
    "reminders. Answer one question: is the sentence about the very same thing as one of the facts (named "
    "differently, like 'the review' for 'Design review with Sam' or 'the dentist' for 'Dental check-up'), and does "
    "it give that thing a day, date or time? Something merely related (a document for an event, a new reminder, a "
    "different meeting) is not the same thing. Do not judge whether the day is right; only link.\n"
    'Answer with JSON only. Not about any fact: {"refers": false}. About one: {"refers": true, "fact": "<the fact '
    'id>", "quote": "<the exact words in the sentence that give its day, date or time, copied character for '
    'character>", "confidence": <0.0 to 1.0>}.\n'
    "Examples, with the fact [event:1] Calendar: Design review with Sam, Tuesday 10 November, 11:00 to 12:00:\n"
    '"See you at the review on Thursday" -> {"refers": true, "fact": "event:1", "quote": "Thursday", '
    '"confidence": 0.9}\n'
    '"I\'ll finish the review slides by Thursday" -> {"refers": false}\n'
    '"Thursday works for lunch" -> {"refers": false}'
)

# Asking for a change on purpose ("push it to Friday", "remind me Wednesday") is not a slip to correct.
_CHANGE = re.compile(
    r"\b(?:move|moved|moving|push|pushed|reschedul\w*|postpone\w*|shift\w*|change\w*|remind me|set a reminder|"
    r"again|instead|swap\w*|delay\w*|bring (?:it|this|that) forward)\b",
    re.IGNORECASE,
)


def _names_the_thing(sentence: str, word: str) -> bool:
    """One shared word links only as a common noun naming the thing itself: "the review on Wednesday" does,
    "the review slides" and "lunch with Sam" do not."""
    for match in _WORD.finditer(sentence):
        if _normal(match.group(0)) != word:
            continue
        if match.group(0)[0].isupper() and match.start() > 0:
            return False  # a name in mid-sentence: a person or place, not the event
        following = _WORD.search(sentence, match.end())
        if following and following.start() - match.end() <= 1 and _normal(following.group(0)) in _ARTIFACTS:
            return False
        return True
    return False


def linked(sentence: str, facts: list[Fact]) -> Fact | None:
    """The fact a sentence names ("the design review", "my flight to Berlin", "the offsite"), if code can tell.
    Two shared words, or all of a short title, link; one shared word links only when no other fact has it and it
    names the thing itself."""
    words = significant_words(sentence)
    owners: dict[str, int] = {}
    for fact in facts:
        for word in fact.words:
            owners[word] = owners.get(word, 0) + 1
    best: tuple[int, Fact] | None = None
    for fact in facts:
        overlap = words & fact.words
        strong = len(overlap) >= 2 or (overlap and overlap == fact.words)
        partial = (
            len(overlap) == 1 and owners[next(iter(overlap))] == 1 and _names_the_thing(sentence, next(iter(overlap)))
        )
        if (strong or partial) and (best is None or len(overlap) > best[0]):
            best = (len(overlap), fact)
    return best[1] if best else None


def build_prompt(sentence: str, facts: list[Fact], today: date) -> str:
    listed = "\n".join(f"[{fact.id}] {fact.text}" for fact in facts)
    return f"Today is {_day_name(today)} {today.year}.\n\nFacts:\n{listed}\n\nSentence:\n{sentence}"


def _span(quote: str, text: str) -> str:
    """The quote as it appears in the text (case and spacing included), or "" when it does not."""
    words = quote.split()
    if not words:
        return ""
    match = re.search(r"\s+".join(re.escape(word) for word in words), text, re.IGNORECASE)
    return match.group(0) if match else ""


def _correction(fact: Fact, quote: str, message: str, fix: str, how: str) -> Correction:
    return Correction(
        kind=KIND,
        key=f"{KIND}:{fact.id}:{quote.casefold()}",
        title=f"Check your {fact.source}",
        message=message,
        evidence=quote,
        detail=f"From your {fact.source}: {fact.text.split(': ', 1)[-1]}. ({'Checked in code' if how == 'clash' else 'Linked by the local model, checked in code'}.)",
        fix=fix,
        fix_kind="replace" if fix else "",
        urgency=NORMAL,
        confidence=0.95 if how == "clash" else 0.85,
    )


def verify(fact: Fact, quote: str, sentence: str, today: date, linked_by_words: bool) -> Correction | None:
    """Is what the sentence states about this fact really different from the fact? Code decides, not the model."""
    day, day_words = stated_day(quote, today)
    if day is None or fact.day is None:
        return None
    if day != fact.day:
        what = {"event": "is on", "todo": "is due", "reminder": "is set for"}[fact.kind]
        message = f"{fact.title} {what} {_day_name(fact.day)}, not {_day_name(day)}."
        fix = _WEEKDAYS[fact.day.weekday()].title() if _WEEKDAY.fullmatch(day_words.strip()) else _day_name(fact.day)
        return _correction(fact, _span(day_words, sentence), message, fix, "link")
    at = stated_time(quote) or stated_time(sentence)
    if fact.kind == "event" and fact.start is not None and at is not None and linked_by_words:
        if not fact.start.time() <= at < fact.end.time():
            message = (
                f"{fact.title} starts at {_clock(fact.start)} on {_day_name(fact.day)}, not {at.strftime('%H:%M')}."
            )
            return _correction(fact, _span(quote, sentence) or _span(day_words, sentence), message, "", "link")
    return None


def judge(raw: str, sentence: str, facts: list[Fact], today: date) -> tuple[Correction | None, str]:
    """The model's link, checked. Returns the correction (or None) and why."""
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        answer = json.loads(raw[start:end])
    except (ValueError, TypeError):
        return None, "rejected: not JSON"
    if not isinstance(answer, dict):
        return None, "rejected: not JSON"
    if answer.get("refers") is not True:
        return None, "no_reference"
    fact = {f.id: f for f in facts}.get(str(answer.get("fact", "")).strip().strip("[]"))
    if fact is None:
        return None, "rejected: unknown fact"
    quote = _span(str(answer.get("quote", ""))[:MAX_QUOTE_CHARS], sentence)
    if not quote:
        return None, "rejected: quote not in the sentence"
    try:
        confidence = float(answer.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    by_words = bool(significant_words(sentence) & fact.words)
    if not by_words and confidence < MIN_LINK_CONFIDENCE:
        return None, "rejected: weak link"
    correction = verify(fact, quote, sentence, today, by_words)
    if correction is None:
        return None, "rejected: the facts agree"
    return correction, "accepted"


def last_sentences(text: str, limit: int = MAX_SENTENCE_CHARS) -> str:
    """The end of what they are writing, cut at a sentence or line boundary."""
    tail = text[-limit:]
    if len(text) > limit:
        cut = max(tail.find(". "), tail.find("\n"))
        if 0 <= cut < limit // 2:
            tail = tail[cut + 1 :]
    return tail.strip()


Complete = Callable[[str, str], Awaitable[str]]  # (system, prompt) -> raw answer


def ollama_complete(model: str, base_url: str = "http://127.0.0.1:11434", keep_alive: int = 300) -> Complete:
    """A local Ollama chat call: JSON out, no thinking, deterministic, short. Talks only to ``base_url``."""
    import httpx

    url = base_url.rstrip("/") + "/api/chat"

    async def complete(system: str, prompt: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_predict": 120},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        # trust_env=False: a system proxy must never see the sentence or the facts.
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", ""))

    return complete


class GroundedChecker:
    """Clash check in code, then a model link checked in code. ``facts`` reads the stores fresh each time."""

    def __init__(
        self, complete: Complete | None, facts: Callable[[], list[Fact]], today: Callable[[], date] = date.today
    ):
        self._complete = complete
        self._facts = facts
        self._today = today
        self.checks = 0
        self.model_calls = 0
        self.found = 0
        self.rejected = 0
        self.last_verdict = ""
        self.last_error = ""

    async def check(self, text: str) -> list[Correction]:
        sentence = last_sentences(text)
        if len(sentence) < 8:
            return []
        today = self._today()
        try:
            facts = self._facts()
        except Exception:
            logger.debug("Reading the personal stores failed", exc_info=True)
            return []
        if not facts:
            return []
        self.checks += 1
        found = clash(sentence, facts, today)
        if found is not None:
            self.found += 1
            self.last_verdict = "clash"
            return [found]
        if _CHANGE.search(sentence):
            self.last_verdict = "a deliberate change"
            return []
        fact = linked(sentence, facts)
        if fact is not None:
            # Named outright: code knows what the sentence is about, so the model is not needed.
            found = verify(fact, sentence, sentence, today, linked_by_words=True)
            self.last_verdict = "named" if found else "named: the facts agree"
            self.found += int(found is not None)
            return [found] if found else []
        chosen = candidates(sentence, facts, today)
        if not chosen or self._complete is None:
            return []
        self.model_calls += 1
        try:
            raw = await self._complete(SYSTEM_PROMPT, build_prompt(sentence, chosen, today))
            self.last_error = ""
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"[:200]
            logger.info("The local model is unavailable for grounded checks: %s", self.last_error)
            return []
        correction, self.last_verdict = judge(raw, sentence, chosen, today)
        if correction is None:
            self.rejected += int(self.last_verdict.startswith("rejected"))
            return []
        self.found += 1
        return [correction]

    def stats(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "model_calls": self.model_calls,
            "found": self.found,
            "rejected": self.rejected,
            "last_verdict": self.last_verdict,
            "error": self.last_error,
        }


__all__ = [
    "KIND",
    "Fact",
    "GroundedChecker",
    "build_facts",
    "build_prompt",
    "candidates",
    "clash",
    "judge",
    "last_sentences",
    "linked",
    "ollama_complete",
    "stated_day",
    "stated_time",
    "verify",
]

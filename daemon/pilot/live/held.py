"""A slip in something said TO Heliox is asked about before Heliox acts on it.

Before this, Live heard a request at the start of dispatch and raised a card, while the request went on down its road:
"Put the review in for Monday the nineteenth" (a Thursday) was booked for whichever day the planner guessed, and
"email the Acme invoice for 4,200" a minute after "the invoice for Acme is 2,400" was emailed with 4,200. Now the
turn is held, one short question is asked, and the answer decides what goes on:

* a weekday that does not match its date: "The nineteenth is a Thursday. Monday the sixteenth, or Thursday the
  nineteenth?" The request goes on with the date the person picks.
* a date that does not exist: "There's no thirty-first of November. The thirtieth of November, or the first of
  December?"
* an amount or a date the person gave a moment ago in this conversation, now different for the same thing: "A
  moment ago you said the Acme invoice is 2,400. 2,400 or 4,200?" Changing it on purpose ("change it to 4,200",
  "actually 4,200") is not asked about.

The answer can be the words of an option ("Monday", "the sixteenth", "2,400"), "the first" or "the second", or "never
mind" to drop the request. An answer that is none of these is asked once more, then the request is dropped with a
sentence saying so. A held request lapses after ``HOLD_SECONDS``. Everything is decided in code from what was said;
nothing here calls a model. The earlier turns are kept in memory for the open conversation only.
"""

from __future__ import annotations

import calendar
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from pilot.live import spoken as sp

HOLD_SECONDS = 60.0
TURNS_SECONDS = 600.0  # earlier turns of the conversation that a request is checked against
MAX_TURNS = 12
NEAR_WORDS = 6

_DAYS = [d.casefold() for d in calendar.day_name]
_MONTHS = [m.casefold() for m in calendar.month_name[1:]]
_DAY_RE = r"(?P<wd>" + "|".join(_DAYS) + r")"
_MONTH_RE = r"(?P<m>" + "|".join(_MONTHS) + r")"
_WEEKDAY_DATE = re.compile(
    rf"\b{_DAY_RE},?\s+(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(?:of\s+)?{_MONTH_RE})?\b", re.IGNORECASE
)
_LONE_DATE = re.compile(rf"\b(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE}\b", re.IGNORECASE)
_DELIBERATE = re.compile(
    r"\b(?:change|changed|update|updated|correct|corrected|actually|instead|make it|now it'?s|new amount|rather)\b",
    re.IGNORECASE,
)
_CANCEL = re.compile(r"^(?:never ?mind|cancel|forget it|stop|neither|none|leave it|don'?t|do not|no)\b", re.IGNORECASE)
_FIRST = re.compile(r"\b(?:first|former|the first one)\b", re.IGNORECASE)
_SECOND = re.compile(r"\b(?:second|latter|the second one|other one)\b", re.IGNORECASE)
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "be",
        "it",
        "this",
        "that",
        "my",
        "our",
        "your",
        "with",
        "from",
        "by",
        "me",
        "please",
        "hey",
        "heliox",
        "email",
        "send",
        "put",
        "book",
        "set",
        "remind",
        "reminder",
        "pay",
        "invoice's",
    ]
)

_UNITS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    w: 10 * (i + 2)
    for i, w in enumerate(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])
}


def words_to_number(text: str) -> int | None:
    """ "two thousand four hundred" -> 2400; None when the words are not a number."""
    total = current = 0
    seen = False
    for word in re.findall(r"[a-z]+", text.casefold().replace("-", " ")):
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
        elif word == "and":
            continue
        else:
            return None
        seen = True
    return total + current if seen else None


_NUMBER_WORDS = (
    r"(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)"
    r"(?:[\s-]+|\b))+"  # a word ends at a space, a hyphen, punctuation ("four hundred.") or the end
)
_AMOUNT = re.compile(
    r"(?P<cur>[$€£₹])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{3,}(?:\.\d+)?)"
    rf"|(?P<words>\b{_NUMBER_WORDS})",
    re.IGNORECASE,
)


def amounts(text: str) -> list[tuple[int, int, int]]:
    """(value, start, end) for each amount: three or more digits ("2,400", "4200") or number words ("two thousand
    four hundred"). Years and clock times are left out."""
    out = []
    for match in _AMOUNT.finditer(text):
        if match["num"]:
            value = float(match["num"].replace(",", ""))
            if match["num"].isdigit() and 1900 <= value <= 2100 and not match["cur"]:
                continue  # a year
            out.append((int(value), match.start("cur") if match["cur"] else match.start("num"), match.end()))
        elif match["words"]:
            value = words_to_number(match["words"])
            if value is not None and value >= 100:
                out.append((value, match.start("words"), match.start("words") + len(match["words"].rstrip())))
    return out


_CLAUSE = re.compile(r"[,;.!?]|\band\b|\bbut\b|\bthen\b", re.IGNORECASE)
# "The invoice is 2,400", "the total comes to 2,400": a statement of an amount, which later requests are checked
# against. A request ("email the invoice for 4,200") is what gets checked.
_STATES_AMOUNT = re.compile(
    r"\b(?:is|are|was|were|comes? to|costs?|totals?|will be|'s)(?:\s+(?:now|actually|really|only|just))?\s*$",
    re.IGNORECASE,
)


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    """The words before and after an amount, within its own clause."""
    before = _CLAUSE.split(text[:start])[-1]
    after = _CLAUSE.split(text[end:])[0]
    return before, after


def _states(text: str, start: int) -> bool:
    before, _ = _clause(text, start, start)
    return bool(_STATES_AMOUNT.search(before.rstrip()))


def _subjects(text: str, start: int, end: int) -> frozenset[str]:
    before_text, after_text = _clause(text, start, end)
    before = re.findall(r"[a-z][a-z'-]+", before_text.casefold())[-NEAR_WORDS:]
    after = re.findall(r"[a-z][a-z'-]+", after_text.casefold())[:NEAR_WORDS]
    return frozenset(w for w in before + after if len(w) >= 4 and w not in _STOP and not words_to_number(w))


# "email it for 4,200", "send that for 4,200": the request names no thing, only points back to one.
_POINTS_BACK = re.compile(r"\b(?:it|that|this|them|those|these)\b", re.IGNORECASE)


def _points_back(text: str, start: int, end: int) -> bool:
    before, after = _clause(text, start, end)
    return bool(_POINTS_BACK.search(before) or _POINTS_BACK.search(after))


def _fmt(value: int) -> str:
    return f"{value:,}"


# ── the question and its answer ──


@dataclass(frozen=True)
class Option:
    say: str  # how the question names it: "Monday the sixteenth"
    keys: tuple[str, ...]  # words in an answer that pick it
    replacement: str  # what replaces ``span`` in the request


@dataclass
class Held:
    kind: str  # "weekday_date", "said_vs_said"
    question: str
    request: str  # the request as written
    span: tuple[int, int]  # the part of it the answer replaces
    options: tuple[Option, Option]
    at: float = 0.0
    asked_again: bool = False
    title: str = ""  # for the card
    evidence: str = ""
    card_id: str = ""


@dataclass(frozen=True)
class Resolution:
    kind: str  # "proceed", "cancel", "again"
    say: str = ""
    request: str = ""


ANSWER_MAX_WORDS = 8  # "oh sorry, Thursday the nineteenth please" is an answer; a whole new request is longer
# How a new request starts, as opposed to an answer ("Thursday", "the second one", "make it 2,400").
_REQUEST_START = re.compile(
    r"^(?:hey heliox[,.]?\s*)?(?:please\s+)?(?:remind|put|book|schedule|send|email|add|create|open|move|call|text|"
    r"message|start|write|tell|find|show|what|when|where|who|how|can you|could you|would you|will you)\b",
    re.IGNORECASE,
)


def is_answer(text: str) -> bool:
    """Whether a turn that comes while a question is pending answers it, or is something new.

    A new request is never taken as the answer, even when it happens to name one of the options ("put the review in
    for Monday the 19th at 10" while "Tuesday the seventeenth, or Thursday the nineteenth?" is pending): the person
    moved on, and acting on the old request with a version they never chose would be the worse slip.
    """
    said = sp.to_written(text).strip()
    if _CANCEL.match(said.casefold().strip(" .!?")):
        return True
    if len(re.findall(r"[\w'’,.-]+", said)) > ANSWER_MAX_WORDS:
        return False
    return not _REQUEST_START.match(said)


def resolve(held: Held, answer: str) -> Resolution:
    """What the person's answer decides."""
    text = sp.to_written(answer).casefold().strip(" .!?")
    if _CANCEL.match(text):
        return Resolution("cancel", "Okay, I've left it.")
    picked = [i for i, option in enumerate(held.options) if any(_has(text, key) for key in option.keys)]
    if len(picked) != 1:
        if _FIRST.search(text) and not _SECOND.search(text):
            picked = [0]
        elif _SECOND.search(text) and not _FIRST.search(text):
            picked = [1]
    if len(picked) == 1:
        option = held.options[picked[0]]
        start, end = held.span
        return Resolution("proceed", request=held.request[:start] + option.replacement + held.request[end:])
    if not held.asked_again:
        held.asked_again = True
        first, second = held.options
        return Resolution("again", f"Sorry, {first.say} or {second.say}?")
    return Resolution("cancel", "I've left it. Say it again when you're ready.")


def _has(text: str, key: str) -> bool:
    key = key.casefold()
    if key.replace(",", "").isdigit():
        digits = re.sub(r"(?<=\d),(?=\d{3})", "", text)
        value = words_to_number(text)
        return bool(re.search(rf"(?<![\d.]){re.escape(key.replace(',', ''))}(?![\d])", digits)) or (
            value is not None and str(value) == key.replace(",", "")
        )
    return bool(re.search(rf"\b{re.escape(key)}\b", text))


def _ordinal_keys(day: int) -> tuple[str, ...]:
    return (sp.ordinal_words(day), f"{day}{_suffix(day)}", f"the {day}")


def _suffix(day: int) -> str:
    return "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _month_for(day: int, month: int | None, today: date) -> date | None:
    """The date a weekday-and-day means: in the named month, else the coming one (a request is about the future)."""
    candidates = []
    if month:
        for year in (today.year, today.year + 1):
            try:
                candidate = date(year, month, day)
            except ValueError:
                return None
            if candidate >= today - timedelta(days=30):
                return candidate
        return None
    y, m = today.year, today.month
    for step in range(0, 3):
        mm, yy = (m - 1 + step) % 12 + 1, y + (m - 1 + step) // 12
        try:
            candidates.append(date(yy, mm, day))
        except ValueError:
            continue
    return next((c for c in candidates if c >= today), None)


def weekday_hold(request: str, today: date) -> Held | None:
    """A weekday and a date that do not agree, in a request: ask which was meant."""
    for match in _WEEKDAY_DATE.finditer(request):
        day = int(match["d"])
        month = _MONTHS.index(match["m"].casefold()) + 1 if match["m"] else None
        when = _month_for(day, month, today)
        if when is None:
            continue
        stated = _DAYS.index(match["wd"].casefold())
        if when.weekday() == stated:
            continue
        actual_name, stated_name = calendar.day_name[when.weekday()], calendar.day_name[stated]
        offset = (stated - when.weekday()) % 7
        near = when + timedelta(days=offset if offset <= 3 else offset - 7)
        month_name = calendar.month_name[when.month]
        a = Option(
            f"{stated_name} the {sp.ordinal_words(near.day)}",
            (stated_name, *_ordinal_keys(near.day)),
            f"{stated_name} {near.day} {calendar.month_name[near.month]}",
        )
        b = Option(
            f"{actual_name} the {sp.ordinal_words(day)}",
            (actual_name, *_ordinal_keys(day)),
            f"{actual_name} {day} {month_name}",
        )
        question = f"The {sp.ordinal_words(day)} is a {actual_name}. {a.say}, or {b.say}?"
        return Held(
            "weekday_date",
            question,
            request,
            match.span(),
            (a, b),
            title=f"The {sp.ordinal_words(day)} of {month_name} is a {actual_name}",
            evidence=match.group(0),
        )
    return None


def impossible_hold(request: str, today: date) -> Held | None:
    for match in _LONE_DATE.finditer(request):
        day, month = int(match["d"]), _MONTHS.index(match["m"].casefold()) + 1
        year = today.year if month >= today.month - 2 else today.year + 1
        last = calendar.monthrange(year, month)[1]
        if day <= last or day > 31:
            continue
        name = calendar.month_name[month]
        following = date(year, month, last) + timedelta(days=1)
        nxt = calendar.month_name[following.month]
        a = Option(f"the {sp.ordinal_words(last)} of {name}", (*_ordinal_keys(last), name), f"{last} {name}")
        b = Option(f"the first of {nxt}", ("first of", "1st", nxt), f"1 {nxt}")
        question = f"There's no {sp.ordinal_words(day)} of {name}. {a.say[0].upper() + a.say[1:]}, or {b.say}?"
        held = Held(
            "weekday_date", question, request, match.span(), (a, b), title=f"There's no {day} {name}",
            evidence=match.group(0),
        )  # fmt: skip
        return held
    return None


# ── said against said, across the turns of one conversation ──


@dataclass
class Turn:
    at: float
    text: str


@dataclass
class Turns:
    clock: Callable[[], float] = time.monotonic
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))

    def add(self, text: str) -> None:
        self.turns.append(Turn(self.clock(), text))

    def clear(self) -> None:
        self.turns.clear()

    def recent(self) -> list[Turn]:
        now = self.clock()
        return [turn for turn in self.turns if now - turn.at <= TURNS_SECONDS]

    def check(self, request: str) -> Held | None:
        """An amount in this request that differs from one the person gave for the same thing a moment ago.

        The thing is matched by the words around both amounts ("the invoice for Acme is 2,400", "email the Acme
        invoice for 4,200"), or, when the request only points back ("email it for 4,200", "send that for 4,200"), it
        is the last thing an amount was stated about in this conversation. That is used for this check only: the
        request is never rewritten beyond the version of the amount the person picks.
        """
        if _DELIBERATE.search(request):
            return None
        recent = self.recent()
        for value, start, end in amounts(request):
            if _states(request, start):
                continue  # a statement is the person telling Heliox, not a request to act on
            subjects = _subjects(request, start, end)
            matched = False
            for turn in reversed(recent):
                for earlier, e_start, e_end in amounts(turn.text):
                    if not _states(turn.text, e_start):
                        continue  # only what the person stated is checked against
                    shared = subjects & _subjects(turn.text, e_start, e_end)
                    if not shared:
                        continue
                    matched = True
                    if earlier == value or any(v == value for v, _, _ in amounts(turn.text)):
                        continue  # the same, or both amounts were said before: two things, not a slip
                    return _amount_hold(request, start, end, value, earlier, _thing(turn.text, e_start, shared))
            if matched or not _points_back(request, start, end):
                continue
            last = _last_stated(recent)
            if last is None:
                continue
            turn, earlier, e_start, e_end = last
            if earlier == value or any(v == value for v, _, _ in amounts(turn.text)):
                continue
            if not _looks_like_slip(earlier, value):
                continue  # "it" may be something new; only the same number gone wrong is worth asking about
            thing = _thing(turn.text, e_start, _subjects(turn.text, e_start, e_end))
            return _amount_hold(request, start, end, value, earlier, thing)
        return None


def _looks_like_slip(said: int, now: int) -> bool:
    """The same number gone wrong: its digits swapped (2,400 and 4,200), one digit different (2,400 and 2,500), or a
    zero added or dropped (2,400 and 24,000)."""
    a, b = str(said), str(now)
    if sorted(a) == sorted(b):
        return True
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) == 1:
        return True
    return said * 10 == now or now * 10 == said


def _last_stated(turns: list[Turn]) -> tuple[Turn, int, int, int] | None:
    """The last amount the person stated about something, in these turns."""
    for turn in reversed(turns):
        stated = [(v, s, e) for v, s, e in amounts(turn.text) if _states(turn.text, s)]
        if stated:
            value, start, end = stated[-1]
            return turn, value, start, end
    return None


def _amount_hold(request: str, start: int, end: int, value: int, earlier: int, thing: str) -> Held:
    a = Option(_fmt(earlier), (_fmt(earlier), str(earlier)), _fmt(earlier))
    b = Option(_fmt(value), (_fmt(value), str(value)), request[start:end].strip())
    return Held(
        "said_vs_said",
        f"A moment ago you said {thing} is {_fmt(earlier)}. {_fmt(earlier)} or {_fmt(value)}?",
        request,
        (start, end),
        (a, b),
        title=f"You said {_fmt(earlier)} a moment ago",
        evidence=request[start:end].strip(),
    )


def _thing(text: str, start: int, shared: frozenset[str]) -> str:
    """ "the Acme invoice" from "the invoice for Acme is 2,400", in the person's own words where possible."""
    before = _clause(text, start, start)[0].rstrip()
    before = re.sub(r"\s+(?:is|was|are|were|comes to|costs|for)$", "", before, flags=re.IGNORECASE)
    words = before.split()[-4:]
    phrase = " ".join(words).strip(" ,.")
    if any(word.casefold().strip(",.") in shared for word in words):
        return re.sub(r"(?i)^(?:hey heliox,?\s*)", "", phrase) or "that"
    return "it"


def hold_for(request: str, today: date, turns: Turns) -> Held | None:
    """The one question to ask before acting on ``request``, or None."""
    return turns.check(request) or weekday_hold(request, today) or impossible_hold(request, today)


__all__ = [
    "HOLD_SECONDS",
    "Held",
    "Option",
    "Resolution",
    "Turns",
    "amounts",
    "is_answer",
    "hold_for",
    "impossible_hold",
    "resolve",
    "weekday_hold",
    "words_to_number",
]

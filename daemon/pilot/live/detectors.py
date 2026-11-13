"""Specific, local mistake detectors for the live layer.

Each detector reads a piece of text the person is writing or saying and returns ``Correction`` objects that
name the exact evidence, say what is wrong in one sentence, and carry a fix when there is one. They are plain
rules over the text: no model call, no network, nothing persisted. A detector that is not sure stays quiet;
a false correction costs more trust than a missed one.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

from pilot.live import spoken as sp

logger = logging.getLogger("pilot.live.detectors")

URGENT, NORMAL, LOW = "urgent", "normal", "low"

KINDS = (
    "risky_command",
    "secret_in_text",
    "weekday_date",
    "arithmetic",
    "profile_fact",
    "error_on_screen",
    "missing_attachment",
    "repeated_word",
    "personal_fact",  # pilot.live.grounded: a local model checks the text against the person's own data
    "community_catch",  # pilot.live.catches: rules from catch packs that passed the precision gate
    "stuck",  # pilot.live.stuck: offers help when the person seems stuck (not a detector over one text)
    "commitment",  # pilot.live.commitments: a promise with a deadline, with an offer to set a reminder
    "said_vs_typed",  # pilot.live.crosscheck: a date said to Heliox against the date just typed for the same thing
    "past_date",  # a date that has already passed, written as if it were still ahead
    "near_name",  # a greeting to someone the person told Heliox about, with their name one letter off
)

KIND_LABELS = {
    "risky_command": "Risky commands before you run them",
    "secret_in_text": "Keys and passwords pasted into messages",
    "weekday_date": "Weekdays, dates and times that can't be right",
    "arithmetic": "Arithmetic slips",
    "profile_fact": "Names that contradict what you told me",
    "error_on_screen": "Errors on screen, with the fix",
    "missing_attachment": "Mentioning an attachment",
    "repeated_word": "Doubled words",
    "personal_fact": "Things that contradict your calendar, to-dos or contacts",
    "community_catch": "Catches from catch packs",
    "stuck": "Offers to help when you seem stuck",
    "commitment": "Promises you make, with a reminder",
    "said_vs_typed": "A date you say that differs from the one you just wrote",
    "past_date": "A date that has already passed, written as if it were ahead",
    "near_name": "A name one letter off someone you told me about",
}
# Kinds found by a model rather than a rule: still there while their quoted evidence is in the text.
MODEL_KINDS = frozenset({"personal_fact"})


@dataclass(frozen=True)
class Correction:
    kind: str
    key: str
    title: str
    message: str
    evidence: str
    detail: str = ""
    fix: str = ""
    fix_kind: str = ""  # "replace" (corrected text), "command" (safer command) or "ask" (a request for Heliox)
    urgency: str = NORMAL
    confidence: float = 0.9
    # The evidence ends where the text ends: the person may still be typing it, so it needs a longer pause.
    trailing: bool = False


@dataclass
class TextContext:
    """What the detectors know besides the text itself."""

    today: date
    app: str = ""
    title: str = ""
    source: str = "focused_field"  # "focused_field", "screen_ocr" or "speech"
    people: dict[str, str] = field(default_factory=dict)  # relation -> the one name the person gave
    # The local hour, for "today" and "tomorrow" (after midnight people still mean the day they are finishing).
    # None: the clock's hour when ``today`` is the real today, otherwise midday.
    hour: int | None = None
    # What a speech recogniser's transcript was turned into (pilot.live.spoken); ``detect`` sets it for speech.
    spoken: sp.Spoken | None = None

    @property
    def is_speech(self) -> bool:
        return self.source == "speech"

    def said(self, text: str, start: int, end: int) -> str:
        """The words the person said for ``text[start:end]``, where ``text`` is what the detectors read."""
        return self.spoken.said(start, end) if self.spoken is not None else text[start:end].strip()

    @property
    def where(self) -> str:
        return f"{self.app} {self.title}".casefold()

    @property
    def is_terminal(self) -> bool:
        return any(word in self.where for word in _TERMINAL_WORDS)

    @property
    def is_code_editor(self) -> bool:
        return any(word in self.where for word in _EDITOR_WORDS)

    @property
    def is_mail_compose(self) -> bool:
        where = self.where
        mail_app = any(word in where for word in ("gmail", "outlook", "thunderbird", "mail", "proton"))
        composing = any(word in where for word in ("compose", "new message", "draft", "reply", "re:", "fwd", "message"))
        return mail_app and composing


_TERMINAL_WORDS = (
    "terminal", "powershell", "cmd.exe", "command prompt", "windowsterminal", "bash", "zsh", "wsl", "ubuntu",
    "iterm", "alacritty", "wezterm", "conemu", "warp", "mintty", "git bash",
)  # fmt: skip
_EDITOR_WORDS = (
    "visual studio code", "code.exe", "cursor", "pycharm", "intellij", "webstorm", "sublime", "notepad++",
    "vim", "neovim", "emacs", "zed", "android studio", "rider", "clion",
)  # fmt: skip


def _at_end(text: str, end: int) -> bool:
    return not text[end:].strip()


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _on(day: int, month: int, ctx: TextContext) -> str:
    """A date as the card says it: "12 November" on screen, "the twelfth of November" aloud."""
    name = calendar.month_name[month]
    return sp.date_words(day, name) if ctx.is_speech else f"{day} {name}"


def _wrote(ctx: TextContext) -> str:
    return "said" if ctx.is_speech else "wrote"


# ── Weekday that doesn't match its date ───────────────────────────────────────────────────────────────────────────

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_RE = r"(?P<wd>mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"
_MONTH_RE = (
    r"(?P<m>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_PATTERNS = (
    # "Tuesday, 14 November 2026", "Tue 14th Nov", "Tuesday the 14th of November"
    re.compile(
        rf"\b{_DAY_RE}\.?,?\s+(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTH_RE}\b\.?"
        rf"(?:,?\s+(?P<y>\d{{4}}))?",
        re.IGNORECASE,
    ),
    # "Tuesday, November 14, 2026", "Tue Nov 14"
    re.compile(
        rf"\b{_DAY_RE}\.?,?\s+{_MONTH_RE}\.?\s+(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b"
        rf"(?:,?\s+(?P<y>\d{{4}}))?",
        re.IGNORECASE,
    ),
)


def _month_number(token: str) -> int:
    token = token[:3].casefold()
    return [m[:3].casefold() for m in calendar.month_name[1:]].index(token) + 1


def _weekday_number(token: str) -> int:
    token = token[:3].casefold()
    return [d[:3] for d in _DAYS].index(token)


def _resolve_year(day: int, month: int, today: date) -> int | None:
    """The year a yearless date most likely means: this year, or next year once it is well past."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today - timedelta(days=60) or year == today.year + 1:
            return year
    return None


def weekday_date(text: str, ctx: TextContext) -> list[Correction]:
    """A weekday that doesn't match its date, and dates and clock times that cannot exist."""
    found: list[Correction] = []
    seen: set[tuple[int, int]] = set()
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            if match.span() in seen:
                continue
            seen.add(match.span())
            stated = _weekday_number(match["wd"])
            day, month = int(match["d"]), _month_number(match["m"])
            month_name = calendar.month_name[month]
            year = int(match["y"]) if match["y"] else _resolve_year(day, month, ctx.today)
            evidence = match.group(0).strip()
            if year is None or day > calendar.monthrange(year, month)[1] or day < 1:
                if 1 <= day <= 31:
                    found.append(_no_such_date(evidence, day, month, int(match["y"]) if match["y"] else None, ctx))
                continue
            actual = date(year, month, day)
            if actual.weekday() == stated:
                continue
            actual_name, stated_name = _DAYS[actual.weekday()].title(), _DAYS[stated].title()
            # The nearest date that really is the weekday they wrote, in case the date is the slip.
            offset = (stated - actual.weekday()) % 7
            forward = actual + timedelta(days=offset)
            backward = forward - timedelta(days=7)
            nearest = forward if offset <= 3 else backward
            corrected = re.sub(re.escape(match["wd"]), _match_case(actual_name, match["wd"]), evidence, count=1)
            found.append(
                Correction(
                    kind="weekday_date",
                    key=f"weekday_date:{evidence.casefold()}",
                    title=_cap(f"{_on(day, month, ctx)} is a {actual_name}"),
                    message=f"You {_wrote(ctx)} {stated_name}.",
                    evidence=evidence,
                    detail=f"If you meant {'' if ctx.is_speech else 'the '}{stated_name}, that's {_on(nearest.day, nearest.month, ctx)}.",
                    fix=corrected,
                    fix_kind="replace",
                    urgency=NORMAL,
                    confidence=0.97,
                    trailing=_at_end(text, match.end()),
                )
            )
    if ctx.is_terminal or ctx.is_code_editor:
        return found  # code and logs are full of test dates, timestamps and ratios that are not claims
    found.extend(_impossible_dates(text, ctx, seen))
    found.extend(_monthless_weekdays(text, ctx, seen))
    found.extend(_numeric_weekdays(text, ctx))
    found.extend(_relative_days(text, ctx))
    found.extend(_clock_times(text, ctx))
    return found


# ── A weekday with a day of the month and no month: "Tuesday the 12th" ──

# The ordinal ending is required, so "Tuesday 12 people" or "Friday 3 pm" are never dates. A month right after it
# belongs to the full-date patterns above.
_MONTHLESS = re.compile(
    rf"\b{_DAY_RE}\.?,?\s+(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)\b(?!\s+(?:of\s+)?{_MONTH_RE}\b)",
    re.IGNORECASE,
)


def _occurrences(day: int, today: date) -> list[date]:
    """The last time this day of the month came round, and the next two times it will."""
    out: list[date] = []
    year, month = today.year, today.month
    for step in range(-1, 3):
        m = month + step
        y = year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        try:
            out.append(date(y, m, day))
        except ValueError:
            continue
    past = [d for d in out if d < today][-1:]
    ahead = [d for d in out if d >= today][:2]
    return past + ahead


def _monthless_weekdays(text: str, ctx: TextContext, covered: set[tuple[int, int]]) -> list[Correction]:
    """ "Tuesday the 12th" when no 12th near today, past or coming, is a Tuesday.

    People leave the month out when they mean the nearest one, so the weekday is checked against the last 12th and
    the next two. Only when it matches none of them is it a slip; the card names the coming one.
    """
    found: list[Correction] = []
    for match in _MONTHLESS.finditer(text):
        if any(match.start() < end and start < match.end() for start, end in covered):
            continue
        day = int(match["d"])
        if not 1 <= day <= 31:
            continue
        stated = _weekday_number(match["wd"])
        dates = _occurrences(day, ctx.today)
        if not dates or any(d.weekday() == stated for d in dates):
            continue
        coming = next((d for d in dates if d >= ctx.today), dates[-1])
        actual_name, stated_name = _DAYS[coming.weekday()].title(), _DAYS[stated].title()
        offset = (stated - coming.weekday()) % 7
        nearest = coming + timedelta(days=offset) if offset <= 3 else coming + timedelta(days=offset - 7)
        evidence = match.group(0).strip()
        found.append(
            Correction(
                kind="weekday_date",
                key=f"weekday_date:{evidence.casefold()}",
                title=_cap(f"{_on(coming.day, coming.month, ctx)} is a {actual_name}"),
                message=f"You {_wrote(ctx)} {stated_name}.",
                evidence=evidence,
                detail=(
                    f"If you meant {'' if ctx.is_speech else 'the '}{stated_name}, that's "
                    f"{_on(nearest.day, nearest.month, ctx)}."
                ),
                fix=re.sub(re.escape(match["wd"]), _match_case(actual_name, match["wd"]), evidence, count=1),
                fix_kind="replace",
                urgency=NORMAL,
                confidence=0.9,
                trailing=_at_end(text, match.end()),
            )
        )
    return found


# ── Dates that do not exist ──

# Only the months shorter than 31 days can be wrong, so "may" and "march" as ordinary words never matter here.
_SHORT_MONTH_RE = r"(?P<m>feb(?:ruary)?|apr(?:il)?|june?|sep(?:t(?:ember)?)?|nov(?:ember)?)"
_LONE_DATES = (
    # "31 November", "the 31st of November 2026", "30 Feb"
    re.compile(
        rf"(?<![\w/.:,-])(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+{_SHORT_MONTH_RE}\b(?![-/]\w)\.?"
        rf"(?:,?\s+(?P<y>\d{{4}})\b)?",
        re.IGNORECASE,
    ),
    # "November 31", "February 30th, 2027"
    re.compile(
        rf"\b{_SHORT_MONTH_RE}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?![/.:]\d)(?:,?\s+(?P<y>\d{{4}})\b)?",
        re.IGNORECASE,
    ),
)
# Someone writing about the impossible date rather than making a slip: "there's no 31 November".
_DATE_NEGATION_BEFORE = re.compile(
    r"\b(?:no|not|never|isn'?t|aren'?t|wasn'?t|doesn'?t|invalid|impossible|non-?existent|fake)\b[^.!?\n]{0,12}$",
    re.IGNORECASE,
)
_LEAP_TALK = re.compile(r"\bleap\s+year", re.IGNORECASE)
_DATE_NEGATION_AFTER = re.compile(r"^\W{0,3}(?:doesn'?t|does not|isn'?t|is not|never)\b", re.IGNORECASE)


def _plausible_year(month: int, today: date) -> int:
    """The year a yearless month most likely means: this year, unless that month is well past."""
    return today.year if month >= today.month - 2 else today.year + 1


def _no_such_date(evidence: str, day: int, month: int, stated_year: int | None, ctx: TextContext) -> Correction:
    month_name = calendar.month_name[month]
    speech = ctx.is_speech
    if month == 2 and day == 29:
        year = stated_year or _plausible_year(month, ctx.today)
        if speech:
            title = f"There's no {sp.date_words(29, month_name, article=False)} {sp.year_words(year)}"
            message = f"{sp.year_words(year)} isn't a leap year."
        else:
            title = f"There's no 29 February {year}"
            message = f"{year} isn't a leap year."
    else:
        days = calendar.monthrange(stated_year or _plausible_year(month, ctx.today), month)[1]
        if speech:
            title = f"There's no {sp.date_words(day, month_name, article=False)}"
            message = f"{month_name} has {sp.cardinal_words(days)} days."
        else:
            title = f"There's no {day} {month_name}"
            message = f"{month_name} has {days} days."
    return Correction(
        kind="weekday_date",
        key=f"weekday_date:{evidence.casefold()}",
        title=title,
        message=_cap(message),
        evidence=evidence,
        urgency=NORMAL,
    )


def _impossible_dates(text: str, ctx: TextContext, covered: set[tuple[int, int]]) -> list[Correction]:
    found: list[Correction] = []
    keys: set[str] = set()
    for pattern in _LONE_DATES:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in covered):
                continue  # already checked together with its weekday
            day, month = int(match["d"]), _month_number(match["m"])
            if not 29 <= day <= 31:
                continue
            year = int(match["y"]) if match["y"] else _plausible_year(month, ctx.today)
            if day <= calendar.monthrange(year, month)[1]:
                continue
            if _DATE_NEGATION_BEFORE.search(text[max(0, match.start() - 40) : match.start()]):
                continue
            if _DATE_NEGATION_AFTER.search(text[match.end() : match.end() + 20]):
                continue
            if day == 29 and month == 2 and _LEAP_TALK.search(text[max(0, match.start() - 60) : match.end() + 60]):
                continue  # "29 February in a year that isn't a leap year": about the rule, not a date
            evidence = match.group(0).strip()
            correction = _no_such_date(evidence, day, month, int(match["y"]) if match["y"] else None, ctx)
            if correction.key not in keys:
                keys.add(correction.key)
                found.append(replace(correction, trailing=_at_end(text, match.end())))
    return found


# ── A weekday next to a numeric date ──

_NUMERIC_DATE = re.compile(
    rf"\b{_DAY_RE}\.?,?\s+(?:the\s+)?(?P<a>\d{{1,2}})/(?P<b>\d{{1,2}})(?:/(?P<y>\d{{4}}|\d{{2}}))?(?![\w/:%]|\.\d)",
    re.IGNORECASE,
)
# What may follow such a date. Anything else ("Friday 3/4 of the team", "Monday 1/2 day") reads as a fraction.
_AFTER_NUMERIC_DATE = re.compile(
    r"^(?:\s*$|\s*[,.;:!?)]|\s+(?:at|@|from|to|until|till|by|before|after|works|morning|afternoon|evening|night|"
    r"so|but|if|unless|then|please|we|i|you|they)\b"
    r"|\s+\d)",
    re.IGNORECASE,
)


def _numeric_weekdays(text: str, ctx: TextContext) -> list[Correction]:
    found: list[Correction] = []
    for match in _NUMERIC_DATE.finditer(text):
        token = match["wd"]
        if len(token) < 6 and not token[:1].isupper():
            continue  # "sat", "wed" and "sun" are also ordinary words
        if not _AFTER_NUMERIC_DATE.match(text[match.end() :]):
            continue
        a, b = int(match["a"]), int(match["b"])
        stated_year = None
        if match["y"]:
            stated_year = int(match["y"]) + (2000 if len(match["y"]) == 2 else 0)
        # Every reading that is a real date: day/month and month/day.
        readings: dict[date, str] = {}
        for day, month in ((a, b), (b, a)):
            if not 1 <= month <= 12 or day < 1:
                continue
            year = stated_year or _resolve_year(day, month, ctx.today)
            if year is None or day > calendar.monthrange(year, month)[1]:
                continue
            readings[date(year, month, day)] = _on(day, month, ctx)
        stated = _weekday_number(token)
        if not readings or any(when.weekday() == stated for when in readings):
            continue
        stated_name = _DAYS[stated].title()
        numeric = f"{match['a']}/{match['b']}"
        evidence = match.group(0).strip()
        common = {
            "kind": "weekday_date",
            "key": f"weekday_date:{evidence.casefold()}",
            "evidence": evidence,
            "urgency": NORMAL,
            "trailing": _at_end(text, match.end()),
        }
        if len(readings) == 1:
            [(when, spoken)] = readings.items()
            actual_name = _DAYS[when.weekday()].title()
            found.append(
                Correction(
                    title=_cap(f"{spoken} is a {actual_name}"),
                    message=f"You {_wrote(ctx)} {stated_name}.",
                    fix=evidence.replace(token, _match_case(actual_name, token), 1),
                    fix_kind="replace",
                    confidence=0.95,
                    **common,
                )
            )
        else:
            (first, first_name), (second, second_name) = sorted(readings.items())
            found.append(
                Correction(
                    title=f"That date isn't a {stated_name}" if ctx.is_speech else f"{numeric} isn't a {stated_name}",
                    message=_cap(
                        f"{first_name} is a {_DAYS[first.weekday()].title()} and {second_name} is a "
                        f"{_DAYS[second.weekday()].title()}."
                    ),
                    confidence=0.9,
                    **common,
                )
            )
    return found


# ── "Tomorrow, Thursday" when tomorrow is a Wednesday ──

_FULL_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
# "tomorrow (Thursday)", "tomorrow, Thursday", "today is Monday"
_RELATIVE = re.compile(
    rf"\b(?P<rel>today|tomorrow|yesterday)"
    rf"(?:\s*\(\s*(?P<wd>{_FULL_DAYS})\s*\)|(?:\s*,\s*|\s+[-–]\s+|\s+(?:is|was)\s+)(?P<wd2>{_FULL_DAYS})\b)",
    re.IGNORECASE,
)
_RELATIVE_OFFSET = {"yesterday": -1, "today": 0, "tomorrow": 1}
# Hypotheticals, alternatives and "the day after tomorrow" do not claim what day it is.
_RELATIVE_NOT_A_CLAIM_BEFORE = re.compile(
    r"(?:\b(?:if|whether|when|unless|assuming|suppose|thought|thinking|forgot|like|not|instead of|rather than|"
    r"either|or|and|after|before)|n't|[,&/])\s*$",
    re.IGNORECASE,
)
# After a comma or dash form, the weekday must end the phrase: "tomorrow, Thursday and Friday" is a list.
_RELATIVE_END = re.compile(
    r"^(?:\s*$|\s*[.!;:)]|,?\s+(?:at|@|from|morning|afternoon|evening|night|first thing|by)\b|\s+(?:the\s+)?\d)",
    re.IGNORECASE,
)


def _local_hour(ctx: TextContext) -> int:
    if ctx.hour is not None:
        return ctx.hour
    return datetime.now().hour if ctx.today == date.today() else 12


def _dated(text: str, start: int, ctx: TextContext) -> date | None:
    """The date written right after a weekday ("Thursday 12 Nov"), when it really is that weekday."""
    for pattern in _DATE_PATTERNS:
        match = pattern.match(text, start)
        if match is None:
            continue
        day, month = int(match["d"]), _month_number(match["m"])
        year = int(match["y"]) if match["y"] else _resolve_year(day, month, ctx.today)
        if year is None or not 1 <= day <= calendar.monthrange(year, month)[1]:
            return None
        written = date(year, month, day)
        return written if written.weekday() == _weekday_number(match["wd"]) else None
    return None


def _from_today(day: date, today: date, speech: bool = False) -> str:
    days = (day - today).days
    named = {-1: "yesterday", 0: "today", 1: "tomorrow", 2: "the day after tomorrow"}
    if days in named:
        return named[days]
    count = sp.cardinal_words(abs(days)) if speech else str(abs(days))
    return f"in {count} days" if days > 0 else f"{count} days ago"


def _relative_days(text: str, ctx: TextContext) -> list[Correction]:
    if _local_hour(ctx) < 5:
        return []  # at 1 a.m. "tomorrow" usually means after some sleep, not the calendar's tomorrow
    found: list[Correction] = []
    for match in _RELATIVE.finditer(text):
        token = match["wd"] or match["wd2"]
        rest = text[match.end() :]
        if rest.startswith("'") or rest.startswith("’") or re.match(r"\s*\?", rest):
            continue  # "today is Monday's deadline", "tomorrow is Thursday?"
        if match["wd2"] and not _RELATIVE_END.match(rest):
            continue
        if _RELATIVE_NOT_A_CLAIM_BEFORE.search(text[max(0, match.start() - 20) : match.start()]):
            continue
        rel = match["rel"].casefold()
        when = ctx.today + timedelta(days=_RELATIVE_OFFSET[rel])
        stated = _weekday_number(token)
        if when.weekday() == stated:
            continue
        actual_name, stated_name = _DAYS[when.weekday()].title(), _DAYS[stated].title()
        verb = "was" if rel == "yesterday" else "is"
        offset = (stated - ctx.today.weekday()) % 7 or 7
        meant = ctx.today + timedelta(days=offset)
        evidence = match.group(0).strip()
        group = "wd" if match["wd"] else "wd2"
        start, end = match.start(group) - match.start(), match.end(group) - match.start()
        detail = "" if rel == "yesterday" else f"The next {stated_name} is {_on(meant.day, meant.month, ctx)}."
        written = _dated(text, match.start(group), ctx)
        if written is not None:
            # "Tomorrow, Thursday 12 Nov" where 12 Nov is a Thursday: the day and date agree, "tomorrow" is the slip.
            # Swapping in tomorrow's weekday would make the date wrong, so the fix drops the relative word.
            on = _on(written.day, written.month, ctx)
            found.append(
                Correction(
                    kind="weekday_date",
                    key=f"weekday_date:{evidence.casefold()}",
                    title=_cap(f"{on} {'wasn' if rel == 'yesterday' else 'isn'}'t {rel}"),
                    message=f"{rel.title()} {verb} {actual_name} {_on(when.day, when.month, ctx)}.",
                    evidence=evidence,
                    detail=f"{stated_name} {on} is {_from_today(written, ctx.today, ctx.is_speech)}.",
                    fix=evidence[start:end],
                    fix_kind="replace",
                    urgency=NORMAL,
                    confidence=0.9,
                    trailing=_at_end(text, match.end()),
                )
            )
            continue
        found.append(
            Correction(
                kind="weekday_date",
                key=f"weekday_date:{evidence.casefold()}",
                title=f"{rel.title()} {verb} {actual_name}",
                message=f"You {_wrote(ctx)} {stated_name}.",
                evidence=evidence,
                detail=detail,
                fix=evidence[:start] + _match_case(actual_name, token) + evidence[end:],
                fix_kind="replace",
                urgency=NORMAL,
                confidence=0.9,
                trailing=_at_end(text, match.end()),
            )
        )
    return found


# ── Clock times that do not exist ──

# "14pm": a 24-hour hour with a 12-hour suffix. With a space only lower-case or dotted, so "14 PM" (people) is left.
_MIXED_CLOCK = re.compile(
    r"(?<![\w:.,])(?P<h>1[3-9]|2[0-3])(?::(?P<mm>[0-5]\d))?(?:(?P<ap>[AaPp][Mm]|[ap]\.m\.)|\s(?P<ap2>[ap]m|[ap]\.m\.))"
    r"(?!\w)",
)
# "at 25:00", "until 13:75": only after a word that makes it a time of day, never a duration or a ratio.
_CLOCK = re.compile(
    r"(?:\b(?:at|by|from|until|till|before|after|around|about)\s+|@\s*)(?P<t>(?P<h>\d{1,2}):(?P<mm>\d{2}))"
    r"(?![\d:]|\.\d|\s*(?:min|mins|minutes|sec|secs|seconds|into|in|mark|left|remaining)\b)",
    re.IGNORECASE,
)


def _clock_times(text: str, ctx: TextContext) -> list[Correction]:
    speech = ctx.is_speech
    mixed_message = (
        "It mixes the twenty-four hour and twelve hour clocks."
        if speech
        else "It mixes the 24-hour and 12-hour clocks."
    )
    found: list[Correction] = []
    for match in _MIXED_CLOCK.finditer(text):
        hour = int(match["h"])
        minutes = match["mm"]
        spelled = match["ap"] or match["ap2"]
        evidence = match.group(0).strip()
        common = {
            "kind": "weekday_date",
            "key": f"weekday_date:{evidence.casefold()}",
            "evidence": evidence,
            "urgency": NORMAL,
            "trailing": _at_end(text, match.end()),
        }
        minute = int(minutes) if minutes else 0
        said_ap = "p m" if spelled[:1].casefold() == "p" else "a m"
        if spelled[:1].casefold() == "p":
            twelve = f"{hour - 12}{':' + minutes if minutes else ''}"
            gap = " " if match["ap2"] else ""
            found.append(
                Correction(
                    title=(
                        _cap(f"{sp.time_words(hour, minute)} {said_ap} is {sp.time_words(hour - 12, minute)} p m")
                        if speech
                        else f"{evidence} is {twelve}{gap}{spelled}"
                    ),
                    message=mixed_message,
                    fix=f"{twelve}{gap}{spelled}",
                    fix_kind="replace",
                    confidence=0.9,
                    **common,
                )
            )
        else:
            found.append(
                Correction(
                    title=f"There's no {sp.time_words(hour, minute)} {said_ap}" if speech else f"There's no {evidence}",
                    message=mixed_message,
                    confidence=0.9,
                    **common,
                )
            )
    for match in _CLOCK.finditer(text):
        hour, minutes = int(match["h"]), int(match["mm"])
        shown = match["t"]
        if minutes >= 60:
            message = "An hour has sixty minutes." if speech else "An hour has 60 minutes."
        elif hour >= 25 or (hour == 24 and minutes > 0):
            message = "A day ends at twenty-three fifty-nine." if speech else "A day ends at 23:59."
        else:
            continue
        found.append(
            Correction(
                kind="weekday_date",
                key=f"weekday_date:{shown}",
                title=f"There's no {sp.time_words(hour, minutes)}" if speech else f"There's no {shown}",
                message=message,
                evidence=shown,
                urgency=NORMAL,
                confidence=0.9,
                trailing=_at_end(text, match.end()),
            )
        )
    return found


def _ordinal(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _match_case(word: str, like: str) -> str:
    if like.isupper():
        return word.upper()
    if len(like) <= 4:  # an abbreviation like "Tue" gets "Sat"
        return word[: len(like.rstrip("."))]
    return word if like[:1].isupper() else word.casefold()


# ── Arithmetic slips ──────────────────────────────────────────────────────────────────────────────────────────────

_NUM = r"-?\d+(?:\.\d+)?"
_ARITH = re.compile(
    rf"(?<![\w.,/:-])(?P<a>{_NUM})\s*(?P<op>[+\-*/×÷]|\s[xX]\s)\s*(?P<b>{_NUM})\s*=\s*(?P<c>{_NUM})(?![\w.%/:])"
)
_PERCENT = re.compile(
    rf"(?<![\w.])(?P<p>{_NUM})\s*%\s*of\s*(?P<base>{_NUM})\s*(?:=|is|equals)\s*(?P<c>{_NUM})(?![\w.%])",
    re.IGNORECASE,
)
# "20% of 150, so 35", "20% of 150, which is 35", "20% of 150 would be 35". The result has to end the claim:
# "so 35 more" or "so 35 + tax" is a different sum, and "so about 35" is not exact.
_PERCENT_SAID = re.compile(
    rf"(?<![\w.])(?P<p>{_NUM})\s*%\s*of\s*(?P<base>{_NUM})"
    r"(?:,?\s+so(?:\s+that['’]s)?|,?\s+which\s+(?:is|makes)|,?\s+that['’]s|,?\s+that\s+is|\s+would\s+be|\s+comes\s+to"
    r"|\s+works\s+out\s+(?:to|at)|\s*(?:->|→|=>))"
    rf"\s*(?P<c>{_NUM})(?![\w%/:]|\.\d)"
    r"(?!\s*(?:more|less|fewer|extra|left|remaining|others?|plus|minus|[-+*/×÷]))",
    re.IGNORECASE,
)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _fmt(value: float, decimals: int) -> str:
    if decimals == 0 and abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.{max(decimals, 2)}f}".rstrip("0").rstrip(".")


def _you_said(text: str, match: re.Match[str], ctx: TextContext) -> str:
    """The result they gave: "You wrote 170." on screen, "You said one seventy." aloud."""
    if ctx.is_speech:
        return f"You said {ctx.said(text, match.start('c'), match.end('c'))}."
    return f"You wrote {match['c']}."


def arithmetic(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.is_code_editor or ctx.is_terminal:
        return []  # code is full of things that look like sums and are not claims
    found: list[Correction] = []
    for match in _ARITH.finditer(text):
        a, b, c = float(match["a"]), float(match["b"]), float(match["c"])
        op = match["op"].strip().casefold()
        if op in ("+",):
            value, symbol = a + b, "+"
        elif op in ("-",):
            value, symbol = a - b, "−"
        elif op in ("*", "×", "x"):
            value, symbol = a * b, "×"
        else:
            if b == 0:
                continue
            value, symbol = a / b, "÷"
        places = _decimals(match["c"])
        if round(value, places) == round(c, places):
            continue
        if symbol == "÷" and abs(value - c) < 10 ** (-places) * 0.51:
            continue
        shown = _fmt(value, places)
        evidence = match.group(0).strip()
        left, right = _fmt(a, _decimals(match["a"])), _fmt(b, _decimals(match["b"]))
        title = f"{left} {symbol} {right} is {shown}"
        if ctx.is_speech:
            word = {"+": "plus", "−": "minus", "×": "times", "÷": "divided by"}[symbol]
            title = _cap(f"{sp.number_words(left)} {word} {sp.number_words(right)} is {sp.number_words(shown)}")
        found.append(
            Correction(
                kind="arithmetic",
                key=f"arithmetic:{evidence}",
                title=title,
                message=_you_said(text, match, ctx),
                evidence=evidence,
                fix=evidence[: match.start("c") - match.start()] + shown,
                fix_kind="replace",
                urgency=NORMAL,
                confidence=0.99,
                trailing=_at_end(text, match.end()),
            )
        )
    percents = [*_PERCENT.finditer(text), *_PERCENT_SAID.finditer(text)]
    for match in percents:
        p, base, c = float(match["p"]), float(match["base"]), float(match["c"])
        value = p * base / 100
        places = _decimals(match["c"])
        if abs(value - c) <= 0.5 * 10**-places + 1e-9:
            continue  # right, or rounded the way people round
        shown = _fmt(value, places)
        evidence = match.group(0).strip()
        title = f"{match['p']}% of {match['base']} is {shown}"
        if ctx.is_speech:
            title = _cap(
                f"{sp.number_words(match['p'])} percent of {sp.number_words(match['base'])} is {sp.number_words(shown)}"
            )
        found.append(
            Correction(
                kind="arithmetic",
                key=f"arithmetic:{evidence}",
                title=title,
                message=_you_said(text, match, ctx),
                evidence=evidence,
                fix=evidence[: match.start("c") - match.start()] + shown,
                fix_kind="replace",
                urgency=NORMAL,
                confidence=0.99,
                trailing=_at_end(text, match.end()),
            )
        )
    return found


# ── Risky commands, caught before Enter ──────────────────────────────────────────────────────────────────────────

_RISKY: tuple[tuple[str, re.Pattern[str], str, str, str], ...] = (
    (
        "rm_root",
        re.compile(r"\brm\s+(?:-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\s+(?:--no-preserve-root\s+)?(?:/|~|/\*|\*)(?=\s|$)", re.I),
        "That deletes the whole drive or your home folder",
        "rm -rf has no undo.",
        "",
    ),
    (
        "force_push",
        re.compile(r"\bgit\s+push\b[^\n]*?(?:\s--force(?!-with-lease)\b|\s-f\b)", re.I),
        "That can overwrite teammates' commits",
        "Use --force-with-lease, which stops if someone else pushed first.",
        "force_with_lease",
    ),
    (
        "reset_hard",
        re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
        "That throws away your changes",
        "git stash would keep them.",
        "",
    ),
    (
        "clean_fdx",
        re.compile(r"\bgit\s+clean\s+-[a-z]*f[a-z]*d[a-z]*x|\bgit\s+clean\s+-[a-z]*x[a-z]*f", re.I),
        "That deletes ignored files too",
        "That includes your .env and build caches.",
        "",
    ),
    (
        "drop_table",
        re.compile(r"\bdrop\s+(?:table|database|schema)\s+(?!if\s+exists\s+\w+_tmp)\S+", re.I),
        "That deletes it for good",
        "Make sure you have a backup first.",
        "",
    ),
    (
        "delete_all_rows",
        re.compile(r"\bdelete\s+from\s+(?P<t>[\w.\"`\[\]]+)\s*;", re.I),
        "That deletes every row",
        "It has no WHERE clause.",
        "",
    ),
    (
        "update_all_rows",
        re.compile(r"\bupdate\s+[\w.\"`\[\]]+\s+set\s+(?:(?!\bwhere\b)[^;])*;", re.I),
        "That updates every row",
        "It has no WHERE clause.",
        "",
    ),
    (
        "chmod_777",
        re.compile(r"\bchmod\s+(?:-R\s+)?777\b", re.I),
        "That makes it writable by anyone",
        "755 for folders or 644 for files is usually enough.",
        "",
    ),
    (
        "pipe_to_shell",
        re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b|\biwr\b[^\n|]*\|\s*iex\b", re.I),
        "That runs a script you haven't read",
        "Download it first and read it before you run it.",
        "",
    ),
    (
        "format_drive",
        re.compile(r"\bformat\s+[a-z]:(?=\s|$)|\bmkfs(?:\.\w+)?\s+/dev/|\bdd\s+if=\S+\s+of=/dev/(?:sd|nvme|disk)", re.I),
        "That wipes a whole drive",
        "Check the drive it names before you press Enter.",
        "",
    ),
)  # fmt: skip

# ── Recursive deletes aimed at something broad: a drive, a home folder, Documents, a system folder ──
#
# The command alone says little: "rm -rf ./build" and "Remove-Item -Recurse .\dist" are routine. What makes a delete
# a likely slip is its target, so this reads the arguments and only speaks when one of them is broad.

# Where a command can start: the start of the line, after a prompt ("PS C:\repo>", "C:\Users\me>", "me@box:~$"),
# or after ";", "&&", "||", "|" or "(". Mid-sentence mentions in prose are left alone.
_DELETE_START = re.compile(
    r"(?:^\s*|^\s*(?:PS\s[^>\n]*>|[^\s>]*[>$#])\s*|[;&|(]\s*)(?:sudo\s+)?(?P<cmd>rm|remove-item|ri|del|erase|rd|rmdir)(?=\s)",
    re.IGNORECASE,
)
_HOME_VARS = r"(?:~|\$home|\$\{home\}|\$env:userprofile|\$env:home|%userprofile%)"
_KNOWN_FOLDERS = ("Documents", "Desktop", "Downloads", "Pictures", "Videos", "Music", "OneDrive")
_KNOWN_RE = "|".join(_KNOWN_FOLDERS)
_SYSTEM_DIRS = re.compile(
    r"[a-z]:\\(?:windows|program files|program files \(x86\)|programdata)|/(?:etc|usr|bin|sbin|lib|lib64|boot|var)",
    re.IGNORECASE,
)
_PS_PATH_PARAMS = ("path", "literalpath", "lp", "pspath")
_PS_VALUE_PARAMS = ("include", "exclude", "filter", "credential", "stream")


def _broad_target(token: str) -> tuple[str, bool] | None:
    """What a delete target covers, as a noun phrase, and whether it was a wildcard; None when it is specific."""
    raw = token.strip("\"'")
    wildcard = False
    star = re.fullmatch(r"(.*?)[\\/]\*(?:\.\*)?", raw)
    if star:
        raw, wildcard = star[1] or "/", True
    path = raw if raw in ("/", "\\") else raw.rstrip("\\/")
    if re.match(r"[a-z]:", path, re.I):
        path = path.replace("/", "\\")
    if re.fullmatch(r"[a-z]:", path, re.I):
        return f"everything on {path.upper()}", wildcard
    if path == "/":
        return "everything on the drive", wildcard
    if re.fullmatch(_HOME_VARS, path, re.I):
        return "your home folder", wildcard
    known = re.fullmatch(rf"{_HOME_VARS}[\\/](?P<k>{_KNOWN_RE})", path, re.I)
    if known:
        return f"your {_canonical_folder(known['k'])} folder", wildcard
    if re.fullmatch(r"[a-z]:\\users|/home|/users", path, re.I):
        return "every user's files", wildcard
    if re.fullmatch(r"[a-z]:\\users\\[^\\]+|/home/[^/]+|/users/[^/]+", path, re.I):
        return "a whole user folder", wildcard
    known = re.fullmatch(rf"(?:[a-z]:\\users\\[^\\]+\\|/(?:home|users)/[^/]+/)(?P<k>{_KNOWN_RE})", path, re.I)
    if known:
        return f"the whole {_canonical_folder(known['k'])} folder", wildcard
    if _SYSTEM_DIRS.fullmatch(path):
        return "a system folder", wildcard
    return None


def _canonical_folder(name: str) -> str:
    return next(folder for folder in _KNOWN_FOLDERS if folder.casefold() == name.casefold())


def _broad_delete(line: str) -> Correction | None:
    for start in _DELETE_START.finditer(line):
        cmd = start["cmd"].casefold()
        segment = re.split(r"[;|]|&&", line[start.end() :], maxsplit=1)[0]
        tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', segment)
        recursive = careful = False
        powershell = cmd in ("remove-item", "ri")
        targets: list[str] = []
        take_next = skip_next = False
        for token in tokens:
            low = token.casefold()
            if skip_next:
                skip_next = False
                continue
            if take_next:
                targets.extend(token.split(","))
                take_next = False
                continue
            if cmd in ("del", "erase", "rd", "rmdir") and re.fullmatch(r"/[a-z](?::\S*)?", low):
                recursive = recursive or low == "/s"
                careful = careful or low == "/p"
                continue
            if low == "--recursive":
                recursive = True
                continue
            if low.startswith("--"):
                careful = careful or low.startswith("--interactive")
                continue
            if cmd == "rm" and re.fullmatch(r"-[rRfvidI]+", token):  # rm's own short options: -rf, -fr, -ri
                recursive = recursive or "r" in low
                careful = careful or "i" in low
                continue
            if low.startswith("-") and len(low) > 1:
                name, _, value = low[1:].partition(":")
                powershell = True
                if len(name) >= 1 and "recurse".startswith(name):
                    recursive = True
                elif name in ("whatif", "wi") or (name in ("confirm", "cf") and value != "$false"):
                    careful = careful or value != "$false"
                elif any(param.startswith(name) and len(name) >= 2 for param in _PS_PATH_PARAMS):
                    take_next = True
                elif any(param.startswith(name) and len(name) >= 2 for param in _PS_VALUE_PARAMS):
                    skip_next = True
                continue
            targets.extend(token.split(",") if powershell else [token])
        if not recursive or careful:
            continue
        for target in targets:
            broad = _broad_target(target)
            if broad is None:
                continue
            what, wildcard = broad
            shown = target.strip("\"'")
            windows = cmd != "rm" or powershell or bool(re.match(r"[a-z]:|%|\$env:", shown, re.I))
            command = line[start.start("cmd") :].strip()
            if wildcard and not what.startswith("everything"):
                what = "everything in " + what.replace("a whole ", "a ").replace("the whole ", "the ")
            tail = "Nothing goes to the Recycle Bin." if windows else "rm has no undo."
            message = tail
            fix = ""
            if powershell and not segment.rstrip().endswith("-WhatIf"):
                fix = line[start.start("cmd") : start.end() + len(segment)].rstrip() + " -WhatIf"
            return Correction(
                kind="risky_command",
                key=f"risky_command:broad_delete:{command}",
                title=f"That deletes {what}",
                message=message,
                evidence=_short(command),
                detail="Adding -WhatIf lists what it would delete without deleting anything." if fix else "",
                fix=fix,
                fix_kind="command" if fix else "",
                urgency=URGENT,
                confidence=0.95,
            )
    return None


def _current_command_lines(text: str, ctx: TextContext) -> list[str]:
    """Only what is about to run: the last lines of a terminal, or a short single-line field."""
    lines = [line for line in text.splitlines() if line.strip()]
    if ctx.is_terminal:
        return lines[-2:]
    if len(text) <= 400:
        return lines[-3:]
    return []


def risky_command(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source == "speech":
        return []
    found: list[Correction] = []
    for line in _current_command_lines(text, ctx):
        if not any(pattern.search(line) for _, pattern, *_ in _RISKY):
            broad = _broad_delete(line)
            if broad is not None:
                found.append(broad)
            continue
        for name, pattern, title, message, fix_name in _RISKY:
            match = pattern.search(line)
            if not match:
                continue
            # The command starts where the rule matched; anything before it is the shell prompt.
            command = line[match.start() :].strip()
            fix = ""
            if fix_name == "force_with_lease":
                fix = re.sub(r"\s--force(?!-with-lease)\b|\s-f\b", " --force-with-lease", command, count=1)
            found.append(
                Correction(
                    kind="risky_command",
                    key=f"risky_command:{name}:{command}",
                    title=title,
                    message=message,
                    evidence=_short(command),
                    fix=fix,
                    fix_kind="command" if fix else "",
                    urgency=URGENT,
                    confidence=0.95,
                )
            )
            break
    return found


# ── Keys and passwords pasted where other people will read them ─────────────────────────────────────────────────

# First match wins, so the specific "sk-" prefixes come before the generic one.
_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    (
        "an OpenAI API key",
        re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}|\bsk-[A-Za-z0-9]*T3BlbkFJ[A-Za-z0-9]*"),
    ),
    # Any other "sk-" key: named neutrally, and only with a long random run, so a slug like "sk-hynix-..." is not one.
    ("an API key", re.compile(r"\bsk-(?:[A-Za-z0-9_-]*-)?(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{20,}")),
    ("a GitHub token", re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("a private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("a password", re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S{6,}", re.I)),
)
_PASSWORD_MANAGERS = ("1password", "bitwarden", "keepass", "lastpass", "dashlane", "keeper", "enpass", "nordpass")


def _redact(secret: str) -> str:
    return secret[:4] + "…" + secret[-4:] if len(secret) > 12 else "…"


def secret_in_text(text: str, ctx: TextContext) -> list[Correction]:
    where = ctx.where
    if ctx.is_terminal or ctx.is_code_editor or any(name in where for name in _PASSWORD_MANAGERS):
        return []  # an .env file or a password manager is where a key belongs
    for label, pattern in _SECRETS:
        match = pattern.search(text)
        if match:
            secret = match.group(0)
            return [
                Correction(
                    kind="secret_in_text",
                    key=f"secret_in_text:{label}:{_redact(secret)}",
                    title=f"That looks like {label}",
                    message="Anyone who reads this can use it.",
                    evidence=_redact(secret),
                    detail="If it has been sent already, rotate it where it was issued.",
                    urgency=URGENT,
                    confidence=0.9,
                )
            ]
    return []


# ── Contradicting what the person told Heliox ─────────────────────────────────────────────────────────────────────

# Relations most people have one of at a time; "my friend Sam" says nothing about "my friend Alex".
SINGULAR_RELATIONS = (
    "manager", "boss", "supervisor", "wife", "husband", "spouse", "fiance", "fiancee", "girlfriend", "boyfriend",
    "mum", "mom", "mother", "dad", "father", "landlord", "therapist", "dentist", "doctor", "accountant", "lawyer",
    "mentor", "co-founder", "cofounder",
)  # fmt: skip
_NAME = r"(?P<name>[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"


def profile_fact(text: str, ctx: TextContext) -> list[Correction]:
    found: list[Correction] = []
    for relation, known in ctx.people.items():
        if relation not in SINGULAR_RELATIONS:
            continue
        pattern = re.compile(rf"\bmy\s+{re.escape(relation)}(?:'s name is|\s+is|,)?\s+{_NAME}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            said = match["name"]
            first = said.split()[0]
            if not first[:1].isupper() or first.casefold() == known.split()[0].casefold():
                continue
            if first.casefold() in {"and", "said", "says", "will", "was", "is", "has", "had", "can", "the"}:
                continue
            found.append(
                Correction(
                    kind="profile_fact",
                    key=f"profile_fact:{relation}:{first.casefold()}",
                    title=f"Your {relation} is {known}",
                    message=f"You wrote {first}.",
                    evidence=_short(match.group(0)),
                    detail=f'If that has changed, say "my {relation} is {first}" and I\'ll update it.',
                    fix=match.group(0).replace(said, known),
                    fix_kind="replace",
                    urgency=NORMAL,
                    confidence=0.85,
                    trailing=_at_end(text, match.end()),
                )
            )
    return found


# ── Errors on screen, with the fix ───────────────────────────────────────────────────────────────────────────────

_PIP_NAMES = {
    "cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "pillow", "yaml": "pyyaml", "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv", "jwt": "pyjwt", "serial": "pyserial", "Crypto": "pycryptodome", "fitz": "pymupdf",
    "dateutil": "python-dateutil", "magic": "python-magic", "win32api": "pywin32", "docx": "python-docx",
}  # fmt: skip

_ERRORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("module_missing", re.compile(r"ModuleNotFoundError: No module named '(?P<mod>[\w.]+)'")),
    ("cannot_find_module", re.compile(r"Error: Cannot find module '(?P<mod>[^']+)'")),
    ("port_in_use", re.compile(r"(?:EADDRINUSE|address already in use)[^\n]*?:(?P<port>\d{2,5})", re.I)),
    ("not_recognized", re.compile(r"'(?P<cmd>[\w.-]+)' is not recognized as an internal or external command", re.I)),
    ("command_not_found", re.compile(r"(?:command not found: (?P<cmd>[\w.-]+)|(?P<cmd2>[\w.-]+): command not found)")),
    ("npm_missing_script", re.compile(r'npm (?:ERR!|error) Missing script: "(?P<script>[\w:.-]+)"')),
    # git's own line goes on: "fatal: not a git repository (or any of the parent directories): .git". Its tail is
    # part of the match, cut short or not (a narrow window, OCR), so the line does not read as a sentence about it.
    (
        "not_git_repo",
        re.compile(r"fatal: not a git repository(?:\s*\(or any of the parent director[^\n]*)?", re.IGNORECASE),
    ),
    (
        "python_syntax",
        re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)[\s\S]{0,200}?SyntaxError: (?P<msg>[^\n]+)'),
    ),
)


def error_on_screen(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source == "speech":
        return []
    tail = text[-4000:]
    for name, pattern in _ERRORS:
        matches = list(pattern.finditer(tail))
        if not matches:
            continue
        match = matches[-1]
        evidence = _short(match.group(0))
        if name == "module_missing":
            module = match["mod"].split(".")[0]
            package = _PIP_NAMES.get(module, module)
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"{module} isn't installed",
                    message=f"Installing {package} should fix it.",
                    evidence=evidence,
                    fix=f"pip install {package}",
                    fix_kind="command",
                )
            ]
        if name == "cannot_find_module":
            module = match["mod"]
            if module.startswith((".", "/")):
                return [
                    Correction(
                        kind="error_on_screen",
                        key=f"error_on_screen:{evidence}",
                        title=f"Node can't find {module}",
                        message="Check the path and the file's extension.",
                        evidence=evidence,
                    )
                ]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"{module} isn't installed",
                    message=f"Installing {module} should fix it.",
                    evidence=evidence,
                    fix=f"npm install {module}",
                    fix_kind="command",
                )
            ]
        if name == "port_in_use":
            port = match["port"]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"Port {port} is already taken",
                    message="Stop that program, or start this one on another port.",
                    evidence=evidence,
                    fix=f"Show me which program is using port {port}",
                    fix_kind="ask",
                )
            ]
        if name in ("not_recognized", "command_not_found"):
            cmd = match.groupdict().get("cmd") or match.groupdict().get("cmd2") or ""
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"The shell can't find {cmd}",
                    message="Install it, or add its folder to PATH.",
                    evidence=evidence,
                    fix=f"Check whether {cmd} is installed on this computer",
                    fix_kind="ask",
                )
            ]
        if name == "npm_missing_script":
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f'There\'s no "{match["script"]}" script',
                    message="npm run lists the scripts you have.",
                    evidence=evidence,
                    fix="npm run",
                    fix_kind="command",
                )
            ]
        if name == "not_git_repo":
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title="This folder isn't a git repository",
                    message="cd into the project, or run git init.",
                    evidence=evidence,
                )
            ]
        if name == "python_syntax":
            file_name = re.split(r"[\\/]", match["file"])[-1]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"Syntax error in {file_name}, line {match['line']}",
                    message=f"Python says: {match['msg'].strip()}.",
                    evidence=evidence,
                )
            ]
    return []


# ── Mentioning an attachment ────────────────────────────────────────────────────────────────────────────────────

_ATTACHMENT = re.compile(
    r"\b(?:i'?ve attached|i have attached|attached (?:is|are|you'll find)|please find attached|see attached|enclosed is)\b",
    re.I,
)


def missing_attachment(text: str, ctx: TextContext) -> list[Correction]:
    if ctx.source != "focused_field" or not ctx.is_mail_compose:
        return []
    match = _ATTACHMENT.search(text)
    if not match:
        return []
    return [
        Correction(
            kind="missing_attachment",
            key=f"missing_attachment:{ctx.title.casefold()}",
            title="You mention an attachment",
            message="Make sure it's attached before you send.",
            evidence=_short(match.group(0)),
            urgency=LOW,
            confidence=0.6,
        )
    ]


# ── Doubled words ─────────────────────────────────────────────────────────────────────────────────────────────────

_REPEAT = re.compile(r"\b(?P<w>[A-Za-z]{2,})\s+(?P=w)\b", re.IGNORECASE)
_LEGIT_REPEATS = {
    "had", "that", "is", "do", "bye", "no", "so", "very", "ha", "go", "knock", "tut", "chop", "yada", "okay", "yes",
    "hey", "well", "now", "come", "there",
}  # fmt: skip


def repeated_word(text: str, ctx: TextContext) -> list[Correction]:
    """Each different doubled word, up to three. Every one is reported so that a doubled word that was already in
    the field (and is left alone by the engine) does not hide one the person types later."""
    if ctx.source != "focused_field" or ctx.is_terminal or ctx.is_code_editor:
        return []
    found: dict[str, Correction] = {}
    for match in _REPEAT.finditer(text):
        word = match["w"]
        key = f"repeated_word:{match.group(0).casefold()}"
        if word.casefold() in _LEGIT_REPEATS or key in found:
            continue
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if before == "-" or after == "-":
            continue  # part of a compound: "read-only only", "opt-in in", "twenty twenty-six", "public public-corpus"
        second = match.group(0).split()[-1]
        if word.islower() and second[:1].isupper():
            continue  # "[object Object]": a different word that happens to be spelled the same
        if len(found) == 3:
            break
        found[key] = Correction(
            kind="repeated_word",
            key=key,
            title=f'"{word}" is there twice',
            message="Delete one of them.",
            evidence=match.group(0),
            fix=word,
            fix_kind="replace",
            urgency=LOW,
            confidence=0.8,
            trailing=_at_end(text, match.end()),
        )
    return list(found.values())


def community_catch(text: str, ctx: TextContext) -> list[Correction]:
    from pilot.live.catches import community_catch as find  # imports this module, so it loads late

    return find(text, ctx)


# ── A date that has already passed, written as if it were ahead ───────────────────────────────────────────────

# "See you on 5 November" written on the 13th. Only a day and month (a weekday and this year's number may come with
# them), only up to PAST_DATE_DAYS back (further back, a yearless date means next year), and only in a sentence
# that looks ahead ("see you", "will", "is due", "let's", "by", "until", "next") and has no word that looks back
# ("was", "had", "met", "last", "ago", "since", "already"...). Minutes, notes from a meeting, logs and changelogs
# are about the past on purpose, so their windows are left alone.
PAST_DATE_DAYS = 45
_PD_DAY = r"(?:(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\.?,?\s+(?:the\s+)?)?"
_PAST_DATE = (
    re.compile(
        rf"\b{_PD_DAY}(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTH_RE}\b\.?(?:,?\s+(?P<y>\d{{4}}))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_PD_DAY}{_MONTH_RE}\.?\s+(?:the\s+)?(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{{4}}))?",
        re.IGNORECASE,
    ),
)
_AHEAD = re.compile(
    r"\b(?:see you|meet(?:ing)?|let'?s|we'?ll|i'?ll|you'?ll|they'?ll|will|shall|going to|is due|are due|due (?:on|by)|"
    r"deadline|by|until|till|planned for|scheduled for|book(?:ed)? for|reserved? for|next|upcoming|"
    r"starts?|kicks? off|is on|are on|happening|please|can we|could we|would|tomorrow)\b",
    re.IGNORECASE,
)
_BEHIND = re.compile(
    r"\b(?:was|were|had|did|met|went|sent|came|ago|last|yesterday|since|earlier|already|held|happened|took place|"
    r"completed|finished|shipped|released|merged|launched|started|ended|closed|signed|paid|received|delivered|"
    r"recap|minutes|back on|as of|been|said|wrote|told)\b",
    re.IGNORECASE,
)
# A date that comes round every year is meant: "the 5 November fireworks next year", "her birthday on 8 November".
_EVERY_YEAR = re.compile(r"\b(?:next year|every year|each year|annual(?:ly)?|anniversary|birthday)\b", re.IGNORECASE)
_PAST_WINDOWS = ("minutes", "meeting notes", "changelog", "release notes", "log", "journal", "diary", "history")


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n"))
    rights = [i for i in (text.find(mark, end) for mark in (".", "!", "?", "\n")) if i >= 0]
    return text[left + 1 : min(rights) if rights else len(text)]


def past_date(text: str, ctx: TextContext) -> list[Correction]:
    """A date that has already passed, in a sentence that looks ahead."""
    if ctx.source != "focused_field" or ctx.is_terminal or ctx.is_code_editor:
        return []
    if any(word in ctx.where for word in _PAST_WINDOWS):
        return []
    found: dict[str, Correction] = {}
    for pattern in _PAST_DATE:
        for match in pattern.finditer(text):
            try:
                day, month = int(match["d"]), _month_number(match["m"])
                stated = int(match["y"]) if match["y"] else None
                when = date(stated or ctx.today.year, month, day)
            except ValueError:
                continue
            if stated is not None and stated != ctx.today.year:
                continue
            gone = (ctx.today - when).days
            if not 1 <= gone <= PAST_DATE_DAYS:
                continue
            # The time words around the date, but not the date itself ("by" must not come from a weekday's "Tue").
            sentence = _sentence_around(text, match.start(), match.end())
            around = sentence.replace(match.group(0), " ")
            if not _AHEAD.search(around) or _BEHIND.search(around) or _EVERY_YEAR.search(around):
                continue
            key = f"past_date:{when.isoformat()}"
            if key in found:
                continue
            ago = "yesterday" if gone == 1 else f"{gone} days ago"
            found[key] = Correction(
                kind="past_date",
                key=key,
                title=f"{_on(day, month, ctx)} has already passed",
                message=f"That was {ago}; today is {ctx.today.strftime('%A')} {_on(ctx.today.day, ctx.today.month, ctx)}.",
                evidence=_short(match.group(0).strip().rstrip(".,")),
                detail="If you meant next year, add the year.",
                urgency=NORMAL,
                confidence=0.8,
                trailing=_at_end(text, match.end()),
            )
    return list(found.values())


# ── A name one letter off someone the person told Heliox about ─────────────────────────────────────────────────

# "Hi Priay," to the manager Priya. Only names the person gave Heliox (one person for the relation), of five letters
# or more, and only where a name is used to address someone: the start of the text or of a line before a comma,
# after "hi", "hello", "dear", "thanks" and the like, or after "@" or "cc". One letter off means two neighbouring
# letters swapped ("Priay"), or one inner letter doubled, added or dropped (never the first or last letter: Daniela,
# Hanna and Michaels are other people or other words), and never one letter swapped for another (Maria, Mario).
# Nothing either when the window (the recipient or subject) spells it that way, or when a surname after it is not the
# surname of the person they know. The card asks ("Did you mean Priya?") and is never spoken (policy.CARD_ONLY_KINDS).
# Each looked for on its own, so "Thanks, Michal," is a greeting and then a name, not one word before a comma.
_ADDRESS = (
    re.compile(r"(?:^|\n)[ \t]*(?P<name>[A-Za-z]{4,})(?=[ \t]*,)"),
    re.compile(
        r"\b(?:hi|hello|hey|dear|thanks|thank you|cheers|morning|afternoon|evening|cc)\b,?[ \t]+(?P<name>[A-Za-z]{4,})\b",
        re.IGNORECASE,
    ),
    re.compile(r"@(?P<name>[A-Za-z]{4,})\b"),
)


def _one_letter_off(word: str, name: str) -> bool:
    w, n = word.casefold(), name.casefold()
    if w == n or w[0] != n[0]:
        return False
    if len(w) == len(n):  # the last two letters may be the swapped pair: "Priay"
        diff = [i for i in range(len(w)) if w[i] != n[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1 and w[diff[0]] == n[diff[1]] and w[diff[1]] == n[diff[0]]
    if w[-1] != n[-1]:
        return False
    if len(w) == len(n) + 1:
        return any(w[:i] + w[i + 1 :] == n for i in range(1, len(w) - 1))
    if len(w) == len(n) - 1:
        return any(n[:i] + n[i + 1 :] == w for i in range(1, len(n) - 1))
    return False


def near_name(text: str, ctx: TextContext) -> list[Correction]:
    """A name in an addressing spot that is one letter off a person the person told Heliox about."""
    if ctx.source != "focused_field" or ctx.is_terminal or ctx.is_code_editor or not ctx.people:
        return []
    known: dict[str, tuple[str, str, str]] = {}  # first name -> (relation, first name, surname or "")
    for relation, name in ctx.people.items():
        parts = name.split()
        first = parts[0] if parts else ""
        if len(first) >= 5 and first.isalpha():
            known[first.casefold()] = (relation, first, parts[-1] if len(parts) > 1 else "")
    where = ctx.title.casefold()
    found: dict[str, Correction] = {}
    for match in (m for pattern in _ADDRESS for m in pattern.finditer(text)):
        group = "name"
        word = match[group]
        if not word[:1].isupper() or word.casefold() in known:
            continue  # the name of someone they told Heliox about, spelled as they told it
        if word.casefold() in where:
            continue  # the window (the recipient, the subject) has it this way: likely a different person
        hits = [entry for entry in known.values() if _one_letter_off(word, entry[1])]
        if len(hits) != 1:
            continue  # nobody, or two people it could be: say nothing
        relation, first, surname = hits[0]
        after = text[match.end(group) :].lstrip(" \t").split(maxsplit=1)
        following = after[0].strip(",.;:!?") if after else ""
        if surname and following[:1].isupper() and following.casefold() != surname.casefold():
            continue  # "Hanah Jones" when they told Heliox about Hannah Smith: someone else
        key = f"near_name:{word.casefold()}"
        if key in found:
            continue
        found[key] = Correction(
            kind="near_name",
            key=key,
            # A question, never a correction: "Jon" may be right, and someone other than the person they know.
            title=f"Did you mean {first}?",
            message=f"You wrote {word}; your {relation} is {first}.",
            evidence=word,
            detail=f"You told me your {relation} is {first}. If {word} is someone else, ignore this.",
            fix=first,
            fix_kind="replace",
            urgency=NORMAL,
            confidence=0.8,
            trailing=_at_end(text, match.end(group)),
        )
    return list(found.values())


DETECTORS = {
    "community_catch": community_catch,
    "risky_command": risky_command,
    "secret_in_text": secret_in_text,
    "weekday_date": weekday_date,
    "arithmetic": arithmetic,
    "profile_fact": profile_fact,
    "error_on_screen": error_on_screen,
    "missing_attachment": missing_attachment,
    "repeated_word": repeated_word,
    "past_date": past_date,
    "near_name": near_name,
}

# What a spoken sentence can be checked for; screen-only checks (commands, errors, typing) do not apply.
SPEECH_KINDS = ("weekday_date", "arithmetic", "profile_fact")


def _said_fix(correction: Correction, norm: sp.Spoken, start: int) -> str:
    """A fix for a spoken card in the person's own words, or "" when it cannot be said that way."""
    evidence, fix = correction.evidence, correction.fix
    if fix in evidence:  # the fix is a piece of the evidence: "Thursday"
        at = start + evidence.index(fix)
        return norm.said(at, at + len(fix))
    shown, new = evidence.split(), fix.split()
    if len(shown) != len(new):
        return ""
    differing = [i for i, (a, b) in enumerate(zip(shown, new, strict=True)) if a != b]
    if len(differing) != 1:
        return ""
    old_word = shown[differing[0]].strip(".,;:!?")
    said = norm.said(start, start + len(evidence))
    swapped = re.sub(rf"\b{re.escape(old_word)}\b", new[differing[0]].strip(".,;:!?"), said, count=1, flags=re.I)
    return "" if swapped == said else swapped


def _in_said_words(correction: Correction, norm: sp.Spoken) -> Correction:
    """The card quotes what the person said, not the written form the detectors read."""
    evidence = correction.evidence
    start = norm.text.find(evidence) if evidence else -1
    if start < 0:
        return correction
    said = norm.said(start, start + len(evidence))
    key = correction.key
    for shown, spoken_form in ((evidence.casefold(), said.casefold()), (evidence, said)):
        if key.endswith(shown):
            key = key[: len(key) - len(shown)] + spoken_form
            break
    fix, fix_kind = correction.fix, correction.fix_kind
    if fix and fix_kind == "replace":
        fix = _said_fix(correction, norm, start)
        fix_kind = "replace" if fix else ""
    return replace(correction, evidence=said, key=key, fix=fix, fix_kind=fix_kind)


# Kinds about how the person uses words and numbers. Inside quotation marks the person is mentioning the words,
# not using them: a commit message quoting "Tuesday 14 November" as an example, an email quoting someone's "could of".
# Risky commands and keys are not in it: quoting one does not make it safe.
MENTION_KINDS = frozenset(
    {
        "weekday_date",
        "arithmetic",
        "repeated_word",
        "community_catch",
        "profile_fact",
        "missing_attachment",
        "past_date",
        "near_name",
    }
)
_QUOTE_MARKS = frozenset('"`\u201c\u201d')
_QUOTED = re.compile(
    r'"[^"\n]{1,300}"'
    "|\u201c[^\u201d\n]{1,300}\u201d"
    "|`[^`\n]{1,300}`"
    r"|(?<![\w])'[^'\n]{1,300}'(?![\w])"
)


# ── What screen reading (OCR) gets wrong, put right before the error checks read it ──

_KNOWN_ERRNO = frozenset(
    {"EADDRINUSE", "ECONNREFUSED", "ECONNRESET", "ENOENT", "EACCES", "EPERM", "ETIMEDOUT", "EADDRNOTAVAIL", "EEXIST"}
)
# A word whose letters were read as digits: "modu1e", "Err0r", "pyth0n" (no other digits in it).
_LETTER_DIGITS = re.compile(r"\b(?=[A-Za-z01]*[A-Za-z]{2})(?=[A-Za-z01]*[01])[A-Za-z01]{3,}\b")
# A number whose digits were read as letters, where a number must be: ":::3OOO", "127.0.0.l:5432", "port 8O80".
_NUMBER_PLACE = re.compile(
    r"(?<=:)[0-9OolI]{2,5}\b|(?<=\bport )[0-9OolI]{2,5}\b|(?<=\.)[0-9OolI]{1,3}(?=:\d|:[0-9OolI])"
)
# An error class name split in two: "ModuleNot FoundError".
_SPLIT_ERROR = re.compile(r"\b([A-Z][A-Za-z]+) ([A-Z][a-z]+(?:Error|Exception))\b")
_SPLIT_ERRNO = re.compile(r"\b(E[A-Z]{2,}) ([A-Z]{2,})\b")


def repair_ocr(text: str) -> str:
    """The usual recognition damage in screen text, undone where the right reading is certain enough."""

    def letters(match: re.Match[str]) -> str:
        word = match.group(0)
        return word.replace("0", "o").replace("1", "l")

    def digits(match: re.Match[str]) -> str:
        return match.group(0).translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"}))

    def errno(match: re.Match[str]) -> str:
        joined = match.group(1) + match.group(2)
        return joined if joined in _KNOWN_ERRNO else match.group(0)

    text = _NUMBER_PLACE.sub(digits, text)
    text = _LETTER_DIGITS.sub(letters, text)
    text = _SPLIT_ERROR.sub(lambda m: m.group(1) + m.group(2), text)
    return _SPLIT_ERRNO.sub(errno, text)


def _line_goes_on(text: str, evidence: str) -> bool:
    """On screen, real error output ends its line; a headline or a sentence about an error goes on ("Fixing
    ModuleNotFoundError: No module named 'requests' in five minutes")."""
    at = text.find(evidence)
    if at < 0 or not evidence:
        return False
    end = text.find("\n", at + len(evidence))
    rest = text[at + len(evidence) : end if end >= 0 else len(text)]
    return len(re.findall(r"[A-Za-z]{2,}", rest)) >= 2


def only_quoted(text: str, evidence: str) -> bool:
    """Every place ``evidence`` appears in ``text`` is inside a quotation (straight, curly or backticks)."""
    if not evidence:
        return False
    spans = [m.span() for m in _QUOTED.finditer(text)]
    start, seen = 0, False
    folded, needle = text.casefold(), evidence.casefold()
    while True:
        at = folded.find(needle, start)
        if at < 0:
            return seen
        seen = True
        end = at + len(needle)
        # A quote mark right before and after also counts: a quotation inside a quotation pairs its marks wrongly.
        hugged = at > 0 and end < len(text) and text[at - 1] in _QUOTE_MARKS and text[end] in _QUOTE_MARKS
        if not hugged and not any(a < at and end <= b for a, b in spans):
            return False
        start = at + 1


def detect(text: str, ctx: TextContext, *, kinds: tuple[str, ...] = KINDS) -> list[Correction]:
    """Every correction the enabled detectors find, most urgent first.

    Speech is first turned into the written form the detectors parse (pilot.live.spoken), and the cards that come
    back quote the words that were said and are worded to be read aloud.
    """
    if not text or not text.strip():
        return []
    allowed = SPEECH_KINDS if ctx.source == "speech" else kinds
    read, where, norm = text, ctx, None
    if ctx.source == "speech":
        norm = sp.normalise(text)
        read, where = norm.text, replace(ctx, spoken=norm)
    elif ctx.source == "screen_ocr":
        read = repair_ocr(text)
    results: list[Correction] = []
    for kind in kinds:
        if kind not in allowed or kind not in DETECTORS:
            continue
        try:
            results.extend(DETECTORS[kind](read, where))
        except Exception:  # a detector bug must never take the live layer down, or the other detectors with it
            logger.warning("Live detector %s failed; the others still ran", kind, exc_info=True)
            continue
    if norm is not None:
        results = [_in_said_words(c, norm) for c in results]
    elif ctx.source == "screen_ocr":
        results = [c for c in results if c.kind != "error_on_screen" or not _line_goes_on(read, c.evidence)]
    else:
        results = [c for c in results if c.kind not in MENTION_KINDS or not only_quoted(read, c.evidence)]
    order = {URGENT: 0, NORMAL: 1, LOW: 2}
    return sorted(results, key=lambda c: order.get(c.urgency, 3))

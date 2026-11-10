"""Specific, local mistake detectors for the live layer.

Each detector reads a piece of text the person is writing or saying and returns ``Correction`` objects that
name the exact evidence, say what is wrong in one sentence, and carry a fix when there is one. They are plain
rules over the text: no model call, no network, nothing persisted. A detector that is not sure stays quiet;
a false correction costs more trust than a missed one.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

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
        rf"\b{_DAY_RE}\.?,?\s+{_MONTH_RE}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{{4}}))?",
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
                    title=f"{day} {month_name} is a {actual_name}",
                    message=f"{day} {month_name} is a {actual_name}, not a {stated_name}.",
                    evidence=evidence,
                    detail=f"If you meant the {stated_name}, that's {nearest.day} {calendar.month_name[nearest.month]}.",
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
    found.extend(_numeric_weekdays(text, ctx))
    found.extend(_relative_days(text, ctx))
    found.extend(_clock_times(text))
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
_DATE_NEGATION_AFTER = re.compile(r"^\W{0,3}(?:doesn'?t|does not|isn'?t|is not|never)\b", re.IGNORECASE)


def _plausible_year(month: int, today: date) -> int:
    """The year a yearless month most likely means: this year, unless that month is well past."""
    return today.year if month >= today.month - 2 else today.year + 1


def _no_such_date(evidence: str, day: int, month: int, stated_year: int | None, ctx: TextContext) -> Correction:
    month_name = calendar.month_name[month]
    if month == 2 and day == 29:
        year = stated_year or _plausible_year(month, ctx.today)
        title = f"There's no 29 February {year}"
        message = f"{year} isn't a leap year, so February has no 29th."
    else:
        title = f"There's no {day} {month_name}"
        message = f"{month_name} doesn't have a {day}{_ordinal(day)}."
    return Correction(
        kind="weekday_date",
        key=f"weekday_date:{evidence.casefold()}",
        title=title,
        message=message,
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
            readings[date(year, month, day)] = f"{day} {calendar.month_name[month]}"
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
                    title=f"{spoken} is a {actual_name}",
                    message=f"{spoken} is a {actual_name}, not a {stated_name}.",
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
                    title=f"{numeric} isn't a {stated_name}",
                    message=(
                        f"{first_name} is a {_DAYS[first.weekday()].title()} and {second_name} is a "
                        f"{_DAYS[second.weekday()].title()}, so {numeric} isn't a {stated_name} either way."
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
    r"^(?:\s*$|\s*[.!;:)]|,?\s+(?:at|@|from|morning|afternoon|evening|night|first thing|by)\b|\s+\d)", re.IGNORECASE
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


def _from_today(day: date, today: date) -> str:
    days = (day - today).days
    named = {-1: "yesterday", 0: "today", 1: "tomorrow", 2: "the day after tomorrow"}
    if days in named:
        return named[days]
    return f"in {days} days" if days > 0 else f"{-days} days ago"


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
        detail = (
            "" if rel == "yesterday" else f"The next {stated_name} is {meant.day} {calendar.month_name[meant.month]}."
        )
        written = _dated(text, match.start(group), ctx)
        if written is not None:
            # "Tomorrow, Thursday 12 Nov" where 12 Nov is a Thursday: the day and date agree, "tomorrow" is the slip.
            # Swapping in tomorrow's weekday would make the date wrong, so the fix drops the relative word.
            on = f"{written.day} {calendar.month_name[written.month]}"
            found.append(
                Correction(
                    kind="weekday_date",
                    key=f"weekday_date:{evidence.casefold()}",
                    title=f"{on} {'wasn' if rel == 'yesterday' else 'isn'}'t {rel}",
                    message=(
                        f"{rel.title()} {verb} {actual_name} {when.day} {calendar.month_name[when.month]}. "
                        f"{stated_name} {on} is {_from_today(written, ctx.today)}."
                    ),
                    evidence=evidence,
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
                message=f"{rel.title()} {verb} {actual_name}, not {stated_name}.",
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


def _clock_times(text: str) -> list[Correction]:
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
        if spelled[:1].casefold() == "p":
            twelve = f"{hour - 12}{':' + minutes if minutes else ''}"
            gap = " " if match["ap2"] else ""
            found.append(
                Correction(
                    title=f"{evidence} is {twelve}{gap}{spelled}",
                    message=f"{evidence} mixes the 24-hour and 12-hour clocks. {hour}:{minutes or '00'} is {twelve}{gap}{spelled}.",
                    fix=f"{twelve}{gap}{spelled}",
                    fix_kind="replace",
                    confidence=0.9,
                    **common,
                )
            )
        else:
            found.append(
                Correction(
                    title=f"There's no {evidence}",
                    message=f"{evidence} mixes the 24-hour and 12-hour clocks, so it isn't a time.",
                    confidence=0.9,
                    **common,
                )
            )
    for match in _CLOCK.finditer(text):
        hour, minutes = int(match["h"]), int(match["mm"])
        shown = match["t"]
        if minutes >= 60:
            message = f"An hour has 60 minutes, so {shown} isn't a time."
        elif hour >= 25 or (hour == 24 and minutes > 0):
            message = f"A day ends at 23:59, so {shown} isn't a time."
        else:
            continue
        found.append(
            Correction(
                kind="weekday_date",
                key=f"weekday_date:{shown}",
                title=f"There's no {shown}",
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
        found.append(
            Correction(
                kind="arithmetic",
                key=f"arithmetic:{evidence}",
                title=f"That comes to {shown}",
                message=f"{_fmt(a, _decimals(match['a']))} {symbol} {_fmt(b, _decimals(match['b']))} is {shown}, not {match['c']}.",
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
        found.append(
            Correction(
                kind="arithmetic",
                key=f"arithmetic:{evidence}",
                title=f"That comes to {shown}",
                message=f"{match['p']}% of {match['base']} is {shown}, not {match['c']}.",
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
        "That deletes everything there",
        "That rm -rf targets the whole drive or your home folder, and there's no undo.",
        "",
    ),
    (
        "force_push",
        re.compile(r"\bgit\s+push\b[^\n]*?(?:\s--force(?!-with-lease)\b|\s-f\b)", re.I),
        "That can overwrite teammates' commits",
        "A plain force-push can overwrite your teammates' commits. --force-with-lease refuses if the remote moved.",
        "force_with_lease",
    ),
    (
        "reset_hard",
        re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
        "That throws away your changes",
        "git reset --hard throws away uncommitted changes. git stash would keep them.",
        "",
    ),
    (
        "clean_fdx",
        re.compile(r"\bgit\s+clean\s+-[a-z]*f[a-z]*d[a-z]*x|\bgit\s+clean\s+-[a-z]*x[a-z]*f", re.I),
        "That deletes ignored files too",
        "git clean -x also deletes ignored files, like your .env and build caches.",
        "",
    ),
    (
        "drop_table",
        re.compile(r"\bdrop\s+(?:table|database|schema)\s+(?!if\s+exists\s+\w+_tmp)\S+", re.I),
        "That drops it for good",
        "DROP removes the table and its data permanently. Make sure you have a backup.",
        "",
    ),
    (
        "delete_all_rows",
        re.compile(r"\bdelete\s+from\s+(?P<t>[\w.\"`\[\]]+)\s*;", re.I),
        "That deletes every row",
        "That DELETE has no WHERE clause, so it removes every row in the table.",
        "",
    ),
    (
        "update_all_rows",
        re.compile(r"\bupdate\s+[\w.\"`\[\]]+\s+set\s+(?:(?!\bwhere\b)[^;])*;", re.I),
        "That updates every row",
        "That UPDATE has no WHERE clause, so it changes every row in the table.",
        "",
    ),
    (
        "chmod_777",
        re.compile(r"\bchmod\s+(?:-R\s+)?777\b", re.I),
        "That makes it writable by anyone",
        "chmod 777 lets every user on the machine change those files.",
        "",
    ),
    (
        "pipe_to_shell",
        re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b|\biwr\b[^\n|]*\|\s*iex\b", re.I),
        "That runs a script you haven't read",
        "Piping a download straight into a shell runs it before you can read it.",
        "",
    ),
    (
        "format_drive",
        re.compile(r"\bformat\s+[a-z]:(?=\s|$)|\bmkfs(?:\.\w+)?\s+/dev/|\bdd\s+if=\S+\s+of=/dev/(?:sd|nvme|disk)", re.I),
        "That wipes a whole drive",
        "That command erases an entire drive.",
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
            if wildcard:
                message = f"That deletes {what}. {tail}"
            else:
                message = f"That deletes {shown} and everything in it. {tail}"
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
                    message=f"That looks like {label}. Anyone who reads this can use it.",
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
                    message=f"You told me your {relation} is {known}, not {first}.",
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
    ("not_git_repo", re.compile(r"fatal: not a git repository")),
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
                    message=f"Python can't find {module}. Installing {package} should fix it.",
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
                        title="A local import path is wrong",
                        message=f"Node can't find {module}. Check the path and the file's extension.",
                        evidence=evidence,
                    )
                ]
            return [
                Correction(
                    kind="error_on_screen",
                    key=f"error_on_screen:{evidence}",
                    title=f"{module} isn't installed",
                    message=f"Node can't find {module}. Installing it should fix that.",
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
                    message=f"Something else is already using port {port}.",
                    evidence=evidence,
                    detail="Stop that program, or start this one on another port.",
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
                    title=f"{cmd} isn't installed or isn't on PATH",
                    message=f"The shell can't find {cmd}. It isn't installed, or its folder isn't on PATH.",
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
                    message=f'package.json has no "{match["script"]}" script. npm run lists the ones it has.',
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
                    message="You're running git outside a repository. cd into the project, or run git init.",
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
                    message=f"Python stopped at line {match['line']} of {file_name}: {match['msg'].strip()}.",
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
            message="You mention an attachment. Make sure it's attached before you send.",
            evidence=_short(match.group(0)),
            urgency=LOW,
            confidence=0.6,
        )
    ]


# ── Doubled words ─────────────────────────────────────────────────────────────────────────────────────────────────

_REPEAT = re.compile(r"\b(?P<w>[A-Za-z]{2,})\s+(?P=w)\b", re.IGNORECASE)
_LEGIT_REPEATS = {"had", "that", "is", "do", "bye", "no", "so", "very", "ha", "go", "knock", "tut", "chop", "yada"}


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
        if len(found) == 3:
            break
        found[key] = Correction(
            kind="repeated_word",
            key=key,
            title="A doubled word",
            message=f'"{word} {word}" has the word twice.',
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
}

# What a spoken sentence can be checked for; screen-only checks (commands, errors, typing) do not apply.
SPEECH_KINDS = ("weekday_date", "arithmetic", "profile_fact")


def detect(text: str, ctx: TextContext, *, kinds: tuple[str, ...] = KINDS) -> list[Correction]:
    """Every correction the enabled detectors find, most urgent first."""
    if not text or not text.strip():
        return []
    allowed = SPEECH_KINDS if ctx.source == "speech" else kinds
    results: list[Correction] = []
    for kind in kinds:
        if kind not in allowed or kind not in DETECTORS:
            continue
        try:
            results.extend(DETECTORS[kind](text, ctx))
        except Exception:  # a detector bug must never take the live layer down
            continue
    order = {URGENT: 0, NORMAL: 1, LOW: 2}
    return sorted(results, key=lambda c: order.get(c.urgency, 3))

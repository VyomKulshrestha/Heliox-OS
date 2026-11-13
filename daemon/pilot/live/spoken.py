"""Spoken forms to written forms, so the live detectors can read what a speech recogniser writes.

A recogniser writes "Monday the twelfth of November at ten" where a person types "Monday 12 November at 10", and it
usually leaves out the punctuation and the capitals. The detectors parse the written form, so speech goes through
:func:`normalise` first: a pure function from the transcript to the text the detectors already understand, with a
map back to the person's own words so a card can quote what was actually said.

It converts only where the surrounding words make the reading certain, because a wrong conversion is a wrong card:

- ordinals, when they sit next to a month or a weekday ("the twelfth of November", "November twelfth", "Friday the
  fourteenth") or follow "by/on/until the" at the end of a clause; "the second before you hit send" stays words;
- times: "ten a m", "ten thirty", "half past two", "quarter to five", "ten o'clock", "at ten";
- years after a month: "twenty twenty six", "two thousand and twenty six", "nineteen ninety nine";
- a sum or a percentage: "twelve times fifteen is one seventy", "twenty percent of three hundred is seventy";
- "tomorrow Thursday" gets the comma that was lost: "tomorrow, Thursday".

"for", "to", "too", "ate" and "won" are read as numbers only inside those shapes (an operand next to an operator,
a number before "percent", an hour before "o'clock", or after "at" with a "p m" behind it).

The module also says numbers, dates and times in words (``cardinal_words``, ``date_words`` and friends) for the
cards Heliox reads aloud. It imports nothing from the detectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

# ── words for numbers ─────────────────────────────────────────────────────────────────────────────────────────────

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]  # fmt: skip
_TENS_WORDS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_UNIT = {word: i for i, word in enumerate(_ONES) if 0 < i < 20}
_TENS = {word: i * 10 for i, word in enumerate(_TENS_WORDS) if word}
_ORD_IRREGULAR = {
    1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth",
}  # fmt: skip
_ORD_UNIT = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
}
_ORD_SIMPLE = {
    **_ORD_UNIT,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip
_HOMOPHONE = {"for": 4, "to": 2, "too": 2, "ate": 8, "won": 1}
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
_AMBIGUOUS_MONTHS = frozenset({"may", "march", "mar"})
_WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
_TIME_PREPS = frozenset({"at", "by", "around", "until", "till", "before", "after", "about", "from"})
_DATE_PREPS = frozenset(
    {"on", "by", "until", "till", "before", "after", "from", "since", "is", "was", "for", "this", "next"}
)
_AFTER_DATE = frozenset(
    {"at", "and", "so", "then", "morning", "afternoon", "evening", "night", "but", "or", "if", "because", "please", "thanks", "to", "for"}
)  # fmt: skip
_RELATIVE = frozenset({"today", "tomorrow", "yesterday"})


def cardinal_words(n: int) -> str:
    """12 -> "twelve", 180 -> "one hundred and eighty", -3 -> "minus three"."""
    if n < 0:
        return "minus " + cardinal_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS_WORDS[tens] + (f"-{_ONES[unit]}" if unit else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" and {cardinal_words(rest)}" if rest else "")
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            high, rest = divmod(n, size)
            tail = "" if not rest else (f" and {cardinal_words(rest)}" if rest < 100 else f" {cardinal_words(rest)}")
            return f"{cardinal_words(high)} {name}{tail}"
    return str(n)  # unreachable


def number_words(text: str) -> str:
    """A written number as words: "170" -> "one hundred and seventy", "-3.5" -> "minus three point five"."""
    text = text.replace(",", "")
    sign = "minus " if text.startswith("-") else ""
    whole, _, decimals = text.lstrip("-").partition(".")
    try:
        words = cardinal_words(int(whole or "0"))
    except ValueError:
        return text
    if decimals:
        words += " point " + " ".join(_ONES[int(d)] for d in decimals if d.isdigit())
    return sign + words


def ordinal_words(n: int) -> str:
    """12 -> "twelfth", 31 -> "thirty-first"."""
    if n in _ORD_IRREGULAR:
        return _ORD_IRREGULAR[n]
    if n < 20:
        return _ONES[n] + "th"
    tens, unit = divmod(n, 10)
    if unit == 0:
        return _TENS_WORDS[tens][:-1] + "ieth"
    return f"{_TENS_WORDS[tens]}-{ordinal_words(unit)}"


def date_words(day: int, month_name: str, *, article: bool = True) -> str:
    """12, "November" -> "the twelfth of November"."""
    return f"{'the ' if article else ''}{ordinal_words(day)} of {month_name}"


def year_words(year: int) -> str:
    """2026 -> "twenty twenty-six", 1999 -> "nineteen ninety-nine", 2005 -> "two thousand and five"."""
    if 2000 <= year <= 2009:
        return "two thousand" + (f" and {_ONES[year - 2000]}" if year > 2000 else "")
    if 1100 <= year <= 2099:
        high, low = divmod(year, 100)
        return cardinal_words(high) + (
            " " + (f"oh {_ONES[low]}" if low < 10 else cardinal_words(low)) if low else " hundred"
        )
    return cardinal_words(year)


# ── written dates and durations, as a person says them (the voice's side) ─────────────────────────────────────────

_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December",
)  # fmt: skip
# Abbreviations only with their capital ("Sat" and "Sun" are words too), full names in any case.
_WD_ABBR = {"Mon": 0, "Tue": 1, "Tues": 1, "Wed": 2, "Thu": 3, "Thur": 3, "Thurs": 3, "Fri": 4, "Sat": 5, "Sun": 6}
_MON_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11,
    "Dec": 12,
}  # fmt: skip
_WD = (
    r"(?P<wd>(?i:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    + "|".join(sorted(_WD_ABBR, key=len, reverse=True))
    + r")\.?"
)
_MON = (
    r"(?P<mon>(?i:january|february|march|april|may|june|july|august|september|october|november|december)|"
    + ("|".join(sorted(_MON_ABBR, key=len, reverse=True)))
    + r")\.?"
)
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?:,? (?P<year>20\d\d))?"
_SPOKEN_DATES = (
    re.compile(r"\b(?:" + _WD + r",? )?(?:the )?" + _DAY + r"(?: of)? " + _MON + r"(?![\w])" + _YEAR + r"\b"),
    re.compile(r"\b(?:" + _WD + r",? )?" + _MON + r" (?:the )?" + _DAY + r"(?![\w:])" + _YEAR + r"\b"),
)
_LONE_WD_ABBR = re.compile(
    r"\b(?P<prep>on|every|next|this|until|from|by|before|after) (?P<wd>"
    + "|".join(sorted(_WD_ABBR, key=len, reverse=True))
    + r")\.?(?![\w])"
)
_DURATION = re.compile(r"(?<![\w.:])(?P<n>\d+) ?(?P<unit>h|hr|hrs|min|mins|sec|secs)\b(?!-)")
_DURATION_WORDS = {"h": "hour", "hr": "hour", "hrs": "hour", "min": "minute", "mins": "minute", "sec": "second",
                   "secs": "second"}  # fmt: skip


def _weekday_index(name: str) -> int:
    return _WD_ABBR[name] if name in _WD_ABBR else _DAY_NAMES.index(name.capitalize())


def _month_index(name: str) -> int:
    return _MON_ABBR[name] if name in _MON_ABBR else _MONTH_NAMES.index(name.capitalize()) + 1


def say_dates(text: str, today: date | None = None) -> str:
    """Written dates as a person says them: "on Thu 19 Nov" -> "on Thursday the nineteenth", "Friday 13 November
    2027" -> "Friday the thirteenth of November twenty twenty-seven". The month is said only when it is not this
    month and the year only when it is not this year; a weekday abbreviation after "on" or "every" is said whole.
    A date that does not exist is left as written."""
    today = today or date.today()

    def spoken(found: re.Match[str]) -> str:
        day, month = int(found["day"]), _month_index(found["mon"])
        year = int(found["year"]) if found["year"] else today.year
        try:
            date(year, month, day)
        except ValueError:
            return found.group(0)
        said = f"the {ordinal_words(day)}"
        if found["wd"]:
            said = f"{_DAY_NAMES[_weekday_index(found['wd'])]} {said}"
        if month != today.month or year != today.year:
            said += f" of {_MONTH_NAMES[month - 1]}"
        if year != today.year:
            said += f" {year_words(year)}"
        return said

    out = text
    for pattern in _SPOKEN_DATES:
        out = pattern.sub(spoken, out)
    return _LONE_WD_ABBR.sub(lambda m: f"{m['prep']} {_DAY_NAMES[_WD_ABBR[m['wd']]]}", out)


def say_durations(text: str) -> str:
    """ "2 h 15 min" -> "2 hours 15 minutes", "1 min" -> "1 minute": the voice reads the number itself."""

    def spoken(found: re.Match[str]) -> str:
        n, unit = int(found["n"]), _DURATION_WORDS[found["unit"]]
        return f"{found['n']} {unit}{'' if n == 1 else 's'}"

    return _DURATION.sub(spoken, text)


def time_words(hour: int, minute: int = 0) -> str:
    """(10, 30) -> "ten thirty", (10, 5) -> "ten oh five", (10, 0) -> "ten"."""
    if minute == 0:
        return cardinal_words(hour)
    return f"{cardinal_words(hour)} {'oh ' + _ONES[minute] if minute < 10 else cardinal_words(minute)}"


# ── tokens ────────────────────────────────────────────────────────────────────────────────────────────────────────


class _Tok(NamedTuple):
    kind: str  # "w" word, "n" digits, "m" a.m./p.m., "o" anything else
    low: str
    start: int
    end: int


_TOKEN = re.compile(
    r"(?P<m>\b[ap]\.m\b\.?)"
    r"|(?P<n>\d+(?:[.,]\d+)*(?::\d{2})?(?:st|nd|rd|th)?)"
    r"|(?P<w>[A-Za-z]+(?:['’][A-Za-z]+)*)"
    r"|(?P<o>\S)",
    re.IGNORECASE,
)
_NUMBER_LEXICON = frozenset({*_UNIT, *_TENS, *_ORD_SIMPLE, "zero"})


def _tokens(text: str) -> list[_Tok]:
    found = [_Tok(m.lastgroup or "o", m.group(0).casefold(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    kept: list[_Tok] = []
    for i, tok in enumerate(found):
        joins = (
            tok.kind == "o"
            and tok.low == "-"
            and 0 < i < len(found) - 1
            and found[i - 1].end == tok.start == found[i + 1].start - 1
            and found[i - 1].low in _NUMBER_LEXICON
            and found[i + 1].low in _NUMBER_LEXICON
        )
        if not joins:  # "twenty-five" reads as "twenty five"
            kept.append(tok)
    return kept


@dataclass(frozen=True)
class _Segment:
    out_start: int
    out_end: int
    src_start: int
    src_end: int
    exact: bool  # the text is the source text, so offsets inside it map one to one


@dataclass(frozen=True)
class Spoken:
    """The written form of a transcript, with a map back to the words that were said."""

    source: str
    text: str
    segments: tuple[_Segment, ...]

    @property
    def changed(self) -> bool:
        return self.text != self.source

    def said(self, start: int, end: int) -> str:
        """The words the person said for ``text[start:end]``."""
        if not self.changed:
            return self.text[start:end].strip()
        low = high = None
        for seg in self.segments:
            if seg.out_end <= start or seg.out_start >= end:
                if not (seg.out_start == seg.out_end and start <= seg.out_start <= end):
                    continue
            if seg.exact:
                lo = seg.src_start + max(start, seg.out_start) - seg.out_start
                hi = seg.src_start + min(end, seg.out_end) - seg.out_start
            else:
                lo, hi = seg.src_start, seg.src_end
            low = lo if low is None else min(low, lo)
            high = hi if high is None else max(high, hi)
        if low is None or high is None:
            return self.text[start:end].strip()
        return self.source[low:high].strip()


class _Scan:
    def __init__(self, text: str) -> None:
        self.text = text
        self.t = _tokens(text)
        self.day_ends: set[int] = set()  # token index after a day that follows its month ("November twelfth")

    def low(self, k: int) -> str:
        return self.t[k].low if 0 <= k < len(self.t) else ""

    def kind(self, k: int) -> str:
        return self.t[k].kind if 0 <= k < len(self.t) else ""

    def is_month(self, k: int) -> bool:
        return self.kind(k) == "w" and self.low(k) in _MONTHS

    def clause_ends_at(self, k: int) -> bool:
        """Token ``k`` ends the clause or starts a word that follows a date ("... the twelfth at ten")."""
        return k >= len(self.t) or self.t[k].kind == "o" or self.t[k].low in _AFTER_DATE


Conversion = tuple[int, int, str]  # first token, token after the last, written form

# ── numbers ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _cardinal(s: _Scan, i: int) -> tuple[int, int] | None:
    """The longest number in words at token ``i``: "one hundred and eighty" -> (180, end)."""
    total = cur = 0
    last = ""
    k = end = i
    while k < len(s.t) and s.t[k].kind == "w":
        w = s.t[k].low
        if w == "zero":
            return (0, k + 1) if k == i else None
        if w in _UNIT:
            v = _UNIT[w]
            if (v < 10 and last in ("unit", "teen")) or (v >= 10 and last in ("unit", "teen", "tens")):
                break
            cur += v
            last = "unit" if v < 10 else "teen"
        elif w in _TENS:
            if last in ("unit", "teen", "tens"):
                break
            cur += _TENS[w]
            last = "tens"
        elif w == "a" and last == "" and s.low(k + 1) in ("hundred", "thousand", "million"):
            cur, last = 1, "unit"
        elif w == "hundred":
            if last not in ("unit", "teen", "tens") or cur >= 100:
                break
            cur *= 100
            last = "hundred"
        elif w in ("thousand", "million"):
            if last not in ("unit", "teen", "tens", "hundred"):
                break
            total += cur * (1000 if w == "thousand" else 10**6)
            cur, last = 0, "scale"
        elif w == "and" and last in ("hundred", "scale") and s.low(k + 1) in _UNIT | _TENS:
            k += 1
            continue
        else:
            break
        k += 1
        end = k
    return (total + cur, end) if end > i else None


def _spoken_number(s: _Scan, i: int) -> tuple[int, int] | None:
    """A number in words, with the habit of "one seventy" for 170."""
    found = _cardinal(s, i)
    if found is None:
        return None
    value, end = found
    if end == i + 1 and 1 <= value <= 9 and s.low(i) in _UNIT:
        rest = _cardinal(s, end)
        if rest and 10 <= rest[0] <= 99 and s.low(end) not in ("a",):
            return value * 100 + rest[0], rest[1]
    return found


_DIGIT_WORDS = {**{w: str(v) for w, v in _UNIT.items() if v < 10}, "zero": "0", "oh": "0"}


def _decimals(s: _Scan, end: int) -> tuple[str, int] | None:
    """ "point five" after a number."""
    if s.low(end) != "point":
        return None
    k = end + 1
    digits = ""
    while s.kind(k) == "w" and s.low(k) in _DIGIT_WORDS:
        digits += _DIGIT_WORDS[s.low(k)]
        k += 1
    return (digits, k) if digits else None


def _operand(s: _Scan, i: int, *, homophone: bool = False) -> tuple[str, int] | None:
    """A number at token ``i``, written with digits: digits as they are, words as ``_spoken_number`` reads them."""
    if s.kind(i) == "n":
        raw = s.t[i].low
        return (raw.replace(",", ""), i + 1) if re.fullmatch(r"\d+(?:[.,]\d+)*", raw) else None
    if s.kind(i) != "w":
        return None
    found = _spoken_number(s, i)
    if found is None:
        if homophone and s.low(i) in _HOMOPHONE:
            return str(_HOMOPHONE[s.low(i)]), i + 1
        return None
    value, end = found
    text = str(value)
    decimal = _decimals(s, end)
    if decimal:
        text, end = f"{text}.{decimal[0]}", decimal[1]
    return text, end


# ── sums and percentages ──────────────────────────────────────────────────────────────────────────────────────────

_OPERATORS = (
    (("plus",), "+"),
    (("minus",), "-"),
    (("times",), "×"),
    (("multiplied", "by"), "×"),
    (("divided", "by"), "÷"),
    (("over",), "÷"),
)
_LINKERS = (
    ("is", "equal", "to"),
    ("comes", "to"),
    ("equals",),
    ("equal",),
    ("makes",),
    ("gives",),
    ("is",),
    ("that's",),
)
_PERCENT_CONNECTORS = (
    ("works", "out", "to"),
    ("works", "out", "at"),
    ("so", "that's"),
    ("which", "is"),
    ("that", "is"),
    ("comes", "to"),
    ("would", "be"),
    ("that's",),
    ("equals",),
    ("makes",),
    ("is",),
    ("so",),
)


def _words_at(s: _Scan, i: int, options: tuple[tuple[str, ...], ...]) -> int | None:
    for words in options:
        if all(s.kind(i + n) == "w" and s.low(i + n) == word for n, word in enumerate(words)):
            return i + len(words)
    return None


def _arithmetic(s: _Scan, i: int) -> list[Conversion] | None:
    first = _operand(s, i, homophone=True)
    if first is None:
        return None
    a, j = first
    for words, symbol in _OPERATORS:
        k = _words_at(s, j, (words,))
        if k is None:
            continue
        second = _operand(s, k, homophone=True)
        if second is None:
            continue
        b, b_end = second
        k = b_end
        link = _words_at(s, k, _LINKERS)
        if link is None:
            continue
        third = _operand(s, link, homophone=False)
        if third is None and s.low(link) in _HOMOPHONE and s.clause_ends_at(link + 1) and s.kind(link + 1) != "w":
            third = (str(_HOMOPHONE[s.low(link)]), link + 1)
        if third is None:
            continue
        c, end = third
        operator_end = _words_at(s, j, (words,)) or j
        return [(i, j, a), (j, operator_end, symbol), (operator_end, b_end, b), (b_end, link, "="), (link, end, c)]
    return None


def _percent(s: _Scan, i: int) -> list[Conversion] | None:
    first = _operand(s, i, homophone=False)
    if first is None and s.low(i) in _HOMOPHONE and s.low(i + 1) in ("percent", "per"):
        first = (str(_HOMOPHONE[s.low(i)]), i + 1)
    if first is None:
        return None
    p, j = first
    if s.low(j) == "percent":
        j += 1
    elif s.low(j) == "per" and s.low(j + 1) == "cent":
        j += 2
    else:
        return None
    done: list[Conversion] = [(i, j, f"{p}%")]
    if s.low(j) != "of":
        return done
    base = _operand(s, j + 1)
    if base is None:
        return done
    done.append((j + 1, base[1], base[0]))
    comma = base[1] + 1 if s.kind(base[1]) == "o" and s.low(base[1]) == "," else base[1]
    link = _words_at(s, comma, _PERCENT_CONNECTORS)
    if link is not None:
        result = _operand(s, link)
        if result is None and s.low(link) in _HOMOPHONE and s.kind(link + 1) != "w" and s.clause_ends_at(link + 1):
            result = (str(_HOMOPHONE[s.low(link)]), link + 1)
        if result is not None:
            done.append((link, result[1], result[0]))
    return done


# ── clock times ───────────────────────────────────────────────────────────────────────────────────────────────────


def _meridiem(s: _Scan, i: int) -> tuple[str, int] | None:
    if s.kind(i) == "m":
        return s.t[i].low[0] + "m", i + 1
    if s.kind(i) != "w":
        return None
    low = s.low(i)
    if low in ("am", "pm"):
        return low, i + 1
    if low in ("a", "p") and s.low(i + 1) == "m" and s.kind(i + 1) == "w":
        return low + "m", i + 2
    return None


def _o_clock(s: _Scan, i: int) -> int | None:
    if s.low(i) in ("o'clock", "o’clock", "oclock"):
        return i + 1
    if s.low(i) == "o" and s.low(i + 1) == "clock":
        return i + 2
    return None


def _hour(s: _Scan, i: int, *, homophone: bool = False) -> tuple[int, int | None, int] | None:
    """An hour (1 to 24) at token ``i``: its value, minutes when written "10:30", and the token after it."""
    if s.kind(i) == "n":
        match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", s.t[i].low)
        if match and 0 < int(match[1]) <= 24:
            return int(match[1]), int(match[2]) if match[2] else None, i + 1
        return None
    if s.kind(i) != "w":
        return None
    found = _cardinal(s, i)
    if found and 0 < found[0] <= 24 and s.low(i) != "a":
        return found[0], None, found[1]
    if homophone and s.low(i) in _HOMOPHONE:
        return _HOMOPHONE[s.low(i)], None, i + 1
    return None


def _minutes(s: _Scan, i: int) -> tuple[int, int] | None:
    """Minutes spoken after an hour: "thirty", "forty five", "oh five", "fifteen"."""
    if s.low(i) == "oh" and s.low(i + 1) in _UNIT and _UNIT[s.low(i + 1)] < 10:
        return _UNIT[s.low(i + 1)], i + 2
    found = _cardinal(s, i)
    if found and 10 <= found[0] <= 59 and s.low(i) != "a":
        return found
    return None


def _written_clock(hour: int, minute: int | None, ap: str | None) -> str:
    shown = f"{hour}:{minute:02d}" if minute is not None else str(hour)
    return f"{shown} {ap}" if ap else shown


def _spoken_clock(s: _Scan, i: int, prep: bool) -> tuple[int, int, int, int] | None:
    """ "half past two", "a quarter to five", "ten past four", "twenty to five": first token, token after, hour, minute."""
    low = s.low(i)
    w = i + 1 if low == "a" and s.low(i + 1) == "quarter" else i
    word = s.low(w)
    if word in ("half", "quarter") and s.kind(w) == "w":
        link = s.low(w + 1)
        if word == "half" and link == "past" and w == i:
            minute, delta = 30, 0
        elif word == "quarter" and link in ("past", "after"):
            minute, delta = 15, 0
        elif word == "quarter" and link == "to":
            minute, delta = 45, -1
        else:
            return None
        hour = _hour(s, w + 2, homophone=True)
        after = hour[2] if hour else 0
    else:
        mins = _cardinal(s, i) if low != "a" else None
        if mins is None:
            return None
        value, after = mins
        link = s.low(after)
        if link == "past" and 1 <= value <= 29:
            minute, delta = value, 0
        elif link == "to" and value in (5, 10, 20, 25) and (prep or s.low(i - 1) in ("it's", "its", "is", "was")):
            minute, delta = 60 - value, -1
        else:
            return None
        hour = _hour(s, after + 1, homophone=True)
        after = hour[2] if hour else 0
    if hour is None:
        return None
    h = hour[0] + delta or 12
    return (i, after, h, minute) if 1 <= h <= 12 else None


def _time(s: _Scan, i: int) -> list[Conversion] | None:
    prep = s.kind(i - 1) == "w" and s.low(i - 1) in _TIME_PREPS
    relative = _spoken_clock(s, i, prep)
    if relative:
        begin, j, h, minute = relative
        ap = _meridiem(s, j)
        return [(begin, ap[1] if ap else j, _written_clock(h, minute, ap[0] if ap else None))]
    hour = _hour(s, i, homophone=prep)
    if hour is None and s.low(i) in _HOMOPHONE and s.kind(i) == "w" and _o_clock(s, i + 1):
        hour = (_HOMOPHONE[s.low(i)], None, i + 1)
    if hour is None:
        return None
    h, written_minutes, j = hour
    digits = s.kind(i) == "n"
    mins = _minutes(s, j) if written_minutes is None else None
    k = mins[1] if mins else j
    minute = written_minutes if written_minutes is not None else (mins[0] if mins else None)
    ap = _meridiem(s, k)
    clock_o = _o_clock(s, j) if minute is None else None
    if s.low(i) in _HOMOPHONE and not (ap or clock_o):
        return None  # "for", "to" and "ate" are numbers only in front of "p m" or "o'clock"
    if ap:
        if digits and ap[1] == k + 1 and s.low(k) == ap[0] and (mins is None):
            return None  # written already: "10 am", "10:30 pm"
        return [(i, ap[1], _written_clock(h, minute, ap[0]))]
    if clock_o is not None:
        ap2 = _meridiem(s, clock_o)
        return [(i, ap2[1] if ap2 else clock_o, f"{h}:00" + (f" {ap2[0]}" if ap2 else ""))]
    if prep and not digits:
        if minute is not None:
            return [(i, k, _written_clock(h, minute, None))]
        if s.low(i - 1) == "at" and 1 <= h <= 12:
            return [(i, j, str(h))]
    return None


# ── dates ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def _suffix(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _ordinal(s: _Scan, i: int) -> tuple[int, int] | None:
    if s.kind(i) != "w":
        return None
    low = s.low(i)
    if low in _ORD_SIMPLE:
        return _ORD_SIMPLE[low], i + 1
    if low in ("twenty", "thirty") and s.low(i + 1) in _ORD_UNIT and s.kind(i + 1) == "w":
        return _TENS[low] + _ORD_UNIT[s.low(i + 1)], i + 2
    return None


def _after_weekday(s: _Scan, i: int) -> bool:
    k = i - 1
    if s.low(k) == "the":
        k -= 1
    while s.kind(k) == "o" and s.low(k) == ",":
        k -= 1
    if s.low(k) in _MISHEARD_WEEKDAY and s.low(k - 1) in _DAY_PREPOSITIONS:
        return True  # "for thirsday the fourteenth": the misheard weekday is read as a weekday too
    return s.low(k) in _WEEKDAYS


def _month_before(s: _Scan, i: int) -> bool:
    if not s.is_month(i - 1):
        return False
    if s.low(i - 1) not in _AMBIGUOUS_MONTHS:
        return True
    before = i - 2
    return before < 0 or s.low(before) in _DATE_PREPS or s.low(before) in _WEEKDAYS or s.kind(before) == "o"


def _month_after(s: _Scan, j: int) -> bool:
    if s.low(j) == "of" and s.is_month(j + 1):
        return True
    return s.is_month(j) and s.low(j) not in _AMBIGUOUS_MONTHS


def _day_context(s: _Scan, i: int, j: int) -> bool:
    if _after_weekday(s, i) or _month_before(s, i) or _month_after(s, j):
        return True
    # "by the twelfth", "on the twentieth": a promise's deadline, but not "on the second floor"
    return (
        s.low(i - 1) == "the"
        and s.low(i - 2) in ("by", "on", "before", "until", "till", "from")
        and s.clause_ends_at(j)
    )


def _year(s: _Scan, i: int) -> tuple[int, int] | None:
    found = _cardinal(s, i)
    if found and 1000 <= found[0] <= 2999 and s.low(i) != "a":
        return found
    if found and found[0] in (19, 20) and s.low(i) != "a":
        value, end = found
        if s.low(end) == "oh" and (digit := _UNIT.get(s.low(end + 1), 0)) and digit < 10:
            return value * 100 + digit, end + 2
        rest = _cardinal(s, end)
        if rest and 10 <= rest[0] <= 99:
            return value * 100 + rest[0], rest[1]
    return None


def _after_month_or_day(s: _Scan, i: int) -> bool:
    k = i - 1
    if s.kind(k) == "o" and s.low(k) == ",":
        k -= 1
    return s.is_month(k) or i in s.day_ends


def _date(s: _Scan, i: int) -> list[Conversion] | None:
    if _after_month_or_day(s, i) and (year := _year(s, i)):
        return [(i, year[1], str(year[0]))]
    ordinal = _ordinal(s, i)
    if ordinal and 1 <= ordinal[0] <= 31 and _day_context(s, i, ordinal[1]):
        n, j = ordinal
        if _month_before(s, i):
            s.day_ends.add(j)
        return [(i, j, f"{n}{_suffix(n)}")]
    found = _spoken_number(s, i) if s.kind(i) == "w" else None
    if found and 1 <= found[0] <= 31 and s.low(i) != "a":
        n, j = found
        if _month_after(s, j) and s.low(j) == "of":
            return [(i, j, str(n))]
        if _month_before(s, i) and s.low(i - 1) not in _AMBIGUOUS_MONTHS and s.clause_ends_at(j):
            s.day_ends.add(j)
            return [(i, j, str(n))]
        if _month_after(s, j) and (_after_weekday(s, i) or s.low(i - 1) == "the"):
            return [(i, j, str(n))]
    return None


# ── the pass over a transcript ────────────────────────────────────────────────────────────────────────────────────


# Weekday names as recognisers mishear them, from transcripts of real speech ("I'll send Priya the deck by frida").
# Only replaced right after a word that introduces a day ("by", "on", "until"...), so "lunch with Frida" stays a
# name. A short explicit list, never fuzzy matching: a guessed weekday that was really a name would be a false catch.
_MISHEARD_WEEKDAY = {
    "frida": "Friday", "fryday": "Friday", "friady": "Friday",
    "wensday": "Wednesday", "wenesday": "Wednesday", "wednsday": "Wednesday", "wendsday": "Wednesday",
    "thurday": "Thursday", "thursdy": "Thursday", "thirsday": "Thursday",
    "tuseday": "Tuesday", "tusday": "Tuesday", "teusday": "Tuesday",
    "munday": "Monday", "mondy": "Monday",
    "saterday": "Saturday", "satuday": "Saturday",
}  # fmt: skip
_DAY_PREPOSITIONS = frozenset({"by", "on", "until", "till", "before", "next", "this", "from", "for", "after"})


def normalise(text: str) -> Spoken:
    """The written form of a transcript. Text with nothing to convert comes back as it is."""
    scan = _Scan(text)
    pieces: list[tuple[str, int, int, bool]] = []  # out text, source start, source end, exact
    pos = 0  # the source text up to here is already in ``pieces``
    consumed = 0  # tokens before this one belong to a conversion
    i = 0
    while i < len(scan.t):
        tok = scan.t[i]
        conversions: list[Conversion] | None = None
        if tok.kind in ("w", "n"):
            for attempt in (_arithmetic, _percent, _time, _date):
                try:
                    conversions = attempt(scan, i)
                except (ValueError, IndexError):
                    conversions = None
                if conversions:
                    break
            if not conversions and tok.low in _MISHEARD_WEEKDAY and i > 0 and scan.low(i - 1) in _DAY_PREPOSITIONS:
                conversions = [(i, i + 1, _MISHEARD_WEEKDAY[tok.low])]
            if (
                not conversions
                and tok.low in _RELATIVE
                and scan.low(i + 1) in _WEEKDAYS
                and scan.t[i + 1].start - tok.end <= 2
            ):
                conversions = [(i, i + 1, f"{text[tok.start : tok.end]},")]
        if not conversions or min(c[0] for c in conversions) < consumed:
            i += 1
            continue
        for first, after, written in conversions:
            start, end = scan.t[first].start, scan.t[after - 1].end
            if start < pos:
                continue
            if start > pos:
                pieces.append((text[pos:start], pos, start, True))
            pieces.append((written, start, end, False))
            pos = end
            consumed = after
        i = max(i + 1, consumed)
    if pos < len(text):
        pieces.append((text[pos:], pos, len(text), True))
    segments: list[_Segment] = []
    out = ""
    for written, src_start, src_end, exact in pieces:
        segments.append(_Segment(len(out), len(out) + len(written), src_start, src_end, exact))
        out += written
    return Spoken(text, out, tuple(segments))


def to_written(text: str) -> str:
    """Just the written form of a transcript."""
    return normalise(text).text


__all__ = [
    "Spoken",
    "cardinal_words",
    "date_words",
    "normalise",
    "number_words",
    "ordinal_words",
    "time_words",
    "to_written",
    "year_words",
]

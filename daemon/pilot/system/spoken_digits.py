"""Every number as words, for a voice that cannot read digits.

Every road's words already pass through ``pilot.conversation.style.speakable_numbers`` on their way to the voice
(PilotServer's speech path): money, times, ordinals up to 31, percentages, dates and thousands become words, and
plain years, counts, decimals, ranges and digit groups are left for the voice, because Kokoro's reader says those
well. Pocket TTS does not: on 19 November it read "called at 3:15" as "three feet" and "March 2026" as "March 2020".
``words_for_voice`` runs the same conversion and then turns every digit left into words, so nothing reaches such a
voice as digits:

    phone-like groups   "555-0142", "+44 20 7946 0018"   digit by digit, a pause between groups
    ranges              "3-5", "10:00–11:00", "2020-2026"  "three to five", "ten to eleven", ...
    money with a size   "$2.5m", "€40k"                  "two point five million dollars", "forty thousand euros"
    years               "in 2026", "March 2026"          "twenty twenty-six" (after a month or a year word)
    versions, addresses "1.2.0", "192.168.1.1"           "one point two point zero", "one nine two dot ..."
    decimals            "3.5"                            "three point five"
    ordinals            "100th"                          "one hundredth"
    degrees             "-3°C"                           "minus three degrees Celsius"
    leading zeros       "007"                            "zero zero seven"
    everything else     "64", "2048"                     "sixty-four", "two thousand and forty-eight"

Pure text, English words only (Pocket speaks English); idempotent.
"""

from __future__ import annotations

import re
from datetime import date

from pilot.live.spoken import cardinal_words, number_words, ordinal_words, year_words

_DIGIT_NAMES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_SIZES = {"k": "thousand", "m": "million", "bn": "billion", "b": "billion", "thousand": "thousand",
          "million": "million", "billion": "billion"}  # fmt: skip
_CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros", "₹": "rupees"}
_MONTH_WORDS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_YEAR_BEFORE = re.compile(
    r"(?:\b(?:" + _MONTH_WORDS + r")\.?|\b(?:in|since|of|year|from|until|till|by|before|after|between|early|late|"
    r"mid|around|circa|during|through)|\b(?:1[1-9]\d\d|20\d\d) to)[,]?\s$",
    re.IGNORECASE,
)

_PHONE = re.compile(
    r"(?<![\w.,:/])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,5}(?:[\s.-]\d{2,5}){1,4}(?![\w,:/]|\.\d)"
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SIZED_MONEY = re.compile(
    r"(?<![\w.])([$£€₹])\s?(\d+(?:\.\d+)?)\s?(k|m|bn|b|thousand|million|billion)\b", re.IGNORECASE
)
_RANGE = re.compile(r"(?<![\w.:/-])(\d+(?:[.:]\d+)?)\s?[-–—]\s?(\d+(?:[.:]\d+)?)(?![\w/]|[.:]\d|-\d)")
_DEGREES = re.compile(r"(-?\d+(?:\.\d+)?)\s?°\s?([CF])?\b")
_DOTTED = re.compile(r"(?<![\w.])v?(\d+(?:\.\d+){2,})(?![\w]|\.\d)")
_DECIMAL = re.compile(r"(?<![\w.])(-?\d+\.\d+)(?![\w]|\.\d)")
_ORDINAL = re.compile(r"(?<![\w.])(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_YEAR = re.compile(r"(?<![\w.,])(1[1-9]\d\d|20\d\d)(?![\w]|[.,]\d)")
_INTEGER = re.compile(r"(?<![\w.])(-?)(\d+)(?![\w]|\.\d)")
_LEFT_DIGITS = re.compile(r"\d+")


def digit_words(digits: str) -> str:
    return " ".join(_DIGIT_NAMES[int(d)] for d in digits if d.isdigit())


def ordinal_any(n: int) -> str:
    """Any ordinal: 3 -> "third", 100 -> "one hundredth", 121 -> "one hundred and twenty-first"."""
    if n <= 31:
        return ordinal_words(n)
    words = cardinal_words(n)
    head, sep, last = words.rpartition(" ")
    unit_head, dash, unit = last.rpartition("-")
    if dash:
        return f"{head}{sep}{unit_head}-{ordinal_words(_number_of(unit))}"
    if last in {"hundred", "thousand", "million", "billion"}:
        return f"{words}th"
    return f"{head}{sep}{ordinal_words(_number_of(last))}"


def _number_of(word: str) -> int:
    for n in range(1, 100):
        if cardinal_words(n) == word:
            return n
    raise ValueError(word)  # pragma: no cover - only called with cardinal_words' own words


def _phone(match: re.Match[str]) -> str:
    text = match.group(0)
    if _ISO_DATE.match(text):
        return text
    groups = re.findall(r"\d+", text)
    digits = sum(len(group) for group in groups)
    separators = re.sub(r"[\d\s]", "", text)
    if digits < 7:
        return text
    if len(groups) == 2 and all(len(group) == 4 and 1100 <= int(group) <= 2099 for group in groups):
        return text  # "2020-2026": a range of years    # "4 200 000" is a number written with thousands spaces, not a phone number.
    if not separators and all(len(group) == 3 for group in groups[1:]) and len(groups[0]) <= 3:
        return text
    plus = "plus " if text.startswith("+") else ""
    return plus + ", ".join(digit_words(group) for group in groups)


def _sized_money(match: re.Match[str]) -> str:
    return f"{number_words(match.group(2))} {_SIZES[match.group(3).casefold()]} {_CURRENCY[match.group(1)]}"


def _range(match: re.Match[str]) -> str:
    return f"{match.group(1)} to {match.group(2)}"


def _degrees(match: re.Match[str]) -> str:
    scale = {"C": " Celsius", "F": " Fahrenheit"}.get(match.group(2) or "", "")
    value = match.group(1)
    unit = "degree" if value.lstrip("-") == "1" else "degrees"
    return f"{number_words(value)} {unit}{scale}"


def _dotted(match: re.Match[str]) -> str:
    parts = match.group(1).split(".")
    if len(parts) == 4 and all(len(part) <= 3 for part in parts):  # an address: digit by digit, "dot" between
        return " dot ".join(digit_words(part) for part in parts)
    return " point ".join(cardinal_words(int(part)) if not part.startswith("0") or part == "0" else digit_words(part)
                          for part in parts)  # fmt: skip


def _integer(match: re.Match[str]) -> str:
    sign, digits = match.groups()
    if len(digits) > 1 and digits.startswith("0"):
        return ("minus " if sign else "") + digit_words(digits)
    if len(digits) > 12:
        return ("minus " if sign else "") + digit_words(digits)
    return ("minus " if sign else "") + cardinal_words(int(digits))


def words_for_voice(text: str, today: date | None = None) -> str:
    """``text`` with no digits left: everything speakable_numbers converts, then every other number as words."""
    from pilot.conversation.style import speakable_numbers

    out = _PHONE.sub(_phone, text)
    out = _SIZED_MONEY.sub(_sized_money, out)
    out = _DEGREES.sub(_degrees, out)
    out = _RANGE.sub(_range, out)
    out = speakable_numbers(out, today)
    out = _DOTTED.sub(_dotted, out)
    out = _DECIMAL.sub(lambda m: number_words(m.group(1)), out)
    out = _ORDINAL.sub(lambda m: ordinal_any(int(m.group(1))), out)

    def year(match: re.Match[str]) -> str:
        before = out_year[: match.start()]
        if _YEAR_BEFORE.search(before[-40:]):
            return year_words(int(match.group(1)))
        return match.group(0)

    out_year = out
    out = _YEAR.sub(year, out_year)
    out = _INTEGER.sub(_integer, out)
    # Whatever digits are still glued to letters ("A4", "3x") are said digit by digit.
    out = _LEFT_DIGITS.sub(lambda m: f" {digit_words(m.group(0))} ", out)
    out = " ".join(out.split()).replace(" ,", ",").replace(" .", ".")
    if text.rstrip().endswith((".", "!", "?")) and not out.endswith((".", "!", "?")):
        out += text.rstrip()[-1]  # "at 9 a.m." keeps its full stop: the voice ends the sentence there
    return out

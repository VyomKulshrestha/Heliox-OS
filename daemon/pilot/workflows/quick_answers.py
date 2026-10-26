"""Quick answers computed in code: sums, percentages, unit conversions, dates and world time.

"What's 15 percent of 240", "convert 5 miles to km", "how many days until Christmas",
"what time is it in Tokyo". These are answered exactly and instantly, without a model, so
they are never rounded wrongly or guessed. Anything that does not parse cleanly is left
to the conversation, so "convert this video to mp4" still reaches the planner.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from datetime import date, datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------- arithmetic

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_WORDS = (
    (r"\bmultiplied by\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bto the power of\b", "**"),
    (r"\braised to\b", "**"),
    (r"\bsquared\b", "**2"),
    (r"\bcubed\b", "**3"),
    (r"\bsquare root of\b", "sqrt "),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\btimes\b", "*"),
    (r"\bover\b", "/"),
    (r"(?<=\d)\s*x\s*(?=\d)", "*"),
    (r"×", "*"),
    (r"÷", "/"),
)
MATH_REQUEST = re.compile(r"^(?:what s|what is|whats|calculate|compute|how much is|work out)\s+(?P<expr>.+)$")
PERCENT_OF = re.compile(r"^(?P<p>[\d.]+)\s*(?:%|percent)\s+of\s+(?P<n>[\d.]+)$")
PERCENT_OFF = re.compile(r"^(?P<p>[\d.]+)\s*(?:%|percent)\s+off\s+(?P<n>[\d.]+)$")
TIP = re.compile(r"^(?:a\s+)?(?P<p>[\d.]+)\s*(?:%|percent)\s+tip\s+on\s+(?P<n>[\d.]+)$")


def _number(value: float) -> str:
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        raise ValueError("not a number")
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    text = f"{value:,.10g}" if abs(value) >= 1e-4 else f"{value:.4g}"
    if "e" not in text and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sqrt"
        and len(node.args) == 1
        and not node.keywords
    ):
        return math.sqrt(_evaluate(node.args[0]))
    raise ValueError("unsupported")


def calculate(expression: str) -> str | None:
    """The answer to a spoken sum, or None when it is not plain arithmetic."""
    text = expression.casefold().strip().rstrip("?.! ").replace(",", "")
    for pattern, kind in ((PERCENT_OF, "of"), (PERCENT_OFF, "off"), (TIP, "tip")):
        match = pattern.match(text)
        if match:
            percent, amount = float(match.group("p")), float(match.group("n"))
            part = percent / 100 * amount
            if kind == "of":
                return f"{_number(percent)}% of {_number(amount)} is {_number(part)}."
            if kind == "off":
                return (
                    f"{_number(percent)}% off {_number(amount)} is {_number(amount - part)} (you save {_number(part)})."
                )
            return f"A {_number(percent)}% tip on {_number(amount)} is {_number(part)}, so {_number(amount + part)} in total."
    text = re.sub(r"^(?:the )", "", text)
    for pattern, replacement in _WORDS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"sqrt\s*([\d.]+)", r"sqrt(\1)", text)
    text = re.sub(r"(\d)\s*%", r"(\1/100)", text)
    if (
        not re.search(r"\d", text)
        or not re.search(r"[-+*/%]|sqrt", text)
        or re.search(r"[a-z_]", text.replace("sqrt", ""))
    ):
        return None
    try:
        tree = ast.parse(text, mode="eval")
        value = _evaluate(tree)
        answer = _number(value)
    except ZeroDivisionError:
        return "You can't divide by zero."
    except (SyntaxError, ValueError, OverflowError, TypeError):
        return None
    return f"That's {answer}."


# ---------------------------------------------------------------------------- units

_UNITS: dict[str, tuple[str, float]] = {}


def _unit(dimension: str, factor: float, *names: str) -> None:
    for name in names:
        _UNITS[name] = (dimension, factor)


_unit("length", 0.001, "mm", "millimeter", "millimeters", "millimetre", "millimetres")
_unit("length", 0.01, "cm", "centimeter", "centimeters", "centimetre", "centimetres")
_unit("length", 1.0, "m", "meter", "meters", "metre", "metres")
_unit("length", 1000.0, "km", "kms", "kilometer", "kilometers", "kilometre", "kilometres")
_unit("length", 0.0254, "inch", "inches")
_unit("length", 0.3048, "ft", "foot", "feet")
_unit("length", 0.9144, "yd", "yard", "yards")
_unit("length", 1609.344, "mi", "mile", "miles")
_unit("mass", 1e-6, "mg", "milligram", "milligrams")
_unit("mass", 0.001, "g", "gram", "grams")
_unit("mass", 1.0, "kg", "kgs", "kilo", "kilos", "kilogram", "kilograms")
_unit("mass", 1000.0, "tonne", "tonnes", "metric ton", "metric tons")
_unit("mass", 0.028349523125, "oz", "ounce", "ounces")
_unit("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_unit("mass", 6.35029318, "stone", "stones")
_unit("volume", 0.001, "ml", "milliliter", "milliliters", "millilitre", "millilitres")
_unit("volume", 1.0, "l", "liter", "liters", "litre", "litres")
_unit("volume", 0.00492892159375, "tsp", "teaspoon", "teaspoons")
_unit("volume", 0.01478676478125, "tbsp", "tablespoon", "tablespoons")
_unit("volume", 0.0295735295625, "fl oz", "fluid ounce", "fluid ounces")
_unit("volume", 0.2365882365, "cup", "cups")
_unit("volume", 0.473176473, "pint", "pints")
_unit("volume", 0.946352946, "quart", "quarts")
_unit("volume", 3.785411784, "gallon", "gallons", "gal")
_unit("speed", 1.0, "m s", "meters per second", "metres per second")
_unit("speed", 1 / 3.6, "km h", "kmh", "kph", "kilometers per hour", "kilometres per hour")
_unit("speed", 0.44704, "mph", "miles per hour")
_unit("speed", 0.514444, "knot", "knots")
_unit("time", 1.0, "second", "seconds", "sec", "secs")
_unit("time", 60.0, "minute", "minutes", "min", "mins")
_unit("time", 3600.0, "hour", "hours", "hr", "hrs")
_unit("time", 86400.0, "day", "days")
_unit("time", 604800.0, "week", "weeks")
_TEMPERATURE = {
    "c": "C", "celsius": "C", "centigrade": "C", "degrees celsius": "C", "degrees c": "C",
    "f": "F", "fahrenheit": "F", "degrees fahrenheit": "F", "degrees f": "F",
    "k": "K", "kelvin": "K", "kelvins": "K",
}  # fmt: skip
_DISPLAY = {"km h": "km/h", "m s": "m/s", "kmh": "km/h", "kph": "km/h"}
_UNIT_NAMES = sorted(list(_UNITS) + list(_TEMPERATURE), key=len, reverse=True)
_UNIT_RE = "|".join(re.escape(name) for name in _UNIT_NAMES)
_AMOUNT = r"(?P<n>-?[\d.]+|a|an|one)"
CONVERT_REQUEST = re.compile(
    rf"^(?:convert |what s |what is |how much is |how many )?{_AMOUNT} (?P<src>{_UNIT_RE}) (?:to|in|into|as) "
    rf"(?P<dst>{_UNIT_RE})$"
    rf"|^how many (?P<dst2>{_UNIT_RE}) (?:are |is )?(?:there )?(?:in|is) {_AMOUNT.replace('n>', 'n2>')} (?P<src2>{_UNIT_RE})$"
)


def _temperature(value: float, src: str, dst: str) -> float:
    kelvin = {"C": value + 273.15, "F": (value - 32) * 5 / 9 + 273.15, "K": value}[src]
    return {"C": kelvin - 273.15, "F": (kelvin - 273.15) * 9 / 5 + 32, "K": kelvin}[dst]


def _round(value: float) -> str:
    if abs(value) >= 100:
        return _number(round(value, 1))
    if abs(value) >= 1:
        return _number(round(value, 2))
    return _number(float(f"{value:.3g}"))


def convert(text: str) -> str | None:
    match = CONVERT_REQUEST.match(text)
    if not match:
        return None
    groups = match.groupdict()
    raw = groups.get("n") or groups.get("n2")
    src, dst = groups.get("src") or groups.get("src2"), groups.get("dst") or groups.get("dst2")
    try:
        value = 1.0 if raw in ("a", "an", "one") else float(raw)
    except (TypeError, ValueError):
        return None
    if src in _TEMPERATURE and dst in _TEMPERATURE:
        result = _temperature(value, _TEMPERATURE[src], _TEMPERATURE[dst])
        return f"{_round(value)}°{_TEMPERATURE[src]} is {_round(result)}°{_TEMPERATURE[dst]}."
    if src not in _UNITS or dst not in _UNITS:
        return None
    (src_dim, src_factor), (dst_dim, dst_factor) = _UNITS[src], _UNITS[dst]
    if src_dim != dst_dim:
        return f"I can't convert {src} to {dst}: one is a {src_dim} and the other a {dst_dim}."
    result = value * src_factor / dst_factor
    return f"{_round(value)} {_DISPLAY.get(src, src)} is {_round(result)} {_DISPLAY.get(dst, dst)}."


# ---------------------------------------------------------------------------- dates and time

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]  # fmt: skip
_MONTH_RE = "|".join(_MONTHS + [month[:3] for month in _MONTHS if month != "may"] + ["sept"])
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_DATE = rf"(?:(?P<month>{_MONTH_RE}) {_DAY}|{_DAY.replace('day>', 'day2>')} (?:of )?(?P<month2>{_MONTH_RE}))(?: (?P<year>\d{{4}}))?"
_HOLIDAYS = {
    "christmas": (12, 25),
    "christmas day": (12, 25),
    "christmas eve": (12, 24),
    "new year": (1, 1),
    "new year s": (1, 1),
    "new year s day": (1, 1),
    "new years": (1, 1),
    "new years day": (1, 1),
    "new year s eve": (12, 31),
    "new years eve": (12, 31),
    "halloween": (10, 31),
    "valentine s day": (2, 14),
    "valentines day": (2, 14),
}
_HOLIDAY_RE = "|".join(sorted((re.escape(name) for name in _HOLIDAYS), key=len, reverse=True))
UNTIL_REQUEST = re.compile(
    rf"^how many (?:days|sleeps) (?:are there |is it |left )?(?:until|till|to|before) (?:(?P<holiday>{_HOLIDAY_RE})|{_DATE})$"
)
SINCE_REQUEST = re.compile(rf"^how many days (?:has it been |is it |have passed )?since {_DATE}$")
WEEKDAY_REQUEST = re.compile(
    rf"^what day (?:of the week )?(?:is|was|will be|falls on) (?:(?P<holiday>{_HOLIDAY_RE})|{_DATE})(?: this year)?$"
)
OFFSET_REQUEST = re.compile(
    r"^(?:what s |what is |what will be )?(?:the )?date (?:in |)(?P<n>\d+|a|one|two|three|four) (?P<unit>days?|weeks?|months?)"
    r"(?: from (?:now|today))?$"
    r"|^what (?:date|day) (?:is|will it be) (?:in )?(?P<n2>\d+|a|one|two|three|four) (?P<unit2>days?|weeks?|months?)"
    r"(?: from (?:now|today))?$"
)
TODAY_REQUEST = re.compile(
    r"^(?:what s|what is|whats) (?:the date|today s date|the date today)(?: today)?$|^what day is (?:it|today)(?: today)?$"
)
TIME_REQUEST = re.compile(r"^(?:what time is it|what s the time|what is the time)(?: now| right now)?$")
ZONE_REQUEST = re.compile(
    r"^(?:what time is it|what s the time|what is the time|time) (?:now |right now )?in (?P<place>[\w ]+)$"
)
_SMALL = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4}
CITIES = {
    "london": "Europe/London", "uk": "Europe/London", "england": "Europe/London", "dublin": "Europe/Dublin",
    "paris": "Europe/Paris", "france": "Europe/Paris", "berlin": "Europe/Berlin", "germany": "Europe/Berlin",
    "madrid": "Europe/Madrid", "rome": "Europe/Rome", "amsterdam": "Europe/Amsterdam", "lisbon": "Europe/Lisbon",
    "zurich": "Europe/Zurich", "stockholm": "Europe/Stockholm", "moscow": "Europe/Moscow",
    "istanbul": "Europe/Istanbul", "dubai": "Asia/Dubai", "cairo": "Africa/Cairo", "lagos": "Africa/Lagos",
    "nairobi": "Africa/Nairobi", "johannesburg": "Africa/Johannesburg", "india": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "bangalore": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata", "kolkata": "Asia/Kolkata", "chennai": "Asia/Kolkata", "hyderabad": "Asia/Kolkata",
    "karachi": "Asia/Karachi", "dhaka": "Asia/Dhaka", "bangkok": "Asia/Bangkok", "singapore": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong", "shanghai": "Asia/Shanghai", "beijing": "Asia/Shanghai", "china": "Asia/Shanghai",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo", "seoul": "Asia/Seoul", "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne", "perth": "Australia/Perth", "auckland": "Pacific/Auckland",
    "new york": "America/New_York", "nyc": "America/New_York", "boston": "America/New_York",
    "washington": "America/New_York", "miami": "America/New_York", "toronto": "America/Toronto",
    "chicago": "America/Chicago", "dallas": "America/Chicago", "houston": "America/Chicago",
    "denver": "America/Denver", "phoenix": "America/Phoenix", "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles", "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "vancouver": "America/Vancouver", "mexico city": "America/Mexico_City", "sao paulo": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires", "utc": "UTC", "gmt": "UTC",
}  # fmt: skip


def _month(name: str) -> int:
    name = name[:3]
    return next(index for index, month in enumerate(_MONTHS, 1) if month.startswith(name))


def _target(groups: dict[str, Any], today: date, *, future: bool) -> date | None:
    if groups.get("holiday"):
        month, day = _HOLIDAYS[groups["holiday"]]
        year = None
    else:
        month = _month(groups.get("month") or groups.get("month2") or "")
        day = int(groups.get("day") or groups.get("day2") or 0)
        year = int(groups["year"]) if groups.get("year") else None
    try:
        target = date(year or today.year, month, day)
        if year is None and future and target < today:
            target = date(today.year + 1, month, day)
        if year is None and not future and target > today:
            target = date(today.year - 1, month, day)
    except ValueError:
        return None
    return target


def _long(day: date) -> str:
    return f"{day.strftime('%A')}, {day.day} {day.strftime('%B %Y')}"


def _clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def date_answer(text: str, now: datetime) -> str | None:
    today = now.date()
    if TIME_REQUEST.match(text):
        return f"It's {_clock(now)}."
    if TODAY_REQUEST.match(text):
        return f"Today is {_long(today)}."
    zone = ZONE_REQUEST.match(text)
    if zone:
        place = zone.group("place").strip()
        name = CITIES.get(place)
        if name is None:
            return None
        try:
            from zoneinfo import ZoneInfo

            there = datetime.now(ZoneInfo(name))
        except Exception:
            return "I can't look up time zones on this computer."
        day_note = ""
        if there.date() > today:
            day_note = " tomorrow"
        elif there.date() < today:
            day_note = " yesterday"
        return f"It's {_clock(there)}{day_note} in {place.title() if len(place) > 3 else place.upper()}."
    until = UNTIL_REQUEST.match(text)
    if until:
        target = _target(until.groupdict(), today, future=True)
        if target is None:
            return "That is not a date I understand."
        days = (target - today).days
        if days == 0:
            return f"That's today, {_long(target)}."
        return f"{days} day{'' if days == 1 else 's'} until {_long(target)}."
    since = SINCE_REQUEST.match(text)
    if since:
        target = _target(since.groupdict(), today, future=False)
        if target is None:
            return "That is not a date I understand."
        days = (today - target).days
        return f"{days} day{'' if days == 1 else 's'} since {_long(target)}."
    weekday = WEEKDAY_REQUEST.match(text)
    if weekday:
        target = _target(weekday.groupdict(), today, future=True)
        return None if target is None else f"{_long(target)}."
    offset = OFFSET_REQUEST.match(text)
    if offset:
        raw = offset.group("n") or offset.group("n2")
        unit = offset.group("unit") or offset.group("unit2")
        amount = int(raw) if raw.isdigit() else _SMALL[raw]
        if unit.startswith("month"):
            month_index = today.month - 1 + amount
            year, month = today.year + month_index // 12, month_index % 12 + 1
            day = min(today.day, [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28, 31, 30, 31,
                                  30, 31, 31, 30, 31, 30, 31][month - 1])  # fmt: skip
            target = date(year, month, day)
        else:
            target = today + timedelta(days=amount * (7 if unit.startswith("week") else 1))
        return f"{_long(target)}."
    return None


# ---------------------------------------------------------------------------- entry point


def _normalize(text: str) -> str:
    text = text.casefold().strip().rstrip("?.! ")
    text = text.replace("'", " ").replace("’", " ").replace("/", " ")
    return " ".join(text.split())


class QuickAnswers:
    """Sums, conversions, dates and world time, answered in code."""

    def __init__(self, now: Any = datetime.now) -> None:
        self._now = now

    def answer(self, text: str) -> str | None:
        normalized = _normalize(text)
        answer = date_answer(normalized, self._now())
        if answer is not None:
            return answer
        answer = convert(normalized)
        if answer is not None:
            return answer
        asked = MATH_REQUEST.match(text.casefold().strip().replace("’", "'").replace("what's", "what s"))
        if asked:
            return calculate(asked.group("expr"))
        if re.fullmatch(r"[\d\s.,()+\-*/x×÷%^]+", text.strip().rstrip("?=")) and re.search(
            r"\d\s*[-+*/x×÷^]\s*\d", text
        ):
            return calculate(text.replace("^", "**"))
        return None

    async def answer_async(self, text: str) -> str | None:
        return self.answer(text)

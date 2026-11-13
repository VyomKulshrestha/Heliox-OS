"""Reminders: "remind me in 20 minutes to stretch", kept on this computer and answered in code.

A reminder is a time and a short text. Setting one never plans or runs an action: it is
stored in ``DATA_DIR/reminders.json`` and, when due, shown as a notification and in
chat and spoken. A reminder that fell due while Heliox was not running is delivered
when it starts again, marked as late, unless it is more than a day old.

Repeating reminders ("remind me every hour to stand up", "remind me every weekday at 9
to check email") are rescheduled each time they go out. Ones that repeat every few
minutes or hours skip the night (10 PM to 7 AM), and a missed one is not delivered late:
the next one comes on time instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pilot.turn_changes import note_change

logger = logging.getLogger("pilot.workflows.reminders")

CHECK_EVERY_SECONDS = 15.0
MAX_REMINDERS = 100
MAX_TEXT = 300
# A reminder more than this late came due while the computer was asleep or Heliox was off. Missed reminders are
# never dropped, whatever their age: they go out once, together (``deliver_missed``).
MISSED_AFTER_SECONDS = 120.0
MIN_REPEAT_MINUTES = 15
NIGHT_START, NIGHT_END = 22, 7  # repeating-by-interval reminders stay quiet in between
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip

SET_REQUEST = re.compile(r"^(?:please |hey heliox |heliox )?remind me (?P<rest>.+)$")
LIST_REQUEST = re.compile(
    r"^(?:hey heliox |heliox )?(?:what are my reminders|what reminders do i have|list my reminders|show my reminders)$"
)
CANCEL_REQUEST = re.compile(
    r"^(?:please )?(?:cancel|delete|remove|clear) (?:all )?(?:my |the )?reminders?(?: (?:to|about) (?P<what>.+))?$"
    r"|^(?:please )?stop reminding me (?:to |about )?(?P<what2>.+)$"
)
TIMER_REQUEST = re.compile(
    r"^(?:please )?(?:set|start) (?:a |an )?(?:timer for (?P<a1>\w+) (?P<u1>seconds?|secs?|minutes?|mins?|hours?)"
    r"|(?P<a2>\w+) (?P<u2>second|sec|minute|min|hour) timer)$"
)
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15}
NUMBER_WORDS |= {"twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "six": 6, "eight": 8, "twelve": 12}
_REPEAT = (
    r"every (?:(?P<n>\d+|two|three|four|five|six|eight|ten|twelve|fifteen|twenty|thirty|forty five|forty) )?"
    r"(?P<unit>minutes?|mins?|hours?)"
    r"|every (?P<days>day|weekday|weekend|morning|evening|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?: (?:at|by) (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))? ?(?P<meridiem>am|pm|a m|p m)?)?"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s:]", " ", text.casefold()).split())


_DAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _clock(moment: datetime) -> str:
    """ "9:00 AM", whatever the system's locale (strftime's %p is empty in a 24-hour locale). The voice says it
    from this written form (pilot.conversation.style.speakable_numbers)."""
    return f"{moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def describe_time(due: float, now: datetime) -> str:
    moment = datetime.fromtimestamp(due)
    if moment.date() == now.date():
        return f"at {_clock(moment)}"
    if moment.date() == (now + timedelta(days=1)).date():
        return f"tomorrow at {_clock(moment)}"
    year = f" {moment.year}" if moment.year != now.year else ""  # so the voice can say it, and the weekday is right
    return (
        f"on {_DAY_ABBR[moment.weekday()]} {moment.day:02d} {_MONTH_ABBR[moment.month - 1]}{year} at {_clock(moment)}"
    )


def _parse_clock(hour: str, minute: str | None, meridiem: str | None, now: datetime) -> tuple[int, int] | None:
    h, m = int(hour), int(minute or 0)
    meridiem = (meridiem or "").replace(".", "").replace(" ", "")
    if meridiem == "pm" and h < 12:
        h += 12
    elif meridiem == "am" and h == 12:
        h = 0
    elif not meridiem and 1 <= h <= 7 and now.hour >= 8:
        h += 12  # "at 5" said during the day means 5 PM
    return (h, m) if 0 <= h <= 23 and 0 <= m <= 59 else None


_SAID_DATE = re.compile(
    r"(?:(?:on|is|for) )?(?:(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday) )?(?:the )?"
    r"(?:(?P<dom>\d{1,2})(?:st|nd|rd|th)? (?:of )?(?P<month>" + "|".join(MONTHS) + r")"
    r"|(?P<month2>" + "|".join(MONTHS) + r") (?:the )?(?P<dom2>\d{1,2})(?:st|nd|rd|th)?)"
    r"(?: (?:at|by) (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))? ?(?P<meridiem>am|pm|a m|p m)?)?"
)


def _said_date(rest: str, now: datetime) -> tuple[datetime, str] | None:
    """ "that the launch call is Monday 12 November at 10": the date and time named anywhere in a "that" sentence,
    and the sentence itself. The date wins over a weekday said with it (Heliox Live asks about a mismatch)."""
    if not rest.startswith("that "):
        return None
    found = _SAID_DATE.search(rest)
    if found is None:
        return None
    month = MONTHS.index(found.group("month") or found.group("month2")) + 1
    clock = (9, 0)
    if found.group("hour"):
        clock = _parse_clock(found.group("hour"), found.group("minute"), found.group("meridiem"), now)
        if clock is None:
            return None
    try:
        due = datetime(now.year, month, int(found.group("dom") or found.group("dom2")), clock[0], clock[1])
    except ValueError:
        return None
    if due <= now:
        due = due.replace(year=now.year + 1)
    if due - now > timedelta(days=366):
        return None
    return due, rest.removeprefix("that ").strip()


_DOW = "|".join(WEEKDAYS)
_MON = "|".join(MONTHS)
_B, _E = r"(?<![\w:])", r"(?![\w:])"
_ON = r"(?:(?:on|for|by|this|next|before|until) )?"
_ORDINAL = r"(?:st|nd|rd|th)"
# Each kind of "when" a person puts around the thing to be reminded of, before it, after it or both. Every match of
# a kind is taken out of the sentence, so what is left is only the thing ("on Thursday 19 November to send the
# report" leaves "send the report").
_DATE_FORMS = (
    # "Thursday 19 November", "the 19th of November", "19 november"
    _B
    + _ON
    + r"(?:(?P<wd>"
    + _DOW
    + r"),? )?(?:the )?(?P<d>\d{1,2})"
    + _ORDINAL
    + r"?(?: of)? (?P<m>"
    + _MON
    + r")"
    + _E,
    # "November 19th", "Thursday, November the 19th"
    _B + _ON + r"(?:(?P<wd>" + _DOW + r"),? )?(?P<m>" + _MON + r") (?:the )?(?P<d>\d{1,2})" + _ORDINAL + r"?" + _E,
    # "Thursday the 19th", "the 19th", "on the 19th": no month, so the ordinal or "the" says it is a day
    _B + _ON + r"(?:(?P<wd>" + _DOW + r"),? )?the (?P<d>\d{1,2})" + _ORDINAL + r"?" + _E,
    _B + _ON + r"(?:(?P<wd>" + _DOW + r"),? )?(?P<d>\d{1,2})" + _ORDINAL + _E,
)
_DATES = tuple(re.compile(form) for form in _DATE_FORMS)
_IN = re.compile(
    _B + r"in (?:(?P<half>half an hour)|(?P<amount>\d+|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty"
    r"|forty five|forty) (?P<unit>minutes?|mins?|hours?))" + _E
)
_DAY_WORD = re.compile(_B + r"(?P<day>tomorrow|today|tonight)(?: (?P<part>morning|afternoon|evening|night))?" + _E)
_WEEKDAY = re.compile(
    _B
    + r"(?:on|this|next|by|for|before|until) (?P<wd>"
    + _DOW
    + r")(?: (?P<part>morning|afternoon|evening|night))?"
    + _E
)
_CLOCK = re.compile(
    _B + r"(?:(?:at|by|around) (?P<h>\d{1,2})(?::(?P<mi>\d{2}))? ?(?P<mer>am|pm|a m|p m)?(?: o clock)?"
    r"|(?:at )?(?P<h2>\d{1,2})(?::(?P<mi2>\d{2}))? ?(?P<mer2>am|pm|a m|p m)|(?:at )?(?P<named>noon|midday|midnight))"
    + _E
)
_PART_HOURS = {"morning": 9, "afternoon": 14, "evening": 18, "night": 20}
_LEAD = re.compile(r"^(?:(?:to|that|about|and|then)\b[ ,]*)+")
_TAIL = re.compile(r"(?:[ ,]+(?:to|on|at|by|for|and|then))+$")


def _date_of(found: re.Match[str], now: datetime) -> datetime | None:
    day, month_name = int(found["d"]), found.groupdict().get("m")
    if month_name:
        try:
            when = datetime(now.year, MONTHS.index(month_name) + 1, day)
        except ValueError:
            return None
        if when.date() < now.date():
            when = when.replace(year=now.year + 1)
        return when
    for step in range(0, 3):  # "the 19th": the next 19th, this month or a later one
        month0 = now.month - 1 + step
        try:
            when = datetime(now.year + month0 // 12, month0 % 12 + 1, day)
        except ValueError:
            continue
        if when.date() >= now.date():
            return when
    return None


def _take(pattern: re.Pattern[str], text: str) -> tuple[list[re.Match[str]], str]:
    found = list(pattern.finditer(text))
    for match in reversed(found):
        text = text[: match.start()] + " " + text[match.end() :]
    return found, " ".join(text.split())


def _when_and_what(rest: str, now: datetime) -> tuple[float, str] | None:
    """When the reminder is due and what it is about, from the words after "remind me", wherever the date and time
    are put: "on Thursday 19 November at 9 to send the report", "to send the report on the 19th at 9", "tomorrow
    morning to call mom", "in an hour to stretch". None when no time is named or the date or time does not exist."""
    text = rest
    dates: list[datetime] = []
    for pattern in _DATES:
        found, text = _take(pattern, text)
        for match in found:
            when = _date_of(match, now)
            if when is None:
                return None
            dates.append(when)
    if len({d.date() for d in dates}) > 1:
        return None  # two different dates in one sentence: which one is for the model or the person to settle
    spans, text = _take(_IN, text)
    days, text = _take(_DAY_WORD, text)
    weekdays, text = _take(_WEEKDAY, text)
    clocks, text = _take(_CLOCK, text)
    if not (dates or spans or days or weekdays or clocks):
        return None
    what = _TAIL.sub("", _LEAD.sub("", text.strip(" ,"))).strip(" ,")
    if not what or len(what) > MAX_TEXT:
        return None
    if spans:
        found = spans[0]
        if found["half"]:
            minutes = 30
        else:
            amount = int(found["amount"]) if found["amount"].isdigit() else NUMBER_WORDS[found["amount"]]
            minutes = amount * (60 if found["unit"].startswith("h") else 1)
        if not 1 <= minutes <= 7 * 24 * 60:
            return None
        return (now + timedelta(minutes=minutes)).timestamp(), what
    part = next((m["part"] for m in (*days, *weekdays) if m["part"]), None)
    clock: tuple[int, int] | None = None
    if clocks:
        found = clocks[0]
        if found["named"]:
            clock = (0, 0) if found["named"] == "midnight" else (12, 0)
        else:
            hour, minute, meridiem = (
                (found["h"], found["mi"], found["mer"]) if found["h"] else (found["h2"], found["mi2"], found["mer2"])
            )
            clock = _parse_clock(hour, minute, meridiem, now)
        if clock is None:
            return None
    elif part or (days and days[0]["day"] == "tonight"):
        clock = (_PART_HOURS.get(part or "night", 9), 0)
    hour, minute = clock or (9, 0)
    if dates:
        due = dates[0].replace(hour=hour, minute=minute)
        if due <= now:
            due = due.replace(year=due.year + 1)
        if due - now > timedelta(days=366):
            return None
    elif days:
        ahead = 1 if days[0]["day"] == "tomorrow" else 0
        due = (now + timedelta(days=ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:
            return None  # "today at 9" said at ten: not guessed
    elif weekdays:
        ahead = (WEEKDAYS.index(weekdays[0]["wd"]) - now.weekday()) % 7
        due = (now + timedelta(days=ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:  # "on monday" said on a Monday afternoon means next Monday
            due += timedelta(days=7)
    else:
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:  # a time already past today means tomorrow
            due += timedelta(days=1)
    return due.timestamp(), what


def parse_reminder(text: str, now: datetime) -> tuple[float, str] | None:
    """(due epoch, reminder text) from "remind me in 20 minutes to stretch" and similar; None if unclear.

    A recogniser writes numbers as words ("Monday the twelfth of November at ten"); they are read as the written
    form first (pilot.live.spoken, which converts only where the words around them make the reading certain)."""
    from pilot.live.spoken import to_written

    asked = SET_REQUEST.match(_normalize(to_written(text)))
    if not asked:
        return None
    rest = asked.group("rest")
    said = _said_date(rest, now)
    if said is not None:
        due, what = said
        return (due.timestamp(), what) if what and len(what) <= MAX_TEXT else None
    return _when_and_what(rest, now)


def _in_night(moment: datetime) -> bool:
    return moment.hour >= NIGHT_START or moment.hour < NIGHT_END


def next_due(repeat: dict[str, Any], after: datetime) -> datetime:
    """When a repeating reminder next goes out, strictly after ``after``."""
    if "minutes" in repeat:
        due = after + timedelta(minutes=int(repeat["minutes"]))
        if _in_night(due):
            morning = due.replace(hour=NIGHT_END, minute=0, second=0, microsecond=0)
            due = morning if due.hour < NIGHT_END else morning + timedelta(days=1)
        return due
    hour, minute = (int(part) for part in repeat["time"].split(":"))
    for offset in range(0, 8):
        day = after + timedelta(days=offset)
        due = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due > after and due.weekday() in repeat["days"]:
            return due
    raise ValueError("A repeating reminder needs at least one day")


_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
SPOKEN_MISSED_LIMIT = 3  # reminders named in the spoken sentence; the card lists them all


def describe_missed(items: list[dict[str, Any]], first_tick: bool, *, now: float) -> tuple[str, str]:
    """The card's line and the one spoken sentence for reminders that came due while away.

    "While this computer was asleep, two reminders came due: call mum and stretch." At the first check after Heliox
    started it says "While Heliox was off" instead. The sentence names at most three; the card lists every one with
    when it was due.
    """
    where = "While Heliox was off" if first_tick else "While this computer was asleep"
    texts = [str(item["text"]).rstrip(".") for item in sorted(items, key=lambda item: item["due"])]
    count = len(texts)
    if count == 1:
        sentence = f"{where}, a reminder came due: {texts[0]}."
    else:
        named = texts[:SPOKEN_MISSED_LIMIT]
        rest = count - len(named)
        listed = ", ".join(named[:-1]) + (f", {named[-1]}" if rest else f" and {named[-1]}")
        if rest:
            listed += f" and {_COUNT_WORDS.get(rest, str(rest))} more"
        sentence = f"{where}, {_COUNT_WORDS.get(count, str(count))} reminders came due: {listed}."
    return sentence, sentence


def describe_repeat(repeat: dict[str, Any]) -> str:
    if "minutes" in repeat:
        minutes = int(repeat["minutes"])
        if minutes % 60 == 0:
            hours = minutes // 60
            return "every hour" if hours == 1 else f"every {hours} hours"
        return f"every {minutes} minutes"
    hour, minute = (int(part) for part in repeat["time"].split(":"))
    clock = _clock(datetime(2000, 1, 1, hour, minute))
    days = sorted(repeat["days"])
    if days == list(range(7)):
        label = "every day"
    elif days == list(range(5)):
        label = "every weekday"
    elif days == [5, 6]:
        label = "every weekend day"
    else:
        label = "every " + " and ".join(WEEKDAYS[day].capitalize() for day in days)
    return f"{label} at {clock}"


def parse_repeating(text: str, now: datetime) -> tuple[float, str, dict[str, Any]] | str | None:
    """(first due, text, repeat) from "remind me every hour to stand up"; a reason when it can't; None otherwise."""
    asked = SET_REQUEST.match(_normalize(text))
    if not asked or not re.search(r"\bevery\b", asked.group("rest")):
        return None
    rest = asked.group("rest")
    for layout in (
        rf"^(?:{_REPEAT}) (?:to |that |about )?(?P<what>.+)$",
        rf"^(?:to |that |about )?(?P<what>.+?) (?:{_REPEAT})$",
    ):
        match = re.match(layout, rest)
        if not match:
            continue
        groups = match.groupdict()
        what = groups["what"].strip()
        if not what or len(what) > MAX_TEXT:
            return None
        if groups.get("unit"):
            amount = groups.get("n")
            count = 1 if not amount else int(amount) if amount.isdigit() else NUMBER_WORDS.get(amount, 0)
            minutes = count * (60 if groups["unit"].startswith("h") else 1)
            if minutes < MIN_REPEAT_MINUTES or minutes > 24 * 60:
                return f"A repeating reminder can come every {MIN_REPEAT_MINUTES} minutes at most often, and once a day at least."
            repeat: dict[str, Any] = {"minutes": minutes}
        else:
            word = groups["days"]
            default_hour = {"morning": 9, "evening": 18, "night": 21}.get(word, 9)
            if groups.get("hour"):
                clock = _parse_clock(groups["hour"], groups.get("minute"), groups.get("meridiem"), now)
                if clock is None:
                    return "That is not a time I understand."
            else:
                clock = (default_hour, 0)
            days = {
                "weekday": [0, 1, 2, 3, 4],
                "weekend": [5, 6],
            }.get(word, [WEEKDAYS.index(word)] if word in WEEKDAYS else list(range(7)))
            repeat = {"days": days, "time": f"{clock[0]:02d}:{clock[1]:02d}"}
        return next_due(repeat, now).timestamp(), what, repeat
    return None


class ReminderStore:
    """Reminders in a small JSON file, delivered by a background loop."""

    def __init__(
        self,
        path: Path,
        *,
        deliver: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        deliver_missed: Callable[[list[dict[str, Any]], bool], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._deliver = deliver
        # Every reminder that came due while the computer slept or Heliox was off, in one call:
        # (items with "due" and "late_seconds", first_tick). ``first_tick``: the first check since Heliox started.
        self._deliver_missed = deliver_missed
        self._clock = clock
        self._ticked = False
        self._task: asyncio.Task[None] | None = None

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items = [item for item in data if isinstance(item, dict) and "due" in item] if isinstance(data, list) else []
        return sorted(items, key=lambda item: item["due"])

    def _save(self, items: list[dict[str, Any]]) -> None:
        note_change("updated your reminders")  # a voice turn that did this cannot be withdrawn
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, due: float, text: str, repeat: dict[str, Any] | None = None) -> dict[str, Any]:
        text = " ".join(str(text).split())
        if not 1 <= len(text) <= MAX_TEXT:
            raise ValueError(f"A reminder must be 1 to {MAX_TEXT} characters")
        if due <= self._clock():
            raise ValueError("That time has already passed")
        items = self.list()
        if len(items) >= MAX_REMINDERS:
            raise ValueError(f"You can keep up to {MAX_REMINDERS} reminders")
        entry = {"id": uuid.uuid4().hex[:10], "due": float(due), "text": text, "created_at": self._clock()}
        if repeat:
            entry["repeat"] = repeat
        items.append(entry)
        self._save(items)
        return entry

    def cancel(self, reminder_id: str | None = None, *, about: str | None = None) -> int:
        """Cancel one reminder by id, those whose text contains ``about``, or (neither given) all."""
        items = self.list()
        if reminder_id:
            kept = [item for item in items if item["id"] != reminder_id]
        elif about:
            needle = about.casefold()
            kept = [item for item in items if needle not in item["text"].casefold()]
        else:
            kept = []
        if len(kept) != len(items):
            self._save(kept)
        return len(items) - len(kept)

    async def tick(self) -> int:
        """Deliver every reminder that is due. Returns how many went out."""
        now = self._clock()
        items = self.list()
        due = [item for item in items if item["due"] <= now]
        if not due:
            self._ticked = True
            return 0
        kept = [item for item in items if item["due"] > now]
        for item in due:
            if item.get("repeat"):  # reschedule before delivering, so none goes out twice
                upcoming = datetime.fromtimestamp(item["due"])
                for _ in range(5000):  # keep the cadence: step from its own time past now
                    upcoming = next_due(item["repeat"], upcoming)
                    if upcoming.timestamp() > now:
                        break
                kept.append({**item, "due": upcoming.timestamp()})
        self._save(kept)  # saved before anything goes out, so no reminder ever fires twice
        first_tick, self._ticked = not self._ticked, True
        sent = 0
        missed = [{**item, "late": True, "late_seconds": now - item["due"]} for item in due
                  if now - item["due"] > MISSED_AFTER_SECONDS]  # fmt: skip
        for item in due:
            if now - item["due"] > MISSED_AFTER_SECONDS and self._deliver_missed is not None:
                continue
            if self._deliver is not None:
                await self._deliver({**item, "late": now - item["due"] > MISSED_AFTER_SECONDS})
            sent += 1
        if missed and self._deliver_missed is not None:
            await self._deliver_missed(missed, first_tick)
            sent += len(missed)
        return sent

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Reminder delivery failed", exc_info=True)
            await asyncio.sleep(CHECK_EVERY_SECONDS)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def answer(self, text: str, *, now: datetime | None = None) -> str | None:
        """Set, list or cancel reminders from a sentence; None when it is not about reminders."""
        now = now or datetime.fromtimestamp(self._clock())
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return "You have no reminders."
            listed = "; ".join(
                f"{describe_repeat(item['repeat']) if item.get('repeat') else describe_time(item['due'], now)}: "
                f"{item['text']}"
                for item in items[:5]
            )
            more = f" And {len(items) - 5} more." if len(items) > 5 else ""
            return f"You have {len(items)} reminder{'' if len(items) == 1 else 's'}. {listed}.{more}"
        timer = TIMER_REQUEST.match(normalized)
        if timer:
            amount_text = timer.group("a1") or timer.group("a2")
            unit = timer.group("u1") or timer.group("u2")
            amount = int(amount_text) if amount_text.isdigit() else NUMBER_WORDS.get(amount_text, 0)
            seconds = amount * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)
            if not 10 <= seconds <= 24 * 3600:
                return "A timer can run from 10 seconds to 24 hours."
            label = f"{amount} {unit.rstrip('s') if amount == 1 else unit if unit.endswith('s') else unit + 's'}"
            label = label.replace("min ", "minute ").replace("mins", "minutes").replace("secs", "seconds")
            try:
                self.add(now.timestamp() + seconds, f"Your {label} timer is done")
            except ValueError as exc:
                return str(exc)
            return f"Timer set for {label}."
        cancel = CANCEL_REQUEST.match(normalized)
        if cancel:
            removed = self.cancel(about=cancel.group("what") or cancel.group("what2"))
            if not removed:
                return "There was no reminder like that to cancel."
            return f"Cancelled {removed} reminder{'' if removed == 1 else 's'}."
        if not SET_REQUEST.match(normalized):
            return None
        repeating = parse_repeating(text, now)
        if isinstance(repeating, str):
            return repeating
        if repeating is not None:
            due, what, repeat = repeating
            try:
                self.add(due, what, repeat)
            except ValueError as exc:
                return str(exc)
            return f"Okay, I'll remind you {describe_repeat(repeat)} to {what}. The first one is {describe_time(due, now)}."
        parsed = parse_reminder(text, now)
        if parsed is None:
            return 'Tell me when, for example "remind me in 20 minutes to stretch" or "remind me at 5 pm to leave".'
        due, what = parsed
        try:
            self.add(due, what)
        except ValueError as exc:
            return str(exc)
        if _normalize(text).split(" remind me ", 1)[-1].removeprefix("remind me ").startswith("that "):
            return f"Okay, I'll remind you {describe_time(due, now)}."  # what it is about was just said
        return f"Okay, I'll remind you {describe_time(due, now)} to {what}."

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)

"""Read calendar events back from the CalDAV server with a fresh client, for the calendar post-conditions.

Every call here only reads (a REPORT or GET through ``caldav``); nothing creates, changes or deletes an event. The
calls block on the network, so the verifiers run them in a worker thread under their own time limit.

Times are compared as instants, never as strings: an event saved for 10:00 in Kolkata and read back as 04:30 UTC is
the same event. A time without a zone is taken in this computer's zone, as ``caldav`` does when it saves one. When the
server holds an all-day event (a date without a time), only the day is compared.

Recurring events: the event and all its changed occurrences share one UID, so "is there an event with this UID" covers
the whole series; the fields compared are those of the series itself (the VEVENT without a RECURRENCE-ID).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

_UID_NOTE = re.compile(r"\(UID ([^\s()]+)\)")

# Pauses before each look for a new event: a server may take a moment to show what it accepted. Three looks and 3 s
# of waiting, inside the verifier's own time limit.
LOOK_PAUSES_SECONDS = (0.0, 1.0, 2.0)

# Replaced in tests by a recorder, so no test waits in real time.
_pause = asyncio.sleep


def event_uid_note(uid: str) -> str:
    """The part of a create's output that names the event, read back by ``event_uid_from_output``."""
    return f" (UID {uid})"


def event_uid_from_output(output: str | None) -> str:
    match = _UID_NOTE.search(output or "")
    return match.group(1) if match else ""


@dataclass(frozen=True)
class CalendarAccount:
    url: str
    username: str
    password: str = field(repr=False)


def open_calendar(account: CalendarAccount, calendar_id: str | None) -> Any:
    """A new client and session, and the calendar the agent would pick: the named one, else the first."""
    import caldav

    client = caldav.DAVClient(url=account.url, username=account.username, password=account.password)
    calendars = client.principal().calendars()
    if not calendars:
        raise LookupError("the CalDAV account has no calendars")
    if not calendar_id:
        return calendars[0]
    for calendar in calendars:
        if str(getattr(calendar, "name", "")) == calendar_id:
            return calendar
    raise LookupError(f"the calendar {calendar_id} is not on the server")


def event_by_uid(calendar: Any, uid: str) -> Any | None:
    """The event with ``uid``, or None when the server has none."""
    from caldav.lib.error import NotFoundError

    try:
        return calendar.event_by_uid(uid)
    except NotFoundError:
        return None


def series_component(event: Any) -> Any | None:
    """The VEVENT that describes the event (for a recurring one, the series, not a changed occurrence)."""
    import icalendar

    parsed = icalendar.Calendar.from_ical(event.data)
    events = [component for component in parsed.walk() if component.name == "VEVENT"]
    for component in events:
        if component.get("recurrence-id") is None:
            return component
    return events[0] if events else None


def property_time(component: Any, name: str) -> date | datetime | None:
    value = component.get(name)
    return getattr(value, "dt", None) if value is not None else None


def same_moment(requested: datetime, found: date | datetime | None) -> bool:
    if found is None:
        return False
    if isinstance(found, datetime):
        return _aware(requested) == _aware(found)
    return requested.date() == found


def describe(moment: date | datetime | None) -> str:
    if moment is None:
        return "none"
    if isinstance(moment, datetime):
        return _aware(moment).isoformat(timespec="minutes")
    return f"{moment.isoformat()} (all day)"


def events_like(calendar: Any, summary: str, start: datetime) -> list[str]:
    """UIDs of events called ``summary`` that start at ``start``: finds an event the server stored under another UID."""
    begin = _aware(start)
    found = calendar.search(start=begin - timedelta(days=1), end=begin + timedelta(days=1), event=True)
    uids = []
    for event in found:
        component = series_component(event)
        if component is None:
            continue
        if str(component.get("summary", "")) == summary and same_moment(start, property_time(component, "dtstart")):
            uids.append(str(component.get("uid", "")))
    return uids


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.astimezone()

"""What the voice says for a date, a time or a duration in a sentence Heliox wrote in code: the written form stays
for the window, and the one gate every road's words pass on their way to the voice (PilotServer._speak_companion_text
-> pilot.conversation.style.speakable_numbers -> pilot.live.spoken.say_dates and say_durations) says it as a person
would. On 13 November the reminder's confirmation was spoken "on Thu 19 Nov at nine a m".
"""

from __future__ import annotations

import locale
from datetime import date, datetime, timedelta

import pytest

from pilot.conversation.style import speakable_numbers
from pilot.live.spoken import say_dates, say_durations
from pilot.workflows.day_briefing import compose_night
from pilot.workflows.focus import describe_quiet_hours, parse_quiet_hours
from pilot.workflows.habits import HabitStore
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.user_profile import UserProfile

NOW = datetime(2026, 11, 12, 10, 0)  # Thursday 12 November 2026
TODAY = NOW.date()


class Clock:
    def __init__(self, when: datetime) -> None:
        self.when = when.timestamp()

    def __call__(self) -> float:
        return self.when


def spoken(text: str) -> str:
    return speakable_numbers(text, TODAY)


# ── reminders, set, listed and due; timers ──


@pytest.mark.parametrize(
    ("said", "written", "voice"),
    [
        ("remind me on 19 November at 9 am to send the report",
         "Okay, I'll remind you on Thu 19 Nov at 9:00 AM to send the report.",
         "Okay, I'll remind you on Thursday the nineteenth at nine a m to send the report."),
        ("remind me on 3 December at 5:30 pm to pay rent",
         "Okay, I'll remind you on Thu 03 Dec at 5:30 PM to pay rent.",
         "Okay, I'll remind you on Thursday the third of December at five thirty p m to pay rent."),
        ("remind me on 2 January to renew the domain",
         "Okay, I'll remind you on Sat 02 Jan 2027 at 9:00 AM to renew the domain.",
         "Okay, I'll remind you on Saturday the second of January twenty twenty-seven at nine a m to renew the domain."),
        ("remind me tomorrow at 7 pm to call mom", "Okay, I'll remind you tomorrow at 7:00 PM to call mom.",
         "Okay, I'll remind you tomorrow at seven p m to call mom."),
        ("remind me in 20 minutes to stretch", "Okay, I'll remind you at 10:20 AM to stretch.",
         "Okay, I'll remind you at ten twenty a m to stretch."),
    ],
)  # fmt: skip
def test_a_reminder_is_written_short_and_said_whole(tmp_path, said, written, voice):
    store = ReminderStore(tmp_path / "r.json", clock=Clock(NOW))
    reply = store.answer(said, now=NOW)
    assert reply == written
    assert spoken(reply) == voice


def test_the_list_and_a_timer(tmp_path):
    store = ReminderStore(tmp_path / "r.json", clock=Clock(NOW))
    store.answer("remind me on 19 November at 9 am to send the report", now=NOW)
    listed = store.answer("what are my reminders", now=NOW)
    assert "Thu 19 Nov at 9:00 AM" in listed
    assert "Thursday the nineteenth at nine a m" in spoken(listed) and "Nov" not in spoken(listed)
    timer = store.answer("set a timer for 5 minutes", now=NOW)
    assert spoken(timer) == timer  # "5 minutes" is already said as written


# ── the day summary, quiet hours, the work day, habits ──


def test_the_night_summary(tmp_path):
    due = (NOW + timedelta(days=1)).replace(hour=8, minute=30).timestamp()
    text = compose_night(
        NOW.replace(hour=21), brief="", focus_seconds=2 * 3600 + 15 * 60, finished_todos=0, open_todos=0,
        reminders=[{"due": due, "text": "stretch"}],
    )  # fmt: skip
    assert "2 h 15 min" in text and "8:30 AM" in text
    assert spoken(text) == (
        "Good night. You focused for 2 hours 15 minutes. Tomorrow your first reminder is at eight thirty a m: stretch."
    )


def test_quiet_hours_in_a_24_hour_written_form():
    window = parse_quiet_hours("focus every weekday from 7 am to 1 pm")
    assert window is not None
    written = f"Quiet hours set: {describe_quiet_hours(window)}."
    assert written == "Quiet hours set: weekdays 07:00 to 13:00."
    assert spoken(written) == "Quiet hours set: weekdays seven a m to one p m."


@pytest.mark.asyncio
async def test_the_end_of_the_working_day(tmp_path):
    reply = await UserProfile(tmp_path / "p.json", clock=Clock(NOW)).answer_async("I finish work at 5:30 pm")
    assert (
        spoken(reply)
        == "Okay, you stop work at five thirty p m. After that I'll hold offers and suggestions until morning."
    )


def test_a_habit_logged(tmp_path):
    habits = HabitStore(tmp_path / "h.json", clock=Clock(NOW))
    reply = habits.answer("log 2 glasses of water")
    assert reply == "Logged water. That's 2 today, and a 1-day streak."
    assert spoken(reply) == reply  # counts and "1-day" are said well as written


# ── the rules, on their own ──


@pytest.mark.parametrize(
    ("written", "voice"),
    [
        ("on Thu 19 Nov", "on Thursday the nineteenth"),
        ("Friday 13 November", "Friday the thirteenth"),
        ("on 20 November", "on the twentieth"),
        ("November 19th", "the nineteenth"),
        ("on Fri 1 Jan 2027", "on Friday the first of January twenty twenty-seven"),
        ("Tue 1 Dec 2026", "Tuesday the first of December"),
        ("every Mon", "every Monday"),
        ("I sat on the sofa on Sunday", "I sat on the sofa on Sunday"),  # "sat" and "Sun" stay words
        ("the 31 November", "the 31 November"),  # no such day: left as written
        ("2 h 15 min", "2 hours 15 minutes"),
        ("1 min", "1 minute"),
        ("a 5-minute timer", "a 5-minute timer"),
    ],
)
def test_say_dates_and_durations(written, voice):
    assert say_durations(say_dates(written, TODAY)) == voice


@pytest.mark.parametrize(
    ("written", "voice"),
    [
        # a 12-hour clock
        ("on Thu 19 Nov at 9:00 AM", "on Thursday the nineteenth at nine a m"),
        ("at 9:30 PM", "at nine thirty p m"),
        # a 24-hour clock
        ("on Thu 19 Nov at 21:00", "on Thursday the nineteenth at nine p m"),
        ("at 07:00", "at seven a m"),
        ("at 00:15", "at twelve fifteen a m"),
    ],
)
def test_twelve_and_twenty_four_hour_clocks_are_said_alike(written, voice):
    assert spoken(written) == voice
    assert spoken(voice) == voice  # said once: saying it again changes nothing


def _set_time_locale(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return locale.setlocale(locale.LC_TIME, name)
        except locale.Error:
            continue
    return None


@pytest.mark.parametrize(
    "names",
    [
        ("en_US.UTF-8", "en_US", "English_United States.1252", "en-US"),  # 12-hour
        ("de_DE.UTF-8", "de_DE", "German_Germany.1252", "de-DE"),  # 24-hour, its own day and month names
    ],
    ids=["12-hour", "24-hour"],
)
def test_the_written_form_does_not_follow_the_system_locale(tmp_path, names):
    """Python's strftime follows LC_TIME once anything sets it: "%p" is empty in a 24-hour locale and "%a %b" are
    in its language, so "9:00 " and "Do 19 Nov" would reach the voice. The reminder's words are built in English."""
    before = locale.setlocale(locale.LC_TIME)
    if _set_time_locale(names) is None:
        pytest.skip("this system has no such locale")
    try:
        store = ReminderStore(tmp_path / "r.json", clock=Clock(NOW))
        reply = store.answer("remind me on 19 November at 9 pm to send the report", now=NOW)
        assert reply == "Okay, I'll remind you on Thu 19 Nov at 9:00 PM to send the report."
        assert spoken(reply) == "Okay, I'll remind you on Thursday the nineteenth at nine p m to send the report."
        assert date(2026, 11, 19).strftime("%p") is not None  # the locale really is in effect for strftime
    finally:
        locale.setlocale(locale.LC_TIME, before)

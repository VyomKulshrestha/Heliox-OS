"""Helpers, habits, focus history and the morning and night briefings."""

from __future__ import annotations

import random
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.clipboard_history import looks_secret
from pilot.workflows.day_briefing import compose_morning, compose_night
from pilot.workflows.focus_log import FocusLog
from pilot.workflows.generators import Generators, make_password
from pilot.workflows.habits import HabitStore
from pilot.workflows.quick_answers import QuickAnswers

NOW = datetime(2026, 10, 27, 8, 30)  # Tuesday morning
GB = 1024**3


# ---------------------------------------------------------------------------- helpers


def test_passwords_are_strong_and_secret():
    for length in (12, 20, 64):
        password = make_password(length)
        assert len(password) == length
        assert looks_secret(password)  # so clipboard history never keeps one


@pytest.mark.asyncio
async def test_password_goes_to_the_clipboard_only():
    write = AsyncMock()
    helpers = Generators(write=write)
    reply = await helpers.answer_async("generate a password")
    password = write.await_args.args[0]
    assert len(password) == 20 and password not in reply
    assert reply.startswith("I put a new 20-character password on the clipboard.")
    assert await helpers.answer_async("make me a 5 character password") == "A password can be 12 to 64 characters long."


@pytest.mark.asyncio
async def test_chance():
    helpers = Generators(rng=random.Random(4))
    assert (await helpers.answer_async("flip a coin")) in ("It's heads.", "It's tails.")
    assert (await helpers.answer_async("roll a die")).startswith("You rolled a ")
    assert (await helpers.answer_async("roll a d20")).startswith("You rolled a ")
    assert (await helpers.answer_async("roll two dice")).endswith(" in total.")
    number = int((await helpers.answer_async("pick a number between 10 and 1")).rstrip("."))
    assert 1 <= number <= 10
    assert (await helpers.answer_async("pick one of red, green or blue")) in (
        "I pick red.",
        "I pick green.",
        "I pick blue.",
    )
    assert await helpers.answer_async("pick a file or folder") is None
    assert await helpers.answer_async("roll back the update") is None


@pytest.mark.asyncio
async def test_stopwatch():
    clock = MagicMock(return_value=100.0)
    helpers = Generators(clock=clock)
    assert (
        await helpers.answer_async("how long has it been") == 'The stopwatch isn\'t running. Say "start a stopwatch".'
    )
    assert await helpers.answer_async("start a stopwatch") == "Stopwatch started."
    clock.return_value = 100.0 + 3725
    assert await helpers.answer_async("how long has it been") == "1 hour, 2 minutes and 5 seconds so far."
    clock.return_value = 100.0 + 90
    assert await helpers.answer_async("stop the stopwatch") == "Stopped at 1 minute and 30 seconds."
    assert await helpers.answer_async("stop the stopwatch") == "The stopwatch isn't running."


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("how long until 5 pm", "8 hours and 30 minutes until 5:00 PM."),
        ("how long until 9", "30 minutes until 9:00 AM."),
        ("how many minutes until 8:45", "15 minutes until 8:45 AM."),
        ("how long until midnight", "15 hours and 30 minutes until 12:00 AM tomorrow."),
        ("how long until 8 am", "23 hours and 30 minutes until 8:00 AM tomorrow."),
    ],
)
def test_time_until(question, answer):
    assert QuickAnswers(now=lambda: NOW).answer(question) == answer


# ---------------------------------------------------------------------------- habits


@pytest.fixture
def clock():
    return MagicMock(return_value=NOW.timestamp())


@pytest.fixture
def habits(tmp_path, clock):
    return HabitStore(tmp_path / "habits.json", clock=clock)


def test_logging_counts_and_streaks(habits, clock):
    for day in (24, 25, 26):
        clock.return_value = datetime(2026, 10, day, 18).timestamp()
        habits.log("workout")
    clock.return_value = NOW.timestamp()
    assert habits.answer("what's my workout streak") == "Your workout streak is 3 days."  # yesterday still counts
    assert habits.answer("log a workout") == "Logged workout. That's 1 today, and a 4-day streak."
    assert habits.answer("how many workouts this week") == "You logged workout 2 times this week."
    assert habits.answer("log 2 glasses of water") == "Logged water. That's 2 today, and a 1-day streak."
    assert habits.answer("I drank a glass of water") == "Logged water. That's 3 today, and a 1-day streak."
    assert habits.answer("how much water today") == "You logged water 3 times today."
    assert habits.answer("show my habits") == (
        "Your habits: workout: 1 today, 4-day streak; water: 3 today, 1-day streak."
    )
    assert habits.answer("stop tracking water") == "I stopped tracking water."


def test_ordinary_sentences_are_not_habits(habits):
    for text in (
        "I did the dishes",
        "track my package",
        "record a video",
        "log in to gmail",
        "log out",
        "log this error to a file",
        "how many files are in my downloads",
    ):
        assert habits.answer(text) is None, text
    assert habits.list() == []


# ---------------------------------------------------------------------------- focus history


def test_focus_history(tmp_path):
    log = FocusLog(tmp_path / "focus.json", now=lambda: NOW)
    monday = datetime(2026, 10, 26, 9).timestamp()
    log.add({"started_at": monday, "ended_at": monday + 50 * 60, "reason": "time"})
    log.add({"started_at": monday + 4 * 3600, "ended_at": monday + 4 * 3600 + 25 * 60, "reason": "asked"})
    assert log.add({"started_at": monday, "ended_at": monday + 30, "reason": "asked"}) is None  # too short
    tuesday = datetime(2026, 10, 27, 7).timestamp()
    log.add({"started_at": tuesday, "ended_at": tuesday + 45 * 60, "reason": "time"})

    assert log.answer("how long did I focus today") == "You focused for 45 min today, in 1 session."
    assert log.answer("how long did I focus this week") == (
        "You focused for 2 h this week, in 3 sessions. The longest was 50 min on Monday."
    )
    assert log.answer("how many focus sessions last week").startswith("You had no focus sessions last week")
    assert log.overview()["this_week"] == {"seconds": 7200, "sessions": 3}


# ---------------------------------------------------------------------------- morning and night


def test_morning_briefing():
    reminders = [
        {"due": datetime(2026, 10, 27, 9, 0).timestamp(), "text": "check email"},
        {"due": datetime(2026, 10, 27, 7, 0).timestamp(), "text": "already past"},
        {"due": datetime(2026, 10, 28, 9, 0).timestamp(), "text": "tomorrow"},
    ]
    text = compose_morning(
        NOW,
        reminders=reminders,
        todos=[{"text": "call the bank"}, {"text": "renew passport"}],
        habits=[{"name": "workout", "streak": 4, "today": 0}, {"name": "water", "streak": 1, "today": 0}],
        health={"disks": [{"drive": "C:\\", "total": 953 * GB, "free": 41 * GB}], "battery": None},
    )
    assert text == (
        "Good morning. It's Tuesday, 27 October. Reminders today: 9:00 AM check email. "
        'You have 2 things to do, starting with "call the bank". Keep your streak going: workout (4 days). '
        "Heads-up: C: is almost full (41 GB free)."
    )
    assert "No reminders today." in compose_morning(NOW, reminders=[], todos=[], habits=[])


def test_night_briefing():
    text = compose_night(
        datetime(2026, 10, 27, 22, 0),
        brief='Today Heliox ran 2 tasks; the latest was "tidy downloads".',
        focus_seconds=5400,
        finished_todos=2,
        open_todos=0,
        reminders=[{"due": datetime(2026, 10, 28, 9, 0).timestamp(), "text": "check email"}],
    )
    assert text == (
        'Good night. Today Heliox ran 2 tasks; the latest was "tidy downloads". You focused for 1 h 30 min. '
        "You finished 2 things on your list, and your list is clear. "
        "Tomorrow your first reminder is at 9:00 AM: check email."
    )


@pytest.mark.asyncio
async def test_rpcs_and_routing(tmp_path, clock):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._habits = HabitStore(tmp_path / "h.json", clock=clock)
    server._focus_log = FocusLog(tmp_path / "f.json", now=lambda: NOW)
    ws = MagicMock()

    logged = await server._handle_habit_log({"name": "Reading", "count": 2}, ws)
    assert logged["habits"][0]["name"] == "reading" and logged["habits"][0]["today"] == 2
    assert (await server._handle_habit_log({"name": "reading", "count": "x"}, ws))["status"] == "error"
    assert (await server._handle_habit_delete({"name": "reading"}, ws))["habits"] == []
    assert (await server._handle_focus_history({}, ws))["today"] == {"seconds": 0, "sessions": 0}

    assert (await server._control_reply("good morning")).startswith("Good ")
    assert (await server._control_reply("good night")).startswith("Good night.")
    assert server._phrase_reserved("good morning")

    await server._on_focus_end(
        {
            "started_at": NOW.timestamp() - 1800,
            "ended_at": NOW.timestamp(),
            "duration": "30 min",
            "reason": "asked",
            "held": [],
            "break_minutes": 0,
        }
    )
    assert len(server._focus_log.list()) == 1

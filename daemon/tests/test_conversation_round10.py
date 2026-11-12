"""What the session suite pinned on files nobody was working in, fixed: a reminder said with its date in words, sums
said in words, the day's greeting on an injectable clock, and numbers said the way a person says them on every
road to the voice."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from pilot.config import PilotConfig
from pilot.conversation.style import say_numbers, speakable_numbers
from pilot.workflows.quick_answers import QuickAnswers, numbers_in_words
from pilot.workflows.reminders import ReminderStore, parse_reminder

NOW = datetime(2026, 11, 12, 9, 0)  # a Thursday


# ── reminders ──


@pytest.mark.parametrize(
    ("said", "due"),
    [
        ("Remind me that the launch call is Monday the twelfth of November at ten.", datetime(2026, 11, 12, 10, 0)),
        ("remind me that my dentist appointment is on the 18th of November at 3 pm", datetime(2026, 11, 18, 15, 0)),
        ("remind me that the review is November 19th", datetime(2026, 11, 19, 9, 0)),
    ],
)
def test_a_reminder_that_names_its_date_in_the_sentence(said, due):
    parsed = parse_reminder(said, NOW)
    assert parsed is not None and datetime.fromtimestamp(parsed[0]) == due


def test_a_that_sentence_without_a_date_is_still_asked_about_and_the_old_shapes_still_work():
    assert parse_reminder("remind me that I am tired", NOW) is None
    stretch = parse_reminder("remind me in 20 minutes to stretch", NOW)
    assert stretch is not None and stretch[1] == "stretch"


def test_the_reply_to_a_that_reminder_is_one_short_sentence(tmp_path):
    store = ReminderStore(tmp_path / "r.json", clock=lambda: NOW.timestamp())
    reply = store.answer("Remind me that the launch call is Monday the twelfth of November at ten.", now=NOW)
    assert reply == "Okay, I'll remind you at 10:00 AM."
    assert speakable_numbers(reply) == "Okay, I'll remind you at ten a m."


# ── sums in words ──


@pytest.mark.parametrize(
    ("asked", "answer"),
    [
        ("what's two plus two?", "That's 4."),
        ("what is twenty five times four", "That's 100."),
        ("what is fifteen percent of eighty", "15% of 80 is 12."),
        ("what is a hundred divided by four", "That's 25."),
        ("what is two thousand four hundred plus one", "That's 2,401."),
        ("convert five miles to km", "5 miles is 8.05 km."),
        ("what's 2 plus 2", "That's 4."),
    ],
)
def test_sums_and_conversions_said_in_words_are_answered_in_code(asked, answer):
    assert QuickAnswers(now=lambda: NOW).answer(asked) == answer


def test_words_that_are_not_numbers_stay_words():
    assert numbers_in_words("a hundred and twelve") == "112"
    assert numbers_in_words("the cat and the hat") == "the cat and the hat"


# ── the day's greeting ──


def test_good_morning_reads_the_servers_clock_which_can_be_pinned():
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._wall_clock = lambda: NOW
    reply = asyncio.run(server._day_request("good morning"))
    assert reply is not None and reply.startswith("Good morning") and "Thursday" in reply


# ── numbers in anything spoken ──


@pytest.mark.parametrize(
    ("written", "said"),
    [
        ("Emailing the Acme invoice for 4 200.", "Emailing the Acme invoice for four thousand two hundred."),
        ("for 4,200", "for four thousand two hundred"),
        ("4200.00 in total", "four thousand two hundred in total"),
        ("$4,200.50 due", "four thousand two hundred dollars and fifty cents due"),
        ("on the 19th", "on the nineteenth"),
        ("at 10:30", "at ten thirty"),
        ("at 3pm", "at three p m"),
        ("at 3:15 p.m.", "at three fifteen p m"),
        ("at 15:00", "at three p m"),
        ("50% done", "fifty percent done"),
        ("12.5% off", "twelve point five percent off"),
    ],
)
def test_amounts_times_ordinals_and_percentages_are_said_as_a_person_says_them(written, said):
    assert say_numbers(written) == said
    assert say_numbers(said) == said  # idempotent: words stay words


@pytest.mark.parametrize("left", ["v1.2.0", "192.168.1.1", "in 2026", "3 items", "10:30:15", "version 4.00.1"])
def test_what_a_voice_already_says_well_is_left_alone(left):
    assert say_numbers(left) == left


def test_every_road_to_the_voice_says_numbers_this_way():
    from pilot.server import PilotServer
    from pilot.system.companion_speech import SpeechChannel, SpeechOutcome

    server = PilotServer(PilotConfig())
    heard = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    server._speech_coordinator.speak = speak
    asyncio.run(server._speak_companion_text("Emailing the Acme invoice for 4 200.", "task_narration", "ack"))
    assert heard == ["Emailing the Acme invoice for four thousand two hundred."]

"""Quick answers: exact sums, conversions, dates and world time, and nothing else."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.quick_answers import QuickAnswers, calculate, convert

NOW = datetime(2026, 10, 26, 14, 10)  # Monday


@pytest.fixture
def quick():
    return QuickAnswers(now=lambda: NOW)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("What is 15 percent of 240?", "15% of 240 is 36."),
        ("what's 20% off 80", "20% off 80 is 64 (you save 16)."),
        ("what is a 15% tip on 64", "A 15% tip on 64 is 9.6, so 73.6 in total."),
        ("what's 12 times 7", "That's 84."),
        ("calculate 3.5 * (2 + 4)", "That's 21."),
        ("what is 2 to the power of 10", "That's 1,024."),
        ("what is the square root of 144", "That's 12."),
        ("what is 1,024 divided by 8", "That's 128."),
        ("what is 7 minus 10", "That's -3."),
        ("15 x 7", "That's 105."),
        ("10/4", "That's 2.5."),
        ("what is 1/0", "You can't divide by zero."),
    ],
)
def test_arithmetic(quick, question, answer):
    assert quick.answer(question) == answer


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("convert 5 miles to km", "5 miles is 8.05 km."),
        ("how many ounces in a pound", "1 pound is 16 ounces."),
        ("what is 70 fahrenheit in celsius", "70°F is 21.11°C."),
        ("convert -40 c to f", "-40°C is -40°F."),
        ("100 km/h to mph", "100 km/h is 62.14 mph."),
        ("how many teaspoons in a tablespoon", "1 tablespoon is 3 teaspoons."),
        ("how many cups in a gallon", "1 gallon is 16 cups."),
        ("5 kg in pounds", "5 kg is 11.02 pounds."),
        ("convert 2 hours to minutes", "2 hours is 120 minutes."),
        ("convert 5 miles to kg", "I can't convert miles to kg: one is a length and the other a mass."),
    ],
)
def test_conversions(quick, question, answer):
    assert quick.answer(question) == answer


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("how many days until Christmas", "60 days until Friday, 25 December 2026."),
        ("how many days until new year's eve", "66 days until Thursday, 31 December 2026."),
        ("how many days until March 3", "128 days until Wednesday, 3 March 2027."),
        ("how many days until October 26", "That's today, Monday, 26 October 2026."),
        ("how many days since January 1", "298 days since Thursday, 1 January 2026."),
        ("what day is December 25", "Friday, 25 December 2026."),
        ("what day of the week is 4th July 2027", "Sunday, 4 July 2027."),
        ("what's the date in 3 weeks", "Monday, 16 November 2026."),
        ("what date will it be in 2 months", "Saturday, 26 December 2026."),
        ("what's the date today", "Today is Monday, 26 October 2026."),
        ("what day is it", "Today is Monday, 26 October 2026."),
        ("what time is it", "It's 2:10 PM."),
        ("how many days until February 30", "That is not a date I understand."),
    ],
)
def test_dates(quick, question, answer):
    assert quick.answer(question) == answer


def test_world_time(quick):
    assert quick.answer("what time is it in Tokyo").endswith(" in Tokyo.")
    assert quick.answer("what's the time in new york").endswith(" in New York.")
    assert quick.answer("time in UTC").endswith(" in UTC.")
    assert quick.answer("time in Narnia") is None


@pytest.mark.parametrize(
    "question",
    [
        "convert this video to mp4",
        "convert report.docx to pdf",
        "what is the weather",
        "what's up",
        "what is python",
        "open 2 tabs",
        "what is 3 times the size of earth",
        "set a timer for 10 minutes",
        "what is my ip address",
        "what's 2 plus",
        "calculate my taxes",
    ],
)
def test_other_questions_are_left_alone(quick, question):
    assert quick.answer(question) is None


def test_huge_powers_are_refused():
    assert calculate("10 ** 1000") is None
    assert convert("convert a mile to furlongs") is None


@pytest.mark.asyncio
async def test_routing_and_shortcut_reservation():
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._quick_answers = QuickAnswers(now=lambda: NOW)

    assert await server._control_reply("what's 15 percent of 240") == "15% of 240 is 36."
    assert await server._control_reply("open the weather app") is None
    assert server._phrase_reserved("what time is it")
    assert not server._phrase_reserved("start work")

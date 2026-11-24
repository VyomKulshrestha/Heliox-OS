"""Times of day as people say them, and as a model is told them: a spoken to-do ("by 9 tomorrow morning") keeps its
time, an item that shares a word with what was just said is offered whatever its date, the time in front of a model
always says the part of the day, and Live's card turns "this evening" into an evening time and "my mom" into "your
mom"."""

from __future__ import annotations

from datetime import datetime, time

import pytest

from pilot.conversation.context import ContextSources, ConversationMemory, build_messages, clock_words
from pilot.conversation.due import due_line, spoken_clock, todo_due
from pilot.live.commitments import find_commitments
from pilot.live.detectors import TextContext

SPOKEN_TODO = {"text": "finish the quarterly report by 9 tomorrow morning", "done": False}
DECISION = "I'm going to watch a series tonight and do the report in the morning"


@pytest.mark.parametrize(
    ("words", "at"),
    [
        ("finish the quarterly report by 9 tomorrow morning", time(9, 0)),
        ("by 9 am tomorrow", time(9, 0)),
        ("call Sam at four this afternoon", time(16, 0)),
        ("ring the plumber half past eight tonight", time(20, 30)),
        ("quarter to nine in the morning", time(8, 45)),
        ("lunch at noon", time(12, 0)),
        ("send it by midnight", time(23, 59)),
        ("at 3", time(15, 0)),  # a plan's bare "at 3" is the afternoon
        ("the review at 14:30", time(14, 30)),
        ("about four hours of work", None),
        ("buy milk", None),
    ],
)
def test_a_time_of_day_as_people_say_it(words, at):
    assert spoken_clock(words) == at


@pytest.mark.parametrize(
    ("hour", "minute"), [(8, 0), (11, 40), (11, 59), (12, 0), (12, 1), (17, 30), (23, 59), (0, 0), (0, 1)]
)
def test_the_spoken_to_do_is_grounds_at_every_hour(hour, minute):
    now = datetime(2026, 11, 24, hour, minute)
    assert todo_due(SPOKEN_TODO["text"], now) == datetime(2026, 11, 25, 9, 0)
    line = due_line(DECISION, todos=[SPOKEN_TODO], reminders=[], now=now, local=False)
    assert 'a to-do about "report", due tomorrow at 9:00 in the morning' in line


def test_a_shared_word_counts_whatever_the_date():
    now = datetime(2026, 11, 24, 10, 0)
    far = {"text": "finish the report by Friday 4 December", "done": False}
    assert "report" in due_line(DECISION, todos=[far], reminders=[], now=now, local=True)
    later = {"id": "r", "text": "report review", "due": datetime(2026, 11, 27, 9, 0).timestamp()}
    assert "report review" in due_line(DECISION, todos=[], reminders=[later], now=now, local=True)
    # Without a shared word, only what is due soon and only when the words speak of a time.
    walk = {"text": "walk the dog by Friday 4 December", "done": False}
    assert due_line(DECISION, todos=[walk], reminders=[], now=now, local=True) == ""


@pytest.mark.parametrize(
    ("hour", "minute", "said"),
    [
        (8, 5, "8:05 in the morning"),
        (11, 21, "11:21 in the morning"),
        (12, 0, "noon"),
        (15, 10, "3:10 in the afternoon"),
        (19, 45, "7:45 in the evening"),
        (23, 40, "11:40 at night"),
        (0, 0, "midnight"),
        (2, 15, "2:15 at night"),
    ],
)
def test_the_time_in_front_of_the_model_says_the_part_of_the_day(hour, minute, said):
    now = datetime(2026, 11, 24, hour, minute)
    assert clock_words(now) == said
    sources = ContextSources(now=lambda: now)
    sent = str(build_messages("hello", sources, ConversationMemory(), screen=None))
    assert f"Now: Tuesday 24 November 2026, {said}." in sent


def _card(said: str, now: datetime):
    return find_commitments(said, TextContext(today=now.date(), source="speech", hour=now.hour), now)


def test_this_evening_is_an_evening_time_and_my_is_your():
    now = datetime(2026, 11, 24, 11, 21)
    [card] = _card("I'm going to call my mum this evening.", now)
    assert "four" not in card.title and "six p m" in card.title  # an hour before seven
    assert "your mum" in card.message and "my mum" not in card.message
    assert card.fix.endswith("to call your mum")

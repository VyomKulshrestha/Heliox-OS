"""A private answer (pilot.conversation.private_answers): a cloud model is told only its kind, from a fixed list, and a
follow-up that swaps one day, time or number is asked of the same handler again in code. The real path, for every
kind of person, is the personal session "a private answer stays on this computer; its follow-up is answered there"
(a reminder about a zebrafish that no request to a cloud model may contain)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig
from pilot.conversation.private_answers import FOLLOW_UP_SECONDS, KINDS, rewrite
from pilot.server import PilotServer, answered_here, asked_here


@pytest.mark.parametrize(
    ("asked", "by", "follow_up", "rewritten"),
    [
        ("what's on today", "_day_request", "And tomorrow?", "what's on tomorrow"),
        ("good morning", "_day_request", "and tomorrow", "what's on tomorrow"),  # the question "good morning" answers
        ("what's on today", "_day_request", "What about Friday?", "what's on Friday"),
        ("what's on tomorrow", "_day_request", "how about on Monday?", "what's on Monday"),
        (
            "what were my last three commands",
            "_local_answer:ActivityTimeline",
            "and the last five?",
            "what were my last five commands",
        ),
        ("what's my reminder at ten", "_local_answer:ReminderStore", "and at four?", "what's my reminder at four"),
    ],
)
def test_the_follow_ups_that_are_asked_again_in_code(asked, by, follow_up, rewritten):
    assert rewrite(asked, by, follow_up) == rewritten


@pytest.mark.parametrize(
    ("asked", "by", "follow_up"),
    [
        ("what reminders do I have", "_local_answer:ReminderStore", "and tomorrow?"),  # no day in the question
        ("what's on today and tomorrow", "_day_request", "and Friday?"),  # two days: which one is meant?
        ("what's on today", "_day_request", "and tomorrow, remind me to call mum"),  # more than a day
        ("what's on today", "_day_request", "tomorrow at ten?"),  # a day and a time
        ("what's on today", "_day_request", "and why?"),
        ("what's on today", "_day_request", "and the bank?"),
        ("what's on today", "_day_request", "what about my sister?"),
        ("what time is it in Tokyo", "_local_answer:QuickAnswers", "and in London?"),  # a place, not a day
    ],
)
def test_the_follow_ups_left_to_the_model(asked, by, follow_up):
    assert rewrite(asked, by, follow_up) is None


def test_the_withheld_lines_come_from_the_fixed_list_never_from_the_answer():
    for by, kind in KINDS.items():
        assert answered_here((), by) == f"(answered on this computer: {kind})"
        assert asked_here(by) == f"(said to Heliox, about {kind})"
    assert answered_here(("updated your reminders",), "_local_answer:ReminderStore") == (
        "(answered on this computer: a reminder; done there: updated your reminders)"
    )
    assert answered_here((), "_a_handler_added_later") == "(answered on this computer)"


def _turn(kept=()):
    return SimpleNamespace(kept=list(kept))


def test_a_follow_up_is_asked_again_only_straight_after_a_question_that_changed_nothing():
    server = PilotServer(PilotConfig())
    now = time.monotonic()
    server._last_code_answer = ("what's on today", "_day_request", True, now)
    assert server._code_follow_up("and tomorrow?", _turn()) == "what's on tomorrow"
    server._last_code_answer = ("what's on today", "_day_request", True, now - FOLLOW_UP_SECONDS - 1)
    assert server._code_follow_up("and tomorrow?", _turn()) is None  # minutes later: not that answer's follow-up
    server._last_code_answer = ("remind me at ten to call mum", "_local_answer:ReminderStore", False, now)
    assert server._code_follow_up("and at four?", _turn()) is None  # a change is never made again this way

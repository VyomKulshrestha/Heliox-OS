"""A follow-up's reach once the room is not quiet (pilot.conversation.reach): the one list of what changes or reveals
the person's data, what never needs the wake phrase, and when the rule applies. Whole sessions: "a second voice ...",
"a television ..." and "the person alone ..." in sessions.json (they need the listener's count)."""

from __future__ import annotations

import pytest

from pilot.conversation.private_answers import KINDS
from pilot.conversation.reach import DATA_HANDLERS, asks_for_data, guarded, keeps_reach


@pytest.mark.parametrize(
    "words",
    [
        "Read me my reminders.",
        "What's on my list?",
        "What do you know about me?",
        "Forget all of it.",
        "Remember that I'm a founder.",
        "Add milk to my list.",
        "What did I copy?",
        "Delete them all.",
        "Send it to Ana.",
        "Email my manager.",
        "Stop suggesting things.",
        "Open my documents folder.",
        "Good morning.",
        "What's on tomorrow?",
        "Undo that.",
    ],
)
def test_what_loses_reach(words):
    assert asks_for_data(words)


@pytest.mark.parametrize(
    "words",
    [
        "What time is it?",
        "What's two plus two?",
        "Why is the sky blue?",
        "And times ten?",
        "Shorter.",
        "Thanks.",
        "Tell me a joke.",
    ],
)
def test_talk_time_sums_and_questions_keep_reach(words):
    assert not asks_for_data(words)


@pytest.mark.parametrize("words", ["Stop.", "No.", "Wait.", "Hold on.", "Cancel.", "Never mind.", "Not now."])
def test_stop_no_and_wait_never_need_the_wake_phrase(words):
    assert keeps_reach(words) and not asks_for_data(words)


def test_every_handler_that_holds_the_persons_data_is_on_the_list():
    kept = {"_live_reply", "_local_answer:SystemHealth"}  # Live's replies to its own card, the computer's health
    assert set(KINDS) - kept <= DATA_HANDLERS
    assert "_local_answer:QuickAnswers" not in DATA_HANDLERS and "_help_request" not in DATA_HANDLERS


@pytest.mark.parametrize(
    ("enabled", "addressed", "overheard", "threshold", "limited"),
    [
        (True, "follow_up", 2, 2, True),
        (True, "follow_up", 1, 2, False),  # one utterance not taken: the person's own muttering, often
        (True, "wake", 5, 2, False),  # the wake phrase keeps full reach
        (True, "button", 5, 2, False),
        (False, "follow_up", 5, 2, False),  # the rule off: logged only
        (True, "follow_up", 1, 0, True),  # a threshold below one is one
    ],
)
def test_when_the_rule_applies(enabled, addressed, overheard, threshold, limited):
    assert guarded(enabled=enabled, addressed=addressed, overheard=overheard, threshold=threshold) is limited


def test_the_server_logs_the_count_on_every_follow_up_and_limits_only_when_on(caplog):
    from types import SimpleNamespace

    from pilot.config import PilotConfig
    from pilot.server import PilotServer
    from pilot.system.voice import voice_turn_addressed

    server = PilotServer(PilotConfig())
    server._voice_listener = SimpleNamespace(addressing=SimpleNamespace(overheard_since_heliox=3))
    token = voice_turn_addressed.set("follow_up")
    try:
        with caplog.at_level("INFO", logger="pilot.server"):
            assert server._reach_limited("read me my reminders") is False  # off by default
        assert "after 3 utterance(s) not taken" in caplog.text and "rule off" in caplog.text
        server.config.conversation.second_voice_guard = True
        assert server._reach_limited("read me my reminders") is True
        assert server._reach_limited("stop") is False
    finally:
        voice_turn_addressed.reset(token)

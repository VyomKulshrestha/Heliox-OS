"""A follow-up's reach once the room is not quiet (pilot.conversation.reach): the one list of what changes or reveals
the person's data, what never needs the wake phrase, and when the rule applies. Whole sessions: "a second voice ...",
"a television ..." and "the person alone ..." in sessions.json (they need the listener's count)."""

from __future__ import annotations

import pytest

from pilot.conversation.private_answers import KINDS
from pilot.conversation.reach import DATA_HANDLERS, asks_for_data, guarded, keeps_reach, overheard_reason


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
        (True, "follow_up", 1, 1, True),  # the default threshold: 2 caught no single second voice
        (True, "follow_up", 1, 2, False),
        (True, "follow_up", 0, 1, False),
        (True, "wake", 5, 1, False),  # the wake phrase keeps full reach
        (True, "button", 5, 1, False),
        (False, "follow_up", 5, 1, False),  # the rule off: logged and counted only
        (True, "follow_up", 1, 0, True),  # a threshold below one is one
    ],
)
def test_when_the_rule_applies(enabled, addressed, overheard, threshold, limited):
    reasons = [overheard_reason(overheard, threshold)]
    assert guarded(enabled=enabled, addressed=addressed, reasons=reasons) is limited


def test_the_room_is_not_quiet_for_any_reason_given_from_outside():
    """A later signal (a voice unlike the one that woke Heliox) plugs in as a reason, as the count does."""
    assert guarded(enabled=True, addressed="follow_up", reasons=["", "a different voice"])
    assert not guarded(enabled=True, addressed="follow_up", reasons=["", ""])


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


def test_the_follow_up_counts_hold_four_numbers_and_a_day_and_reset(tmp_path):
    import json
    from datetime import date

    from pilot.conversation.follow_up_counts import FollowUpCounts

    path = tmp_path / "follow_up_counts.json"
    counts = FollowUpCounts(path, today=lambda: date(2026, 11, 13))
    for overheard in (0, 1, 2, 3, 5):
        counts.record(overheard)
    assert counts.read() == {"since": "2026-11-13", "taken": 5, "at_least_1": 4, "at_least_2": 3, "at_least_3": 2}
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert set(kept) == {"since", "taken", "at_least_1", "at_least_2", "at_least_3"}
    assert all(isinstance(value, int) for key, value in kept.items() if key != "since")  # no words, no times
    later = FollowUpCounts(path, today=lambda: date(2026, 12, 1))
    assert later.reset() == {"since": "2026-12-01", "taken": 0, "at_least_1": 0, "at_least_2": 0, "at_least_3": 0}
    assert later.read()["taken"] == 0


def test_the_counts_are_kept_on_every_follow_up_and_shown_in_the_voice_diagnostics(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from pilot.config import PilotConfig
    from pilot.conversation.follow_up_counts import FollowUpCounts
    from pilot.server import PilotServer
    from pilot.system.voice import voice_turn_addressed

    server = PilotServer(PilotConfig())
    server._follow_up_counts_store = FollowUpCounts(tmp_path / "counts.json")
    server._voice_listener = SimpleNamespace(addressing=SimpleNamespace(overheard_since_heliox=2))
    token = voice_turn_addressed.set("follow_up")
    try:
        server._reach_limited("what time is it")  # the rule is off: counted all the same
    finally:
        voice_turn_addressed.reset(token)
    server._voice_listener = None
    stats = asyncio.run(server._handle_voice_listener_stats({}, None))
    assert stats["follow_up_counts"]["taken"] == 1 and stats["follow_up_counts"]["at_least_2"] == 1
    reset = asyncio.run(server._handle_voice_follow_up_counts_reset({}, None))
    assert reset["follow_up_counts"]["taken"] == 0

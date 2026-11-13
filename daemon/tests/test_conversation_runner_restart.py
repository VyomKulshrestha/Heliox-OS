"""The session runner as the daemon: a restart starts on what was saved, "wrap up my day" runs, and with no model at
all "what do you know about me" does not say the profile goes anywhere (found by the Live agent's day, 13 November)."""

from __future__ import annotations

import asyncio

from benchmarks.conversation_eval.sessions import run_session


def _said(record, turn: int) -> str:
    return " ".join(s.text for s in record.said if s.turn == turn)


def test_a_setting_changed_by_voice_survives_a_restart():
    session = {
        "name": "profile off, then a restart",
        "turns": [
            {"say": "Hey Heliox, keep my profile on this computer."},
            {"event": "restart"},
            {"say": "Hey Heliox, call me Sam.", "gap": 2},
        ],
    }
    record, failures = asyncio.run(run_session(session, "claude"))
    assert not failures, failures
    assert record.config_after[0]["conversation"]["share_profile"] == "off"
    # After the restart the saved "off" stands: the next save (if any) keeps it, and nothing turned it back on.
    assert record.config_after[2].get("conversation", {}).get("share_profile", "off") == "off"


def test_two_sessions_never_read_each_others_saves():
    off = {"name": "off", "turns": [{"say": "Hey Heliox, keep my profile on this computer."}]}
    plain = {"name": "plain", "turns": [{"event": "restart"}, {"say": "Hey Heliox, what time is it?"}]}
    asyncio.run(run_session(off, "claude"))
    record, failures = asyncio.run(run_session(plain, "claude"))
    assert not failures, failures
    assert record.config_after[1].get("conversation", {}).get("share_profile", "auto") != "off"


def test_wrap_up_my_day_runs_in_the_runner():
    session = {"name": "wrap up", "turns": [{"say": "Hey Heliox, wrap up my day."}]}
    record, failures = asyncio.run(run_session(session, "none"))
    assert not failures, failures
    assert _said(record, 0)


def test_with_no_model_nothing_is_said_to_go_anywhere():
    session = {
        "name": "no model, what do you know",
        "turns": [
            {"say": "Hey Heliox, call me Sam."},
            {"say": "Hey Heliox, what do you know about me?", "gap": 2},
        ],
    }
    record, failures = asyncio.run(run_session(session, "none"))
    assert not failures, failures
    reply = _said(record, 1)
    assert "Sam" in reply
    assert " go to " not in reply and "None of it leaves this computer." in reply

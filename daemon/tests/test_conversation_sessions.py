"""Whole sessions through the real server, for each kind of person (benchmarks/conversation_eval/sessions.py).

A session that fails because of another workstream's code is listed in KNOWN with the turn and the cause; it is run
and must still fail the same way (strict xfail), so a fix there shows up here as a pass to take off the list."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.conversation_eval.sessions import KINDS, Record, check_session, load_pack, run_session

SESSIONS = load_pack()

# (session, kind) -> the cause, owned elsewhere.
KNOWN: dict[tuple[str, str], str] = {
    ("a long answer and enough", "none"): (
        "pilot/workflows/quick_answers.py: 'what's two plus two?' (numbers in words) is not answered in code, so a "
        "person with no model gets no answer"
    ),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("session", SESSIONS, ids=[s["name"] for s in SESSIONS])
def test_a_whole_session(session, kind, request):
    cause = KNOWN.get((session["name"], kind))
    if cause:
        request.applymarker(pytest.mark.xfail(reason=cause, strict=True))
    _, failures = asyncio.run(run_session(session, kind))
    assert failures == [], "\n".join(failures)


def test_the_pack_covers_what_the_round_asked_for():
    assert 15 <= len(SESSIONS) <= 25
    assert all(6 <= len([t for t in s["turns"] if "say" in t]) <= 15 for s in SESSIONS)
    names = " ".join(s["name"] for s in SESSIONS).casefold()
    for topic in (
        "morning",
        "correction",
        "shorter",
        "again",
        "live",
        "vague destructive",
        "approval",
        "passing",
        "misheard",
        "someone else",
        "call",
        "wake phrase",
        "misheard right after the wake",
        "undo",
    ):
        assert topic in names, topic


def test_the_cross_session_checks_catch_what_no_single_test_owns():
    from benchmarks.conversation_eval.sessions import Said, Shown

    session = {"turns": [{"say": "a"}, {"say": "b"}, {"say": "c"}]}
    record = Record()
    record.said = [Said(1, 0, "Okay.", "final_answer", False), Said(2, 1, "Okay.", "final_answer", False)]
    assert any("said twice running" in f for f in check_session(session, "claude", record))
    record.said = [Said(1, 0, " ".join(["word"] * 26), "final_answer", False)]
    assert any("26 words" in f for f in check_session(session, "claude", record))
    record.said = [Said(1, 0, "Lisbon.", "final_answer", True)]
    assert any("mid-turn" in f for f in check_session(session, "claude", record))
    record.said = [Said(1, 1, "Madrid.", "final_answer", False), Said(2, 0, "Lisbon.", "final_answer", False)]
    record.shown = [Shown(3, 1, "voice_result", {})]
    assert any("turn 1 spoke after turn 2" in f for f in check_session(session, "claude", record))
    # A task of turn 1 that ends while turn 2 runs may still update the screen: that is its own, and in order.
    record.said = [Said(1, 0, "Lisbon.", "final_answer", False), Said(2, 1, "Madrid.", "final_answer", False)]
    record.shown = [Shown(3, 1, "voice_result", {}), Shown(4, 0, "task_complete", {})]
    assert not any("output order" in f for f in check_session(session, "claude", record))

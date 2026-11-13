"""The Live Bench's measure of when a delivery landed (pilot.live.bench.moment_of): a measurement only."""

from __future__ import annotations

import random

from pilot.live.bench import PERSONAS, Session, expand, keep_typing, moment_of, when_table

STEADY = PERSONAS["steady"]


def session_of(steps: list[dict]) -> Session:
    scenario = {"name": "t", "app": "notepad.exe", "title": "a.txt - Notepad", "steps": steps}
    return Session(scenario, expand(scenario, STEADY, random.Random(1)), 0.0)


def play_to(session: Session, at: float) -> None:
    t = 0.0
    while t <= at:
        session.advance(t)
        t = round(t + 0.2, 3)


def test_a_card_in_a_pause_after_the_slip():
    session = session_of([{"type": "Meet on Thursday 19 November."}, {"pause": 4.0}])
    done = len("Meet on Thursday 19 November.") / STEADY.chars_per_second
    play_to(session, done + 2.0)
    found = moment_of(session, {"evidence": "Thursday 19 November", "source": "focused_field"}, done + 2.0)
    assert found["doing"] == "pause" and 1.5 <= found["paused"] <= 2.5
    assert 2.0 <= found["after_slip"] <= 3.5  # the evidence was whole before the full stop was typed


def test_mid_word_between_words_and_text_gone():
    session = session_of([{"type": "Meet on Thursday 19 November and then"}, {"send": True}, {"pause": 3.0}])
    typing = len("Meet on Thursday 19 November and then") / STEADY.chars_per_second
    play_to(session, typing - 0.3)
    payload = {"evidence": "Thursday 19 November", "source": "focused_field"}
    assert moment_of(session, payload, typing - 0.3)["doing"] in ("mid-word", "between words")
    play_to(session, typing + 1.5)
    assert moment_of(session, payload, typing + 1.5)["doing"] == "text gone"  # sent


def test_left_the_field():
    session = session_of([{"type": "Meet on Thursday 19 November."}, {"switch": "chrome.exe|Inbox"}, {"pause": 3.0}])
    play_to(session, 9.0)
    found = moment_of(session, {"evidence": "Thursday 19 November", "source": "focused_field"}, 9.0)
    assert found["doing"] == "left the field"


def test_a_spoken_reply_to_speech_and_the_person_speaking_again():
    session = session_of([{"say": "meet on Thursday the nineteenth", "seconds": 2.0}, {"pause": 1.0},
                          {"say": "and book the room", "seconds": 2.0}])  # fmt: skip
    play_to(session, 2.4)
    payload = {"evidence": "Thursday the nineteenth", "source": "speech", "spoken": True,
               "title": "The nineteenth is a Wednesday.", "message": "You said Thursday."}  # fmt: skip
    found = moment_of(session, payload, 2.4)
    assert found["doing"] == "after speaking" and abs(found["after_slip"] - 0.4) < 0.01
    assert found["overlaps_person"]  # nine words take 3.6 s: the person starts again at 3.0 s
    assert "said, caught: 1 deliveries (1 spoken)" in when_table(
        [{**found, "source": "speech", "expected": True, "spoken": True}]
    )


def test_keep_typing_leaves_speech_and_screen_alone():
    typed = {"name": "a", "app": "x", "title": "y", "steps": [{"type": "a"}]}
    said = {"name": "b", "app": "x", "title": "y", "steps": [{"say": "a"}]}
    assert len(keep_typing(typed)["steps"]) == 4 and keep_typing(said) is said

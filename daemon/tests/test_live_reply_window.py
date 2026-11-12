"""Answering Live without the wake phrase, right after it spoke, with no conversation open.

The rule (decided 12 November): Live spoke to the person, so for about 6 s its own closed list of replies ("why?",
"thanks", "not now", "don't mention these", "fix it", "never mind", stop words) is taken without the wake phrase;
nothing else is, and nothing opens the conversation. The listener's side (pilot.system.addressing and
pilot.system.voice) belongs to the speech workstream; the case it does not do yet is marked xfail (strict), so the
test turns red the day it does, as a reminder to drop the mark.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.live.engine import is_live_reply
from pilot.system.addressing import Addressing, invites_reply
from pilot.system.voice import ContinuousVoiceListener

REMARK = "The nineteenth of November is a Thursday. You wrote Monday."


@pytest.mark.parametrize(
    "said",
    [
        "Why?",
        "why did you say that",
        "Thanks.",
        "not now",
        "Don't mention these.",
        "fix it",
        "never mind",
        "stop",
        "What do you mean?",
    ],
)
def test_lives_own_replies(said):
    assert is_live_reply(said)


@pytest.mark.parametrize(
    "said",
    [
        "open the door",
        "why don't you open the door",
        "thanks for the file, send it to Sam",
        "not now, I'm driving to Leeds",
        "stop the music",
        "",
        "why is the sky blue",
    ],
)
def test_anything_else_is_not_a_reply_to_live(said):
    assert not is_live_reply(said)


def listener_after_a_live_remark():
    """The real listener and Addressing, after the server's own call when Live has spoken (no conversation open)."""

    async def nothing(_text):
        return None

    clock = [1000.0]
    listener = ContinuousVoiceListener(
        wake_words=["hey heliox"], on_command=nothing, on_overheard=nothing, config=PilotConfig()
    )
    listener._wake_calibrator = MagicMock()
    listener._addressing = Addressing(clock=lambda: clock[0])
    listener._recorder = SimpleNamespace(last_voiced_seconds=0.6, last_turn_ended_at=None, speech_started_at=None)
    listener.arm_follow_up_window(asked=invites_reply(REMARK))  # _speak_companion_text does this after speaking
    note = getattr(listener, "note_live_spoke", None)
    if callable(note):
        note()  # pilot.personal.wiring.live_speak does this once the listener has it
    return listener, clock


def hear(listener, clock, at: float, said: str):
    clock[0] = 1000.0 + at
    listener._recorder.speech_started_at = clock[0]
    listener._recorder.last_turn_ended_at = clock[0] + 0.6
    decision, _ = listener._decide_addressed(said, "", True)
    return decision


def test_why_two_seconds_after_live_spoke_is_taken_and_opens_nothing():
    listener, clock = listener_after_a_live_remark()
    decision = hear(listener, clock, 2.0, "why?")
    assert decision.act
    assert not listener._addressing.conversation_open


@pytest.mark.parametrize(("at", "said"), [(2.0, "open the door"), (9.0, "why?")])
def test_nothing_else_and_nothing_later_is_taken(at, said):
    listener, clock = listener_after_a_live_remark()
    decision = hear(listener, clock, at, said)
    assert not decision.act and decision.why == "no conversation open"
    assert not listener._addressing.conversation_open


@pytest.mark.asyncio
async def test_live_speak_tells_the_listener_only_when_it_was_heard():
    from pilot.personal.wiring import live_speak

    for status, told in (("spoken", 1), ("interrupted", 1), ("suppressed", 0), ("held", 0)):
        listener = SimpleNamespace(is_running=False, note_live_spoke=MagicMock())

        async def speak(_text, *, channel, dedupe_key, status=status):
            return SimpleNamespace(status=status)

        server = SimpleNamespace(_voice_listener=listener, _live=None, _speak_companion_text=speak)
        assert await live_speak(server, REMARK, False) == status
        assert listener.note_live_spoke.call_count == told, status

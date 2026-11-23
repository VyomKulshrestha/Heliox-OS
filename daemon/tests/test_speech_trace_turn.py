"""The stamps for the turn's own story (pilot.system.speech_trace): the opening reading, an early request started,
kept or dropped, and when the hum was decided, so the next end-to-end measurement shows each of them. Fakes only."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.conversation import provisional as prov
from pilot.conversation.layer import ConversationLayer
from pilot.system import speech_trace, voice

WAKE = {"addressed": "wake", "said": "Hey Heliocs, what's the capital of Peru?", "wake_text": "Hey Heliox, what's the"}


@pytest.fixture
def stamps(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(speech_trace, "sink", lambda event, data: seen.append((event, data)))
    return seen


def names(seen):
    return [event for event, _data in seen]


@pytest.mark.asyncio
async def test_the_opening_reading_is_stamped_with_its_wait_and_its_second_read(stamps, monkeypatch):
    item = voice.ContinuousVoiceListener(config=PilotConfig())
    monkeypatch.setattr(item, "_parakeet", lambda: object())
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 0)
    monkeypatch.setattr(voice, "_write_wav", lambda _frames, _rate: "o.wav")
    replies = iter(["Hey Heliox.", "Hey Heliox, what's the"])

    async def whisper(_path, **_kwargs):
        item._recorder.current_frames = [np.zeros(800, dtype=np.int16)] * 48  # 2.4 s said by now
        return next(replies)

    item._transcribe_whisper_text = whisper
    item._recorder.user_is_speaking = True
    item._start_early_wake([np.zeros(800, dtype=np.int16)] * 32)
    assert await item._early_wake == "Hey Heliox, what's the"
    assert names(stamps) == ["opening read started", "opening read done"]
    assert stamps[0][1]["audio_s"] == 1.6 and stamps[1][1] == {**stamps[1][1], "confirms": True, "reread": True}


def test_an_early_request_is_stamped_started_and_dropped_with_its_reason(stamps):
    started: list[str] = []
    item = voice.ContinuousVoiceListener(config=PilotConfig())
    item.early_request = lambda turn, words, addressed: started.append(turn) or True
    layer = ConversationLayer(sink=None, model=lambda: None)

    def cancel(turn: str, reason: str) -> None:
        layer.provisional.add(
            prov.Provisional(turn_id=turn, words="w w w", addressed="wake", started=1.0, model=None, before=[])
        )
        layer.cancel_provisional(turn, reason)

    item.cancel_early_request = cancel
    item._begin_early_request("Hey Heliox, what's the capital of Peru?", WAKE)
    item._drop_early_request("carried_on")
    assert names(stamps) == ["early request started", "early request dropped"]
    assert stamps[0][1]["turn"] == "v1" and stamps[0][1]["n"] == 1
    assert stamps[1][1]["reason"] == "carried_on"


def test_a_kept_early_request_is_stamped_with_its_start_and_first_text(stamps):
    answers = prov.ProvisionalAnswers()
    held = prov.Provisional(turn_id="v3", words="w w w", addressed="wake", started=10.0, model=None, before=[])
    held.first_text_at = 10.9
    answers.note_adopted(held)
    assert names(stamps) == ["early request kept"]
    assert stamps[0][1]["started"] == 10.0 and stamps[0][1]["first_text_at"] == 10.9


@pytest.mark.asyncio
async def test_the_hum_decision_is_stamped_at_the_turn_end_with_or_without_a_hum(stamps):
    heard: list[str] = []

    async def ear(text, channel, key):
        heard.append(text)
        return "spoken"

    layer = ConversationLayer(
        sink=ear,
        model=lambda: SimpleNamespace(local=False),
        clock=lambda: 100.5,
        acknowledge=lambda: True,
        ack_draw=lambda: 0.0,
    )
    assert layer.hum_at_turn_end("v1", "Why is my laptop fan so loud?", 100.0)
    for _ in range(10):
        await asyncio.sleep(0)
    assert names(stamps) == ["hum decided", "hum to the voice"] and heard == ["Mm."]
    assert stamps[0][1]["path"] == "turn end" and stamps[0][1]["hum"] == "Mm."
    held = prov.Provisional(turn_id="v2", words="w w w", addressed="wake", started=1.0, model=None, before=[])
    held.first_text_at = 100.2
    layer.provisional.add(held)
    assert not layer.hum_at_turn_end("v2", "Why is my laptop fan so loud?", 100.0)
    assert stamps[-1][1]["hum"] == "" and stamps[-1][1]["reason"] == "answer ready"

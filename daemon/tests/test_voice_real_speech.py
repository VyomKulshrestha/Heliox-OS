"""Regression cases from a real rehearsal: the owner's voice through a Bluetooth headset in hands-free mode
(narrowband, quiet). Text only; the recording itself stays on the owner's machine.

What the recognisers heard for "Hey Heliox, remind me that the launch call is Monday the twelfth of November at ten":
  streaming (fast)  "Do you know my way down the lawn spot is monday that will do no"
  Whisper base      "Here Heliox, remind me that the launch call is Monday, the 12th of November at 10."
and for noise between sentences, Whisper produced "Thanks for watching." and "Hey Heliox, GitHub, Google, Google…".
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.live.detectors import TextContext, detect
from pilot.system import voice
from pilot.system.voice import (
    ContinuousVoiceListener,
    is_safe_fast_phrase,
    looks_hallucinated,
    loose_wake_prefix,
)

FINAL = "Here Heliox, remind me that the launch call is Monday, the 12th of November at 10."
STREAMING = "Do you know my way down the lawn spot is monday that will do no"


@pytest.mark.parametrize(
    ("heard", "prefix"),
    [
        (FINAL, "Here Heliox"),
        ("Hey Heliox. Remind me that the launch call is Monday, the 12th of November at 10th.", "Hey Heliox"),
        ("Hey Heliox, what can I say?", "Hey Heliox"),
        ("Dear Heliox, Remind me that the launch call is Monday the 12th of November at time.", "Dear Heliox"),
        ("Hello Heliox. Nice to meet you.", "Hello Heliox"),
    ],
)
def test_the_wake_phrase_as_whisper_wrote_the_owners_voice(heard, prefix):
    assert loose_wake_prefix(heard) == prefix


@pytest.mark.parametrize(
    "heard",
    [
        STREAMING,  # the fast recogniser: no wake phrase to find, and none is invented
        "He listen what can they serve",
        "Here it works, what can I say?",
        "Here he looks and what can I say.",
        "The three units remind me that the launch call is Monday the 12th of November at 10.",
    ],
)
def test_misheard_lines_do_not_wake_heliox(heard):
    assert loose_wake_prefix(heard) == ""


def test_the_slip_is_caught_in_the_final_pass_words_and_not_in_the_fast_ones():
    ctx = TextContext(today=date(2026, 11, 10), hour=12, source="speech")
    command = FINAL[len("Here Heliox, ") :]
    [card] = detect(command, ctx)
    assert card.title == "The twelfth of November is a Thursday" and card.message == "You said Monday."
    assert detect(STREAMING, ctx) == []


@pytest.mark.parametrize(
    "text",
    [
        "Thanks for watching.",
        "Thank you.",
        "Hey Heliox, GitHub, Google, Google, Google, Google, Google, Google,",
        "Hey Heliox. Desktop command vocabulary.",
        "Hello Heliox. Good morning. Good morning. Good morning. Good morning.",
    ],
)
def test_noise_read_as_words_is_dropped(text):
    assert looks_hallucinated(text)


@pytest.mark.parametrize("text", [FINAL, "Hey Heliox, what can I say?", "No.", "Stop stop", "Thank you, Priya"])
def test_real_sentences_are_kept(text):
    assert not looks_hallucinated(text)


def test_segments_whisper_marks_as_not_speech_are_dropped():
    segments = [
        SimpleNamespace(text=" Hey Heliox, what can I say?", no_speech_prob=0.14),
        SimpleNamespace(text=" Thanks for watching.", no_speech_prob=0.77),
    ]
    assert voice._speech_text(segments) == "Hey Heliox, what can I say?"
    assert voice._speech_text([SimpleNamespace(text=" Hey Heliox, GitHub", no_speech_prob=0.68)]) == ""


def test_only_stopping_and_denying_are_taken_from_the_fast_recogniser():
    for safe in ("No.", "Stop", "Wait, stop", "Cancel", "Hold on"):
        assert is_safe_fast_phrase(safe)
    for unsafe in ("Yes", "Approve", "Delete it", STREAMING, "no problem go ahead"):
        assert not is_safe_fast_phrase(unsafe)


def listener_with(heard: str, final: str):
    listener = ContinuousVoiceListener(wake_words=["hey heliox", "heliox"], config=PilotConfig())
    listener._recorder._soundcard_recorder = object()  # an open stream, so the utterance path is used
    listener._recorder.record_utterance = AsyncMock(return_value="utterance.wav")
    streaming = MagicMock()
    streaming.finish = AsyncMock(return_value=heard)
    listener._configure_turn_taking = lambda: streaming
    listener._transcribe_path = AsyncMock(return_value=final)
    captions: list[str] = []

    async def caption(text):
        captions.append(text)

    listener._on_partial = caption
    listener._loop = asyncio.get_running_loop()
    return listener, captions


@pytest.mark.asyncio
async def test_the_words_acted_on_come_from_the_accurate_pass_and_the_caption_is_corrected():
    listener, captions = listener_with(STREAMING, FINAL)
    assert await listener._record_and_transcribe() == FINAL
    assert listener.last_transcription_engine == "whisper"
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert captions == [FINAL]


@pytest.mark.asyncio
async def test_a_stop_or_a_no_is_taken_from_the_fast_pass_without_waiting():
    listener, _ = listener_with("No", "Know")
    assert await listener._record_and_transcribe() == "No"
    listener._transcribe_path.assert_not_awaited()
    assert listener.last_transcription_engine == "streaming"

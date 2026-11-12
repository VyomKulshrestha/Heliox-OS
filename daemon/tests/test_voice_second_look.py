"""The second look at telephone bandwidth: on a headset in call mode, when the wake check heard no wake phrase,
Whisper reads the utterance once more through a 16 -> 8 -> 16 kHz round trip, and that reading counts only if it
holds one. Never on a full-band microphone, and never after an utterance the first look took. The two utterances of
the bench replay whose outcome it changes (12 November): the owner's "Hey Heliox, remind me that the launch call is
Monday the twelfth of November at ten." on the call-mode headset, which nothing heard at 16 kHz and which is now
taken; and the synthetic "Helix is a nice name for a project.", which the second look reads as "Heliox is a nice name
for a project." and the rule still refuses (talk about Heliox). Replay with the second look on every utterance:
0 of 981 unaddressed dispatched, 12 of 13 addressed taken (11 before). Fakes only: no recogniser, no audio."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.system.addressing import Addressing, Heard
from tests.test_parakeet_final_pass import _parakeet_listener

UTT00_16K = ""  # Whisper base heard nothing it rated as speech
UTT00_8K = "Here Heliox, invite me to enter launch. Polish Monday, 12th November at 10."


def call_mode(item, verdict: str = "", name: str = "Headset (Rockerz 415 Hands-Free AG Audio)"):
    item._mic_quality.verdict = verdict
    item._recorder.input_device_name = name


@pytest.mark.asyncio
async def test_a_call_mode_headset_gets_a_second_look_that_finds_the_wake_phrase(tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "")  # Parakeet heard nothing either
    item._transcribe_whisper_text = AsyncMock(return_value=UTT00_16K)
    item._telephone_band_wake = AsyncMock(return_value=UTT00_8K)
    call_mode(item)
    text = await item._transcribe_path("utterance.wav")
    item._telephone_band_wake.assert_awaited_once()
    assert item.last_wake_source == "telephone band"
    decision = Addressing(clock=lambda: 0.0).decide(Heard(text, 0.0, 5.0, 3.8, wake_text=item._last_wake_text))
    assert decision.act and decision.why == "wake"


@pytest.mark.asyncio
async def test_the_microphone_check_s_narrowband_verdict_is_enough_where_the_name_does_not_say(tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "Here is what can I say?")
    item._transcribe_whisper_text = AsyncMock(return_value="Here we go to work the nice video.")
    item._telephone_band_wake = AsyncMock(return_value="Here we go, the work can I fill it?")
    call_mode(item, verdict="narrowband", name="AirPods Pro")  # macOS shows the plain headset name
    await item._transcribe_path("utterance.wav")
    item._telephone_band_wake.assert_awaited_once()
    assert item.last_wake_source == "whole"  # no wake phrase in the second look either: utt04 stays a miss


@pytest.mark.asyncio
async def test_talk_about_heliox_found_by_the_second_look_is_still_refused(tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "Helix is a nice name for a project.")
    item._transcribe_whisper_text = AsyncMock(return_value="Helix is a nice name for a project.")
    item._telephone_band_wake = AsyncMock(return_value="Heliox is a nice name for a project.")
    call_mode(item)
    text = await item._transcribe_path("utterance.wav")
    decision = Addressing(clock=lambda: 0.0).decide(Heard(text, 0.0, 2.5, 2.0, wake_text=item._last_wake_text))
    assert not decision.act


@pytest.mark.asyncio
@pytest.mark.parametrize(("verdict", "name"), [("ok", "Microphone (USB Audio Device)"), ("", "MacBook Pro Microphone")])
async def test_a_full_band_microphone_never_gets_a_second_look(verdict, name, tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "")
    item._transcribe_whisper_text = AsyncMock(return_value="")
    item._telephone_band_wake = AsyncMock(return_value=UTT00_8K)
    call_mode(item, verdict=verdict, name=name)
    await item._transcribe_path("utterance.wav")
    item._telephone_band_wake.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_utterance_the_first_look_took_is_never_delayed_by_a_second(tmp_path, monkeypatch):
    item = _parakeet_listener(tmp_path, monkeypatch, "Hey, what's not connecting?")
    item._transcribe_whisper_text = AsyncMock(return_value="Hey Heliox, what can I do?")
    item._telephone_band_wake = AsyncMock(return_value="")
    call_mode(item)
    await item._transcribe_path("utterance.wav")
    item._telephone_band_wake.assert_not_awaited()
    assert item.last_wake_source == "whole"

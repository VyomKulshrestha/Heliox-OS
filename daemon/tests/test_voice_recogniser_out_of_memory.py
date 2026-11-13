"""A recogniser that runs out of memory never leaves Heliox deaf without saying so: the bar says it once, health shows
hearing as a fault with that sentence (capture_notice), and Heliox listens with the smallest recogniser present.
Stand-in loads that raise MemoryError: no model."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import voice


def listener(whisper_model: str = "small", final_pass: str = "whisper") -> voice.ContinuousVoiceListener:
    config = PilotConfig()
    config.voice.whisper_model = whisper_model
    config.voice.final_pass = final_pass
    config.voice.transcription_engine = "faster_whisper"
    config.voice.turn_detection = "silence"
    return voice.ContinuousVoiceListener(wake_words=["hey heliox"], config=config)


def present(monkeypatch, sizes: set[str]) -> None:
    monkeypatch.setattr("pilot.system.model_presence.whisper_present", lambda size: size in sizes)
    monkeypatch.setattr("pilot.system.model_presence.usable_whisper_model", lambda configured: configured)


def statuses_of(item: voice.ContinuousVoiceListener) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []

    async def on_status(status, data):
        seen.append((status, data))

    item._on_status = on_status
    item._loop = asyncio.get_running_loop()
    return seen


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_whisper_out_of_memory_falls_back_to_the_smallest_present_and_says_so_once(monkeypatch):
    item = listener("small")
    present(monkeypatch, {"small", "tiny", "base"})
    seen = statuses_of(item)
    loaded: list[str] = []

    async def transcribe(path, language, model, engine="auto"):
        loaded.append(model)
        if model == "small":
            raise MemoryError()
        return {"text": "what time is it", "language": "en"}

    monkeypatch.setattr(voice, "_transcribe_speech", transcribe)
    assert await item._transcribe_whisper_text("u.wav") == "what time is it"
    assert loaded == ["small", "tiny"]  # the smallest present, not OpenAI Whisper, not the same model again
    assert "Whisper small" in item.capture_notice and "Whisper tiny" in item.capture_notice
    assert await item._transcribe_whisper_text("u.wav") == "what time is it"
    assert loaded == ["small", "tiny", "tiny"]  # from then on the smaller one
    await settle()
    assert [status for status, _ in seen] == ["recogniser_out_of_memory"]  # said once


@pytest.mark.asyncio
async def test_with_nothing_smaller_it_says_heliox_cannot_hear(monkeypatch):
    item = listener("tiny")
    present(monkeypatch, {"tiny"})
    statuses_of(item)

    async def transcribe(path, language, model, engine="auto"):
        raise MemoryError()

    monkeypatch.setattr(voice, "_transcribe_speech", transcribe)
    assert await item._transcribe_whisper_text("u.wav") == ""
    assert "can't hear you" in item.capture_notice


@pytest.mark.asyncio
async def test_openai_whisper_is_not_tried_after_running_out_of_memory(monkeypatch):
    async def out_of_memory(*_args):
        raise MemoryError()

    fallback = AsyncMock()
    monkeypatch.setattr(voice, "_transcribe_faster_whisper", out_of_memory)
    monkeypatch.setattr(voice, "_transcribe_whisper", fallback)
    with pytest.raises(MemoryError):
        await voice._transcribe_speech("u.wav", "en", "small", "auto")
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_parakeet_out_of_memory_is_set_aside_and_whisper_reads_the_sentence(monkeypatch, tmp_path):
    item = listener("base", final_pass="parakeet")
    seen = statuses_of(item)

    class Parakeet:
        available = True

        def transcribe(self, samples):
            raise MemoryError()

    parakeet = Parakeet()
    monkeypatch.setattr(type(item), "_parakeet", lambda self: None if self._parakeet_out_of_memory else parakeet)
    path = Path(tmp_path / "u.wav")
    with wave.open(str(path), "w") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes((np.random.default_rng(0).normal(0, 300, 16000)).astype(np.int16).tobytes())

    async def whisper(audio_path, language=None):
        return "hey heliox what time is it"

    item._transcribe_whisper_text = whisper
    assert await item._transcribe_path(str(path)) == "hey heliox what time is it"  # the sentence is not lost
    assert item._parakeet() is None  # set aside for the session
    assert "Parakeet" in item.capture_notice
    await settle()
    assert [status for status, _ in seen] == ["recogniser_out_of_memory"]

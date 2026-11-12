"""Tests that the Whisper model is loaded once and reused across calls,
instead of being reloaded from disk on every transcription (as it was
before this fix -- see _get_whisper_model's docstring in voice.py)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import pilot.system.voice as voice


@pytest.fixture(autouse=True)
def _clear_model_cache(monkeypatch):
    voice._whisper_model_cache.clear()
    # Whisper is loaded only from a file already on this computer (pilot.system.model_presence); here every
    # model counts as cached, at a made-up path, so the tests do not depend on this machine's cache.
    monkeypatch.setattr("pilot.system.model_presence.openai_whisper_cached", lambda name: f"cache/{name}.pt")
    yield
    voice._whisper_model_cache.clear()


def _fake_whisper_module(load_model_mock: MagicMock) -> ModuleType:
    module = ModuleType("whisper")
    module.load_model = load_model_mock
    return module


def test_loads_model_once_across_repeated_calls():
    load_model = MagicMock(return_value=MagicMock())

    with patch.dict(sys.modules, {"whisper": _fake_whisper_module(load_model)}):
        first = voice._get_whisper_model("base")
        second = voice._get_whisper_model("base")

    load_model.assert_called_once_with("cache/base.pt")
    assert first is second


def test_a_model_that_is_not_on_this_computer_is_never_fetched(monkeypatch):
    from pilot.system.model_presence import SpeechModelMissing

    monkeypatch.setattr("pilot.system.model_presence.openai_whisper_cached", lambda name: None)
    load_model = MagicMock()

    with patch.dict(sys.modules, {"whisper": _fake_whisper_module(load_model)}), pytest.raises(SpeechModelMissing):
        voice._get_whisper_model("base")

    load_model.assert_not_called()


def test_different_model_names_get_separate_cache_entries():
    load_model = MagicMock(side_effect=lambda name: MagicMock(name=name))

    with patch.dict(sys.modules, {"whisper": _fake_whisper_module(load_model)}):
        base = voice._get_whisper_model("base")
        small = voice._get_whisper_model("small")
        base_again = voice._get_whisper_model("base")

    assert load_model.call_count == 2
    assert base is base_again
    assert base is not small


@pytest.mark.asyncio
async def test_transcribe_whisper_uses_cached_model():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": "hello world", "language": "en"}
    load_model = MagicMock(return_value=fake_model)

    with patch.dict(sys.modules, {"whisper": _fake_whisper_module(load_model)}):
        result1 = await voice._transcribe_whisper("a.wav", "auto", "base")
        result2 = await voice._transcribe_whisper("b.wav", "auto", "base")

    assert result1["text"] == "hello world"
    assert result2["text"] == "hello world"
    load_model.assert_called_once_with("cache/base.pt")
    assert fake_model.transcribe.call_count == 2
    kwargs = fake_model.transcribe.call_args.kwargs
    assert kwargs["initial_prompt"] == voice._VOICE_TRANSCRIPTION_PROMPT
    assert "Names may include" not in kwargs["initial_prompt"]

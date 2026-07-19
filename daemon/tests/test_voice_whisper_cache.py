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
def _clear_model_cache():
    voice._whisper_model_cache.clear()
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

    load_model.assert_called_once_with("base")
    assert first is second


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
    load_model.assert_called_once_with("base")
    assert fake_model.transcribe.call_count == 2

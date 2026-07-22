"""Tests for _speak_impl's Pocket TTS engine branch: selecting it by
default, falling back to the existing OS-native TTS dispatch on any
failure or when tts_engine == "os_native", and propagating CancelledError
(barge-in) without falling back mid-interruption. Mirrors
test_voice_supersede.py's patch.object(voice, ...) style for isolating
_speak_impl from the rest of speak()'s supersede machinery, which is
untouched by this feature and already covered there.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.system import voice
from pilot.system.platform_detect import Platform


def _config_with(tts_engine: str = "pocket_tts", tts_voice: str = "alba") -> PilotConfig:
    config = PilotConfig()
    config.voice = VoiceConfig(tts_engine=tts_engine, tts_voice=tts_voice)
    return config


@pytest.mark.asyncio
async def test_default_engine_uses_pocket_tts(monkeypatch):
    monkeypatch.setattr(PilotConfig, "load", lambda: _config_with())

    with patch("pilot.system.pocket_tts.synthesize_and_play", new=AsyncMock()) as mock_synth:
        result = await voice._speak_impl("hello world")

    mock_synth.assert_awaited_once_with("hello world", "alba")
    assert result == "Spoken: hello world..."


@pytest.mark.asyncio
async def test_pocket_tts_writes_to_output_file_when_given(monkeypatch):
    monkeypatch.setattr(PilotConfig, "load", lambda: _config_with())

    with patch("pilot.system.pocket_tts.synthesize_to_file", new=AsyncMock()) as mock_write:
        result = await voice._speak_impl("hello", output_file="out.wav")

    mock_write.assert_awaited_once_with("hello", "alba", "out.wav")
    assert result == "Speech saved to out.wav"


@pytest.mark.asyncio
async def test_os_native_engine_skips_pocket_tts_entirely(monkeypatch):
    monkeypatch.setattr(PilotConfig, "load", lambda: _config_with(tts_engine="os_native"))
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)

    with (
        patch("pilot.system.pocket_tts.synthesize_and_play", new=AsyncMock()) as mock_synth,
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="os-native result")) as mock_os_tts,
    ):
        result = await voice._speak_impl("hello")

    mock_synth.assert_not_awaited()
    mock_os_tts.assert_awaited_once()
    assert result == "os-native result"


@pytest.mark.asyncio
async def test_pocket_tts_failure_falls_back_to_os_native(monkeypatch):
    monkeypatch.setattr(PilotConfig, "load", lambda: _config_with())
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)

    with (
        patch(
            "pilot.system.pocket_tts.synthesize_and_play",
            new=AsyncMock(side_effect=RuntimeError("model failed to load")),
        ) as mock_synth,
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="os-native result")) as mock_os_tts,
    ):
        result = await voice._speak_impl("hello")

    mock_synth.assert_awaited_once()
    mock_os_tts.assert_awaited_once()
    assert result == "os-native result"


@pytest.mark.asyncio
async def test_pocket_tts_import_error_falls_back_to_os_native(monkeypatch):
    """The common case: the optional `pocket-tts` package isn't installed."""
    monkeypatch.setattr(PilotConfig, "load", lambda: _config_with())
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)

    def _raise_import_error(*_args, **_kwargs):
        raise ImportError("No module named 'pocket_tts'")

    with (
        patch("pilot.system.pocket_tts.synthesize_and_play", side_effect=_raise_import_error),
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="os-native result")) as mock_os_tts,
    ):
        result = await voice._speak_impl("hello")

    mock_os_tts.assert_awaited_once()
    assert result == "os-native result"


@pytest.mark.asyncio
async def test_pocket_tts_cancellation_propagates_without_falling_back(monkeypatch):
    monkeypatch.setattr(PilotConfig, "load", lambda: _config_with())
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)

    with (
        patch(
            "pilot.system.pocket_tts.synthesize_and_play",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ),
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="os-native result")) as mock_os_tts,
        pytest.raises(asyncio.CancelledError),
    ):
        await voice._speak_impl("hello")

    mock_os_tts.assert_not_awaited()

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import pilot.system.voice as voice


@pytest.fixture(autouse=True)
def _clear_model_cache():
    voice._faster_whisper_model_cache.clear()
    yield
    voice._faster_whisper_model_cache.clear()


def test_faster_whisper_model_is_cached_by_runtime():
    model_type = MagicMock(return_value=object())
    module = SimpleNamespace(WhisperModel=model_type)

    with (
        patch.dict("sys.modules", {"faster_whisper": module}),
        patch.object(voice, "_faster_whisper_runtime", return_value=("cpu", "int8")),
    ):
        first = voice._get_faster_whisper_model("base")
        second = voice._get_faster_whisper_model("base")

    assert first is second
    model_type.assert_called_once()
    args, kwargs = model_type.call_args
    assert args == ("base",) and kwargs["device"] == "cpu" and kwargs["compute_type"] == "int8"
    assert 1 <= kwargs["cpu_threads"] <= 8  # measured: 8 threads halve the pass on a 6 s sentence


@pytest.mark.asyncio
async def test_faster_whisper_uses_command_aware_deterministic_decoding():
    model = MagicMock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(text=" open "), SimpleNamespace(text=" GitHub ")]),
        SimpleNamespace(language="en"),
    )

    with patch.object(voice, "_get_faster_whisper_model", return_value=model):
        result = await voice._transcribe_faster_whisper("voice.wav", "auto", "base")

    assert result == {"text": "open GitHub", "language": "en"}
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["beam_size"] == 5
    assert kwargs["temperature"] == 0
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is False
    assert kwargs["initial_prompt"] == voice._VOICE_TRANSCRIPTION_PROMPT
    assert kwargs["without_timestamps"] is True and "hotwords" not in kwargs
    assert "language" not in kwargs


@pytest.mark.asyncio
async def test_auto_engine_falls_back_to_openai_whisper():
    expected = {"text": "show system information", "language": "en"}
    with (
        patch.object(voice, "_transcribe_faster_whisper", side_effect=ImportError),
        patch.object(voice, "_transcribe_whisper", return_value=expected) as fallback,
    ):
        result = await voice._transcribe_speech("voice.wav", "auto", "base", "auto")

    assert result == expected
    fallback.assert_awaited_once_with("voice.wav", "auto", "base")


@pytest.mark.asyncio
async def test_explicit_faster_whisper_does_not_hide_backend_failure():
    with (
        patch.object(voice, "_transcribe_faster_whisper", side_effect=ImportError),
        pytest.raises(ImportError),
    ):
        await voice._transcribe_speech("voice.wav", "auto", "base", "faster_whisper")


@pytest.mark.asyncio
async def test_unknown_transcription_engine_is_rejected():
    with pytest.raises(ValueError, match="Unsupported transcription engine"):
        await voice._transcribe_speech("voice.wav", "auto", "base", "remote_magic")

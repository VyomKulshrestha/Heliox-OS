"""Tests for pilot.system.pocket_tts -- Kyutai Pocket TTS's model/voice-state
caching (mirrors test_voice_whisper_cache.py's fake-module-injection
technique for voice._get_whisper_model, since pocket_tts is never actually
installed in this dev/CI environment), audio generation, and playback
(including barge-in cancellation calling sounddevice.stop()).
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import pilot.system.pocket_tts as pocket_tts


@pytest.fixture(autouse=True)
def _clear_caches():
    pocket_tts._model_cache.clear()
    pocket_tts._voice_state_cache.clear()
    yield
    pocket_tts._model_cache.clear()
    pocket_tts._voice_state_cache.clear()


def _fake_audio_tensor(values: list[int]) -> MagicMock:
    """A stand-in for the 1D torch.Tensor generate_audio() returns --
    only .numpy() is ever called on it by this module."""
    tensor = MagicMock()
    tensor.numpy.return_value = np.array(values, dtype=np.float32)
    return tensor


def _fake_pocket_tts_module(model_mock: MagicMock) -> ModuleType:
    module = ModuleType("pocket_tts")
    module.TTSModel = MagicMock()
    module.TTSModel.load_model.return_value = model_mock
    return module


def _fake_model(sample_rate: int = 24000) -> MagicMock:
    model = MagicMock()
    model.sample_rate = sample_rate
    model.get_state_for_audio_prompt.side_effect = lambda voice: f"state-for-{voice}"
    model.generate_audio.return_value = _fake_audio_tensor([1, 2, 3])
    return model


def test_get_model_loads_once_across_repeated_calls():
    model = _fake_model()
    fake_module = _fake_pocket_tts_module(model)

    with patch.dict(sys.modules, {"pocket_tts": fake_module}):
        first = pocket_tts._get_model()
        second = pocket_tts._get_model()

    fake_module.TTSModel.load_model.assert_called_once()
    assert first is second


def test_get_voice_state_caches_per_voice_name():
    model = _fake_model()

    first = pocket_tts._get_voice_state(model, "alba")
    second = pocket_tts._get_voice_state(model, "alba")
    third = pocket_tts._get_voice_state(model, "giovanni")

    assert model.get_state_for_audio_prompt.call_count == 2
    assert first == second == "state-for-alba"
    assert third == "state-for-giovanni"


@pytest.mark.asyncio
async def test_synthesize_returns_numpy_audio_and_sample_rate():
    model = _fake_model(sample_rate=24000)
    fake_module = _fake_pocket_tts_module(model)

    with patch.dict(sys.modules, {"pocket_tts": fake_module}):
        audio, sample_rate = await pocket_tts.synthesize("hello", "alba")

    assert isinstance(audio, np.ndarray)
    assert list(audio) == [1, 2, 3]
    assert sample_rate == 24000
    model.generate_audio.assert_called_once_with("state-for-alba", "hello")


@pytest.mark.asyncio
async def test_warmup_loads_model_and_voice_without_generating_audio():
    model = _fake_model(sample_rate=24000)
    fake_module = _fake_pocket_tts_module(model)

    with patch.dict(sys.modules, {"pocket_tts": fake_module}):
        await pocket_tts.warmup("alba")

    fake_module.TTSModel.load_model.assert_called_once()
    model.get_state_for_audio_prompt.assert_called_once_with("alba")
    model.generate_audio.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_propagates_import_error_when_package_missing():
    with patch.dict(sys.modules, {"pocket_tts": None}), pytest.raises(ImportError):
        await pocket_tts.synthesize("hello", "alba")


def _fake_sounddevice_module(play=None, wait=None, stop=None) -> ModuleType:
    """sounddevice is, like pocket_tts, an optional dependency never
    installed in CI (see test_continuous_recorder.py's equivalent
    real-hardware avoidance) -- patching "sounddevice.play" directly would
    require the real package to be importable just to resolve the patch
    target, which fails with ModuleNotFoundError when it isn't installed.
    Injecting a fake module via sys.modules, mirroring
    _fake_pocket_tts_module above, works regardless of whether the real
    package is present."""
    module = ModuleType("sounddevice")
    module.play = play or MagicMock()
    module.wait = wait or MagicMock()
    module.stop = stop or MagicMock()
    return module


@pytest.mark.asyncio
async def test_play_goes_through_the_shared_output_line(monkeypatch):
    """Never sd.play: a second output stream in the daemon stops the microphone on a Mac whose microphone is on the
    output's device. The shared line (voice.play_on_output_line) fades on a cut."""
    from pilot.system import voice

    audio = np.array([1, 2, 3], dtype=np.float32)
    line = AsyncMock()
    monkeypatch.setattr(voice, "play_on_output_line", line)
    await pocket_tts.play(audio, 24000)
    line.assert_awaited_once_with(audio, 24000)


@pytest.mark.asyncio
async def test_synthesize_and_play_composes_synthesize_and_play(monkeypatch):
    from pilot.system import voice

    model = _fake_model(sample_rate=24000)
    fake_pocket_tts = _fake_pocket_tts_module(model)
    line = AsyncMock()
    monkeypatch.setattr(voice, "play_on_output_line", line)

    with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}):
        await pocket_tts.synthesize_and_play("hello", "alba")

    model.generate_audio.assert_called_once_with("state-for-alba", "hello")
    line.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesize_to_file_writes_wav(tmp_path):
    model = _fake_model(sample_rate=24000)
    fake_module = _fake_pocket_tts_module(model)
    output_file = str(tmp_path / "out.wav")

    with patch.dict(sys.modules, {"pocket_tts": fake_module}), patch("scipy.io.wavfile.write") as mock_write:
        await pocket_tts.synthesize_to_file("hello", "alba", output_file)

    assert mock_write.call_count == 1
    args, _kwargs = mock_write.call_args
    assert args[0] == output_file
    assert args[1] == 24000
    assert list(args[2]) == [1, 2, 3]

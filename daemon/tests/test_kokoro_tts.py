from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pilot.system import kokoro_tts


@pytest.fixture(autouse=True)
def _clear_pipeline_cache():
    kokoro_tts._pipeline_cache.clear()
    yield
    kokoro_tts._pipeline_cache.clear()


def _fake_kokoro_module(pipeline: MagicMock) -> ModuleType:
    module = ModuleType("kokoro")
    module.KPipeline = MagicMock(return_value=pipeline)
    return module


def _fake_sounddevice(play=None, wait=None, stop=None) -> ModuleType:
    module = ModuleType("sounddevice")
    module.play = play or MagicMock()
    module.wait = wait or MagicMock()
    module.stop = stop or MagicMock()
    return module


def test_pipeline_is_cached():
    pipeline = MagicMock()
    fake_module = _fake_kokoro_module(pipeline)

    with patch.dict(sys.modules, {"kokoro": fake_module}):
        first = kokoro_tts._get_pipeline()
        second = kokoro_tts._get_pipeline()

    fake_module.KPipeline.assert_called_once_with(
        lang_code="a",
        repo_id="hexgrad/Kokoro-82M",
    )
    assert first is second is pipeline


@pytest.mark.asyncio
async def test_synthesize_concatenates_generated_audio():
    pipeline = MagicMock(
        return_value=[
            ("Hello", "həloʊ", np.array([0.1, 0.2], dtype=np.float32)),
            ("there", "ðɛɹ", np.array([0.3], dtype=np.float32)),
        ]
    )
    fake_module = _fake_kokoro_module(pipeline)

    with patch.dict(sys.modules, {"kokoro": fake_module}):
        audio, sample_rate = await kokoro_tts.synthesize("Hello\nthere", "af_heart")

    assert sample_rate == 24000
    assert np.allclose(audio, [0.1, 0.2, 0.3])
    pipeline.assert_called_once_with(
        "Hello\nthere",
        voice="af_heart",
        speed=1.0,
        split_pattern=r"\n+",
    )


@pytest.mark.asyncio
async def test_warmup_loads_pipeline_without_synthesizing():
    pipeline = MagicMock()
    fake_module = _fake_kokoro_module(pipeline)

    with patch.dict(sys.modules, {"kokoro": fake_module}):
        await kokoro_tts.warmup("af_heart")

    fake_module.KPipeline.assert_called_once_with(
        lang_code="a",
        repo_id="hexgrad/Kokoro-82M",
    )
    pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_empty_generation_is_an_error():
    pipeline = MagicMock(return_value=[])
    fake_module = _fake_kokoro_module(pipeline)

    with (
        patch.dict(sys.modules, {"kokoro": fake_module}),
        pytest.raises(RuntimeError, match="produced no audio"),
    ):
        await kokoro_tts.synthesize("Hello", "af_heart")


@pytest.mark.asyncio
async def test_play_goes_through_the_shared_output_line(monkeypatch):
    """Never a stream of its own: the daemon's output line (voice.play_on_output_line), which fades on a cut."""
    from pilot.system import voice

    audio = np.array([0.1], dtype=np.float32)
    line = AsyncMock()
    monkeypatch.setattr(voice, "play_on_output_line", line)
    await kokoro_tts.play(audio, 24000)
    line.assert_awaited_once_with(audio, 24000)


@pytest.mark.asyncio
async def test_synthesize_to_file_uses_soundfile(tmp_path):
    pipeline = MagicMock(return_value=[("Hi", "haɪ", np.array([0.1], dtype=np.float32))])
    fake_kokoro = _fake_kokoro_module(pipeline)
    fake_soundfile = ModuleType("soundfile")
    fake_soundfile.write = MagicMock()
    output = str(tmp_path / "voice.wav")

    with patch.dict(sys.modules, {"kokoro": fake_kokoro, "soundfile": fake_soundfile}):
        await kokoro_tts.synthesize_to_file("Hi", "af_heart", output)

    fake_soundfile.write.assert_called_once()
    args = fake_soundfile.write.call_args.args
    assert args[0] == output
    assert args[2] == 24000

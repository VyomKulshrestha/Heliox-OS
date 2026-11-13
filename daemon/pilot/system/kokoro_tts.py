"""Natural local text-to-speech using the open-weight Kokoro model.

Kokoro is optional at import time. The voice dispatcher falls back to the
platform's native TTS if its package, weights, or audio device are unavailable.
Model inference and playback stay off the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any

SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"
_WEIGHT_FILES = ("config.json", "kokoro-v1_0.pth")
_pipeline_cache: dict[str, Any] = {}
_pipeline_lock = threading.Lock()
_device = "auto"
_offline_by_us = False


def configure(*, device: str = "auto") -> None:
    """Choose where the model runs: "auto" (CUDA when present), "cpu" or "cuda"."""
    global _device
    _device = device if device in {"auto", "cpu", "cuda"} else "auto"
    from pilot.system.local_tts_worker import cuda_hidden

    if _device != "cpu" and cuda_hidden():
        _device = "cpu"  # CUDA_VISIBLE_DEVICES hides every card


def use_cached_weights_offline(voice: str) -> bool:
    """Skip Hugging Face's network checks when the weights and voice are already on disk.

    Each hf_hub_download call otherwise asks the Hub whether a newer file exists; on this machine that added about
    1.4 s to every cold start. Must run before the model loads. Returns whether offline mode is now on.
    """
    if os.environ.get("HF_HUB_OFFLINE"):
        return True
    try:
        from huggingface_hub import constants, try_to_load_from_cache
    except ImportError:
        return False
    files = [*_WEIGHT_FILES, f"voices/{voice}.pt"] if voice else list(_WEIGHT_FILES)
    if not all(isinstance(try_to_load_from_cache(REPO_ID, name), str) for name in files):
        return False
    global _offline_by_us
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants.HF_HUB_OFFLINE = True
    _offline_by_us = True
    return True


def _allow_fetching_missing_voice(voice: str) -> None:
    """A voice picked after the worker went offline is fetched once, as it was before offline mode."""
    global _offline_by_us
    if not _offline_by_us:
        return
    from huggingface_hub import constants, try_to_load_from_cache

    if isinstance(try_to_load_from_cache(REPO_ID, f"voices/{voice}.pt"), str):
        return
    os.environ.pop("HF_HUB_OFFLINE", None)
    constants.HF_HUB_OFFLINE = False
    _offline_by_us = False


def _get_pipeline(language: str = "a") -> Any:
    """The pipeline that reads ``language`` (a voice's first letter). Every language's pipeline uses the one model the
    first pipeline loaded: a pipeline per language is only its reader, so a person who hears Heliox in Spanish after
    English does not hold a second copy of the model (about 330 MB each)."""
    with _pipeline_lock:
        if language not in _pipeline_cache:
            from kokoro import KPipeline

            shared = next((p.model for p in _pipeline_cache.values() if getattr(p, "model", None) is not None), None)
            if shared is not None:
                _pipeline_cache[language] = KPipeline(lang_code=language, repo_id=REPO_ID, model=shared)
            elif _device == "auto":
                try:
                    _pipeline_cache[language] = KPipeline(lang_code=language, repo_id=REPO_ID)
                except RuntimeError as exc:
                    # A broken GPU driver should cost speed, not the voice.
                    print(f"Kokoro could not use the GPU ({exc}); loading on the CPU", file=sys.stderr)
                    _pipeline_cache[language] = KPipeline(lang_code=language, repo_id=REPO_ID, device="cpu")
            else:
                _pipeline_cache[language] = KPipeline(lang_code=language, repo_id=REPO_ID, device=_device)
        return _pipeline_cache[language]


def _to_numpy(audio: Any) -> Any:
    value = audio.detach().cpu() if hasattr(audio, "detach") else audio
    return value.numpy() if hasattr(value, "numpy") else value


def _generate(text: str, voice: str) -> tuple[Any, int]:
    import numpy as np

    from pilot.system.voice_by_language import kokoro_lang_code

    pipeline = _get_pipeline(kokoro_lang_code(voice))  # the reader of the voice's own language ("ef_dora": Spanish)
    _allow_fetching_missing_voice(voice)
    chunks = [
        _to_numpy(audio)
        for _graphemes, _phonemes, audio in pipeline(
            text,
            voice=voice,
            speed=1.0,
            split_pattern=r"\n+",
        )
    ]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    return np.concatenate(chunks), SAMPLE_RATE


def prepare(voice: str) -> None:
    """Load weights and the voice, then make one throwaway sentence so the first real one is warm (worker side)."""
    from pilot.system.voice_by_language import kokoro_lang_code

    use_cached_weights_offline(voice)
    _get_pipeline(kokoro_lang_code(voice))
    if voice:
        _generate("Okay.", voice)


async def warmup(_voice: str) -> None:
    """Load model weights before the first spoken intervention."""
    await asyncio.to_thread(_get_pipeline)


async def synthesize(text: str, voice: str) -> tuple[Any, int]:
    return await asyncio.to_thread(_generate, text, voice)


def _play_blocking(audio: Any, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(audio, sample_rate)
    sd.wait()


async def play(audio: Any, sample_rate: int) -> None:
    try:
        await asyncio.to_thread(_play_blocking, audio, sample_rate)
    except asyncio.CancelledError:
        import sounddevice as sd

        sd.stop()
        raise


async def synthesize_and_play(text: str, voice: str) -> None:
    audio, sample_rate = await synthesize(text, voice)
    await play(audio, sample_rate)


def synthesize_to_file_here(text: str, voice: str, output_file: str) -> None:
    """Make ``text`` and write it as a WAV, on the calling thread (the voice worker's own: no event loop, no thread
    hop)."""
    import soundfile

    audio, sample_rate = _generate(text, voice)
    soundfile.write(output_file, audio, sample_rate)


async def synthesize_to_file(text: str, voice: str, output_file: str) -> None:
    await asyncio.to_thread(synthesize_to_file_here, text, voice, output_file)

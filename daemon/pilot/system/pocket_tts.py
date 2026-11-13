"""Kyutai Pocket TTS -- CPU-only, cross-platform local text-to-speech.

Optional: only exercised when pilot.config.VoiceConfig.tts_engine ==
"pocket_tts" AND the `pocket-tts` package (pip install pocket-tts, or the
`pilot-daemon[voice]` extra) is actually installed. pilot.system.voice's
_speak_impl falls back to the existing OS-native TTS (SAPI/say/espeak) on
any ImportError or other failure raised from here -- this module is never
a hard dependency, mirroring how openai-whisper is treated elsewhere in
this file's sibling voice.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("pilot.system.pocket_tts")

# Module-level caches -- mirrors voice._get_whisper_model's plain dict-cache
# pattern rather than CognitiveEngine's singleton/loading-state-machine, since
# Pocket TTS has no async loading phases or cognitive-state API to
# coordinate; a synchronous load-once-and-cache is all that's needed.
_model_cache: dict[str, Any] = {}
_voice_state_cache: dict[str, Any] = {}


def _get_model() -> Any:
    """Loads (and caches) the default Pocket TTS model. Only the default
    English model is wired up in this pass -- see module docstring."""
    key = "default"
    if key not in _model_cache:
        from pocket_tts import TTSModel

        _model_cache[key] = TTSModel.load_model()
    return _model_cache[key]


def _get_voice_state(model: Any, voice: str) -> Any:
    """Loads (and caches) the reusable voice-prompt state for a given
    built-in voice preset name."""
    if voice not in _voice_state_cache:
        _voice_state_cache[voice] = model.get_state_for_audio_prompt(voice)
    return _voice_state_cache[voice]


def _generate(text: str, voice: str) -> tuple[Any, int]:
    """Blocking model load + inference -- always called via asyncio.to_thread
    (see synthesize()), the same reasoning as _transcribe_whisper's own
    asyncio.to_thread(_do) for genuinely CPU-bound work."""
    model = _get_model()
    state = _get_voice_state(model, voice)
    audio = model.generate_audio(state, text)
    return audio.numpy(), model.sample_rate


def _warmup_sync(voice: str) -> None:
    """Load model weights and the selected reusable voice state without
    synthesizing audio, so the first real intervention does not pay the full
    cold-start cost."""
    model = _get_model()
    _get_voice_state(model, voice)


async def warmup(voice: str) -> None:
    """Preload the selected Pocket TTS model and voice off the event loop."""
    await asyncio.to_thread(_warmup_sync, voice)


async def synthesize(text: str, voice: str) -> tuple[Any, int]:
    """Generates audio for `text` spoken in `voice`, returning (PCM numpy
    array, sample_rate). Raises ImportError if pocket_tts isn't installed,
    or whatever pocket_tts/huggingface_hub itself raises on a model-load or
    weight-download failure -- callers decide the fallback."""
    return await asyncio.to_thread(_generate, text, voice)


async def play(audio: Any, sample_rate: int) -> None:
    """Play already-synthesised audio through the daemon's shared output line (voice.play_on_output_line), never a
    stream of its own: on a Mac whose microphone is on the output's device a second output stream in this process
    stops the microphone. Cancelling (barge-in) cuts it at once."""
    from pilot.system.voice import play_on_output_line

    await play_on_output_line(audio, sample_rate)


async def synthesize_and_play(text: str, voice: str) -> None:
    """Generates and plays `text` aloud in one call -- the composed
    operation voice.py's _speak_impl pocket_tts branch actually calls."""
    audio, sample_rate = await synthesize(text, voice)
    await play(audio, sample_rate)


async def synthesize_to_file(text: str, voice: str, output_file: str) -> None:
    """Generates `text` and writes it to output_file as a WAV file, for
    parity with speak()'s existing output_file parameter on the OS-native
    TTS paths."""
    await asyncio.to_thread(synthesize_to_file_here, text, voice, output_file)


def synthesize_to_file_here(text: str, voice: str, output_file: str) -> None:
    """The same on the calling thread (the voice worker's own: no event loop, no thread hop)."""
    from scipy.io import wavfile

    audio, sample_rate = _generate(text, voice)
    wavfile.write(output_file, sample_rate, audio)

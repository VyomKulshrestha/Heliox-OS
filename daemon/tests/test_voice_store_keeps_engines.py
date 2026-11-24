"""Switching the voice engine keeps the other voices that spoke recently in the store of made speech, bounded, and never
plays it for the chosen voice.

On 24 November a measurement switched from Kokoro to Pocket TTS and the store kept only Pocket's entries: switching
back, no hum and no fixed sentence was ready until they had been made again. Files in a temporary folder only.
"""

from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import LocalTTSWorkerManager, SpeechStore

KOKORO = SpeechStore.signature("kokoro_tts", "af_heart")
BELLA = SpeechStore.signature("kokoro_tts", "af_bella")
POCKET = SpeechStore.signature("pocket_tts", "alba")


def _write_wav(path, samples: np.ndarray, rate: int) -> None:
    """A 16-bit mono WAV with the standard library (soundfile, an optional audio package, is not needed here)."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def _fill(store: SpeechStore, tmp_path, signature: str, texts: list[str], seconds: float = 0.3) -> None:
    for number, text in enumerate(texts):
        made = tmp_path / f"{abs(hash((signature, text)))}-{number}.wav"
        _write_wav(made, np.full(int(seconds * 24000), 0.1, dtype=np.float32), 24000)
        assert store.add_fixed(text, signature, made)


def test_switching_engine_and_back_finds_the_hums_ready_and_never_the_other_engine_s_audio(tmp_path):
    store = SpeechStore(tmp_path / "voice_heads")
    store.keep_voices(KOKORO)
    _fill(store, tmp_path, KOKORO, ["Mm.", "Hmm.", "Okay."])
    store.keep_voices(POCKET)  # the person chose Pocket TTS
    assert all(store.has(text, KOKORO) for text in ("Mm.", "Hmm.", "Okay."))  # kept for later
    assert not store.has("Hmm.", POCKET) and store.lookup("Hmm.", POCKET) is None  # never played for Pocket
    _fill(store, tmp_path, POCKET, ["Mm.", "Hmm."])
    store.keep_voices(KOKORO)  # and back
    assert all(store.has(text, KOKORO) for text in ("Mm.", "Hmm.", "Okay."))  # ready at once, nothing made again
    assert store.has("Hmm.", POCKET)
    assert store.lookup("Hmm.", KOKORO) != store.lookup("Hmm.", POCKET)  # each voice its own audio
    reopened = SpeechStore(store.folder)  # after a restart the same
    reopened.keep_voices(KOKORO)
    assert reopened.has("Okay.", KOKORO) and reopened.has("Mm.", POCKET)


def test_another_voice_of_the_same_engine_is_kept_too(tmp_path):
    store = SpeechStore(tmp_path / "voice_heads")
    store.keep_voices(KOKORO)
    _fill(store, tmp_path, KOKORO, ["Mm."])
    store.keep_voices(POCKET)
    _fill(store, tmp_path, POCKET, ["Mm."])
    store.keep_voices(BELLA)  # Kokoro again, with another voice: af_heart's audio stays for going back
    assert store.has("Mm.", KOKORO) and store.has("Mm.", POCKET) and not store.has("Mm.", BELLA)
    store.keep_voices("not a signature")  # an unreadable chosen signature drops none of the voices kept
    assert store.has("Mm.", POCKET) and store.has("Mm.", KOKORO)


def test_the_other_engine_keeps_its_hums_first_under_the_cap(tmp_path, monkeypatch):
    store = SpeechStore(tmp_path / "voice_heads")
    store.keep_voices(KOKORO)
    _fill(store, tmp_path, KOKORO, ["One.", "Two.", "Three.", "Mm.", "Hmm."])
    one = (store.folder / f"{SpeechStore.key('One.', KOKORO)}.wav").stat().st_size
    monkeypatch.setattr(local_tts_worker, "OTHER_ENGINE_BYTES", 3 * one)
    store.keep_voices(POCKET)
    assert store.has("Mm.", KOKORO) and store.has("Hmm.", KOKORO)
    assert sum(store.has(text, KOKORO) for text in ("One.", "Two.", "Three.")) == 1
    assert store.fixed_bytes(KOKORO) <= 3 * one


def test_the_other_engine_s_audio_does_not_use_up_the_chosen_voice_s_room(tmp_path, monkeypatch):
    store = SpeechStore(tmp_path / "voice_heads")
    store.keep_voices(KOKORO)
    _fill(store, tmp_path, KOKORO, ["One.", "Two."])
    monkeypatch.setattr(local_tts_worker, "FIXED_SPEECH_BYTES", store.fixed_bytes(KOKORO))
    store.keep_voices(POCKET)
    _fill(store, tmp_path, POCKET, ["One.", "Two."])  # Pocket's own room, whatever Kokoro holds
    assert store.fixed_bytes(POCKET) == store.fixed_bytes(KOKORO)


def test_configuring_the_voice_keeps_the_other_engine_and_says_ready_only_for_the_chosen_one(tmp_path, monkeypatch):
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: None)
    manager.environ = {}
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    monkeypatch.setattr(manager, "speaks_locally_now", lambda engine: True)
    kokoro = SimpleNamespace(tts_engine="kokoro_tts", tts_voice="af_heart")
    pocket = SimpleNamespace(tts_engine="pocket_tts", tts_voice="alba")
    local_tts_worker.configure_local_voice(kokoro)
    _fill(manager.head_cache, tmp_path, KOKORO, ["Mm.", "Hmm."])
    local_tts_worker.configure_local_voice(pocket)
    assert not local_tts_worker.made_ahead(pocket, "Hmm.")  # Kokoro's hum is never Pocket's
    local_tts_worker.configure_local_voice(kokoro)
    assert local_tts_worker.made_ahead(kokoro, "Hmm.") and local_tts_worker.made_ahead(kokoro, "Mm.")


@pytest.mark.asyncio
async def test_a_reply_in_the_chosen_voice_never_plays_the_other_engine_s_kept_audio(tmp_path):
    """Speaking with Pocket while Kokoro's "Okay." is kept: the piece is made in Pocket's voice, not played from it."""
    store = SpeechStore(tmp_path / "voice_heads")
    store.keep_voices(KOKORO)
    _fill(store, tmp_path, KOKORO, ["Okay."])
    store.keep_voices(POCKET)
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: None)
    manager.head_cache = store
    assert not manager._all_kept("pocket_tts", "alba", ["Okay."])
    assert manager._all_kept("kokoro_tts", "af_heart", ["Okay."])

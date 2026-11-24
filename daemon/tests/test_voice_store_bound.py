"""The speech store's size is bounded whatever voices pass through it, the oldest voice goes first, and a store
written by the version before (one voice per engine in voices.json) loses nothing. The caps are set small here so the
files stay a few kilobytes; the rule is the same at 64 MB and 32 MB."""

from __future__ import annotations

import json
import wave

import numpy as np

from pilot.system import local_tts_worker as ltw
from pilot.system.local_tts_worker import SpeechStore

PIECES = ["Mm.", "Hmm.", "Okay.", "One moment.", "Got it."]


def _fill(store: SpeechStore, tmp_path, signature: str, texts: list[str]) -> int:
    """Each text as a 0.3 s piece (about 14 KB) offered to the store; returns how many it took (its room may end)."""
    taken = 0
    for number, text in enumerate(texts):
        made = tmp_path / f"{abs(hash((signature, text)))}-{number}.wav"
        with wave.open(str(made), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(24000)
            out.writeframes((np.full(7200, 0.1) * 32767).astype("<i2").tobytes())
        taken += bool(store.add_fixed(text, signature, made))
    return taken


def _bytes(store: SpeechStore) -> int:
    return sum(path.stat().st_size for path in store.folder.glob("*.wav"))


def _bytes_of(store: SpeechStore, signature: str) -> int:
    return sum(
        (store.folder / f"{key}.wav").stat().st_size
        for key, entry in store._load().items()
        if entry.get("signature") == signature and (store.folder / f"{key}.wav").is_file()
    )


def test_the_store_never_holds_more_than_the_chosen_voice_and_six_capped_others(tmp_path, monkeypatch):
    monkeypatch.setattr(ltw, "FIXED_SPEECH_BYTES", 50_000)  # the chosen voice's room (64 MB in use)
    monkeypatch.setattr(ltw, "OTHER_ENGINE_BYTES", 32_000)  # each other voice's room (32 MB in use)
    store = SpeechStore(tmp_path / "voice_heads")
    voices = [SpeechStore.signature("kokoro_tts", f"voice_{n}") for n in range(9)]
    for signature in voices:  # nine voices chosen in turn, each making as much as its room allows
        store.keep_voices(signature)
        assert _fill(store, tmp_path, signature, PIECES) == 3  # its room holds three of the five
        bound = ltw.FIXED_SPEECH_BYTES + ltw.KEEP_OTHER_VOICES * ltw.OTHER_ENGINE_BYTES
        assert _bytes(store) <= bound
    store.keep_voices(voices[-1])
    kept = [store.has("Mm.", signature) for signature in voices]
    assert kept == [False, False] + [True] * 7  # the chosen one and the six before it; the two oldest went first
    for signature in voices[2:-1]:
        assert _bytes_of(store, signature) <= ltw.OTHER_ENGINE_BYTES
        assert store.has("Mm.", signature) and store.has("Hmm.", signature)  # the hums are kept first
    recent = json.loads((store.folder / "voices.json").read_text(encoding="utf-8"))["recent"]
    assert recent == list(reversed(voices[2:]))  # newest first


def test_a_store_from_before_keeps_both_its_voices(tmp_path):
    folder = tmp_path / "voice_heads"
    kokoro, pocket = SpeechStore.signature("kokoro_tts", "af_heart"), SpeechStore.signature("pocket_tts", "alba")
    old = SpeechStore(folder)
    for signature in (kokoro, pocket):
        old.keep_voices(signature)
        _fill(old, tmp_path, signature, ["Mm.", "Hmm.", "Okay."])
    # As the version before wrote it: one voice per engine.
    (folder / "voices.json").write_text(json.dumps({"kokoro_tts": kokoro, "pocket_tts": pocket}), encoding="utf-8")
    reopened = SpeechStore(folder)
    reopened.keep_voices(SpeechStore.signature("kokoro_tts", "am_michael"))  # a third voice is chosen
    for signature in (kokoro, pocket):
        assert all(reopened.has(text, signature) for text in ("Mm.", "Hmm.", "Okay."))

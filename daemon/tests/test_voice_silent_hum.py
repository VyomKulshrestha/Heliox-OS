"""A hum that comes out silent is no hum: it is not kept, not made again, and the face's voice hums with the hum it
does have, or not at all. Pocket's michael made "Hmm." at a peak of 0.008 (about -64 dBFS) on 24 November.

Made-up managers and stores only: no worker process, no model, no audio package; orders are counted."""

from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.conversation.acknowledge import ACKNOWLEDGEMENTS, next_acknowledgement
from pilot.system import local_tts_worker as ltw
from pilot.system.local_tts_worker import LocalTTSWorkerManager, Piece, SpeechStore, audible

MICHAEL = SpeechStore.signature("pocket_tts", "michael")


def _wav(path, peak: float) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes((np.full(2400, peak) * 32767).astype("<i2").tobytes())


def test_a_hum_is_audible_only_above_the_floor(tmp_path):
    assert audible(Piece("Hmm.", audio=np.full(2400, 0.008, dtype=np.float32), rate=24000)) is False  # michael's
    assert audible(Piece("Mm.", audio=np.full(2400, 0.3, dtype=np.float32), rate=24000)) is True
    assert audible(Piece("Mm.", audio=np.zeros(0, dtype=np.float32), rate=24000)) is False
    quiet, loud = tmp_path / "quiet.wav", tmp_path / "loud.wav"
    _wav(quiet, 0.008)
    _wav(loud, 0.4)
    assert audible(quiet) is False and audible(Piece("Hmm.", path=loud)) is True
    assert audible(tmp_path / "missing.wav") is False


def test_the_store_remembers_a_silent_hum_per_voice(tmp_path):
    store = SpeechStore(tmp_path / "voice_heads")
    store.mark_silent("Hmm.", MICHAEL)
    assert store.is_silent("Hmm.", MICHAEL) and not store.is_silent("Mm.", MICHAEL)
    assert not store.is_silent("Hmm.", SpeechStore.signature("pocket_tts", "alba"))  # alba's own "Hmm." is fine
    assert SpeechStore(store.folder).is_silent("Hmm.", MICHAEL)  # after a restart too
    assert not store.has("Hmm.", MICHAEL)  # never kept: nothing silent ever plays


@pytest.mark.asyncio
async def test_a_silent_hum_is_not_kept_and_not_made_again():
    manager = LocalTTSWorkerManager(spawn=lambda *_: None, clock=lambda: 100.0)
    manager.process, manager.engine, manager.loaded = SimpleNamespace(poll=lambda: None), "pocket_tts", True
    kept: set[str] = set()
    silent: set[str] = set()
    manager.head_cache = SimpleNamespace(
        has=lambda text, sig: text in kept,
        add_fixed=lambda text, sig, piece: kept.add(text) or True,
        is_silent=lambda text, sig: text in silent,
        mark_silent=lambda text, sig: silent.add(text),
    )
    made: list[str] = []

    async def make(engine, text, voice, path, folder, **_given):
        made.append(text)
        peak = 0.008 if text == "Hmm." else 0.4
        return Piece(text, audio=np.full(2400, peak, dtype=np.float32), rate=24000)

    async def never_warm():
        return False

    async def sleep(_seconds):
        return None

    manager._make, manager.too_warm_now, manager._sleep = make, never_warm, sleep
    count = await manager.make_ahead("pocket_tts", "michael", list(ACKNOWLEDGEMENTS), idle_seconds=0.0,
                                     quiet=lambda: True, battery=lambda: False, rest_factor=0.0)  # fmt: skip
    assert count == 1 and kept == {"Mm."} and silent == {"Hmm."}
    await manager.make_ahead("pocket_tts", "michael", list(ACKNOWLEDGEMENTS), idle_seconds=0.0,
                             quiet=lambda: True, battery=lambda: False, rest_factor=0.0)  # fmt: skip
    assert made == ["Mm.", "Hmm."]  # the second run made nothing: "Mm." is kept and "Hmm." is known silent


def test_the_face_hums_with_the_hum_it_has_or_not_at_all():
    only_mm = {"Mm."}.__contains__
    assert next_acknowledgement("", only_mm) == "Mm."
    assert (
        next_acknowledgement("Mm.", only_mm) == "Mm."
    )  # its only hum, said again rather than half a second of nothing
    assert next_acknowledgement("Mm.", set().__contains__) == ""  # none ready: no hum
    both = set(ACKNOWLEDGEMENTS).__contains__
    assert next_acknowledgement("Mm.", both) == "Hmm." and next_acknowledgement("Hmm.", both) == "Mm."
    assert next_acknowledgement("Mm.") == "Hmm."  # without the check, in turn as before


def test_a_standby_whose_hum_is_silent_still_takes_over():
    standby = LocalTTSWorkerManager(spawn=lambda *_: None)
    standby.head_cache = SimpleNamespace(
        has=lambda text, sig: text == "Mm.", is_silent=lambda text, sig: text == "Hmm."
    )
    current = ltw.Handover("pocket_tts", "michael", "pocket_tts", "alba", standby)
    assert ltw._hums_done(current) is True

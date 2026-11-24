"""How a hum ends when the answer comes: made (and kept) with a short quiet tail, and a cut piece always fades.

In the 24 November live run the answer cut 5 of 18 hums 0.70 to 0.95 s in, all in the hum's quiet tail (Kokoro
af_heart's "Hmm." was 0.95 s long, its sound over at 0.47 s). No audio device: the output line's callback is called
by the test, block by block.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from pilot.conversation.acknowledge import ACKNOWLEDGEMENTS
from pilot.system import local_tts_worker, voice
from pilot.system.local_tts_worker import (
    ACK_TAIL_SECONDS,
    KEEP_CLAUSE_SECONDS,
    KEEP_LEAD_SECONDS,
    LocalTTSWorkerManager,
    SpeechStore,
    tail_for,
)

RATE = 24000


def test_a_hum_keeps_a_short_tail_and_a_sentence_keeps_its_pause():
    for hum in ACKNOWLEDGEMENTS:
        assert tail_for(hum) == ACK_TAIL_SECONDS
    assert tail_for("That is the capital.") is None  # a sentence's own pause before the next sentence
    assert tail_for("When you are ready") == KEEP_CLAUSE_SECONDS


@pytest.mark.asyncio
async def test_a_hum_made_ahead_asks_the_worker_for_the_short_tail(tmp_path):
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: None)
    asked: dict[str, float | None] = {}

    async def synthesize(engine, text, voice_name, target, folder, **options):
        asked[text] = options["keep_tail"]
        return local_tts_worker.Piece(text, audio=np.zeros(10, dtype=np.float32), rate=RATE)

    manager.synthesize = synthesize  # type: ignore[method-assign]
    await manager._make("kokoro_tts", "Hmm.", "af_heart", tmp_path / "a.wav", tmp_path, kind="ahead")
    await manager._make("kokoro_tts", "Okay, done.", "af_heart", tmp_path / "b.wav", tmp_path, kind="ahead")
    assert asked == {"Hmm.": ACK_TAIL_SECONDS, "Okay, done.": None}


def _hum(seconds_of_sound: float, seconds_long: float) -> np.ndarray:
    audio = np.zeros(int(seconds_long * RATE), dtype=np.float32)
    start, stop = int(0.03 * RATE), int(seconds_of_sound * RATE)
    audio[start:stop] = 0.2 * np.sin(np.arange(stop - start) * 2 * np.pi * 150 / RATE)
    return audio


def test_hums_kept_with_their_whole_tail_are_cut_once_and_other_pieces_are_left_alone(tmp_path):
    # The cut itself (trim_silence) reads and writes the WAV with soundfile, an optional audio package; without it
    # the store leaves kept audio as it was. Skipped where it is not installed (CI's test install).
    soundfile = pytest.importorskip("soundfile")
    store = SpeechStore(tmp_path / "voice_heads")
    sig = SpeechStore.signature("kokoro_tts", "af_heart")
    hum, sentence = tmp_path / "hum.wav", tmp_path / "sentence.wav"
    soundfile.write(str(hum), _hum(0.47, 0.95), RATE)  # as Kokoro's "Hmm." was kept before
    soundfile.write(str(sentence), _hum(0.6, 1.2), RATE)
    assert store.add_fixed("Okay.", sig, sentence)
    assert store.add_fixed("Hmm.", sig, hum)
    key = SpeechStore.key("Hmm.", sig)
    store._load()[key].pop("tail")  # an entry from before hums were made with the short tail
    timing = {"rate": RATE, "samples": int(0.95 * RATE), "phonemes": [["h", 0.03, 0.2], ["m", 0.2, 0.47]], "words": []}
    (store.folder / f"{key}.wav.timing.json").write_text(json.dumps(timing), encoding="utf-8")

    assert store.trim_short_tails() == 1
    kept, _rate = soundfile.read(str(store.lookup("Hmm.", sig)), dtype="float32")
    assert len(kept) / RATE == pytest.approx(0.47 - 0.03 + KEEP_LEAD_SECONDS + ACK_TAIL_SECONDS, abs=0.01)
    moved = json.loads((store.folder / f"{key}.wav.timing.json").read_text(encoding="utf-8"))
    assert moved["samples"] == len(kept)  # the face's mouth timeline follows the cut
    other, _rate = soundfile.read(str(store.lookup("Okay.", sig)), dtype="float32")
    assert len(other) == int(1.2 * RATE)  # a sentence keeps its pause
    assert store.trim_short_tails() == 0  # once
    assert SpeechStore(store.folder).trim_short_tails() == 0  # and remembered in the index


def test_a_cut_fades_over_the_next_block_and_never_stops_mid_sample():
    playback = voice._Playback(np.full(RATE, 0.5, dtype=np.float32), RATE)
    block = np.zeros((480, 1), dtype=np.float32)
    playback.callback(block, 480, None, None)
    assert np.all(block == 0.5)
    playback.cut()
    playback.callback(block, 480, None, None)
    fade = int(RATE * voice.FADE_SECONDS)
    assert block[0, 0] == pytest.approx(0.5, abs=0.01) and np.all(np.diff(block[:fade, 0]) <= 0)  # a ramp down
    assert np.all(block[fade:] == 0) and playback.faded.is_set()


def _line_with(previous) -> voice._OutputLine:
    line = voice._OutputLine()
    line.running.set()  # the device is asking for audio
    line.current = previous
    return line


def test_the_next_piece_waits_for_a_cut_piece_s_fade_before_it_takes_the_line(monkeypatch):
    monkeypatch.setattr(voice, "CUT_FADE_WAIT_SECONDS", 10.0)  # no real waiting decides this test
    hum = voice._Playback(np.full(RATE, 0.5, dtype=np.float32), RATE)
    line = _line_with(hum)
    hum.cut()  # the answer cut the hum; the device has not asked for the next block yet
    answer = voice._Playback(np.full(RATE, 0.3, dtype=np.float32), RATE)
    attaching = threading.Thread(target=line.attach, args=(answer,))
    attaching.start()
    attaching.join(0.05)
    assert attaching.is_alive() and line.current is hum  # the hum still has the line: its fade is still to play
    block = np.zeros((480, 1), dtype=np.float32)
    line.callback(block, 480, None, None)  # the device asks: the fade plays
    attaching.join(5)
    assert not attaching.is_alive() and line.current is answer
    assert block[0, 0] > 0 and np.all(block[int(RATE * voice.FADE_SECONDS) :] == 0)


def test_the_next_piece_never_waits_on_a_line_whose_device_is_not_asking(monkeypatch):
    monkeypatch.setattr(voice, "CUT_FADE_WAIT_SECONDS", 10.0)
    hum = voice._Playback(np.full(RATE, 0.5, dtype=np.float32), RATE)
    line = _line_with(hum)
    line.running.clear()  # nothing will play the fade: the next piece takes the line at once
    hum.cut()
    answer = voice._Playback(np.zeros(10, dtype=np.float32), RATE)
    line.attach(answer)
    assert line.current is answer
    playing = voice._Playback(np.zeros(10, dtype=np.float32), RATE)  # not cut: replaced at once as before
    line.running.set()
    line.attach(playing)
    assert line.current is playing


def test_the_hum_texts_are_the_acknowledgements():
    assert set(ACKNOWLEDGEMENTS) == {"Mm.", "Hmm."}
    assert all(local_tts_worker.keeps_short_tail(hum) for hum in ACKNOWLEDGEMENTS)
    assert not local_tts_worker.keeps_short_tail("Mm, sure.")


def test_trim_is_skipped_without_a_file(tmp_path: Path):
    store = SpeechStore(tmp_path / "voice_heads")
    sig = SpeechStore.signature("kokoro_tts", "af_heart")
    store._load()[SpeechStore.key("Mm.", sig)] = {"text": "Mm.", "signature": sig, "kept": True, "fixed": True}
    assert store.trim_short_tails() == 0

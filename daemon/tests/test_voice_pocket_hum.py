"""A hum is short and ends cleanly in any engine's voice, and an answer that cuts it fades it out.

Pocket TTS (alba) made "Hmm." loud for its whole 880 ms with a burst in its last 30 ms; in a live run on 24 November
the answer cut it 623 to 854 ms in on 10 of 21 turns, up to about 260 ms of it still sounding. Made-up audio only: no
model, no audio device.
"""

from __future__ import annotations

import numpy as np
import pytest

from pilot.system import local_tts_worker, voice
from pilot.system.local_tts_worker import (
    ACK_TAIL_SECONDS,
    HUM_FADE_SECONDS,
    HUM_SOUND_SECONDS,
    SILENCE_LEVEL,
    LocalTTSWorkerManager,
    Piece,
    shape_hum,
)

RATE = 24000


def _sound(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    return (amplitude * np.sin(np.arange(int(seconds * RATE)) * 2 * np.pi * 140 / RATE + 0.5)).astype(np.float32)


def _sounding(audio: np.ndarray) -> float:
    loud = np.flatnonzero(np.abs(audio) > SILENCE_LEVEL)
    return (loud[-1] + 1 - loud[0]) / RATE if loud.size else 0.0


def test_a_long_loud_hum_with_a_burst_at_its_end_is_cut_short_and_faded():
    lead = np.zeros(int(0.03 * RATE), np.float32)
    hum = np.concatenate([lead, _sound(0.82), _sound(0.03, 0.9)])  # as Pocket's "Hmm.": loud, then a burst
    shaped = shape_hum(hum, RATE)
    assert _sounding(shaped) <= HUM_SOUND_SECONDS + 1 / RATE
    assert np.abs(shaped).max() <= 0.3 + 1e-6  # the burst is gone
    assert len(shaped) / RATE == pytest.approx(0.03 + HUM_SOUND_SECONDS + ACK_TAIL_SECONDS, abs=0.002)
    end = int((0.03 + HUM_SOUND_SECONDS) * RATE)
    fade = int(HUM_FADE_SECONDS * RATE)
    window = np.abs(shaped[end - fade : end])
    assert window[-20:].max() < 0.01 and np.abs(shaped[end - fade - 200 : end - fade]).max() > 0.25  # a fade to nothing
    assert not shaped[end:].any()  # then silence


def test_a_short_hum_keeps_its_sound_and_only_its_end_is_smoothed():
    hum = np.concatenate([np.zeros(720, np.float32), _sound(0.27), np.zeros(int(0.5 * RATE), np.float32)])
    shaped = shape_hum(hum, RATE)
    assert _sounding(shaped) == pytest.approx(0.27, abs=0.01)  # its last few samples faded below the silence level
    assert len(shaped) / RATE == pytest.approx(0.03 + 0.27 + ACK_TAIL_SECONDS, abs=0.002)
    assert np.allclose(
        shaped[720 : 720 + int(0.1 * RATE)], hum[720 : 720 + int(0.1 * RATE)]
    )  # untouched before the fade


def test_silence_stays_as_it_is():
    quiet = np.zeros(1000, np.float32)
    assert np.array_equal(shape_hum(quiet, RATE), quiet)


@pytest.mark.asyncio
async def test_a_hum_made_now_is_shaped_and_a_sentence_is_not(tmp_path):
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: None)
    made = np.concatenate([np.zeros(720, np.float32), _sound(0.85)])

    async def synthesize(engine, text, voice_name, target, folder, **options):
        timing = {"rate": RATE, "samples": len(made), "phonemes": [["m", 0.03, 0.85]], "words": []}
        return Piece(text, audio=made.copy(), rate=RATE, timing=timing)

    manager.synthesize = synthesize  # type: ignore[method-assign]
    hum = await manager._make("pocket_tts", "Hmm.", "alba", tmp_path / "a.wav", tmp_path, kind="ahead")
    sentence = await manager._make("pocket_tts", "Okay, done.", "alba", tmp_path / "b.wav", tmp_path)
    assert _sounding(hum.audio) <= HUM_SOUND_SECONDS + 1 / RATE
    assert hum.timing["samples"] == len(hum.audio)  # the face's timeline follows the cut
    assert len(sentence.audio) == len(made)


def test_a_cut_hum_fades_out_over_the_longer_fade_and_a_cut_sentence_keeps_the_short_one():
    for seconds in (voice.HUM_CUT_FADE_SECONDS, voice.FADE_SECONDS):
        playback = voice._Playback(np.full(RATE, 0.5, dtype=np.float32), RATE)
        playback.fade_seconds = seconds
        playback.cut()
        block = np.zeros((2048, 1), dtype=np.float32)
        playback.callback(block, 2048, None, None)
        fade = int(RATE * seconds)
        assert np.all(np.diff(block[:fade, 0]) <= 0) and block[fade // 2, 0] > 0.1  # still sounding halfway down
        assert not block[fade:].any() and playback.faded.is_set()
    assert voice.HUM_CUT_FADE_SECONDS > voice.FADE_SECONDS
    assert voice.CUT_FADE_WAIT_SECONDS > voice.HUM_CUT_FADE_SECONDS


@pytest.mark.asyncio
async def test_the_player_gives_a_hum_the_longer_fade(monkeypatch):
    made: list[voice._Playback] = []

    class Quiet(voice._Playback):
        def __init__(self, audio, rate):
            super().__init__(audio, rate)
            made.append(self)

        def play(self) -> None:
            return None

    monkeypatch.setattr(voice, "_Playback", Quiet)
    await voice._play_local_tts_file(Piece("Mm.", audio=_sound(0.2), rate=RATE))
    await voice._play_local_tts_file(Piece("Here it is.", audio=_sound(0.2), rate=RATE))
    assert [p.fade_seconds for p in made] == [voice.HUM_CUT_FADE_SECONDS, voice.FADE_SECONDS]
    assert local_tts_worker.keeps_short_tail("Mm.")

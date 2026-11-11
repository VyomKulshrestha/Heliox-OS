"""Barge-in cuts Heliox's voice at once: a few milliseconds of fade, then silence on the shared output line. The
line stays open between sentences, because opening a stream took 1 to 3 s on the owner's machine and put that
pause before "Okay, stopped." after a cut-off.

The audio packages (sounddevice, soundfile) are optional and absent in CI, so the device and the file reader are
fakes here; the one test against a real file is skipped when soundfile is not installed."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import voice


def test_fade_block_ramps_down_then_is_silent():
    block = np.ones((10, 1), dtype=np.float32)
    faded = voice.fade_block(block, fade_left=4, fade_total=4)
    assert faded[0, 0] == pytest.approx(1.0) and faded[3, 0] == pytest.approx(0.25)
    assert not faded[4:].any()
    assert not voice.fade_block(block, 0, 4).any()
    mono = voice.fade_block(np.ones(4, dtype=np.float32), 2, 4)
    assert mono[0] == pytest.approx(0.5) and mono[-1] == 0


class FakeStream:
    instances: list[FakeStream] = []

    def __init__(self, samplerate, channels, dtype, callback, latency=None):
        self.callback = callback
        self.channels = channels
        self.latency = 0.0
        self.aborted_at: float | None = None
        self.blocks: list[tuple[float, np.ndarray]] = []
        self._stop = threading.Event()
        FakeStream.instances.append(self)

    def start(self):
        def run():
            while not self._stop.is_set():
                out = np.zeros((480, self.channels), dtype=np.float32)
                self.callback(out, 480, None, None)
                self.blocks.append((time.perf_counter(), out.copy()))
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def abort(self):
        self.aborted_at = time.perf_counter()
        self._stop.set()

    def close(self):
        self._stop.set()


@pytest.fixture(autouse=True)
def fresh_line(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    monkeypatch.setattr(voice, "_output_line", voice._OutputLine())
    yield
    voice._output_line.close()


def fake_audio(monkeypatch, samples: int) -> None:
    monkeypatch.setattr(voice, "_read_audio_file", lambda _path: (np.full(samples, 0.5, dtype=np.float32), 48000))


@pytest.mark.asyncio
async def test_cutting_heliox_off_fades_to_silence_within_tens_of_milliseconds(tmp_path, monkeypatch):
    path = tmp_path / "speech.wav"
    fake_audio(monkeypatch, 48000 * 3)
    task = asyncio.ensure_future(voice._play_local_tts_file(path))
    await asyncio.sleep(0.2)
    cut_at = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)
    stream = FakeStream.instances[0]
    voiced = [(at, block) for at, block in stream.blocks if block.any()]
    assert voiced[-1][0] - cut_at < 0.08  # nothing is voiced after the fade
    tail = voiced[-1][1][voiced[-1][1][:, 0] > 0, 0]
    # The voice ramps down to nothing instead of being cut mid-wave (a click).
    assert tail[-1] < 0.01 and tail[0] > tail[-1]
    assert stream.aborted_at is None  # the line stays open for what Heliox says next


@pytest.mark.asyncio
async def test_an_uninterrupted_line_plays_to_the_end(tmp_path, monkeypatch):
    path = tmp_path / "speech.wav"
    fake_audio(monkeypatch, 4800)
    await voice._play_local_tts_file(path)
    stream = FakeStream.instances[0]
    assert stream.aborted_at is None
    played = np.concatenate([block for _at, block in stream.blocks])
    assert (played > 0.4).sum() == 4800


@pytest.mark.asyncio
async def test_the_next_sentence_after_a_cut_off_reuses_the_open_line(tmp_path, monkeypatch):
    path = tmp_path / "speech.wav"
    fake_audio(monkeypatch, 48000 * 3)
    answer = asyncio.ensure_future(voice._play_local_tts_file(path))
    await asyncio.sleep(0.2)
    answer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await answer
    fake_audio(monkeypatch, 4800)
    await voice._play_local_tts_file(path)  # "Okay, stopped."
    assert len(FakeStream.instances) == 1  # no second open: that took 1 to 3 s on the real device


def test_warming_opens_the_line_once_and_another_rate_reopens_it():
    line = voice._output_line
    first = line.ensure(24000, 1)
    assert line.ensure(24000, 1) is first
    assert line.ensure(48000, 1) is not first and first.aborted_at is not None


def test_a_real_wav_is_read_as_float_samples(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "speech.wav"
    soundfile.write(str(path), np.full(480, 0.25, dtype=np.float32), 24000)
    audio, rate = voice._read_audio_file(path)
    assert rate == 24000 and audio.dtype == np.float32 and abs(float(audio[0]) - 0.25) < 1e-3

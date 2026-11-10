"""Barge-in cuts Heliox's voice at once: a few milliseconds of fade, then the output stream is aborted instead of
draining what the device has buffered (sd.stop() took 175 to 200 ms)."""

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

    def __init__(self, samplerate, channels, dtype, callback):
        self.callback = callback
        self.channels = channels
        self.latency = 0.0
        self.aborted_at: float | None = None
        self.blocks: list[np.ndarray] = []
        self._stop = threading.Event()
        FakeStream.instances.append(self)

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                out = np.zeros((480, self.channels), dtype=np.float32)
                self.callback(out, 480, None, None)
                self.blocks.append(out.copy())
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def abort(self):
        self.aborted_at = time.perf_counter()
        self._stop.set()


@pytest.mark.asyncio
async def test_cutting_heliox_off_fades_and_aborts_within_tens_of_milliseconds(tmp_path, monkeypatch):
    import soundfile

    path = tmp_path / "speech.wav"
    soundfile.write(str(path), np.full(48000 * 3, 0.5, dtype=np.float32), 48000)
    FakeStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    task = asyncio.ensure_future(voice._play_local_tts_file(path))
    await asyncio.sleep(0.2)
    cut_at = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stream = FakeStream.instances[0]
    assert stream.aborted_at is not None and stream.aborted_at - cut_at < 0.08
    voiced = [block for block in stream.blocks if block.any()]
    tail = voiced[-1][voiced[-1][:, 0] > 0, 0]
    # The voice ramps down to nothing before the stream stops, instead of being cut mid-wave (a click).
    assert tail[-1] < 0.01 and tail[0] > tail[-1]


@pytest.mark.asyncio
async def test_an_uninterrupted_line_plays_to_the_end(tmp_path, monkeypatch):
    import soundfile

    path = tmp_path / "speech.wav"
    soundfile.write(str(path), np.full(4800, 0.5, dtype=np.float32), 48000)
    FakeStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    await voice._play_local_tts_file(path)
    stream = FakeStream.instances[0]
    assert stream.aborted_at is None
    played = np.concatenate(stream.blocks)
    assert (played > 0.4).sum() == 4800

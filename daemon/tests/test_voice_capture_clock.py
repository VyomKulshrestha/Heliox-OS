"""Decisions are made on the time the sound arrived, not the time the listener got to it: the same utterances,
fed through the real recorder and listener, decide the same way with no backlog and with 7 or 25 s of recognition
backlog (a CPU-only laptop). Generated tones, scripted words, an injected clock; no models, no audio device."""

from __future__ import annotations

import asyncio
import time as real_time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy")

from pilot.config import PilotConfig  # noqa: E402
from pilot.system import voice  # noqa: E402
from pilot.system.voice import ContinuousVoiceListener  # noqa: E402

FRAME = 0.05
RATE = 16000


def tone(seconds: float) -> list:
    t = np.arange(int(seconds * RATE)) / RATE
    return [(0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)]


def quiet(seconds: float) -> list:
    return [np.zeros(int(seconds * RATE), np.int16)]


class NoSpeculation:
    def start(self, _frames):
        return None

    def cancel(self):
        return None

    def take(self):
        return None


async def run(script: list[tuple[list, str]], backlog: float, monkeypatch) -> list[tuple[str, str]]:
    """Play ``script`` (audio pieces, and the words for each voiced piece) through the listener. With ``backlog``,
    every frame is captured first and the listener only starts ``backlog`` seconds after the last one."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        voice,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, time=real_time.time, perf_counter=real_time.perf_counter),
    )
    dispatched: list[tuple[str, str]] = []

    async def on_command(text: str) -> None:
        dispatched.append((text, voice.voice_turn_addressed.get()))

    config = PilotConfig()
    config.voice.turn_detection = "silence"
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], on_command=on_command, config=config)
    listener._wake_calibrator = MagicMock()
    listener._wake_calibrator.promoted.return_value = []
    listener._wake_calibrator.check_near_miss.return_value = None
    listener._addressing.clock = lambda: clock.now
    listener._speculate_final_pass = lambda: NoSpeculation()
    words = iter([text for _audio, text in script if text])

    async def transcribe(_path, speculative=False):
        return next(words, "")

    listener._transcribe_path = transcribe
    recorder = listener._recorder
    recorder.use_file_source()
    recorder._soundcard_recorder = object()  # "active"
    recorder._loop = asyncio.get_running_loop()
    recorder._queue = asyncio.Queue()
    size = int(RATE * FRAME)
    frames = [
        piece[i : i + size] for audio, _text in script for piece in audio for i in range(0, len(piece) - size + 1, size)
    ]
    captured = [1000.0 + (k + 1) * FRAME for k in range(len(frames))]
    listener._running = True
    loop_task = asyncio.ensure_future(listener._listen_loop())

    async def settle():
        for _ in range(50):
            await asyncio.sleep(0)

    if backlog:
        for frame, at in zip(frames, captured, strict=True):
            recorder.feed(frame, captured=at)
        clock.now = captured[-1] + backlog
        await asyncio.sleep(0.2)
    else:
        for frame, at in zip(frames, captured, strict=True):
            clock.now = at
            recorder.feed(frame, captured=at)
            await settle()
    await asyncio.sleep(0.3)
    listener._running = False
    recorder._queue.put_nowait(voice._RECORDER_STOP)
    loop_task.cancel()
    try:
        await loop_task
    except (asyncio.CancelledError, Exception):
        pass
    return dispatched


SPLIT = [
    (quiet(0.3), ""),
    (tone(1.0), "Hey Heliox, remind me to"),
    (quiet(1.4), ""),
    (tone(1.0), "call the bank tomorrow morning."),
    (quiet(1.5), ""),
]
NEW_TURN = [
    (quiet(0.3), ""),
    (tone(1.0), "Hey Heliox, what time is it?"),
    (quiet(4.5), ""),
    (tone(1.0), "And tomorrow?"),
    (quiet(1.5), ""),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backlog", [0.0, 7.0, 25.0])
async def test_a_continuation_is_the_same_with_and_without_a_recognition_backlog(backlog, monkeypatch):
    dispatched = await run(SPLIT, backlog, monkeypatch)
    assert dispatched == [("remind me to", "wake"), ("call the bank tomorrow morning", "follow_up")]


@pytest.mark.asyncio
@pytest.mark.parametrize("backlog", [0.0, 7.0])
async def test_a_late_second_sentence_is_refused_the_same_with_and_without_a_backlog(backlog, monkeypatch):
    # 4.5 s after the turn and Heliox never answered: not a continuation, and no follow-up window either way.
    dispatched = await run(NEW_TURN, backlog, monkeypatch)
    assert dispatched == [("what time is it", "wake")]


def test_portaudio_s_capture_time_is_moved_onto_the_monotonic_clock():
    info = SimpleNamespace(inputBufferAdcTime=50.0, currentTime=50.02)
    end = voice._portaudio_capture_end(info, 800, 16000)
    assert end is not None and abs((real_time.monotonic() - 0.02 + 0.05) - end) < 0.01
    assert voice._portaudio_capture_end(SimpleNamespace(inputBufferAdcTime=0.0, currentTime=0.0), 800, 16000) is None


def test_without_capture_times_the_stream_s_sample_count_is_the_clock(monkeypatch):
    from pilot.system.voice import _ContinuousRecorder

    clock = SimpleNamespace(now=500.0)
    monkeypatch.setattr(voice, "time", SimpleNamespace(monotonic=lambda: clock.now, time=real_time.time))
    recorder = _ContinuousRecorder()
    first = recorder._capture_stamp(800, None)
    clock.now += 0.3  # the next callback is late
    second = recorder._capture_stamp(800, None)
    assert abs(second - first - 0.05) < 1e-9  # the audio is 50 ms on, whenever the callback ran
    clock.now += 5.0  # the device restarted
    assert recorder._capture_stamp(800, None) == clock.now

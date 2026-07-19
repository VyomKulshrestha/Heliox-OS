"""Tests for _ContinuousRecorder's queue-consuming logic (wait_for_speech_start /
record_utterance), independent of the actual sounddevice.InputStream --
frames are pushed directly onto the recorder's internal queue, matching
what the real audio callback would hand off via call_soon_threadsafe."""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from pilot.system.voice import _ContinuousRecorder


def _loud_frame(size: int = 800) -> np.ndarray:
    return np.full(size, 10000, dtype=np.int16)


def _silent_frame(size: int = 800) -> np.ndarray:
    return np.zeros(size, dtype=np.int16)


def _recorder(**kwargs) -> _ContinuousRecorder:
    recorder = _ContinuousRecorder(**kwargs)
    recorder._queue = asyncio.Queue()
    return recorder


async def _feed(queue: asyncio.Queue, frames: list[np.ndarray]) -> None:
    """Pushes frames onto the queue AFTER the consumer has started waiting
    (wait_for_speech_start/record_utterance discard any backlog queued
    before they start listening — see _drain_stale_frames), mirroring how
    the real audio callback feeds frames in over time rather than all at
    once."""
    for f in frames:
        await asyncio.sleep(0.01)
        queue.put_nowait(f)


@pytest.mark.asyncio
async def test_wait_for_speech_start_detects_sustained_loud_frames():
    recorder = _recorder(start_frames=2, silence_frames=3)
    feeder = asyncio.create_task(
        _feed(recorder._queue, [_silent_frame(), _silent_frame(), _loud_frame(), _loud_frame()])
    )

    result = await recorder.wait_for_speech_start(timeout=2.0)
    await feeder
    assert result is True


@pytest.mark.asyncio
async def test_wait_for_speech_start_times_out_on_silence():
    recorder = _recorder(start_frames=2, silence_frames=3)
    for _ in range(5):
        recorder._queue.put_nowait(_silent_frame())

    assert await recorder.wait_for_speech_start(timeout=0.2) is False


@pytest.mark.asyncio
async def test_wait_for_speech_start_returns_false_when_queue_is_none():
    recorder = _ContinuousRecorder()
    assert await recorder.wait_for_speech_start(timeout=0.1) is False


@pytest.mark.asyncio
async def test_record_utterance_writes_wav_and_returns_path():
    recorder = _recorder(start_frames=1, silence_frames=2, sample_rate=16000)
    feeder = asyncio.create_task(
        _feed(recorder._queue, [_loud_frame(), _loud_frame(), _silent_frame(), _silent_frame()])
    )

    path = await recorder.record_utterance(timeout=2.0)
    await feeder
    assert path is not None
    assert os.path.exists(path)
    os.remove(path)


@pytest.mark.asyncio
async def test_record_utterance_returns_none_on_timeout_without_speech():
    recorder = _recorder(start_frames=3, silence_frames=3)
    for _ in range(3):
        recorder._queue.put_nowait(_silent_frame())

    assert await recorder.record_utterance(timeout=0.2) is None


@pytest.mark.asyncio
async def test_record_utterance_returns_none_when_queue_is_none():
    recorder = _ContinuousRecorder()
    assert await recorder.record_utterance(timeout=0.1) is None


def test_stop_is_safe_when_never_started():
    recorder = _ContinuousRecorder()
    recorder.stop()  # must not raise
    assert recorder.is_active is False

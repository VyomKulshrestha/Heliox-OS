"""When the speculative final pass may start: at 100 ms of quiet only while the last passes ran well under real time
(a wasted one costs little), at 200 ms otherwise and until three passes are known. Injected numbers only."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.voice import CHEAP_PASS_RTF, ContinuousVoiceListener, _Speculation


def listener() -> ContinuousVoiceListener:
    return ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())


def test_200_ms_until_three_passes_are_known_and_on_a_slow_machine():
    item = listener()
    assert item.speculation_frames() == 4
    for _ in range(2):
        item.note_pass(0.1, 2.0)  # fast, but only two
    assert item.speculation_frames() == 4
    for _ in range(5):
        item.note_pass(0.9, 3.0)  # 0.3 of real time: a CPU-only laptop
    assert item.speculation_frames() == 4


def test_100_ms_when_the_last_passes_ran_well_under_real_time_and_back_when_they_slow():
    item = listener()
    for _ in range(3):
        item.note_pass(0.12, 3.0)  # 0.04 of real time
    assert item.speculation_frames() == 2
    for _ in range(3):
        item.note_pass(3 * CHEAP_PASS_RTF * 3.0, 3.0)  # the machine got busy
    assert item.speculation_frames() == 4


def test_a_very_short_pass_does_not_count():
    item = listener()
    for _ in range(5):
        item.note_pass(0.01, 0.2)
    assert item.speculation_frames() == 4


@pytest.mark.asyncio
async def test_a_finished_speculative_pass_is_timed_against_its_audio(monkeypatch):
    item = listener()

    async def transcribe(path, *, speculative=False):
        await asyncio.sleep(0)
        return "what time is it"

    item._transcribe_path = transcribe
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 0)
    speculation = _Speculation(item)
    frames = [np.zeros(800, dtype=np.int16) for _ in range(40)]  # 2 s of audio
    speculation.start(frames)
    assert await speculation.take() == "what time is it"
    assert len(item._pass_rtfs) == 1 and item._pass_rtfs[0] < CHEAP_PASS_RTF


def test_the_recorder_gets_the_choice_with_each_utterance(monkeypatch):
    item = listener()
    for _ in range(3):
        item.note_pass(0.1, 3.0)
    item._loop = object()
    item._turn_detector = None
    item._streaming = None
    monkeypatch.setattr(asyncio, "ensure_future", lambda coro: coro.close())
    item._configure_turn_taking()
    assert item._recorder.speculate_frames == 2


@pytest.mark.parametrize(("cores", "threads"), [(1, 1), (2, 1), (4, 2), (6, 3), (8, 4), (24, 4), (64, 4)])
def test_parakeet_uses_half_the_cores_at_most_four(cores, threads):
    from pilot.system.parakeet_asr import parakeet_threads

    assert parakeet_threads(cores) == threads

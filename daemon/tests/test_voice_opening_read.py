"""The wake check and the final words (pilot.system.voice): the opening is read while the person is still talking even
when another recogniser holds the processor, a reading that ends at the wake phrase is made once more, and the final
words wait for a whole-utterance check only when one is needed, run beside Parakeet, never after it.

The 24 November live run found the opening reading skipped on 20 of 26 turns: a pass started at the pause after "Hey
Heliox," was still running when the opening came due. The whole-utterance check then held the final words for about a
quarter of a second. Fakes only: no models, no audio device. Order is checked by events and counts, never by time."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.voice import ContinuousVoiceListener, opening_can_confirm, opening_cannot_confirm

RATE = 16000
FRAME = 800  # 50 ms


def frames(seconds: float) -> list[np.ndarray]:
    return [np.zeros(FRAME, dtype=np.int16) for _ in range(int(seconds * RATE / FRAME))]


def listener(tmp_path, monkeypatch, said: str = "Hey Heliocs, what's a good name for a grey cat?", seconds=3.0):
    config = PilotConfig()
    config.voice.final_pass = "parakeet"
    config.voice.parakeet_model_dir = str(tmp_path)
    item = ContinuousVoiceListener(config=config)
    fake = MagicMock()
    fake.transcribe.return_value = said
    fake.available = True
    monkeypatch.setattr(item, "_parakeet", lambda: fake)
    monkeypatch.setattr(voice, "_load_pcm_wav_for_whisper", lambda _path: np.zeros(int(seconds * RATE), np.float32))
    return item


async def settle(times: int = 10) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("text", "confirms"),
    [
        ("Hey Heliox, what's a-", True),
        ("Hey Heliox. How long?", True),
        ("Hey Heliox.", False),  # 9 of the run's 27 clips read only this at 1.6 s
        ("Hey Heliox", False),
        ("So I told her", False),
        ("", False),
    ],
)
def test_a_reading_confirms_only_with_a_word_after_the_wake_phrase(text, confirms):
    assert opening_can_confirm(text) is confirms


@pytest.mark.asyncio
async def test_a_finished_reading_of_the_phrase_alone_cannot_confirm():
    done = asyncio.get_running_loop().create_future()
    done.set_result("Hey Heliox.")
    assert opening_cannot_confirm(done)
    pending = asyncio.get_running_loop().create_future()
    assert not opening_cannot_confirm(pending)  # still reading: it may yet confirm
    pending.cancel()


@pytest.mark.asyncio
async def test_the_opening_waits_for_a_running_recogniser_and_reads_what_was_said_by_then(tmp_path, monkeypatch):
    item = listener(tmp_path, monkeypatch)
    busy = {"calls": 0}

    def running() -> int:
        busy["calls"] += 1
        return 1 if busy["calls"] <= 5 else 0  # a dropped pass's thread, then free

    monkeypatch.setattr(voice, "_recognisers_running", running)
    written: list[int] = []
    monkeypatch.setattr(voice, "_write_wav", lambda fs, _rate: written.append(sum(len(f) for f in fs)) or "o.wav")

    async def whisper(_path, **_kwargs):
        return "Hey Heliox, what's a good"

    item._transcribe_whisper_text = whisper
    item._recorder.user_is_speaking = True
    item._recorder.current_frames = frames(2.2)  # said by the time the processor is free
    item._start_early_wake(frames(1.6))
    assert await item._early_wake == "Hey Heliox, what's a good"
    assert written == [int(2.2 * RATE)]  # read, not skipped, and on everything said by then


@pytest.mark.asyncio
async def test_a_reading_that_ends_at_the_wake_phrase_is_made_once_more_while_they_talk(tmp_path, monkeypatch):
    item = listener(tmp_path, monkeypatch)
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 0)
    written: list[int] = []
    monkeypatch.setattr(voice, "_write_wav", lambda fs, _rate: written.append(sum(len(f) for f in fs)) or "o.wav")
    replies = iter(["Hey Heliox.", "Hey Heliox, what's a good name for a-"])
    first_read = asyncio.Event()

    async def whisper(_path, **_kwargs):
        first_read.set()
        return next(replies)

    item._transcribe_whisper_text = whisper
    item._recorder.user_is_speaking = True
    item._recorder.current_frames = frames(1.6)
    item._start_early_wake(frames(1.6))
    await first_read.wait()
    await settle()
    assert len(written) == 1  # nothing more said yet: no second reading
    item._recorder.current_frames = frames(2.4)  # 0.8 s more
    assert await item._early_wake == "Hey Heliox, what's a good name for a-"
    assert written == [int(1.6 * RATE), int(2.4 * RATE)]


@pytest.mark.asyncio
async def test_no_second_reading_once_the_person_has_stopped(tmp_path, monkeypatch):
    item = listener(tmp_path, monkeypatch)
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 0)
    calls: list[str] = []

    async def whisper(path, **_kwargs):
        calls.append(path)
        return "Hey Heliox."

    item._transcribe_whisper_text = whisper
    item._recorder.user_is_speaking = False
    item._start_early_wake(frames(1.6))
    assert await item._early_wake == "Hey Heliox."
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_reading_still_waiting_for_the_processor_is_not_waited_for(tmp_path, monkeypatch):
    """The final pass starts while the opening still waits: the whole-utterance check runs beside Parakeet."""
    item = listener(tmp_path, monkeypatch, said="Hey Heliocs, what time is it?")
    whisper_started = threading.Event()
    seen: list[bool] = []

    def parakeet(_samples):
        seen.append(whisper_started.wait(5.0))
        return "Hey Heliocs, what time is it?"

    item._parakeet().transcribe.side_effect = parakeet

    async def whisper(_path, **_kwargs):
        whisper_started.set()
        return "Hey Heliox, what time is it?"

    item._transcribe_whisper_text = whisper
    item._early_wake = asyncio.get_running_loop().create_future()  # never resolves: still waiting
    item._opening_waiting = True
    assert await item._transcribe_path("utterance.wav") == "Hey Heliox, what time is it?"
    assert seen == [True] and item.last_wake_source == "whole"
    item._early_wake.cancel()


@pytest.mark.asyncio
async def test_a_reading_that_ends_unable_to_confirm_starts_the_whole_check_before_parakeet_is_done(
    tmp_path, monkeypatch
):
    item = listener(tmp_path, monkeypatch, said="Hey Heliocs, what time is it?")
    whisper_started = threading.Event()
    seen: list[bool] = []

    def parakeet(_samples):
        seen.append(whisper_started.wait(5.0))  # Parakeet ends only once the whole check has begun
        return "Hey Heliocs, what time is it?"

    item._parakeet().transcribe.side_effect = parakeet
    reads: list[str] = []

    async def whisper(_path, **_kwargs):
        reads.append("whole")
        whisper_started.set()
        return "Hey Heliox, what time is it?"

    item._transcribe_whisper_text = whisper
    opening = asyncio.get_running_loop().create_future()
    item._early_wake = opening
    asyncio.get_running_loop().call_soon(opening.set_result, "Hey Heliox.")  # ends while Parakeet runs
    assert await item._transcribe_path("utterance.wav") == "Hey Heliox, what time is it?"
    assert seen == [True] and reads == ["whole"] and item.last_wake_source == "whole"


@pytest.mark.asyncio
async def test_an_agreeing_reading_still_needs_no_whole_check(tmp_path, monkeypatch):
    item = listener(tmp_path, monkeypatch, said="Hey Heliocs, what time is it?")
    reads: list[str] = []

    async def whisper(_path, **_kwargs):
        reads.append("whole")
        return ""

    item._transcribe_whisper_text = whisper
    opening = asyncio.get_running_loop().create_future()
    item._early_wake = opening
    asyncio.get_running_loop().call_soon(opening.set_result, "Hey Heliox, what time")
    assert await item._transcribe_path("utterance.wav") == "Hey Heliox, what time is it?"
    await settle()
    assert reads == [] and item.last_wake_source == "opening"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "whisper_runs"),
    [
        ("Skip the backup and the logs.", False),  # a reply: Parakeet's words settle it
        ("Hey Heliocs, remind me that the launch call is Monday.", True),  # not a reply as written: checked after
    ],
)
async def test_inside_a_conversation_a_longer_utterance_is_read_by_parakeet_alone_first(
    tmp_path, monkeypatch, said, whisper_runs
):
    item = listener(tmp_path, monkeypatch, said=said, seconds=2.5)
    order: list[str] = []

    def parakeet(_samples):
        order.append("parakeet")
        return said

    item._parakeet().transcribe.side_effect = parakeet

    async def whisper(_path, **_kwargs):
        order.append("whisper")
        return "Hey Heliox, remind me that the launch call is Monday."

    item._transcribe_whisper_text = whisper
    item._addressing._addressed()  # a conversation someone opened with the wake phrase
    item.arm_follow_up_window(30)
    await item._transcribe_path("utterance.wav")
    assert order == (["parakeet", "whisper"] if whisper_runs else ["parakeet"])


@pytest.mark.asyncio
async def test_a_short_utterance_in_a_conversation_keeps_whisper_beside_parakeet(tmp_path, monkeypatch):
    item = listener(tmp_path, monkeypatch, said="Okay.", seconds=1.0)
    whisper_started = threading.Event()
    seen: list[bool] = []

    def parakeet(_samples):
        seen.append(whisper_started.wait(5.0))
        return "Okay."

    item._parakeet().transcribe.side_effect = parakeet

    async def whisper(_path, **_kwargs):
        whisper_started.set()
        return ""  # Whisper rates it not speech: a lone word on noise

    item._transcribe_whisper_text = whisper
    item._addressing._addressed()
    item.arm_follow_up_window(30)
    assert await item._transcribe_path("utterance.wav") == ""
    assert seen == [True]


@pytest.mark.asyncio
async def test_a_reading_of_the_opening_does_not_hold_back_the_speculative_pass(monkeypatch):
    item = ContinuousVoiceListener(config=PilotConfig())
    started: list[str] = []

    async def transcribe(_path, *, speculative=False):
        started.append("pass")
        return ""

    item._transcribe_path = transcribe
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 1)
    monkeypatch.setattr(voice, "_write_wav", lambda _frames, _rate: "candidate.wav")
    item._opening_threads = 1  # the one running is the opening reading
    speculation = voice._Speculation(item)
    speculation.start(frames(1.0))
    assert speculation.task is not None
    await speculation.take()
    assert started == ["pass"]
    item._opening_threads = 0  # a dropped pass's thread: held back, as before
    speculation.start(frames(1.0))
    assert speculation.task is None

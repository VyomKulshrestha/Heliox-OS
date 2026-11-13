"""Every recogniser the listener uses runs one real inference at start, off the event loop and bounded, and the
listener is ready only when that is done: a sentence said in the first seconds waits for warm recognisers instead of
being their (slow) first call. Fakes only: no model, no microphone."""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.streaming_asr import StreamingTranscriber


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []


def listener(engine: str = "auto") -> voice.ContinuousVoiceListener:
    config = PilotConfig()
    config.voice.transcription_engine = engine
    config.voice.turn_detection = "silence"  # no Smart Turn loaded in the background by these tests
    return voice.ContinuousVoiceListener(wake_words=["hey heliox"], config=config)


def fakes(item: voice.ContinuousVoiceListener, seen: Recorder, monkeypatch, *, whisper_fails: bool = False) -> None:
    async def whisper(path, language, model):
        seen.calls.append("whisper")
        assert os.path.exists(path) and language == "en"
        if whisper_fails:
            raise RuntimeError("no model")
        return {"text": ""}

    monkeypatch.setattr(voice, "_transcribe_faster_whisper", whisper)

    class Parakeet:
        def transcribe(self, samples):
            seen.calls.append("parakeet")
            assert samples.dtype == np.float32 and len(samples) == 16000
            return ""

    class Streaming:
        ready = True
        error = ""

        def warm(self):
            seen.calls.append("streaming")
            return True

    class Detector:
        ready = True
        error = ""

        def probability(self, audio):
            seen.calls.append("smart_turn")
            return 0.1

    item._parakeet = lambda: Parakeet()
    item._streaming = Streaming()
    item._turn_detector = Detector()


@pytest.mark.asyncio
async def test_each_recogniser_in_use_runs_one_inference(monkeypatch):
    item, seen = listener(), Recorder()
    fakes(item, seen, monkeypatch)
    timings = await item._warm_recognisers()
    assert seen.calls == ["whisper", "parakeet", "streaming", "smart_turn"]
    assert set(timings) == {"whisper", "parakeet", "streaming", "smart_turn"}


@pytest.mark.asyncio
async def test_a_failing_or_stuck_step_is_skipped_and_the_rest_still_warm(monkeypatch):
    item, seen = listener(), Recorder()
    fakes(item, seen, monkeypatch, whisper_fails=True)
    monkeypatch.setattr(voice, "WARM_STEP_SECONDS", 0.05)

    class Stuck:
        ready = True

        def warm(self):
            time.sleep(0.3)
            return True

    item._streaming = Stuck()
    timings = await item._warm_recognisers()
    assert "whisper" not in timings and "streaming" not in timings
    assert {"parakeet", "smart_turn"} <= set(timings)


@pytest.mark.asyncio
async def test_ready_only_when_warm_and_the_voice_bar_is_told(monkeypatch):
    item, seen = listener(), Recorder()
    fakes(item, seen, monkeypatch)
    statuses: list[str] = []

    async def on_status(status, data):
        statuses.append(status)

    item._on_status = on_status
    item._loop = asyncio.get_running_loop()
    item._warmed = asyncio.Event()

    async def nothing():
        return None

    item._warm_turn_taking = nothing
    monkeypatch.setattr(voice, "_get_faster_whisper_model", lambda name: None)
    await item._warm_transcriber()
    for _ in range(5):
        await asyncio.sleep(0)
    assert item._warmed.is_set() and statuses == ["hearing_ready"]
    assert "whisper" in seen.calls
    item._running = True
    stats = item.get_stats()
    assert stats["hears"] is True and "whisper" in stats["warm_ms"]


def test_it_does_not_hear_before_the_warm_up_or_when_stopped():
    item = listener()
    assert item.get_stats()["hears"] is False  # not started
    item._running, item._warmed = True, asyncio.Event()
    assert item.get_stats()["hears"] is False  # listening, still warming
    item._warmed.set()
    item._running = False
    assert item.get_stats()["hears"] is False  # stopped


@pytest.mark.asyncio
async def test_a_sentence_said_while_warming_is_read_after_the_warm_up(monkeypatch):
    item = listener("whisper")
    item._recorder._soundcard_recorder = object()  # the stream is "open"
    item._warmed = asyncio.Event()
    read_at: list[float] = []

    async def transcribe(path, *, speculative=False):
        read_at.append(time.monotonic())
        return "hey heliox what time is it"

    async def record(**_kwargs):
        return "utterance.wav"

    item._transcribe_path = transcribe
    item._recorder.record_utterance = record

    async def warm_later():
        await asyncio.sleep(0.15)
        item._warmed.set()
        return time.monotonic()

    warmed_at, text = await asyncio.gather(
        warm_later(), item._record_and_transcribe(duration=3, timeout=None, keep_backlog=True)
    )
    assert text == "hey heliox what time is it" and read_at[0] >= warmed_at


@pytest.mark.asyncio
async def test_a_load_slower_than_the_wait_is_never_read_cold_and_the_person_is_asked_again(monkeypatch):
    """A slow machine: the models take longer to load than WARM_WAIT_SECONDS. The sentence said meanwhile is not read
    by a cold recogniser, not acted on late, and not dropped silently: the bar asks the person to say it again, and the
    next sentence, once the recognisers are ready, is read as usual."""
    item = listener("whisper")
    item._recorder._soundcard_recorder = object()
    item._loop = asyncio.get_running_loop()
    item._warmed = asyncio.Event()
    monkeypatch.setattr(voice, "WARM_WAIT_SECONDS", 0.05)
    statuses: list[tuple[str, dict]] = []

    async def on_status(status, data):
        statuses.append((status, data))

    item._on_status = on_status
    read: list[str] = []

    async def transcribe(path, *, speculative=False):
        read.append(path)
        return "turn off the lights"

    async def record(**_kwargs):
        return "utterance.wav"

    item._transcribe_path = transcribe
    item._recorder.record_utterance = record

    async def slow_load():  # the stand-in load: longer than the wait
        await asyncio.sleep(0.3)
        item._warmed.set()

    loading = asyncio.ensure_future(slow_load())
    assert await item._record_and_transcribe(duration=3, timeout=None, keep_backlog=True) == ""
    for _ in range(5):
        await asyncio.sleep(0)
    assert read == []  # never read cold, never acted on
    assert [status for status, _ in statuses] == ["hearing_not_ready"]
    assert "say that again" in statuses[0][1]["message"].casefold()
    await loading
    assert await item._record_and_transcribe(duration=3, timeout=None, keep_backlog=True) == "turn off the lights"


def test_the_streaming_warm_up_decodes_its_own_stream_and_leaves_the_utterance_alone():
    class Stream:
        def __init__(self) -> None:
            self.samples = 0
            self.finished = False

        def accept_waveform(self, rate, samples):
            self.samples += len(samples)

        def input_finished(self):
            self.finished = True

    class Recognizer:
        def __init__(self) -> None:
            self.streams: list[Stream] = []
            self.decoded = 0

        def create_stream(self):
            self.streams.append(Stream())
            return self.streams[-1]

        def is_ready(self, stream):
            if stream.finished and not getattr(stream, "done", False):
                stream.done = True
                return True
            return False

        def decode_stream(self, stream):
            self.decoded += 1

    recognizer = Recognizer()
    transcriber = StreamingTranscriber(recognizer=recognizer)
    utterance = object()
    transcriber._stream = utterance
    assert transcriber.warm(timeout=5.0) is True
    assert transcriber._stream is utterance  # the utterance's stream is untouched
    assert len(recognizer.streams) == 1 and recognizer.streams[0].samples >= 16000 and recognizer.decoded == 1
    assert StreamingTranscriber().warm() is False  # not loaded: nothing to warm

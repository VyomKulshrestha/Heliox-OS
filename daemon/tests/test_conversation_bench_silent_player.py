"""The bench's silent player (driver.play_silently) takes every kind of piece the voice hands over: a WAV path, a
local_tts_worker.Piece with its samples in memory or a kept WAV in ``path``, and a StreamedPiece. It never opens a
sound device, reports its position to the speak() in progress as the real player does, and gives the real player's
"piece read" and "playback start" stamps. Before, it read only a path, so every in-memory piece failed and the local
voice fell to the system voice in every bench run. A clock that moves only when the player waits; nothing is timed."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from benchmarks.conversation_bench import driver, wavio
from benchmarks.conversation_bench import timeline as tl
from pilot.system import speech_trace
from pilot.system.local_tts_worker import Piece, SpeechProgress, StreamedPiece, speech_progress

RATE = 10000
STEP_SAMPLES = round(driver.SILENT_STEP_SECONDS * RATE)


class Clock:
    """Moves only when the player waits (``sleep``), by the time it asked for; counts its waits."""

    def __init__(self) -> None:
        self.now = 100.0
        self.waits = 0
        self.positions: list[int] = []
        self.progress: SpeechProgress | None = None

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits += 1
        self.now += seconds
        if self.progress is not None and self.progress.playing is not None:
            self.positions.append(self.progress.playing.position)


@pytest.fixture
def traced():
    events: list[tuple[str, dict]] = []
    before = speech_trace.sink
    speech_trace.sink = lambda event, data: events.append((event, dict(data)))
    yield events
    speech_trace.sink = before


async def _play(piece, clock: Clock) -> tuple[list[tuple[str, dict]], SpeechProgress]:
    logged: list[tuple[str, dict]] = []
    progress = SpeechProgress("A sentence.", ["A sentence."], 1)
    clock.progress = progress
    token = speech_progress.set(progress)
    try:
        await driver.play_silently(
            piece, lambda kind, _text="", **data: logged.append((kind, data)), clock=clock, sleep=clock.sleep
        )
    finally:
        speech_progress.reset(token)
    return logged, progress


def _samples(count: int) -> np.ndarray:
    return np.full(count, 0.25, dtype=np.float32)


@pytest.mark.parametrize("kind", ["path", "in_memory", "kept_wav"])
async def test_a_whole_piece_of_each_kind_plays_silently_at_its_own_pace(kind, tmp_path, traced):
    count = 10 * STEP_SAMPLES  # ten steps of the player
    wav = tmp_path / "piece.wav"
    wavio.write(wav, _samples(count), RATE)
    piece = {
        "path": wav,
        "in_memory": Piece("A sentence.", audio=_samples(count), rate=RATE),
        "kept_wav": Piece("A sentence.", path=wav, source="store"),
    }[kind]
    clock = Clock()
    logged, progress = await _play(piece, clock)
    assert logged[0][0] == tl.AUDIO_STARTS
    assert clock.waits == 10  # consumed at the pace of its audio, by the clock the player was given
    assert clock.positions == [STEP_SAMPLES * n for n in range(1, 11)]  # its position, as the real player reports it
    assert progress.playing.position == count == len(progress.playing.audio)
    names = [event for event, _data in traced]
    assert names == ["piece read", "playback start"]
    read, start = traced[0][1], traced[1][1]
    assert read["in_memory"] is (kind == "in_memory")
    assert start["rate"] == RATE and start["seconds"] == pytest.approx(count / RATE) and start["silent"] is True


class _ScriptedStream(StreamedPiece):
    """A stream whose chunks arrive as the test says: each take returns the next scripted step, and may move the clock
    (a chunk that arrived late)."""

    def __init__(self, steps: list[tuple[int, bool, float]], clock: Clock, on_take=None) -> None:
        super().__init__("A streamed sentence.", stopper=self._stopped)
        self.rate = RATE
        self.steps = list(steps)
        self.clock = clock
        self.takes = 0
        self.stops = 0
        self.on_take = on_take

    def _stopped(self, _stream_id: int) -> None:
        self.stops += 1

    def take(self, start: int, timeout: float):
        self.takes += 1
        if self.on_take is not None:
            self.on_take(self.takes)
        if not self.steps:
            return [], False  # nothing more yet
        samples, ended, late = self.steps.pop(0)
        self.clock.now += late
        return ([_samples(samples)] if samples else []), ended


async def test_a_streamed_piece_plays_all_it_received_and_reports_as_it_goes(traced):
    clock = Clock()
    stream = _ScriptedStream([(4 * STEP_SAMPLES, False, 0.0), (6 * STEP_SAMPLES, True, 0.0)], clock)
    logged, progress = await _play(stream, clock)
    playback = progress.playing
    assert playback.received == 10 * STEP_SAMPLES and playback.complete
    assert clock.waits == 10 and playback.position == playback.received
    # A streamed piece's position is moved on after each wait (read here as each wait starts): one step at a time.
    assert clock.positions == [STEP_SAMPLES * n for n in range(10)]
    played = [data for kind, data in logged if kind == "silent stream played"]
    assert played == [{"chunks": 2, "samples": 10 * STEP_SAMPLES, "underruns": 0, "audio_s": 0.2, "complete": True,
                       "error": ""}]  # fmt: skip
    assert stream.stops == 0 and traced == []  # a stream has no file to read: no "piece read"


async def test_a_streamed_chunk_that_arrives_late_counts_one_dry_spell():
    clock = Clock()
    late = 3 * driver.SILENT_STEP_SECONDS  # the second chunk came three steps after the first had played out
    stream = _ScriptedStream([(STEP_SAMPLES, False, 0.0), (STEP_SAMPLES, False, late + driver.SILENT_STEP_SECONDS),
                              (STEP_SAMPLES, True, 0.0)], clock)  # fmt: skip
    logged, _progress = await _play(stream, clock)
    played = next(data for kind, data in logged if kind == "silent stream played")
    assert played["underruns"] == 1 and played["samples"] == 3 * STEP_SAMPLES


async def test_a_cut_streamed_piece_asks_the_worker_to_stop_it():
    clock = Clock()
    loop = asyncio.get_running_loop()
    taken_three = asyncio.Event()

    def on_take(takes: int) -> None:
        if takes == 3:
            loop.call_soon_threadsafe(taken_three.set)  # take runs on a worker thread

    stream = _ScriptedStream([(STEP_SAMPLES, False, 0.0)], clock, on_take)
    progress = SpeechProgress("A streamed sentence.", ["A streamed sentence."], 1)
    token = speech_progress.set(progress)
    logged: list[str] = []
    try:
        playing = asyncio.ensure_future(
            driver.play_silently(stream, lambda kind, *_a, **_k: logged.append(kind), clock=clock, sleep=clock.sleep)
        )
        await taken_three.wait()  # the player has asked three times for chunks that have not come
        playing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await playing
    finally:
        speech_progress.reset(token)
    assert stream.stops == 1 and not progress.playing.complete
    assert logged == [tl.AUDIO_STARTS, "silent stream played"]


def test_the_harness_hands_every_piece_to_the_silent_player():
    import inspect

    source = inspect.getsource(driver.Harness)
    assert "await play_silently(piece, log, setup=self._speaking_setup)" in source
    assert "wavio.decode(Path(path)" not in source  # the old player read only a path

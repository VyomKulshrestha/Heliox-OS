"""A reply's sentence never waits behind the next one (or a piece made for the store) on the voice worker's pipe.

On 24 November (live run e2e2_a2, turn e2e-19) the whole answer was in before its first sentence was handed over: the
conversation asked for the second sentence to be made ahead (prepare) 18 ms before it handed over the first, the
second took the worker for 1.15 s, and the first played only when both were made. These tests replay that order with
a fake worker that answers each request only when the test lets it, and count what happens; nothing is timed.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import PIPE_AHEAD, PIPE_NEXT, PIPE_NOW, LocalTTSWorkerManager, _Pipe


class GatedWorker:
    """Answers a load at once; each speech request only once the test opens its gate (in the order they arrive)."""

    def __init__(self) -> None:
        self.asked: list[str] = []  # texts in the order the worker was given them
        self.gates: dict[str, threading.Event] = {}
        self.returncode: int | None = None
        self._payload = b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, read=self._read, close=lambda: None)
        self._pending = b""
        self._lock = threading.Lock()

    def gate(self, text: str) -> threading.Event:
        with self._lock:
            return self.gates.setdefault(text, threading.Event())

    def _write(self, payload: bytes) -> None:
        self._payload = payload

    def _close(self) -> None:
        self.stdin.closed = True
        self.returncode = 0

    def _readline(self) -> bytes:
        request = json.loads(self._payload)
        if request.get("op") == "load":
            return b'{"status":"ok","op":"load"}\n'
        text = request["text"]
        self.asked.append(text)
        assert self.gate(text).wait(10), f"the test never let {text!r} finish"
        samples = np.full(240, 0.1, dtype="<f4").tobytes()
        self._pending = samples
        line = {"status": "ok", "rate": 24000, "samples": 240, "format": "f32le", "bytes": len(samples)}
        return (json.dumps(line) + "\n").encode()

    def _read(self, size: int) -> bytes:
        data, self._pending = self._pending[:size], self._pending[size:]
        return data

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


async def until(condition, rounds: int = 500) -> None:
    """Let the loop (and the fake worker's thread) run until ``condition`` holds; counts rounds, never reads a clock."""
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


async def loaded_manager(worker: GatedWorker) -> LocalTTSWorkerManager:
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: worker)
    manager.environ = {}
    await manager.ensure_ready("kokoro_tts", "af_heart")
    return manager


@pytest.mark.asyncio
async def test_the_sentence_handed_over_goes_before_the_next_one_asked_for_just_before_it():
    """e2e-19's order: prepare(second), then speak(first). The worker is given the first sentence first, it plays as
    soon as its own audio exists, and the second is made while it plays (then played without being made again)."""
    worker = GatedWorker()
    manager = await loaded_manager(worker)
    events: list[str] = []

    async def play(piece) -> None:
        events.append(f"play {piece.text}")

    first, second = "It sends your traffic through a tunnel,", "so websites see that server's address."
    manager.prepare("kokoro_tts", second, "af_heart")  # the conversation asks for the next one first
    speaking = asyncio.ensure_future(manager.speak("kokoro_tts", first, "af_heart", None, play))
    await until(lambda: worker.asked)
    assert worker.asked == [first]  # the next sentence waits for its turn
    worker.gate(first).set()
    await until(lambda: events)
    assert events == [f"play {first}"]  # played the moment its own audio was in
    await speaking
    await until(lambda: len(worker.asked) == 2)
    assert worker.asked == [first, second]  # made while the first played
    worker.gate(second).set()
    await manager.speak("kokoro_tts", second, "af_heart", None, play)
    assert events == [f"play {first}", f"play {second}"] and worker.asked == [first, second]  # not made again
    kinds = [record[0] for record in manager.synth_log]
    assert kinds == ["reply", "next"]
    (_k1, _w1, _req1, started1, done1), (_k2, _w2, _req2, started2, _done2) = manager.synth_log
    assert started1 <= done1 <= started2  # each record has its own start on the pipe, not the next request's
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_prepared_sentence_a_reply_waits_for_goes_first_and_one_never_spoken_waits():
    """The next sentence is wanted before the current speech has all its pieces: the reply waits for it, so it goes
    at once. A sentence prepared and never handed over is made only after the speech that followed it."""
    worker = GatedWorker()
    manager = await loaded_manager(worker)

    async def play(piece) -> None:
        return None

    manager.prepare("kokoro_tts", "Wanted now.", "af_heart")
    speaking = asyncio.ensure_future(manager.speak("kokoro_tts", "Wanted now.", "af_heart", None, play))
    await until(lambda: worker.asked)
    assert worker.asked == ["Wanted now."]  # adopted by the reply: made at once, not left waiting for a turn
    worker.gate("Wanted now.").set()
    await speaking

    manager.prepare("kokoro_tts", "Never handed over.", "af_heart")
    for _ in range(20):
        await asyncio.sleep(0)
    assert worker.asked == ["Wanted now."]  # waits for the speech that follows it
    worker.gate("Something else.").set()
    await manager.speak("kokoro_tts", "Something else.", "af_heart", None, play)
    await until(lambda: len(worker.asked) == 3)
    assert worker.asked == ["Wanted now.", "Something else.", "Never handed over."]
    worker.gate("Never handed over.").set()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_pipe_serves_a_waiting_reply_before_the_next_sentence_and_both_before_the_store():
    pipe = _Pipe()
    await pipe.acquire(PIPE_AHEAD)  # held by a piece made for the store
    order: list[str] = []

    async def wait(name: str, rank) -> None:
        await pipe.acquire(rank)
        order.append(name)

    ahead = asyncio.ensure_future(wait("ahead", PIPE_AHEAD))
    nxt = asyncio.ensure_future(wait("next", PIPE_NEXT))
    promoted = {"wanted": False}
    late = asyncio.ensure_future(wait("next, then wanted", lambda: PIPE_NOW if promoted["wanted"] else PIPE_NEXT))
    now = asyncio.ensure_future(wait("now", PIPE_NOW))
    for _ in range(5):
        await asyncio.sleep(0)
    promoted["wanted"] = True  # a reply now waits for it: it ranks with a reply's piece, in the order they asked
    for _ in range(4):
        pipe.release()
        for _ in range(5):
            await asyncio.sleep(0)
    assert order == ["next, then wanted", "now", "next", "ahead"]
    assert pipe.locked()
    pipe.release()
    assert not pipe.locked()
    await asyncio.gather(ahead, nxt, late, now)


@pytest.mark.asyncio
async def test_a_waiter_cancelled_after_it_was_given_the_pipe_passes_it_on():
    pipe = _Pipe()
    await pipe.acquire()
    given = asyncio.ensure_future(pipe.acquire(PIPE_NOW))
    other = asyncio.ensure_future(pipe.acquire(PIPE_NEXT))
    gone = asyncio.ensure_future(pipe.acquire(PIPE_AHEAD))
    for _ in range(3):
        await asyncio.sleep(0)
    gone.cancel()  # cancelled while waiting: removed
    pipe.release()  # given to the first...
    given.cancel()  # ...which is cancelled before it runs: the pipe goes to the next
    with pytest.raises(asyncio.CancelledError):
        await given
    with pytest.raises(asyncio.CancelledError):
        await gone
    await other
    assert pipe.locked()
    pipe.release()
    assert not pipe.locked()


def test_the_bench_takes_the_first_piece_a_reply_waited_for_not_the_next_sentence():
    from benchmarks.conversation_bench.driver import synth_wait

    records = [("next", 12, 1.77, 1.77, 2.92), ("reply", 12, 1.79, 2.92, 4.14)]
    found = synth_wait(records, 0)
    assert found["synth_wait_s"] == round(2.92 - 1.79, 3) and found["behind_next_piece"] is True
    assert local_tts_worker.PIPE_NOW < local_tts_worker.PIPE_NEXT < local_tts_worker.PIPE_AHEAD

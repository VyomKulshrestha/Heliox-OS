"""A reload of Heliox's voice that a reply waits for comes before any work made ahead, and the worker says where a
load's time went.

In a live run on 24 November Heliox's own voice took 46.4 s to load again after a forced release, against 13.7 s for
the first load of the same run; nothing was on the worker's pipe meanwhile, so the time was inside the worker's load.
Work made ahead (fixed sentences for the store, the next sentence of a reply) now never starts the worker itself, and
the load answer carries its steps. Fake workers and a stand-in engine only: no model, no audio device.
"""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pilot.system import local_tts_worker, speech_trace
from pilot.system.local_tts_worker import PIPE_AHEAD, LocalTTSUnavailable, LocalTTSWorkerManager

RATE = 24000


class _Bytes(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = io.BytesIO()


def test_the_worker_says_how_long_each_step_of_a_load_took(monkeypatch):
    module = SimpleNamespace(prepare=lambda voice: None, last_load={"pipeline_s": 2.5, "warm_s": 0.4})
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: module)
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda *_threads: None)
    monkeypatch.setattr(
        local_tts_worker.sys, "stdin", io.StringIO(json.dumps({"op": "load", "voice": "af_heart"}) + "\n")
    )
    out = _Bytes()
    monkeypatch.setattr(local_tts_worker.sys, "stdout", out)
    assert local_tts_worker._serve("kokoro_tts") == 0
    head = json.loads(out.buffer.getvalue().split(b"\n", 1)[0])
    assert head["status"] == "ok" and head["op"] == "load"
    took = head["took"]
    assert {"waited_s", "import_s", "prepare_s"} <= set(took) and took["pipeline_s"] == 2.5 and took["warm_s"] == 0.4
    assert all(value >= 0 for value in took.values())


class RecordingWorker:
    """Answers every request at once, as the real worker does, and keeps what it was asked, in order."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self._pending = b""
        self.returncode: int | None = None
        worker = self

        class Stdin:
            closed = False

            def write(self, payload: bytes) -> None:
                worker._answer(json.loads(payload))

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True
                worker.returncode = 0

        class Stdout:
            def readline(self) -> bytes:
                line, _, rest = worker._pending.partition(b"\n")
                worker._pending = rest
                return line + b"\n"

            def read(self, size: int) -> bytes:
                part, worker._pending = worker._pending[:size], worker._pending[size:]
                return part

            def close(self) -> None:
                return None

        self.stdin, self.stdout = Stdin(), Stdout()

    def _answer(self, request: dict[str, Any]) -> None:
        if request.get("op") == "load":
            self.asked.append("load")
            took = {"waited_s": 0.2, "import_s": 3.0, "prepare_s": 9.0}
            self._pending += (json.dumps({"status": "ok", "op": "load", "took": took}) + "\n").encode()
            return
        self.asked.append(request["text"])
        payload = np.full(RATE // 10, 0.5, np.float32).tobytes()
        head = {"status": "ok", "rate": RATE, "samples": RATE // 10, "format": "f32le", "bytes": len(payload),
                "timing": None, "took": {"total_s": 0.0}}  # fmt: skip
        self._pending += (json.dumps(head) + "\n").encode() + payload

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_work_made_ahead_never_starts_the_voice_and_a_reply_s_reload_goes_first(monkeypatch, tmp_path):
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)  # this reply waits for the load
    workers: list[RecordingWorker] = []

    def spawn(engine, device, gpu_lease=False):
        workers.append(RecordingWorker())
        return workers[-1]

    manager = LocalTTSWorkerManager(spawn=spawn)
    manager.environ = {}
    stamps: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(speech_trace, "sink", lambda event, data: stamps.append((event, dict(data))))

    # The voice was released: a fixed sentence for the store and the next sentence of a reply come first.
    with pytest.raises(LocalTTSUnavailable):
        await manager._make(
            "kokoro_tts", "Okay.", "af_heart", tmp_path / "a.wav", tmp_path, rank=PIPE_AHEAD, kind="ahead",
            start_voice=False,
        )  # fmt: skip
    manager.prepare("kokoro_tts", "And the second.", "af_heart")
    for _ in range(10):
        await asyncio.sleep(0)
    assert workers == []  # neither started the voice

    async def play(piece) -> None:
        return None

    await manager.speak("kokoro_tts", "The first.", "af_heart", None, play)  # the reply loads it
    for _ in range(50):
        if len(workers[0].asked) == 3:
            break
        await asyncio.sleep(0.01)
    assert len(workers) == 1 and workers[0].asked == ["load", "The first.", "And the second."]
    [loaded] = [data for event, data in stamps if event == "voice loaded"]
    assert loaded["worker_import_s"] == 3.0 and loaded["worker_prepare_s"] == 9.0 and loaded["asked"] <= loaded["done"]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_next_sentence_that_could_not_be_made_ahead_is_made_for_the_reply_that_wants_it(monkeypatch):
    """Prepared while the voice was not loaded, it gave up; the reply that reaches it makes it itself."""
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)
    worker = RecordingWorker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.environ = {}

    async def play(piece) -> None:
        return None

    manager.prepare("kokoro_tts", "Second.", "af_heart")
    manager._speech_made(10**6)  # its turn has come, with the voice still not loaded: it gives up
    for _ in range(20):
        await asyncio.sleep(0)
    await manager.speak("kokoro_tts", "Second.", "af_heart", None, play)
    assert worker.asked == ["load", "Second."]
    await manager.shutdown()

"""Every stage of a piece made now is stamped, as plain numbers, so a run on any computer names where the time went
between the sentence being handed over and its samples being ready: the request written to the pipe, read by the
worker, taken by its loop, the synthesis and trim, the answer written, read back here, and the event loop back on it.

A live run with the voice on the graphics card left about 140 ms between "first sentence to the voice" and its first
sound unaccounted for (0.25 s in all), because the bench's sink keeps only plain numbers and the worker's own step
times came as a dict. Stand-in engine and stand-in worker only: no model, no card, no sound device.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pilot.system import local_tts_worker, speech_trace

RATE = 24000
# The worker's loop configures the Kokoro module in this process; it is put back after each test (conftest).
pytestmark = pytest.mark.usefixtures("kokoro_module_restored")


class _Bytes(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = io.BytesIO()


def test_the_worker_says_when_it_read_took_and_answered_each_request(monkeypatch, tmp_path):
    module = SimpleNamespace(
        generate_here=lambda text, voice, *, timing=False: (np.full(2400, 0.5, np.float32), RATE, None)
    )
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: module)
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda *_threads: None)
    request = {"text": "Hi.", "voice": "af_heart", "output_file": str(tmp_path / "o.wav"), "reply": "samples"}
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    out = _Bytes()
    monkeypatch.setattr(local_tts_worker.sys, "stdout", out)
    assert local_tts_worker._serve("kokoro_tts") == 0
    head = json.loads(out.buffer.getvalue().split(b"\n", 1)[0])
    at = head["at"]
    assert 0 < at["got"] <= at["began"] <= at["sent"]  # read off the pipe, taken by the loop, answered: in that order


class _StampedWorker:
    """Answers with samples and the worker's own stamps, as the real worker does."""

    def __init__(self) -> None:
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
        import time

        if request.get("op") == "load":
            self._pending += b'{"status": "ok", "op": "load"}\n'
            return
        got = time.monotonic()
        payload = np.full(RATE // 4, 0.5, np.float32).tobytes()
        head = {"status": "ok", "rate": RATE, "samples": RATE // 4, "format": "f32le", "bytes": len(payload),
                "timing": None, "took": {"read_s": 0.0, "synth_s": 0.0, "trim_s": 0.0, "total_s": 0.0},
                "at": {"got": got, "began": got, "sent": time.monotonic()}}  # fmt: skip
        self._pending += (json.dumps(head) + "\n").encode() + payload

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_each_stage_of_a_piece_made_now_is_a_plain_number_in_order(monkeypatch):
    worker = _StampedWorker()
    manager = local_tts_worker.LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.environ = {}
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)
    stamps: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(speech_trace, "sink", lambda event, data: stamps.append((event, dict(data))))

    async def play(piece: Any) -> None:
        return None

    await manager.speak("kokoro_tts", "Hello there.", "af_heart", None, play)
    await manager.shutdown()
    [made] = [data for event, data in stamps if event == "piece synthesized"]
    order = ["requested", "worker_started", "worker_got", "worker_began", "worker_sent", "answer_line_at",
             "answer_read_at", "resumed_at", "done"]  # fmt: skip
    values = [made[name] for name in order]
    assert all(isinstance(value, float) for value in values)
    assert values == sorted(values)  # each stage after the one before it, on the one clock
    # The write returns once the worker has the line, which it may already have read: after the pipe was taken.
    assert made["worker_started"] <= made["sent_at"] <= made["answer_line_at"]
    assert made["worker_synth_s"] == 0.0 and made["payload_bytes"] == 4 * (RATE // 4)
    assert made["kind"] == "reply"


@pytest.mark.asyncio
async def test_a_worker_that_gives_no_stamps_leaves_them_empty(monkeypatch):
    worker = _StampedWorker()
    original = worker._answer

    def without_stamps(request: dict[str, Any]) -> None:
        original(request)
        if request.get("op") != "load":
            line, _, rest = worker._pending.partition(b"\n")
            head = json.loads(line)
            head.pop("at")
            worker._pending = (json.dumps(head) + "\n").encode() + rest

    worker._answer = without_stamps  # type: ignore[method-assign]
    manager = local_tts_worker.LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.environ = {}
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)
    stamps: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(speech_trace, "sink", lambda event, data: stamps.append((event, dict(data))))

    async def play(piece: Any) -> None:
        return None

    await manager.speak("kokoro_tts", "Hello there.", "af_heart", None, play)
    await manager.shutdown()
    [made] = [data for event, data in stamps if event == "piece synthesized"]
    assert made["worker_got"] is None and made["worker_sent"] is None and isinstance(made["answer_read_at"], float)

"""The first sound of a made piece: the worker sends its samples back on the pipe, already trimmed, with the phoneme
timing in its answer; nothing is written to disk before the person hears it, the player plays the samples as they
are, and the face's mouth data is made once they are on the output line. The store still gets its copy, after the
piece was said.

The audio packages are optional and absent in CI: the worker's engine, the worker process and the output device are
fakes. Order is checked on ordered logs, never on clock stamps."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pilot.system import local_tts_worker, speech_notices, speech_trace, voice

RATE = 24000


def _made(lead: int = 1000, sound: int = 2400, tail: int = 12000) -> np.ndarray:
    """A made piece: quiet lead, sound, quiet tail (as Kokoro pads every piece)."""
    return np.concatenate([np.zeros(lead, np.float32), np.full(sound, 0.5, np.float32), np.zeros(tail, np.float32)])


class _Bytes:
    """stdout with a binary buffer, as the worker's own."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def write(self, text: str) -> None:
        self.buffer.write(text.encode("utf-8"))

    def flush(self) -> None:
        return None


def _answers(raw: bytes) -> list[tuple[dict[str, Any], bytes]]:
    out = []
    stream = io.BytesIO(raw)
    while True:
        line = stream.readline()
        if not line:
            return out
        head = json.loads(line)
        out.append((head, stream.read(int(head.get("bytes") or 0))))


def test_the_worker_sends_the_samples_trimmed_with_their_timing_and_writes_nothing(monkeypatch, tmp_path):
    timing = {"rate": RATE, "samples": 15400, "phonemes": [["h", 0.0417, 0.06], ["i", 0.06, 0.14]], "words": []}
    asked: list[tuple[str, str, bool]] = []

    def generate_here(text, voice_name, *, timing=False):
        asked.append((text, voice_name, timing))
        return _made(), RATE, (dict(timing_made) if timing else None)

    timing_made = timing
    module = SimpleNamespace(generate_here=generate_here)
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: module)
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda: None)
    text_file = tmp_path / "t.txt"
    text_file.write_text("Hi.", encoding="utf-8")
    request = {"text_file": str(text_file), "voice": "af_heart", "output_file": str(tmp_path / "o.wav"),
               "reply": "samples", "keep_lead": 0.01, "keep_tail": None,
               "timing_file": str(tmp_path / "o.wav.timing.json")}  # fmt: skip
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    out = _Bytes()
    monkeypatch.setattr(local_tts_worker.sys, "stdout", out)

    assert local_tts_worker._serve("kokoro_tts") == 0

    [(head, payload)] = _answers(out.buffer.getvalue())
    assert head["status"] == "ok" and head["rate"] == RATE and head["format"] == "f32le"
    samples = np.frombuffer(payload, dtype="<f4")
    lead = int(0.01 * RATE)
    assert len(samples) == head["samples"] == 15400 - (1000 - lead)  # the quiet lead cut, the end kept (a sentence)
    assert samples[lead] == pytest.approx(0.5) and not samples[:lead].any()
    assert head["timing"]["samples"] == len(samples)
    assert head["timing"]["phonemes"][0][1] == pytest.approx(0.0417 - (1000 - lead) / RATE, abs=1e-4)
    assert set(head["took"]) >= {"read_s", "synth_s", "trim_s", "total_s"}
    assert asked == [("Hi.", "af_heart", True)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt"]  # no WAV and no timing file written


def test_a_clause_piece_keeps_only_its_short_tail(monkeypatch, tmp_path):
    module = SimpleNamespace(generate_here=lambda text, voice_name, *, timing=False: (_made(), RATE, None))
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: module)
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda: None)
    text_file = tmp_path / "t.txt"
    text_file.write_text("When you are ready", encoding="utf-8")
    request = {"text_file": str(text_file), "voice": "af_heart", "output_file": str(tmp_path / "o.wav"),
               "reply": "samples", "keep_lead": 0.0, "keep_tail": 0.1}  # fmt: skip
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    out = _Bytes()
    monkeypatch.setattr(local_tts_worker.sys, "stdout", out)
    local_tts_worker._serve("kokoro_tts")
    [(head, payload)] = _answers(out.buffer.getvalue())
    assert head["samples"] == 2400 + int(0.1 * RATE) and head["timing"] is None
    assert len(payload) == 4 * head["samples"]


def test_a_failed_piece_sends_no_bytes(monkeypatch, tmp_path):
    def broken(text, voice_name, *, timing=False):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: SimpleNamespace(generate_here=broken))
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda: None)
    text_file = tmp_path / "t.txt"
    text_file.write_text("Hi.", encoding="utf-8")
    request = {"text_file": str(text_file), "voice": "af_heart", "output_file": str(tmp_path / "o.wav"),
               "reply": "samples"}  # fmt: skip
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    out = _Bytes()
    monkeypatch.setattr(local_tts_worker.sys, "stdout", out)
    local_tts_worker._serve("kokoro_tts")
    [(head, payload)] = _answers(out.buffer.getvalue())
    assert head["status"] == "error" and "CUDA" in head["error"] and payload == b""


# ── The daemon's side ─────────────────────────────────────────────────


class _SamplesWorker:
    """A worker process that answers speech with samples on its pipe, as the real one does."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
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

        self.stdin = Stdin()
        self.stdout = Stdout()

    def _answer(self, request: dict[str, Any]) -> None:
        self.requests.append(request)
        if request.get("op") == "load":
            self._pending += b'{"status": "ok", "op": "load"}\n'
            return
        text = Path(request["text_file"]).read_text(encoding="utf-8")
        audio = np.full(RATE // 2, 0.25 if text.startswith("Hello") else 0.5, np.float32)
        payload = audio.tobytes()
        head = {"status": "ok", "rate": RATE, "samples": len(audio), "format": "f32le", "bytes": len(payload),
                "timing": None, "took": {"read_s": 0.0, "synth_s": 0.1, "trim_s": 0.0, "total_s": 0.1}}  # fmt: skip
        self._pending += (json.dumps(head) + "\n").encode() + payload

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


@pytest.fixture
def manager(monkeypatch):
    fresh = local_tts_worker.LocalTTSWorkerManager()
    fresh.environ = {}  # a made-up machine with every card visible; the test process keeps its own hidden
    monkeypatch.setattr(local_tts_worker, "_manager", fresh)
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)  # the grace is not what is tested here
    return fresh


@pytest.mark.asyncio
async def test_a_reply_plays_from_memory_and_no_file_is_written(manager, monkeypatch):
    worker = _SamplesWorker()
    manager._spawn = lambda engine, device, gpu_lease=False: worker
    played: list[Any] = []
    stamps: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(speech_trace, "sink", lambda event, data: stamps.append((event, data)))

    async def play(piece: Any) -> None:
        played.append(piece)

    await manager.speak("kokoro_tts", "Hello there. Second one.", "af_heart", None, play)
    await manager.shutdown()

    assert [p.text for p in played] == ["Hello there.", "Second one."]
    assert all(p.path is None and p.audio is not None and p.rate == RATE for p in played)
    assert played[0].audio[0] == pytest.approx(0.25) and played[1].audio[0] == pytest.approx(0.5)
    speech = [r for r in worker.requests if "text_file" in r]
    assert all(r["reply"] == "samples" and not Path(r["output_file"]).exists() for r in speech)
    assert speech[0]["keep_lead"] == local_tts_worker.KEEP_LEAD_SECONDS and speech[0]["keep_tail"] is None
    made = [data for event, data in stamps if event == "piece synthesized"]
    assert [d["in_memory"] for d in made] == [True, True] and made[0]["took"]["synth_s"] == 0.1


def test_the_store_keeps_a_piece_from_memory_after_it_was_said_twice(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    store = local_tts_worker.SpeechStore(tmp_path)
    signature = local_tts_worker.SpeechStore.signature("kokoro_tts", "af_heart")
    piece = local_tts_worker.Piece("Okay.", audio=np.full(2400, 0.5, np.float32), rate=RATE)
    assert store.said("Okay.", signature, piece) is False  # once: counted only
    assert store.said("Okay.", signature, piece) is True  # twice: kept
    kept = store.lookup("Okay.", signature)
    audio, rate = soundfile.read(str(kept), dtype="float32")
    assert rate == RATE and len(audio) == 2400 and audio[0] == pytest.approx(0.5, abs=1e-3)
    assert store.add_fixed("Done.", signature, piece) and store.has("Done.", signature)


# ── The player ────────────────────────────────────────────────────────


class FakeStream:
    instances: list[FakeStream] = []

    def __init__(self, samplerate, channels, dtype, callback, latency=None):
        self.callback = callback
        self.channels = channels
        self.latency = 0.0
        self.blocks: list[np.ndarray] = []
        self._stop = threading.Event()
        FakeStream.instances.append(self)

    def start(self):
        def run():
            while not self._stop.is_set():
                out = np.zeros((480, self.channels), dtype=np.float32)
                self.callback(out, 480, None, None)
                self.blocks.append(out.copy())
                time.sleep(0.005)

        threading.Thread(target=run, daemon=True).start()

    def abort(self):
        self._stop.set()

    def close(self):
        self._stop.set()


@pytest.fixture
def line(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    monkeypatch.setattr(voice, "_output_line", voice._OutputLine())
    yield
    voice._output_line.close()


@pytest.mark.asyncio
async def test_samples_in_memory_play_without_reading_a_file_and_the_mouth_is_made_after_they_are_on_the_line(
    line, monkeypatch
):
    def no_file(_path):
        raise AssertionError("a piece made in memory is never read from disk")

    monkeypatch.setattr(voice, "_read_audio_file", no_file)
    log: list[str] = []
    sent: list[tuple[str, dict[str, Any]]] = []
    real_mouth = speech_notices.mouth_for

    def mouth_for(path, audio, rate, timing=None):
        log.append("mouth, on the line" if voice._output_line.current is not None else "mouth, before the line")
        assert path is None and timing == {"rate": RATE, "samples": 4800, "phonemes": [["a", 0.0, 0.2]]}
        return real_mouth(path, audio, rate, timing)

    monkeypatch.setattr(speech_notices, "mouth_for", mouth_for)

    async def send(method, params):
        sent.append((method, params))

    speech_notices.install(send, face_on=lambda: True)
    try:
        piece = local_tts_worker.Piece(
            "Ah.", audio=np.full(4800, 0.5, np.float32), rate=RATE,
            timing={"rate": RATE, "samples": 4800, "phonemes": [["a", 0.0, 0.2]]},
        )  # fmt: skip
        await voice._play_local_tts_file(piece)
        for _ in range(400):
            if any(m == "voice_end" for m, _p in sent):
                break
            await asyncio.sleep(0.005)
    finally:
        task = speech_notices._hub.task
        speech_notices.uninstall()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    assert log == ["mouth, on the line"]
    assert any(block.any() for block in FakeStream.instances[-1].blocks)
    [piece_notice] = [p for m, p in sent if m == "voice_piece"]
    assert piece_notice["mouth"] == "visemes" and piece_notice["seconds"] == pytest.approx(0.2)


def test_the_time_spent_making_the_mouth_comes_off_the_lead(monkeypatch):
    clock = iter([10.0, 10.03])
    monkeypatch.setattr(speech_notices.time, "monotonic", lambda: next(clock))
    sent: list[dict[str, Any]] = []
    reply = SimpleNamespace(take_index=lambda: 0, piece=lambda params: sent.append(params) or True, voice="")
    token = speech_notices._current.set(reply)  # type: ignore[arg-type]
    monkeypatch.setattr(speech_notices, "installed", lambda: True)
    try:
        hook = speech_notices.handed(lambda: {"mouth": "envelope", "envelope": [1], "envelope_rate": 25}, 0.5)
        assert hook is not None
        hook(0.09)
    finally:
        speech_notices._current.reset(token)
    assert sent[0]["lead_ms"] == pytest.approx(60.0) and sent[0]["mouth"] == "envelope"


# ── The store keeps the timing ────────────────────────────────────────

TIMING = {"rate": RATE, "samples": 4800, "phonemes": [["O", 0.0, 0.1], ["k", 0.1, 0.15], ["eI", 0.15, 0.2]],
          "words": [["Okay", 0.0, 0.2]]}  # fmt: skip


def test_a_kept_piece_carries_its_timing_and_plays_to_the_face_as_a_fresh_one(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    monkeypatch.setattr(speech_notices, "face_on", lambda: True)
    store = local_tts_worker.SpeechStore(tmp_path)
    signature = local_tts_worker.SpeechStore.signature("kokoro_tts", "af_heart")
    audio = np.full(4800, 0.5, np.float32)
    piece = local_tts_worker.Piece("Okay.", audio=audio, rate=RATE, timing=TIMING)
    assert store.add_fixed("Okay.", signature, piece)
    kept = store.lookup("Okay.", signature)
    assert json.loads(speech_notices.timing_path(kept).read_text(encoding="utf-8")) == TIMING
    fresh = speech_notices.mouth_for(None, audio, RATE, TIMING)
    _audio, _rate, from_store = voice._read_piece(kept)
    assert from_store["mouth"] == fresh["mouth"] == "visemes"
    assert from_store["visemes"] == fresh["visemes"] and from_store["words"] == fresh["words"]


def test_an_old_kept_piece_without_timing_still_plays_with_the_envelope(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    monkeypatch.setattr(speech_notices, "face_on", lambda: True)
    store = local_tts_worker.SpeechStore(tmp_path)
    signature = local_tts_worker.SpeechStore.signature("kokoro_tts", "af_heart")
    piece = local_tts_worker.Piece("Done.", audio=np.full(4800, 0.5, np.float32), rate=RATE)
    assert store.add_fixed("Done.", signature, piece)
    kept = store.lookup("Done.", signature)
    assert not speech_notices.timing_path(kept).exists()
    assert voice._read_piece(kept)[2]["mouth"] == "envelope"
    store.keep_only("another voice")  # dropped with its timing
    assert not kept.exists()


def test_a_file_from_an_older_worker_is_kept_with_the_timing_file_beside_it(tmp_path):
    store = local_tts_worker.SpeechStore(tmp_path / "store")
    signature = local_tts_worker.SpeechStore.signature("kokoro_tts", "af_heart")
    made = tmp_path / "made.wav"
    made.write_bytes(b"RIFF-made")
    speech_notices.timing_path(made).write_text(json.dumps(TIMING), encoding="utf-8")
    assert store.add_fixed("Okay.", signature, made)
    kept = store.lookup("Okay.", signature)
    assert json.loads(speech_notices.timing_path(kept).read_text(encoding="utf-8")) == TIMING
    for _ in range(2):
        store.said("Hi.", signature, made)
    assert speech_notices.timing_path(store.lookup("Hi.", signature)).is_file()
    store._drop(store.key("Hi.", signature))
    assert not list((tmp_path / "store").glob(f"{store.key('Hi.', signature)}*"))


@pytest.mark.asyncio
async def test_speech_always_asks_for_the_timing_so_the_store_has_it_for_later(manager):
    worker = _SamplesWorker()
    manager._spawn = lambda engine, device, gpu_lease=False: worker

    async def play(piece: Any) -> None:
        return None

    await manager.speak("kokoro_tts", "Hello there.", "af_heart", None, play)  # no face shown
    await manager.shutdown()
    speech = [r for r in worker.requests if "text_file" in r]
    assert speech and all(r["timing"] is True and "timing_file" not in r for r in speech)

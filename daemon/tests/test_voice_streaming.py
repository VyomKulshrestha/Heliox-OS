"""The streamed voice (Pocket TTS): the worker sends each chunk as the engine makes it, playback starts on the first
chunk and appends the rest, a cut stops the stream within one chunk, the finished piece goes to the store, and the
window hears of every part as it goes to the line.

The audio packages are optional and absent in CI: the engine, the worker process and the output device are fakes.
Order is checked on ordered logs and events, never on clock stamps."""

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

from pilot.system import local_tts_worker, speech_notices, voice

RATE = 24000
CHUNK = 1920  # 80 ms, as Pocket's


def _frames(raw: bytes) -> list[tuple[dict[str, Any], bytes]]:
    out = []
    stream = io.BytesIO(raw)
    while True:
        line = stream.readline()
        if not line:
            return out
        head = json.loads(line)
        out.append((head, stream.read(int(head.get("bytes") or 0))))


class _Bytes:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def write(self, text: str) -> None:
        self.buffer.write(text.encode("utf-8"))

    def flush(self) -> None:
        return None


class _Lines:
    """stdin that gives its lines one at a time, each when the test allows it."""

    def __init__(self, lines: list[tuple[str, threading.Event | None]]) -> None:
        self.lines = lines

    def __iter__(self):
        for line, gate in self.lines:
            if gate is not None:
                assert gate.wait(5), "the test never let this line through"
            yield line + "\n"


def _serve_with(monkeypatch, engine_module: Any, stdin: Any) -> list[tuple[dict[str, Any], bytes]]:
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: engine_module)
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda *_threads: None)
    monkeypatch.setattr(local_tts_worker.sys, "stdin", stdin)
    out = _Bytes()
    monkeypatch.setattr(local_tts_worker.sys, "stdout", out)
    assert local_tts_worker._serve("pocket_tts") == 0
    return _frames(out.buffer.getvalue())


def _text(tmp_path: Path, text: str = "Hello there.") -> str:
    path = tmp_path / "t.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_worker_sends_each_chunk_as_it_is_made_then_an_end(monkeypatch, tmp_path):
    def stream_here(text, voice_name, stop):
        return RATE, (np.full(CHUNK, k / 10, np.float32) for k in range(1, 4))

    request = {"op": "stream", "id": 7, "text_file": _text(tmp_path), "voice": "alba"}
    frames = _serve_with(monkeypatch, SimpleNamespace(stream_here=stream_here), io.StringIO(json.dumps(request) + "\n"))

    assert [h["frame"] for h, _ in frames] == ["audio", "audio", "audio", "end"]
    assert [h["seq"] for h, _ in frames[:3]] == [0, 1, 2] and all(
        h["id"] == 7 and h["rate"] == RATE for h, _ in frames[:3]
    )
    assert np.frombuffer(frames[1][1], dtype="<f4")[0] == pytest.approx(0.2)
    end = frames[-1][0]
    assert end["chunks"] == 3 and end["cut"] is False and end["audio_s"] == pytest.approx(3 * CHUNK / RATE)
    assert "error" not in end


def test_a_stop_ends_the_stream_before_its_next_chunk(monkeypatch, tmp_path):
    two_out = threading.Event()
    asked: list[int] = []

    def stream_here(text, voice_name, stop):
        def chunks():
            for k in range(100):
                if stop.is_set():
                    return  # as PocketStream reads it before asking the model for each chunk
                asked.append(k)
                yield np.full(CHUNK, 0.5, np.float32)
                if k == 1:
                    two_out.set()
                    assert stop.wait(5), "the stop never arrived"

        return RATE, chunks()

    request = {"op": "stream", "id": 3, "text_file": _text(tmp_path), "voice": "alba"}
    stdin = _Lines([(json.dumps(request), None), (json.dumps({"op": "stop", "id": 3}), two_out)])
    frames = _serve_with(monkeypatch, SimpleNamespace(stream_here=stream_here), stdin)

    assert [h["frame"] for h, _ in frames] == ["audio", "audio", "end"]
    assert frames[-1][0]["cut"] is True and frames[-1][0]["chunks"] == 2
    assert asked == [0, 1]  # nothing more was asked of the model after the stop


def test_a_stop_that_comes_first_stops_the_stream_before_any_chunk(monkeypatch, tmp_path):
    def stream_here(text, voice_name, stop):
        return RATE, (np.full(CHUNK, 0.5, np.float32) for _ in range(5) if not stop.is_set())

    request = {"op": "stream", "id": 4, "text_file": _text(tmp_path), "voice": "alba"}
    lines = json.dumps({"op": "stop", "id": 4}) + "\n" + json.dumps(request) + "\n"
    frames = _serve_with(monkeypatch, SimpleNamespace(stream_here=stream_here), io.StringIO(lines))
    assert [h["frame"] for h, _ in frames] == ["end"] and frames[0][0]["cut"] is True


def test_an_engine_that_cannot_stream_says_so_in_the_end_frame(monkeypatch, tmp_path):
    request = {"op": "stream", "id": 1, "text_file": _text(tmp_path), "voice": "af_heart"}
    frames = _serve_with(monkeypatch, SimpleNamespace(), io.StringIO(json.dumps(request) + "\n"))
    assert frames[0][0]["frame"] == "end" and "does not stream" in frames[0][0]["error"]


def test_pocket_whole_pieces_are_the_stream_joined_and_its_stream_uses_the_shared_model(monkeypatch):
    from pilot.system import pocket_tts

    made: list[tuple[str, Any, str]] = []

    class FakeStreams:
        sample_rate = RATE

        def stream(self, text, stop=None, *, voice=""):
            made.append((text, stop, voice))
            return iter([np.full(CHUNK, 0.1, np.float32), np.full(CHUNK, 0.2, np.float32)])

    monkeypatch.setattr(pocket_tts, "_streams", FakeStreams())
    audio, rate, timing = pocket_tts.generate_here("Hi.", "alba")
    assert rate == RATE and timing is None and len(audio) == 2 * CHUNK and audio[-1] == pytest.approx(0.2)
    stop = threading.Event()
    rate, chunks = pocket_tts.stream_here("Again.", "alba", stop)
    assert rate == RATE and len(list(chunks)) == 2 and made[-1] == ("Again.", stop, "alba")


# ── The daemon's side ─────────────────────────────────────────────────


class _StreamingWorker:
    """A worker process that streams: each chunk is sent when the test releases it (``release``), and a stop on
    stdin ends the stream before the next one, as the real worker does."""

    def __init__(self, chunks: int = 4) -> None:
        self.chunks = chunks
        self.requests: list[dict[str, Any]] = []
        self.made: list[tuple[int, int]] = []  # (stream id, chunk) in the order sent
        self.returncode: int | None = None
        self._out = bytearray()
        self._cond = threading.Condition()
        self._stops: dict[int, threading.Event] = {}
        self.allowed = threading.Semaphore(0)
        worker = self

        class Stdin:
            closed = False

            def write(self, payload: bytes) -> None:
                for line in payload.decode("utf-8").splitlines():
                    worker._got(json.loads(line))

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True
                worker._push(b"")

        class Stdout:
            def readline(self) -> bytes:
                with worker._cond:
                    worker._cond.wait_for(lambda: b"\n" in worker._out or worker.returncode is not None, 5)
                    if b"\n" not in worker._out:
                        return b""
                    index = worker._out.index(b"\n") + 1
                    line = bytes(worker._out[:index])
                    del worker._out[:index]
                    return line

            def read(self, size: int) -> bytes:
                with worker._cond:
                    worker._cond.wait_for(lambda: len(worker._out) >= size, 5)
                    part = bytes(worker._out[:size])
                    del worker._out[:size]
                    return part

            def close(self) -> None:
                return None

        self.stdin = Stdin()
        self.stdout = Stdout()

    def release(self, count: int = 1) -> None:
        for _ in range(count):
            self.allowed.release()

    def _push(self, data: bytes) -> None:
        with self._cond:
            self._out += data
            self._cond.notify_all()

    def _got(self, request: dict[str, Any]) -> None:
        self.requests.append(request)
        if request.get("op") == "stop":
            self._stops.setdefault(int(request["id"]), threading.Event()).set()
            self.allowed.release()  # wake a stream waiting for its next chunk
            return
        if request.get("op") == "load":
            self._push(b'{"status": "ok", "op": "load"}\n')
            return
        if request.get("op") == "stream":
            stop = self._stops.setdefault(int(request["id"]), threading.Event())
            threading.Thread(target=self._stream, args=(int(request["id"]), stop), daemon=True).start()
            return
        audio = np.full(RATE // 4, 0.5, np.float32)  # a whole piece (made ahead for the store)
        head = {"status": "ok", "rate": RATE, "samples": len(audio), "format": "f32le", "bytes": 4 * len(audio),
                "timing": None}  # fmt: skip
        self._push((json.dumps(head) + "\n").encode() + audio.tobytes())

    def _stream(self, stream_id: int, stop: threading.Event) -> None:
        sent = 0
        for k in range(self.chunks):
            assert self.allowed.acquire(timeout=5), "the test never released the next chunk"
            if stop.is_set():
                break
            data = np.full(CHUNK, 0.1 * (k + 1), np.float32).tobytes()
            head = {"frame": "audio", "id": stream_id, "seq": k, "rate": RATE, "bytes": len(data)}
            self.made.append((stream_id, k))
            self._push((json.dumps(head) + "\n").encode() + data)
            sent += 1
        end = {"frame": "end", "id": stream_id, "chunks": sent, "cut": stop.is_set(), "audio_s": sent * CHUNK / RATE}
        self._push((json.dumps(end) + "\n").encode())

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        self._push(b"")
        return 0


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
                time.sleep(0.002)

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


@pytest.fixture
def manager(monkeypatch):
    fresh = local_tts_worker.LocalTTSWorkerManager()
    fresh.environ = {}  # a made-up machine with every card visible; the test process keeps its own hidden
    monkeypatch.setattr(local_tts_worker, "_manager", fresh)
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)  # the grace is not what is tested here
    return fresh


async def _until(condition, what: str) -> None:
    for _ in range(4000):
        if condition():
            return
        await asyncio.sleep(0.002)
    raise AssertionError(f"never happened: {what}")


def _voiced() -> list[float]:
    """The first sample of each voiced block the fake device played, in order."""
    blocks = FakeStream.instances[-1].blocks if FakeStream.instances else []
    return [float(block[np.flatnonzero(block[:, 0])[0], 0]) for block in list(blocks) if block.any()]


@pytest.mark.asyncio
async def test_playback_starts_on_the_first_chunk_and_appends_the_rest(manager, line):
    worker = _StreamingWorker(chunks=4)
    spawned: list[tuple[str, str]] = []

    def spawn(engine, device, gpu_lease=False):
        spawned.append((device, gpu_lease))
        return worker

    manager._spawn = spawn
    manager.choose_device = lambda device: ("cuda", True)  # the arbiter would give a card: Pocket does not take it
    worker.release()
    speaking = asyncio.ensure_future(
        manager.speak(
            "pocket_tts",
            "A sentence long enough to be cut at a comma, if it were Kokoro speaking it.",
            "alba",
            None,
            voice._play_local_tts_file,
        )  # fmt: skip
    )
    await _until(lambda: _voiced(), "the first chunk heard")
    assert worker.made == [(1, 0)]  # heard while the rest is still to be made
    worker.release(3)
    await speaking
    await manager.shutdown()

    heard = _voiced()
    assert heard[0] == pytest.approx(0.1) and heard[-1] == pytest.approx(0.4)  # every chunk, in order
    assert sorted(set(round(v, 1) for v in heard)) == [0.1, 0.2, 0.3, 0.4]
    streams = [r for r in worker.requests if r.get("op") == "stream"]
    assert len(streams) == 1  # one sentence, one stream: no clause cut for a streamed engine
    assert spawned == [("cpu", False)] and manager.device_reason.startswith("On the processor")
    progress = manager.last_progress
    assert progress is not None and progress.finished and progress.heard_words() == 16


@pytest.mark.asyncio
async def test_the_next_sentence_streams_while_this_one_plays(manager, line):
    worker = _StreamingWorker(chunks=2)
    manager._spawn = lambda engine, device, gpu_lease=False: worker
    worker.release(2)
    speaking = asyncio.ensure_future(manager.speak("pocket_tts", "First one. Second one.", "alba", None,
                                                   voice._play_local_tts_file))  # fmt: skip
    await _until(lambda: len([r for r in worker.requests if r.get("op") == "stream"]) == 2, "the second stream")
    worker.release(2)
    await speaking
    await manager.shutdown()
    assert worker.made == [(1, 0), (1, 1), (2, 0), (2, 1)]


@pytest.mark.asyncio
async def test_a_cut_stops_the_stream_within_a_chunk_and_the_pipe_is_free_at_once(manager, line, monkeypatch):
    worker = _StreamingWorker(chunks=50)
    manager._spawn = lambda engine, device, gpu_lease=False: worker
    sent: list[tuple[str, dict[str, Any]]] = []

    async def send(method, params):
        sent.append((method, params))

    speech_notices.install(send, face_on=lambda: True)
    try:
        worker.release(2)
        long = "Here is a long answer that the person talks over."
        speaking = asyncio.ensure_future(_noticed(manager.speak("pocket_tts", long, "alba", None,
                                                                voice._play_local_tts_file)))  # fmt: skip
        await _until(lambda: len(_voiced()) > 0 and len(worker.made) == 2, "two chunks made and the first heard")
        speaking.cancel()
        with pytest.raises(asyncio.CancelledError):
            await speaking
        await _until(lambda: any(r.get("op") == "stop" for r in worker.requests), "the stop sent to the worker")
        # The pipe is free again as soon as the worker has ended the stream: the next reply streams at once.
        nxt = asyncio.ensure_future(manager.speak("pocket_tts", "Okay.", "alba", None, voice._play_local_tts_file))
        await _until(lambda: len([r for r in worker.requests if r.get("op") == "stream"]) == 2, "the next stream")
        worker.release(60)
        await nxt
        await _until(lambda: any(m == "voice_cut" for m, _p in sent), "the cut notice")
    finally:
        task = speech_notices._hub.task
        speech_notices.uninstall()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await manager.shutdown()
    first = [k for stream_id, k in worker.made if stream_id == 1]
    assert first == [0, 1]  # nothing more was made of the cut sentence
    cut_reply = next(p["reply"] for m, p in sent if m == "voice_cut")
    of_cut = [m for m, p in sent if p.get("reply") == cut_reply]
    assert of_cut[0] == "voice_piece" and of_cut[-1] == "voice_cut" and "voice_end" not in of_cut
    pieces = [p for m, p in sent if m == "voice_piece" and p["reply"] == cut_reply]
    chunks_in_first = pieces[0]["seconds"] * RATE / CHUNK  # the chunks already in when it started: one or two
    assert pieces[0]["mouth"] == "envelope" and round(chunks_in_first) in (1, 2)
    assert chunks_in_first == pytest.approx(round(chunks_in_first), abs=0.01)
    assert [p["index"] for p in pieces] == list(range(len(pieces)))


async def _noticed(coro):
    with speech_notices.reply():
        return await coro


@pytest.mark.asyncio
async def test_a_streamed_sentence_said_twice_is_kept_in_the_store(manager, line, tmp_path):
    pytest.importorskip("soundfile")
    worker = _StreamingWorker(chunks=2)
    manager._spawn = lambda engine, device, gpu_lease=False: worker
    manager.head_cache = local_tts_worker.SpeechStore(tmp_path)
    for _ in range(2):
        worker.release(2)
        await manager.speak("pocket_tts", "Sure.", "alba", None, voice._play_local_tts_file)
    signature = local_tts_worker.SpeechStore.signature("pocket_tts", "alba")
    assert manager.head_cache.has("Sure.", signature)
    assert not manager.head_cache.has("Sure.", local_tts_worker.SpeechStore.signature("kokoro_tts", "alba"))
    streams = len([r for r in worker.requests if r.get("op") == "stream"])
    await manager.speak("pocket_tts", "Sure.", "alba", None, voice._play_local_tts_file)  # from the store now
    assert len([r for r in worker.requests if r.get("op") == "stream"]) == streams
    await manager.shutdown()


def test_what_was_heard_of_a_stream_still_arriving_is_never_more_than_was_played():
    text = "one two three four five six seven eight nine ten"
    progress = local_tts_worker.SpeechProgress(text, [text])
    # 1.2 s received, 1.0 s played, the end not here yet: placed over the received audio, nine words would count.
    progress.playing = SimpleNamespace(received=int(1.2 * RATE), position=int(1.0 * RATE), sample_rate=RATE,
                                       complete=False)  # fmt: skip
    letters = sum(len(word) + 1 for word in text.split())
    longest = letters * local_tts_worker.SLOW_SECONDS_PER_LETTER
    heard = progress.heard_words(margin=0.0)
    assert heard <= int(10 * 1.0 / longest) + 1 and heard < 9
    progress.playing.complete = True  # the whole piece is 1.2 s: now its real length places the words
    assert progress.heard_words(margin=0.0) >= 7


def test_a_streamed_engine_speaks_whole_sentences_and_makes_nothing_ahead(manager):
    long = "When you get a moment, could you check the second draft, because the numbers changed again today."
    assert local_tts_worker.split_for_speech(long, clauses=False) == [long]
    assert len(local_tts_worker.split_for_speech(long)) > 1  # Kokoro's first clause comes first
    manager.prepare("pocket_tts", "Next sentence.", "alba")
    assert manager._prepared == {}


def test_a_real_worker_loads_while_its_request_reader_waits():
    """A worker process whose load imports numpy (a DLL that asks the standard input for its type as it starts) while
    the request reader waits for the next line: on Windows that hung for ever before the pipe was moved off the
    standard input. No model, no card: the load only imports numpy."""
    import subprocess

    code = (
        "import pilot.system.local_tts_worker as w; "
        "w._load = lambda engine, voice: __import__('numpy'); w._serve('pocket_tts', 'cpu')"
    )
    worker = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    timer = threading.Timer(60.0, worker.kill)  # a hang fails the test instead of holding the suite
    timer.start()
    try:
        worker.stdin.write(b'{"op": "load", "voice": "alba"}\n')
        worker.stdin.flush()
        answer = worker.stdout.readline()
    finally:
        timer.cancel()
        with contextlib.suppress(OSError):
            worker.stdin.close()
        worker.wait(30)
        worker.stdout.close()
    assert json.loads(answer) == {"status": "ok", "op": "load"}

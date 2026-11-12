"""The local-voice worker seen through voice.py: one reusable process, sentence-by-sentence playback, barge-in."""

from __future__ import annotations

import asyncio
import io
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pilot.system import local_tts_worker, voice


class _FakeStdin:
    def __init__(self) -> None:
        self.payload = b""
        self.closed = False

    def write(self, payload: bytes) -> None:
        self.payload = payload

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    def __init__(self, worker: _SuccessfulWorker) -> None:
        self.worker = worker
        self.closed = False
        self.requests: list[dict] = []

    def readline(self) -> bytes:
        request = json.loads(self.worker.stdin.payload)
        self.requests.append(request)
        if request.get("op") == "load":
            return b'{"status":"ok","op":"load"}\n'
        Path(request["output_file"]).write_bytes(b"RIFF-test-audio")
        return b'{"status":"ok"}\n'

    def close(self) -> None:
        self.closed = True


class _SuccessfulWorker:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(self)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return self.returncode


def _spoken(worker: _SuccessfulWorker) -> list[str]:
    return [
        Path(r["text_file"]).read_text(encoding="utf-8") if Path(r["text_file"]).exists() else "?"
        for r in worker.stdout.requests
        if "text_file" in r
    ]


@pytest.fixture(autouse=True)
def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> local_tts_worker.LocalTTSWorkerManager:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)  # every card visible: "auto" stays "auto"
    fresh = local_tts_worker.LocalTTSWorkerManager()
    monkeypatch.setattr(local_tts_worker, "_manager", fresh)
    return fresh


@pytest.mark.asyncio
async def test_local_tts_inference_runs_in_isolated_reusable_worker() -> None:
    worker = _SuccessfulWorker()

    with (
        patch.object(voice.subprocess, "Popen", return_value=worker) as spawn,
        patch.object(voice, "_play_local_tts_file", new=AsyncMock()) as play,
    ):
        await voice._run_local_tts_worker("kokoro_tts", "hello", "af_heart", None)
        await voice._run_local_tts_worker("kokoro_tts", "again", "af_heart", None)
        await voice._shutdown_local_tts_worker()

    command = spawn.call_args.args[0]
    assert command[1:3] == ["-m", "pilot.system.local_tts_worker"]
    assert command[-3:] == ["--engine", "kokoro_tts", "--serve"]
    assert command[command.index("--device") + 1] == "auto"
    spawn.assert_called_once()
    assert worker.stdout.requests[0] == {"op": "load", "voice": "af_heart"}  # cold: load first, then speak
    assert play.await_count == 2
    assert worker.stdin.closed is True


@pytest.mark.asyncio
async def test_a_long_reply_starts_playing_after_its_first_sentence() -> None:
    worker = _SuccessfulWorker()
    events: list[str] = []
    original = worker.stdout.readline

    def readline() -> bytes:
        line = original()
        request = worker.stdout.requests[-1]
        if "text_file" in request:
            events.append("made: " + Path(request["text_file"]).read_text(encoding="utf-8"))
        return line

    worker.stdout.readline = readline  # type: ignore[method-assign]

    async def play(path: Path) -> None:
        events.append("play")
        await asyncio.sleep(0.05)

    with (
        patch.object(voice.subprocess, "Popen", return_value=worker),
        patch.object(voice, "_play_local_tts_file", new=play),
    ):
        await voice._run_local_tts_worker("kokoro_tts", "First one. Second one! Third one?", "af_heart", None)
        await voice._shutdown_local_tts_worker()

    assert events.count("play") == 3
    assert events.index("play") < events.index("made: Third one?")  # speaking began before the end was made
    assert [e for e in events if e.startswith("made")] == ["made: First one.", "made: Second one!", "made: Third one?"]


class _SlowStdout(_FakeStdout):
    """Answers each speech request only when the test lets it, like a worker still synthesising."""

    def __init__(self, worker: _SuccessfulWorker, release: list) -> None:
        super().__init__(worker)
        self.release = release
        self.answered: list[str] = []

    def readline(self) -> bytes:
        request = json.loads(self.worker.stdin.payload)
        if request.get("op") != "load":
            self.answered.append(Path(request["text_file"]).read_text(encoding="utf-8"))  # read on arrival
            gate = threading.Event()
            self.release.append(gate)
            gate.wait(5)
        return super().readline()


@pytest.mark.asyncio
async def test_speech_cut_off_mid_synthesis_keeps_the_worker_warm_for_the_next_reply() -> None:
    worker = _SuccessfulWorker()
    gates: list = []
    worker.stdout = _SlowStdout(worker, gates)
    with (
        patch.object(voice.subprocess, "Popen", return_value=worker) as spawn,
        patch.object(voice, "_play_local_tts_file", new=AsyncMock()) as play,
    ):
        first = asyncio.ensure_future(voice._run_local_tts_worker("kokoro_tts", "a long answer", "af_heart", None))
        while not gates:
            await asyncio.sleep(0.01)
        first.cancel()  # the person talked over Heliox while it was still synthesising
        with pytest.raises(asyncio.CancelledError):
            await first
        assert local_tts_worker.manager().process is worker and worker.returncode is None
        second = asyncio.ensure_future(voice._run_local_tts_worker("kokoro_tts", "okay", "af_heart", None))
        await asyncio.sleep(0.05)
        gates[0].set()  # the worker finishes the abandoned request first, unheard
        while len(gates) < 2:
            await asyncio.sleep(0.01)
        gates[1].set()
        await second
        await voice._shutdown_local_tts_worker()
    spawn.assert_called_once()  # no cold start
    assert worker.stdout.answered == ["a long answer", "okay"]
    assert play.await_count == 1  # only the reply that was not cut off is played


@pytest.mark.asyncio
async def test_stopping_mid_reply_stops_playback_and_skips_the_remaining_sentences() -> None:
    worker = _SuccessfulWorker()
    playing = asyncio.Event()

    async def play(path: Path) -> None:
        playing.set()
        await asyncio.sleep(30)  # cancelled by the stop

    with (
        patch.object(voice.subprocess, "Popen", return_value=worker),
        patch.object(voice, "_play_local_tts_file", new=play),
    ):
        speech = asyncio.ensure_future(voice._run_local_tts_worker("kokoro_tts", "One. Two. Three.", "af_heart", None))
        await asyncio.wait_for(playing.wait(), 5)
        speech.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(speech, 1)
        await asyncio.sleep(0.05)
        assert local_tts_worker.manager().running
        await voice._shutdown_local_tts_worker()
    assert "Three." not in _spoken(worker)


@pytest.mark.asyncio
async def test_local_tts_worker_preserves_requested_output_file(tmp_path: Path) -> None:
    output = tmp_path / "speech.wav"
    worker = _SuccessfulWorker()

    with (
        patch.object(voice.subprocess, "Popen", return_value=worker),
        patch.object(voice, "_play_local_tts_file", new=AsyncMock()) as play,
    ):
        await voice._run_local_tts_worker("kokoro_tts", "hello. There.", "af_heart", str(output))
        await voice._shutdown_local_tts_worker()

    assert output.read_bytes() == b"RIFF-test-audio"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["speech.wav"]  # nothing written beside it
    assert _spoken(worker) == ["?"]  # one request for the whole text (its temp file is gone now)
    play.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_tts_shutdown_closes_streams_after_worker_already_exited() -> None:
    worker = _SuccessfulWorker()
    worker.returncode = 1
    manager = local_tts_worker.manager()
    manager.process = worker
    manager.engine = "kokoro_tts"

    await voice._shutdown_local_tts_worker()

    assert worker.stdin.closed is True
    assert worker.stdout.closed is True
    assert manager.process is None


def test_worker_answers_a_load_request_before_any_speech(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    loaded: list[tuple[str, str]] = []
    monkeypatch.setattr(local_tts_worker, "_load", lambda engine, voice_name: loaded.append((engine, voice_name)))
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO('{"op": "load", "voice": "alba"}\n'))

    assert local_tts_worker._serve("pocket_tts") == 0

    assert loaded == [("pocket_tts", "alba")]
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "op": "load"}


def test_worker_reports_a_failed_load_without_exiting(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def broken(engine: str, voice_name: str) -> None:
        raise ModuleNotFoundError("No module named 'kokoro'")

    monkeypatch.setattr(local_tts_worker, "_load", broken)
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO('{"op": "load", "voice": "x"}\n'))

    assert local_tts_worker._serve("pocket_tts") == 0

    reply = json.loads(capsys.readouterr().out)
    assert reply["status"] == "error" and "kokoro" in reply["error"]


def test_voice_state_says_when_heliox_can_speak_without_a_load(monkeypatch):
    import asyncio as _asyncio
    from types import SimpleNamespace

    from pilot.system import local_tts_worker as ltw

    kokoro = SimpleNamespace(tts_engine="kokoro_tts")
    assert ltw.local_voice_state(SimpleNamespace(tts_engine="os_native")) == {
        "voice_loaded": True,
        "voice_loading": False,
    }
    loop = _asyncio.new_event_loop()
    pending = loop.create_future()
    worker = SimpleNamespace(running=True, engine="kokoro_tts", loaded=False, _failed_until=None, _load_task=pending)
    monkeypatch.setattr(ltw, "_manager", worker)
    assert ltw.local_voice_state(kokoro) == {"voice_loaded": False, "voice_loading": True}
    worker.loaded, worker._load_task = True, None
    assert ltw.local_voice_state(kokoro) == {"voice_loaded": True, "voice_loading": False}
    worker._failed_until = 10.0  # failed: Heliox would use the system voice until it recovers
    assert ltw.local_voice_state(kokoro)["voice_loaded"] is False
    worker.running = False
    assert ltw.local_voice_state(kokoro) == {"voice_loaded": False, "voice_loading": False}
    loop.close()


@pytest.mark.asyncio
async def test_a_sentence_prepared_while_the_last_one_plays_is_not_synthesised_again() -> None:
    from types import SimpleNamespace

    worker = _SuccessfulWorker()
    played: list[Path] = []

    async def play(path: Path) -> None:
        played.append(path)

    config = SimpleNamespace(tts_engine="kokoro_tts", tts_voice="af_heart")
    with (
        patch.object(voice.subprocess, "Popen", return_value=worker),
        patch.object(voice, "_play_local_tts_file", new=play),
    ):
        await voice._run_local_tts_worker("kokoro_tts", "Here is the first part.", "af_heart", None)
        local_tts_worker.prepare_local_speech(config, "And here is the second.")  # while the first plays
        for _ in range(100):
            if "And here is the second." in _spoken(worker):
                break
            await asyncio.sleep(0.01)
        made_before = len(_spoken(worker))
        await voice._run_local_tts_worker("kokoro_tts", "And here is the second.", "af_heart", None)
        assert len(_spoken(worker)) == made_before  # played from the prepared audio, no second synthesis
        assert len(played) == 2 and not played[1].exists()  # the prepared file is removed once spoken
        local_tts_worker.prepare_local_speech(SimpleNamespace(tts_engine="os_native", tts_voice=""), "Hi.")  # no-op
        await voice._shutdown_local_tts_worker()


def test_the_daemon_gives_its_voice_worker_its_own_process_id_to_watch() -> None:
    import os

    from pilot.system import local_tts_worker

    with patch.object(local_tts_worker.subprocess, "Popen") as spawn:
        local_tts_worker._default_spawn("kokoro_tts", "cpu")
    command = spawn.call_args.args[0]
    assert command[command.index("--parent-pid") + 1] == str(os.getpid())
    assert command[-3:] == ["--engine", "kokoro_tts", "--serve"]


def test_a_voice_worker_whose_daemon_is_killed_ends_even_with_its_input_still_open(tmp_path: Path) -> None:
    """The worker's stdin is held open by another process, as a worker busy loading never reads its end: only the
    parent watch can end it, on every system."""
    import subprocess
    import sys
    import time

    import psutil

    daemon = (
        "import os, subprocess, sys, time\n"
        "read, write = os.pipe()\n"
        "os.set_inheritable(write, True)\n"
        "holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], close_fds=False)\n"
        "worker = subprocess.Popen([sys.executable, '-m', 'pilot.system.local_tts_worker', '--device', 'cpu',"
        " '--parent-pid', str(os.getpid()), '--engine', 'pocket_tts', '--serve'], stdin=read)\n"
        "print(worker.pid, holder.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", daemon], stdout=subprocess.PIPE, text=True)
    worker_pid, holder_pid = (int(part) for part in parent.stdout.readline().split())
    try:
        time.sleep(1.5)  # the worker is up and watching
        assert psutil.pid_exists(worker_pid)
        parent.kill()
        parent.wait(timeout=10)

        def gone() -> bool:
            try:
                return psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return True

        deadline = time.monotonic() + 30  # the watch looks every second; the rest is room for a slow machine
        while time.monotonic() < deadline and not gone():
            time.sleep(0.1)
        assert gone(), "the voice worker outlived its daemon"
    finally:
        parent.stdout.close()
        for pid in (worker_pid, holder_pid):
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass

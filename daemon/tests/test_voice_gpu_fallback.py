"""The local voice and the graphics card. CUDA_VISIBLE_DEVICES that hides every card keeps the voice on the CPU (on
Windows an empty value never reaches the worker, so the daemon decides); the GPU arbiter's word decides where it
starts; after a GPU error the same voice moves to the CPU for the rest of the session, said once. Fake workers
only: no model, no card, no audio."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import LocalTTSWorkerManager, cuda_hidden

CUDNN = "RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"


@pytest.mark.parametrize(
    ("value", "hidden"),
    [(None, False), ("", True), ("  ", True), ("-1", True), ("-1,-2", True), ("0", False), ("1,0", False)]
    + [("GPU-8f3c", False)],
)
def test_cuda_visible_devices_hides_every_card_only_when_empty_or_negative(value, hidden):
    environ = {} if value is None else {"CUDA_VISIBLE_DEVICES": value}
    assert cuda_hidden(environ) is hidden


class FakeWorker:
    """A worker process. On the GPU its card fails ``fail_on`` ("load" or "speak") with a cuDNN error, or the
    process dies ("exit"); on the CPU everything works."""

    def __init__(self, device: str, fail_on: str) -> None:
        self.device = device
        self.fail_on = fail_on if device != "cpu" else ""
        self.requests: list[dict] = []
        self.returncode: int | None = None
        self.stdin = self
        self.stdout = self
        self.closed = False
        self._payload = b""

    # stdin
    def write(self, payload: bytes) -> None:
        self._payload = payload

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    # stdout
    def readline(self) -> bytes:
        request = json.loads(self._payload)
        self.requests.append(request)
        kind = "load" if request.get("op") == "load" else "speak"
        if self.fail_on == kind:
            return (json.dumps({"status": "error", "error": CUDNN}) + "\n").encode()
        if self.fail_on == "exit" and kind == "speak":
            self.returncode = 3221225477
            return b""
        if kind == "speak":
            Path(request["output_file"]).write_bytes(b"RIFF-test-audio")
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def manager_on_a_failing_card(fail_on: str, arbiter: tuple[str, bool] = ("auto", True)):
    started: list[FakeWorker] = []
    leases: list[bool] = []

    def spawn(engine: str, device: str, gpu_lease: bool = False) -> FakeWorker:
        leases.append(gpu_lease)
        worker = FakeWorker(device, fail_on)
        started.append(worker)
        return worker

    manager = LocalTTSWorkerManager(spawn=spawn)
    asked: list[str] = []

    def choose(configured: str) -> tuple[str, bool]:
        asked.append(configured)
        return arbiter

    manager.choose_device = choose
    faults: list[tuple[str, str]] = []
    manager.on_gpu_fault = lambda component, reason: faults.append((component, reason))
    said: list[tuple[str, dict]] = []

    async def on_status(status: str, data: dict) -> None:
        said.append((status, data))

    manager.on_status = on_status
    return manager, started, leases, asked, faults, said


async def speak(manager: LocalTTSWorkerManager, text: str) -> list[Path]:
    played: list[Path] = []

    async def play(path: Path) -> None:
        played.append(path)

    await manager.speak("kokoro_tts", text, "af_heart", None, play)
    return played


@pytest.fixture(autouse=True)
def every_card_visible(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", ["speak", "load", "exit"])
async def test_after_a_gpu_error_the_same_voice_goes_on_on_the_cpu_for_the_session_said_once(fail_on):
    manager, started, leases, _asked, faults, said = manager_on_a_failing_card(fail_on)
    assert len(await speak(manager, "The backup is done.")) == 1  # spoken, by the local voice, not the system one
    assert [w.device for w in started] == ["auto", "cpu"] and leases == [True, False]
    assert faults and faults[0][0] == "voice"  # the arbiter hears about the fault
    assert [s for s, _ in said].count("voice_on_cpu") == 1
    assert "processor" in dict(said)["voice_on_cpu"]["message"]
    assert not manager.failing  # no system-voice fallback
    assert len(await speak(manager, "And the next one.")) == 1
    await manager.shutdown()
    assert len(await speak(manager, "After a restart of the worker.")) == 1
    assert [w.device for w in started] == ["auto", "cpu", "cpu"]  # remembered for the session
    assert [s for s, _ in said].count("voice_on_cpu") == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_an_empty_cuda_visible_devices_keeps_the_voice_off_the_card(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    manager, started, leases, asked, _faults, said = manager_on_a_failing_card("speak")
    assert len(await speak(manager, "Hello.")) == 1
    assert [w.device for w in started] == ["cpu"] and leases == [False]
    assert asked == []  # every card hidden: nothing to ask the arbiter
    assert "voice_on_cpu" not in [s for s, _ in said]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_arbiter_s_word_decides_where_the_voice_starts():
    manager, started, leases, asked, _faults, _said = manager_on_a_failing_card("speak", arbiter=("cpu", False))
    assert len(await speak(manager, "Hello.")) == 1
    assert asked == ["auto"] and [w.device for w in started] == ["cpu"] and leases == [False]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_voice_error_that_is_not_the_card_still_falls_back_to_the_system_voice():
    manager, _started, _leases, _asked, _faults, said = manager_on_a_failing_card("")

    class BadVoice(FakeWorker):
        def readline(self) -> bytes:
            request = json.loads(self._payload)
            if request.get("op") == "load":
                return b'{"status":"ok"}\n'
            return b'{"status":"error","error":"KeyError: voice not found"}\n'

    manager._spawn = lambda engine, device, gpu_lease=False: BadVoice(device, "")
    with pytest.raises(local_tts_worker.LocalTTSUnavailable):
        await speak(manager, "Hello.")
    assert manager.failing and not manager.cpu_for_session
    assert "voice_on_cpu" not in [s for s, _ in said]
    await manager.shutdown()


def test_the_worker_itself_stays_off_a_hidden_card(monkeypatch):
    from pilot.system import kokoro_tts

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(local_tts_worker.sys, "stdin", io.StringIO(""))
    chosen: list[str] = []
    monkeypatch.setattr(kokoro_tts, "configure", lambda *, device: chosen.append(device))
    assert local_tts_worker._serve("kokoro_tts", "auto", gpu_lease=True) == 0
    assert chosen == ["cpu"]


def test_kokoro_configured_for_the_card_runs_on_the_cpu_when_every_card_is_hidden(monkeypatch):
    from pilot.system import kokoro_tts

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    try:
        kokoro_tts.configure(device="cuda")
        assert kokoro_tts._device == "cpu"
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        kokoro_tts.configure(device="cuda")
        assert kokoro_tts._device == "cuda"
    finally:
        kokoro_tts.configure(device="auto")

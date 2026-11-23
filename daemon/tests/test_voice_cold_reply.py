"""No reply sits silent for a load that will not end soon: with no load under way the reply goes to the system voice at
once and the load starts behind it; a load already under way is waited for only when it is expected to end within
LOAD_GRACE_SECONDS, judged from how long this voice's loads have taken on this computer (kept beside the store).

Fake workers and a clock the test moves. The one wait the rule allows is recorded and given a generous real timeout,
so no outcome here depends on how fast the machine running the test is."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import LocalTTSWorkerManager, LocalVoiceNotReady, SpeechStore


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class HeldLoadWorker:
    def __init__(self) -> None:
        self.loaded = threading.Event()
        self.load_asked = threading.Event()
        self.texts: list[str] = []
        self.returncode: int | None = None
        self._payload = b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, close=lambda: None)

    def _write(self, payload: bytes) -> None:
        self._payload = payload

    def _close(self) -> None:
        self.stdin.closed = True
        self.returncode = 0

    def _readline(self) -> bytes:
        request = json.loads(self._payload)
        if request.get("op") == "load":
            self.load_asked.set()
            self.loaded.wait(10)
            return b'{"status":"ok","op":"load"}\n'
        text = Path(request["text_file"]).read_text(encoding="utf-8")
        self.texts.append(text)
        Path(request["output_file"]).write_bytes(b"RIFF")
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


async def settle(condition) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("never happened")


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker = HeldLoadWorker()
    clock = Clock()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: worker, clock=clock)
    manager.environ = {}
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    monkeypatch.setattr(local_tts_worker, "_manager", manager)  # where record_reply_voice notes the reply
    waits: list[float] = []
    real_bounded = local_tts_worker.bounded

    async def recorded(awaitable, timeout):
        waits.append(timeout)
        if timeout <= local_tts_worker.LOAD_GRACE_SECONDS:  # the reply's wait for a nearly finished load
            worker.loaded.set()  # it ends within the wait
            return await real_bounded(awaitable, 30.0)
        return await real_bounded(awaitable, timeout)

    monkeypatch.setattr(local_tts_worker, "bounded", recorded)
    return manager, worker, clock, waits


async def say(manager: LocalTTSWorkerManager, text: str) -> list[str]:
    played: list[str] = []

    async def play(piece) -> None:
        played.append(piece.text)

    await manager.speak("kokoro_tts", text, "af_heart", None, play)
    return played


def short_waits(waits: list[float]) -> list[float]:
    return [w for w in waits if w <= local_tts_worker.LOAD_GRACE_SECONDS]


@pytest.mark.asyncio
async def test_with_no_load_under_way_the_reply_goes_to_the_system_voice_at_once_and_the_load_starts(setup):
    manager, worker, _clock, waits = setup
    statuses: list[str] = []

    async def on_status(status: str, data: dict) -> None:
        statuses.append(status)

    manager.on_status = on_status
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "It's 10:42.")
    assert short_waits(waits) == []  # not even a short wait: a cold load takes 15 to 25 s
    assert statuses == ["voice_loading"]  # "Getting my voice ready" while the system voice speaks
    await settle(worker.load_asked.is_set)  # the load goes on behind the reply
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_load_nearly_done_is_waited_for_and_speaks_the_reply(setup):
    manager, worker, clock, waits = setup
    manager.load_times = [["kokoro_tts", "card", 10.0], ["kokoro_tts", "card", 12.0], ["kokoro_tts", "card", 11.0]]
    manager._load_times_read = True
    manager.prewarm("kokoro_tts", "af_heart")  # the person was heard: the load starts
    await settle(worker.load_asked.is_set)
    clock.now += 10.8  # loads here take 11 s: about 0.2 s to go
    assert manager.expected_load_left("kokoro_tts") == pytest.approx(0.2)
    assert await say(manager, "Done.") == ["Done."]
    assert short_waits(waits) == [pytest.approx(0.2 + local_tts_worker.LOAD_WAIT_MARGIN_SECONDS)]
    assert manager.last_reply.branch == "loaded_within_grace"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_load_early_in_its_run_is_not_waited_for(setup):
    manager, worker, clock, waits = setup
    manager.load_times = [["kokoro_tts", "card", 11.0]]
    manager._load_times_read = True
    manager.prewarm("kokoro_tts", "af_heart")
    await settle(worker.load_asked.is_set)
    clock.now += 3.0
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "Done.")
    assert short_waits(waits) == []
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_each_load_s_length_is_kept_beside_the_store_and_read_after_a_restart(setup, tmp_path):
    manager, worker, clock, _waits = setup
    manager.prewarm("kokoro_tts", "af_heart")
    await settle(worker.load_asked.is_set)
    clock.now += 14.5
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    await settle(lambda: (tmp_path / "voice_heads" / "loads.json").is_file())
    where = "cpu" if manager.effective_device == "cpu" else "card"
    assert manager.load_times == [["kokoro_tts", where, 14.5]]
    await manager.shutdown()

    restarted = LocalTTSWorkerManager(spawn=lambda engine, device: worker, clock=clock)
    restarted.head_cache = SpeechStore(tmp_path / "voice_heads")
    assert restarted.expected_load_left("kokoro_tts") == pytest.approx(14.5)  # nothing under way: a whole load


def test_only_the_latest_loads_count(tmp_path):
    manager = LocalTTSWorkerManager(clock=Clock())
    manager.effective_device = "cpu"
    for seconds in range(1, 16):
        manager._note_load_time("pocket_tts", float(seconds))
    assert [r[2] for r in manager.load_times] == [float(s) for s in range(6, 16)]
    assert manager.expected_load_left("pocket_tts") == pytest.approx(10.5)
    assert manager.expected_load_left("kokoro_tts") == local_tts_worker.DEFAULT_LOAD_SECONDS  # nothing recorded

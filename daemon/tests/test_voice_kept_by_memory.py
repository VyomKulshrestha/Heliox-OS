"""An idle local voice stays loaded while the computer has the memory to spare, and is released when it does not or
when the person chose the clock (voice.tts_release = "time"). A fake worker and a fake clock: no model."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.system import local_tts_worker, speech_budget
from pilot.system.local_tts_worker import LocalTTSWorkerManager

GB = 1024**3


@pytest.mark.parametrize(
    ("free", "total", "kept"),
    [
        (4 * GB, 16 * GB, True),  # plenty: stays loaded
        (1.5 * GB, 16 * GB, True),
        (0.8 * GB, 16 * GB, False),  # other programs are short: released
        (4 * GB, 8 * GB, False),  # a small computer: released (it would not load at all by default)
        (None, None, True),  # cannot tell: kept
    ],
)
def test_the_memory_rule(free, total, kept):
    assert speech_budget.keep_voice_loaded(free, total) is kept


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self._moved = asyncio.Event()

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        wake = self.now + seconds
        while self.now < wake:
            self._moved.clear()
            await self._moved.wait()

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        self._moved.set()
        for _ in range(30):
            await asyncio.sleep(0)


def worker():
    import json
    from pathlib import Path

    state = {"payload": b""}
    process = SimpleNamespace(returncode=None)

    def write(payload):
        state["payload"] = payload

    def readline():
        request = json.loads(state["payload"])
        if request.get("op") != "load":
            Path(request["output_file"]).write_bytes(b"RIFF")
        return b'{"status":"ok"}\n'

    def close():
        process.stdin.closed = True
        process.returncode = 0

    process.stdin = SimpleNamespace(write=write, flush=lambda: None, close=close, closed=False)
    process.stdout = SimpleNamespace(readline=readline, close=lambda: None)
    process.poll = lambda: process.returncode
    process.wait = lambda timeout=None: 0
    return process


@pytest.mark.asyncio
async def test_an_idle_voice_stays_loaded_while_memory_allows_and_goes_when_it_is_needed():
    clock, process = Clock(), worker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: process, clock=clock, sleep=clock.sleep)
    memory = {"enough": True}
    manager.keep_while = lambda: memory["enough"]
    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    await clock.advance(301)  # five idle minutes: by the clock it would go
    await clock.advance(60 * 30)  # half an hour later, memory to spare: still loaded
    assert manager.running
    memory["enough"] = False  # the person opens something big
    await clock.advance(local_tts_worker.MEMORY_RECHECK_SECONDS + 1)
    await clock.advance(1)
    assert not manager.running


@pytest.mark.asyncio
async def test_never_released_sooner_than_keep_loaded_seconds_and_the_clock_rule_still_works():
    clock, process = Clock(), worker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: process, clock=clock, sleep=clock.sleep)
    manager.keep_while = lambda: False  # memory short from the start
    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    await clock.advance(200)
    assert manager.running  # mid-conversation: kept whatever the memory
    await clock.advance(101)
    await clock.advance(1)
    assert not manager.running


def test_the_setting_chooses_the_rule(monkeypatch):
    shared = local_tts_worker.manager()
    for name in ("keep_while", "light_reason", "choose_device", "on_gpu_fault", "free_memory", "keep_loaded_seconds"):
        monkeypatch.setattr(shared, name, getattr(shared, name))  # put back after the test
    monkeypatch.setattr(shared, "device", shared.device)
    for release, expect_rule in (("memory", True), ("time", False)):
        config = PilotConfig()
        config.voice = VoiceConfig(tts_engine="os_native", tts_release=release)
        local_tts_worker.configure_local_voice(config.voice)
        assert (local_tts_worker.manager().keep_while is not None) is expect_rule
    assert PilotConfig().voice.tts_release == "memory"


def on_the_card(memory: dict, wants: dict):
    """A manager whose worker started on the card holding the lease, with the arbiter's count of components that
    found the card held (GpuArbiter.wants) and the memory rule as the test sets them."""
    clock, process = Clock(), worker()
    manager = LocalTTSWorkerManager(
        spawn=lambda engine, device, gpu_lease=False: process, clock=clock, sleep=clock.sleep
    )
    manager.keep_while = lambda: memory["enough"]
    manager.lease_wants = lambda: wants["count"]
    manager.choose_device = lambda configured: ("auto", True)
    return clock, manager


@pytest.fixture
def cuda_torch(monkeypatch):
    from pilot.system import gpu_paths

    monkeypatch.setattr(gpu_paths, "torch_build", lambda platform=None: "cuda")
    monkeypatch.setattr(local_tts_worker, "cuda_hidden", lambda environ=None: False)  # the suite hides every card


@pytest.mark.asyncio
async def test_a_voice_on_the_card_stays_loaded_while_idle_until_another_component_asks_for_the_card(cuda_torch):
    """19 November, one development laptop: released by the clock after 300 s, the next reply waited for a 15 s reload
    in the system voice. Now it stays, and gives the lease up once Gnomon, Whisper or screen reading finds the card
    held by it (the lease is not taken from its holder)."""
    memory, wants = {"enough": True}, {"count": 4}  # four asked before this voice took the card: they do not count
    clock, manager = on_the_card(memory, wants)
    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    assert manager.gpu_lease
    await clock.advance(301)
    await clock.advance(60 * 30)  # half an hour idle, nobody asked: still loaded
    assert manager.running
    wants["count"] += 1  # Gnomon found the card held by the voice
    await clock.advance(local_tts_worker.MEMORY_RECHECK_SECONDS + 1)
    await clock.advance(1)
    assert not manager.running
    assert [kind for kind, _detail, _at in manager.reload_log].count("released") == 1


@pytest.mark.asyncio
async def test_a_voice_on_the_card_goes_when_memory_runs_short_as_on_the_processor(cuda_torch):
    memory, wants = {"enough": True}, {"count": 0}
    clock, manager = on_the_card(memory, wants)
    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    await clock.advance(301)
    assert manager.running
    memory["enough"] = False
    await clock.advance(local_tts_worker.MEMORY_RECHECK_SECONDS + 1)
    await clock.advance(1)
    assert not manager.running


@pytest.mark.asyncio
async def test_a_request_for_the_card_mid_conversation_waits_for_the_idle_time(cuda_torch):
    memory, wants = {"enough": True}, {"count": 0}
    clock, manager = on_the_card(memory, wants)
    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    wants["count"] += 1
    await clock.advance(200)
    assert manager.running  # mid-conversation: the voice is not taken away
    await clock.advance(101)
    await clock.advance(1)
    assert not manager.running


def test_the_arbiter_counts_each_component_that_finds_the_card_held_by_another():
    from pilot.system import gpu_arbiter

    holder = {"value": None}
    shared = gpu_arbiter.GpuArbiter(
        read=lambda: gpu_arbiter.MachineReading(
            "nvidia", gpus=(gpu_arbiter.GpuInfo(0, "card", 60.0, 6 * gpu_arbiter.GIB, 8 * gpu_arbiter.GIB),)
        ),
        holder=lambda: holder["value"],
        environ={},  # the suite hides every card; this arbiter sees the made-up one
    )
    assert shared.check("gnomon").code == "ok" and shared.wants() == 0
    holder["value"] = {"component": "voice", "pid": 1}
    assert shared.check("voice").code == "ok" and shared.wants() == 0  # the holder itself
    assert shared.check("gnomon").code == "busy"
    assert shared.check("whisper").code == "busy"
    assert shared.wants() == 2 and shared.last_wanted_by == "whisper"


def test_the_daemon_wires_the_arbiter_s_count_to_the_voice(monkeypatch):
    from pilot.system import gpu_arbiter

    shared = local_tts_worker.manager()
    for name in ("keep_while", "light_reason", "choose_device", "on_gpu_fault", "free_memory", "lease_wants"):
        monkeypatch.setattr(shared, name, getattr(shared, name))
    local_tts_worker.configure_local_voice(VoiceConfig(tts_engine="os_native"))
    assert shared.lease_wants == gpu_arbiter.arbiter().wants

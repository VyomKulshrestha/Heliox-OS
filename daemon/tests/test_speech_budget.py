"""A computer with little memory: Heliox speaks with the system voice by itself, says why once, and a voice that
cannot load for memory never leaves Heliox mute (pilot.system.speech_budget, LocalTTSWorkerManager, health's ready).

Stand-in loads only: the fake workers of test_local_voice_lifecycle.py, total memory reported as 4, 8 and 16 GB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import local_tts_worker, speech_budget
from pilot.system.local_tts_worker import LocalTTSUnavailable
from tests.test_local_voice_lifecycle import FakeClock, FakeWorker, Spawner, _manager

GB = 1024**3


@pytest.mark.parametrize(
    ("total", "light"),
    [(4 * GB, True), (int(7.7 * GB), True), (8 * GB, True), (12 * GB, False), (16 * GB, False), (None, False)],
)
def test_the_rule_by_total_memory(total, light):
    reason = speech_budget.light_voice_reason(total, True)
    assert bool(reason) is light
    if light:
        assert (
            reason.startswith(f"This computer has {round(total / GB)} GB of memory") and speech_budget.SETTING in reason
        )
    assert speech_budget.light_voice_reason(total, False) == ""  # the setting turned off: the local voice everywhere


def _statuses(manager) -> list[tuple[str, dict]]:
    said: list[tuple[str, dict]] = []

    async def on_status(status: str, data: dict) -> None:
        said.append((status, data))

    manager.on_status = on_status
    return said


@pytest.mark.asyncio
async def test_an_8_gb_computer_speaks_with_the_system_voice_says_why_once_and_starts_no_worker():
    clock, spawner = FakeClock(), Spawner(FakeWorker())
    manager = _manager(spawner, clock)
    manager.light_reason = lambda: speech_budget.light_voice_reason(8 * GB, True)
    said = _statuses(manager)

    for _ in range(3):
        with pytest.raises(LocalTTSUnavailable):
            await manager.speak("kokoro_tts", "Hello.", "af_heart", None, AsyncMock())

    assert spawner.calls == []  # nothing of the 1.6-1.9 GB is loaded
    unavailable = [data["message"] for status, data in said if status == "voice_unavailable"]
    assert unavailable == [speech_budget.light_voice_reason(8 * GB, True)]  # said once
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_16_gb_computer_loads_the_local_voice():
    clock, spawner = FakeClock(), Spawner(FakeWorker())
    manager = _manager(spawner, clock)
    manager.light_reason = lambda: speech_budget.light_voice_reason(16 * GB, True)
    await manager.speak("kokoro_tts", "Hello.", "af_heart", None, AsyncMock())
    assert spawner.calls and manager.loaded and manager.unavailable_reason == ""
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_load_that_raises_memory_error_falls_back_and_says_why():
    clock = FakeClock()

    def spawn(engine: str, device: str):
        raise MemoryError("cannot allocate the model")

    manager = _manager(spawn, clock)
    said = _statuses(manager)
    with pytest.raises(LocalTTSUnavailable):
        await manager.speak("kokoro_tts", "Hello.", "af_heart", None, AsyncMock())

    assert manager.failing and "could not start" in manager.unavailable_reason
    assert [d["message"] for s, d in said if s == "voice_unavailable"] == [speech_budget.VOICE_FAILED]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_too_little_free_memory_says_so_in_plain_words():
    clock, spawner = FakeClock(), Spawner(FakeWorker())
    manager = _manager(spawner, clock)
    manager.free_memory = lambda: int(0.8 * GB)
    said = _statuses(manager)
    with pytest.raises(LocalTTSUnavailable):
        await manager.speak("kokoro_tts", "Hello.", "af_heart", None, AsyncMock())
    assert [d["message"] for s, d in said if s == "voice_unavailable"] == [speech_budget.MEMORY_SHORT]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_worker_that_dies_leaves_the_system_voice_and_ready_says_why(monkeypatch):
    clock, worker = FakeClock(), FakeWorker()
    manager = _manager(Spawner(worker), clock)
    await manager.speak("kokoro_tts", "One.", "af_heart", None, AsyncMock())
    worker.stdout.readline = lambda: b""  # the worker died (the system killed it for memory, say)
    with pytest.raises(LocalTTSUnavailable):
        await manager.speak("kokoro_tts", "Two.", "af_heart", None, AsyncMock())

    server = PilotServer(PilotConfig())
    server.config.voice.tts_engine = "kokoro_tts"
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    ready = (await server._handle_health({}, MagicMock()))["ready"]
    assert ready["speaks"] == {"ok": True, "state": "system_voice", "reason": speech_budget.VOICE_FAILED}
    await manager.shutdown()


@pytest.mark.asyncio
async def test_ready_says_the_small_machine_reason(monkeypatch):
    clock, spawner = FakeClock(), Spawner(FakeWorker())
    manager = _manager(spawner, clock)
    manager.light_reason = lambda: speech_budget.light_voice_reason(4 * GB, True)
    with pytest.raises(LocalTTSUnavailable):
        await manager.speak("kokoro_tts", "Hello.", "af_heart", None, AsyncMock())

    server = PilotServer(PilotConfig())
    server.config.voice.tts_engine = "kokoro_tts"
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    speaks = (await server._handle_health({}, MagicMock()))["ready"]["speaks"]
    assert speaks["state"] == "system_voice" and speaks["reason"].startswith("This computer has 4 GB of memory")
    await manager.shutdown()

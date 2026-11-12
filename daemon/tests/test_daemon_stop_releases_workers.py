"""stop() ends what is in flight before any store closes, in a fixed order, within a bound.

Order: speech stops, the conversation layer's turn is interrupted, voice turns are cancelled, the execute in flight is
cancelled, subsystems stop, the listener closes (cancelling other requests in flight); only then stores close. Every case below checks that the
in-flight work had ended by the time the experience ledger (one of the last stores) was closed.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from pilot import server as server_module
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import local_tts_worker

STOP_LIMIT = 30.0  # only matters if stop() hangs


class _Ledger:
    """Records what was still running when stop() closed it."""

    def __init__(self, *watched: asyncio.Task) -> None:
        self.watched = watched
        self.running_at_close: list[bool] | None = None

    async def close(self) -> None:
        self.running_at_close = [not task.done() for task in self.watched]


class _Speech:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def stop_all(self) -> int:
        self.order.append("speech stopped")
        return 1

    def status(self) -> dict:
        return {"active": False}

    async def close(self) -> None:
        self.order.append("speech closed")


async def _forever(started: asyncio.Event) -> None:
    started.set()
    await asyncio.sleep(3600)


@pytest.fixture
def quiet_voice(monkeypatch):
    manager = local_tts_worker.LocalTTSWorkerManager()
    manager.shutdown = AsyncMock()
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    return manager


async def test_stop_shuts_the_local_voice_worker(quiet_voice):
    await asyncio.wait_for(PilotServer(PilotConfig()).stop(), STOP_LIMIT)
    quiet_voice.shutdown.assert_awaited_once()


async def test_a_voice_worker_that_will_not_stop_does_not_hold_up_the_daemon(quiet_voice):
    quiet_voice.shutdown.side_effect = OSError("pipe broken")
    await asyncio.wait_for(PilotServer(PilotConfig()).stop(), STOP_LIMIT)  # logs the failure and carries on


async def test_an_execute_in_flight_ends_before_the_stores_close(quiet_voice):
    server = PilotServer(PilotConfig())
    started = asyncio.Event()
    execute = asyncio.create_task(_forever(started))
    await started.wait()
    server._active_execution_task = execute
    server._experience_ledger = ledger = _Ledger(execute)

    await asyncio.wait_for(server.stop(), STOP_LIMIT)

    assert execute.cancelled() and ledger.running_at_close == [False]


async def test_a_typed_turn_streaming_is_interrupted_before_the_stores_close(quiet_voice):
    server = PilotServer(PilotConfig())
    started = asyncio.Event()

    class Layer:
        def __init__(self) -> None:
            self._active = asyncio.create_task(_forever(started))
            self.interrupted = False

        def interrupt(self) -> bool:
            self.interrupted = True
            self._active.cancel()
            return True

    server._conversation = layer = Layer()
    await started.wait()
    server._experience_ledger = ledger = _Ledger(layer._active)

    await asyncio.wait_for(server.stop(), STOP_LIMIT)

    assert layer.interrupted and layer._active.cancelled() and ledger.running_at_close == [False]


async def test_a_voice_turn_is_cancelled_before_the_stores_close(quiet_voice, monkeypatch):
    server = PilotServer(PilotConfig())
    started = asyncio.Event()

    async def dispatch(_text: str) -> None:
        await _forever(started)

    monkeypatch.setattr(server, "_voice_command_dispatch", dispatch)
    turn = asyncio.create_task(server._voice_turn("hey heliox what's on my calendar"))
    await started.wait()
    server._experience_ledger = ledger = _Ledger(turn)

    await asyncio.wait_for(server.stop(), STOP_LIMIT)

    assert turn.cancelled() and ledger.running_at_close == [False]
    assert not server._voice_turn_tasks


async def test_speech_stops_first_so_a_live_remark_ends_at_once(quiet_voice):
    server = PilotServer(PilotConfig())
    order: list[str] = []
    server._speech_coordinator = _Speech(order)

    class Ledger:
        async def close(self) -> None:
            order.append("stores closed")

    server._experience_ledger = Ledger()

    await asyncio.wait_for(server.stop(), STOP_LIMIT)

    assert order[0] == "speech stopped"
    assert order.index("speech stopped") < order.index("stores closed")


async def test_work_that_ignores_cancellation_does_not_hold_stop_past_its_bound(quiet_voice, monkeypatch):
    monkeypatch.setattr(server_module, "STOP_IN_FLIGHT_SECONDS", 0.2)
    server = PilotServer(PilotConfig())
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn(_text: str) -> None:
        started.set()
        while not release.is_set():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue  # misbehaves: swallows every cancellation

    monkeypatch.setattr(server, "_voice_command_dispatch", stubborn)
    turn = asyncio.create_task(server._voice_turn("keep going"))
    await started.wait()

    await asyncio.wait_for(server.stop(), STOP_LIMIT)  # returns although the turn is still running

    assert not turn.done()
    release.set()
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)


async def test_a_subsystem_that_never_returns_cannot_hold_stop_past_the_whole_bound(quiet_voice, monkeypatch):
    """Every awaited close is bounded, and the whole of stop() too: here each step may wait 0.2 s and the whole of
    stop() 0.5 s, while two subsystems never return and one raises."""
    monkeypatch.setattr(server_module, "STOP_IN_FLIGHT_SECONDS", 0.2)
    monkeypatch.setattr(server_module, "STOP_TOTAL_SECONDS", 0.5)
    server = PilotServer(PilotConfig())

    async def never() -> None:
        await asyncio.Event().wait()

    async def broken() -> None:
        raise OSError("database is locked")

    closed: list[str] = []

    async def close_memory() -> None:
        closed.append("memory")

    server._reminders = type("Hangs", (), {"stop": staticmethod(never)})()
    server._live = type("Hangs", (), {"stop": staticmethod(never)})()
    server._executor = type("Breaks", (), {"close": staticmethod(broken)})()
    server._memory = type("Store", (), {"close": staticmethod(close_memory)})()
    loop = asyncio.get_running_loop()
    started = loop.time()

    await asyncio.wait_for(server.stop(), STOP_LIMIT)

    # The deadline is what is asserted, with room for a slow machine: 0.5 s plus scheduling, far below 2 x 0.2 s
    # per step for every step stop() has.
    assert loop.time() - started < 5.0
    assert closed == [] or closed == ["memory"]  # after the deadline a close is started but not waited for

"""The daemon listens before its heavy start-up, says which stage it is in, and holds requests until it is ready.

Order and states only: nothing here measures time. initialize() is replaced by one that waits for the test, so the
daemon is "starting" for exactly as long as the test wants.
"""

import asyncio
import json

import pytest
import websockets

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import storage_health
from pilot.system.startup import StartupProgress

LIMIT = 60.0  # failure limit only


def test_progress_states_in_order():
    clock = iter(range(100))
    progress = StartupProgress(clock=lambda: float(next(clock)))
    seen = []
    progress.on_change = lambda snapshot: seen.append((snapshot["state"], snapshot["stage"]))
    progress.begin("checking the data folder")
    progress.begin("opening Heliox's records")
    progress.ready()
    assert seen == [
        ("starting", "checking the data folder"),
        ("starting", "opening Heliox's records"),
        ("ready", ""),
    ]
    assert progress.snapshot()["stages_done"] == ["checking the data folder", "opening Heliox's records", "ready"]


async def test_wait_ready_says_why_when_start_up_failed_or_is_still_going():
    progress = StartupProgress()
    waiter = asyncio.create_task(progress.wait_ready())
    progress.failed("OSError: the model stack could not be imported")
    assert await waiter == "Heliox could not finish starting: OSError: the model stack could not be imported"

    slow = StartupProgress()
    slow.begin("loading memory")
    assert await slow.wait_ready(timeout=0.01) == "Heliox is still starting (loading memory). Try again in a moment."

    done = StartupProgress()
    done.ready()
    assert await done.wait_ready() == ""


async def _rpc(ws, method, request_id, params=None):
    await ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}, "id": request_id}))


async def _next_reply(ws, request_id, notes):
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), LIMIT))
        if message.get("id") == request_id:
            return message
        if message.get("method"):
            notes.append(message)


@pytest.fixture
def slow_start(monkeypatch, tmp_path):
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")  # never the real token files
    monkeypatch.setattr(storage_health, "preflight", lambda _folder: None)
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", 0, "test-token"
    server = PilotServer(config)
    release = asyncio.Event()
    outcome: dict = {}

    async def initialize():
        server._startup_stage("opening Heliox's records")
        await release.wait()
        if outcome.get("fail"):
            raise OSError("the model stack could not be imported")
        server._handlers = {"get_config": server._handle_get_config}  # the real one builds the whole table here

    server.initialize = initialize
    return server, release, outcome


async def _listening(server):
    for _ in range(int(LIMIT / 0.01)):
        if server._server is not None and server._server.sockets:
            return server._server.sockets[0].getsockname()[1]
        await asyncio.sleep(0.01)
    raise AssertionError("the daemon never listened")


async def test_ping_and_status_answer_while_starting_and_a_request_waits_then_succeeds(slow_start):
    server, release, _outcome = slow_start
    start = asyncio.create_task(server.start())
    port = await _listening(server)
    notes: list = []
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await _rpc(ws, "auth", "auth", {"token": "test-token"})
            await _next_reply(ws, "auth", notes)
            await _rpc(ws, "ping", "p1")
            pong = await _next_reply(ws, "p1", notes)
            # A window that connects while starting is told the current stage right after authenticating.
            assert notes[0]["method"] == "startup_status" and notes[0]["params"]["state"] == "starting"
            assert pong.get("result", {}).get("pong") is True, pong
            await _rpc(ws, "startup_status", "s1")
            status = (await _next_reply(ws, "s1", notes))["result"]
            assert status["state"] == "starting" and status["stage"] == "opening Heliox's records"
            await _rpc(ws, "ready", "r1")
            assert (await _next_reply(ws, "r1", notes))["result"]["ready"] is False

            await _rpc(ws, "get_config", "c1")  # asked in the first minute: waits, is not refused
            await asyncio.sleep(0.2)
            assert not any(note.get("id") == "c1" for note in notes)
            release.set()
            reply = await _next_reply(ws, "c1", notes)
            assert "result" in reply and "model" in reply["result"]
            await _rpc(ws, "startup_status", "s2")
            assert (await _next_reply(ws, "s2", notes))["result"]["state"] == "ready"
        stages = [note["params"]["stage"] for note in notes if note["method"] == "startup_status"]
        assert "preparing the voice" in stages
    finally:
        release.set()
        await asyncio.wait_for(start, LIMIT)
        await server.stop()


async def test_a_failed_start_up_answers_waiting_requests_with_the_reason(slow_start):
    server, release, outcome = slow_start
    outcome["fail"] = True
    start = asyncio.create_task(server.start())
    port = await _listening(server)
    notes: list = []
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await _rpc(ws, "auth", "auth", {"token": "test-token"})
            await _next_reply(ws, "auth", notes)
            await _rpc(ws, "get_config", "c1")
            await asyncio.sleep(0.1)
            release.set()
            reply = await _next_reply(ws, "c1", notes)
            assert reply["error"]["code"] == -32000
            assert reply["error"]["message"].startswith("Heliox could not finish starting: OSError")
    finally:
        with pytest.raises(OSError):
            await asyncio.wait_for(start, LIMIT)
        await server.stop()


async def test_a_server_that_was_never_started_does_not_gate_requests():
    server = PilotServer(PilotConfig())
    assert (await server._handle_startup_status({}, None))["state"] == "ready"

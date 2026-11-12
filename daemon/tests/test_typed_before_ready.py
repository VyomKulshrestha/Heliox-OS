"""The first thing a person types while Heliox is still starting is kept and answered once it is ready, exactly once:
end to end over a real socket against a server held before ready. (The window's side, a request typed before the
daemon even listens, is tested in tauri-app/ui/src/lib/api/daemon.test.ts.)"""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import storage_health

LIMIT = 30.0  # failure limit only


@pytest.fixture
def held_server(monkeypatch, tmp_path):
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(storage_health, "preflight", lambda _folder: None)
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", 0, "test-token"
    server = PilotServer(config)
    release = asyncio.Event()
    executed: list[str] = []

    async def execute(params, ws):
        executed.append(str(params.get("input", "")))
        return {"status": "success", "message": f"Opened: {params.get('input')}"}

    async def initialize():
        server._startup_stage("opening Heliox's records")
        await release.wait()  # held before ready until the test lets it go
        server._handlers = {"execute": execute}

    server.initialize = initialize
    return server, release, executed


async def _reply(ws, request_id):
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), LIMIT))
        if message.get("id") == request_id:
            return message


async def test_a_request_typed_while_starting_is_answered_once_ready_and_only_once(held_server):
    server, release, executed = held_server
    start = asyncio.create_task(server.start())
    for _ in range(int(LIMIT / 0.01)):
        if server._server is not None and server._server.sockets:
            break
        await asyncio.sleep(0.01)
    port = server._server.sockets[0].getsockname()[1]
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": 1}))
            await _reply(ws, 1)
            await ws.send(
                json.dumps({"jsonrpc": "2.0", "method": "execute", "params": {"input": "open my mail"}, "id": 2})
            )
            await asyncio.sleep(0.2)
            assert executed == []  # held: start-up has not finished
            release.set()
            reply = await _reply(ws, 2)
            assert reply["result"] == {"status": "success", "message": "Opened: open my mail"}
            await asyncio.sleep(0.2)
        assert executed == ["open my mail"]  # answered once ready, and only once
    finally:
        release.set()
        await asyncio.wait_for(start, LIMIT)
        await server.stop()

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.fixture
async def server_port(unused_tcp_port):
    """Provides a random unused TCP port for the test server."""
    return unused_tcp_port


@pytest.fixture
async def daemon_server(server_port, tmp_path, monkeypatch):
    """
    Fixture that starts a PilotServer in the background with a
    clean temporary environment.
    """
    # Isolate the test environment using temporary directories
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    config_dir.mkdir()
    data_dir.mkdir()
    state_dir.mkdir()

    monkeypatch.setattr("pilot.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("pilot.config.DATA_DIR", data_dir)
    monkeypatch.setattr("pilot.config.STATE_DIR", state_dir)
    monkeypatch.setattr("pilot.config.DB_FILE", data_dir / "pilot.db")
    monkeypatch.setattr("pilot.config.LOG_FILE", state_dir / "pilot.log")

    config = PilotConfig()
    config.server.host = "127.0.0.1"
    config.server.port = server_port

    # Mock heavy subsystems to avoid requiring a real LLM or OCR
    # These mocks ensure the test focuses on the IPC layer.
    with (
        patch("pilot.models.router.ModelRouter.initialize", new_callable=AsyncMock),
        patch("pilot.models.cache.LLMCache.initialize", new_callable=AsyncMock),
        patch("pilot.memory.store.MemoryStore.initialize", new_callable=AsyncMock),
        patch("pilot.agents.screen_vision.ScreenVisionAgent.start", new_callable=AsyncMock),
        patch("pilot.cognitive.tribe_engine.TribeEngine.load_model", new_callable=AsyncMock),
        patch("pilot.models.budget_tracker.BudgetTracker.initialize", new_callable=AsyncMock),
        patch("pilot.agents.prompt_improver.PromptImprover.initialize", new_callable=AsyncMock),
        patch("pilot.agents.subconscious.SubconsciousAgent.initialize", new_callable=AsyncMock),
        patch("pilot.security.audit.AuditLogger", return_value=MagicMock()),
    ):
        server = PilotServer(config)
        server_task = asyncio.create_task(server.start())

        # Give the server a moment to start the websocket listener
        await asyncio.sleep(0.2)

        uri = f"ws://127.0.0.1:{server_port}"
        yield uri

        # Cleanup
        await server.stop()
        await server_task


@pytest.mark.asyncio
async def test_ipc_ping_pong(daemon_server):
    """Verify that the basic ping-pong handshake works."""
    async with websockets.connect(daemon_server) as ws:
        request = {"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "test-1"}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        assert response["id"] == "test-1"
        assert response["result"]["pong"] is True
        assert "version" in response["result"]


@pytest.mark.asyncio
async def test_ipc_parse_error(daemon_server):
    """Verify that invalid JSON returns a Parse error (-32700)."""
    async with websockets.connect(daemon_server) as ws:
        await ws.send("invalid json {")

        response = json.loads(await ws.recv())
        assert response["id"] is None
        assert response["error"]["code"] == -32700
        assert "Parse error" in response["error"]["message"]


@pytest.mark.asyncio
async def test_ipc_method_not_found(daemon_server):
    """Verify that calling a non-existent method returns -32601."""
    async with websockets.connect(daemon_server) as ws:
        request = {"jsonrpc": "2.0", "method": "non_existent_method", "id": 99}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        assert response["id"] == 99
        assert response["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_ipc_get_config(daemon_server):
    """Verify that get_config returns the expected configuration sections."""
    async with websockets.connect(daemon_server) as ws:
        request = {"jsonrpc": "2.0", "method": "get_config", "id": "cfg-1"}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        result = response["result"]
        assert "model" in result
        assert "security" in result
        assert "first_run_complete" in result


@pytest.mark.asyncio
async def test_ipc_execute_flow_broadcast(daemon_server):
    """
    Verify that notifications (status updates) are broadcast correctly.
    """
    async with websockets.connect(daemon_server) as ws:
        # Mock the planner so it doesn't try to call an LLM
        with patch("pilot.agents.planner.Planner.plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = MagicMock(error=None, actions=[], explanation="Mocked plan")

            request = {"jsonrpc": "2.0", "method": "execute", "params": {"input": "test command"}, "id": "exec-1"}
            await ws.send(json.dumps(request))

            # We expect several notifications and eventually a response
            # 1. status: receiving input
            # 2. status: recalling memory
            # 3. status: routing agents
            # ...

            messages = []
            for _ in range(5):
                msg = json.loads(await ws.recv())
                messages.append(msg)
                if "id" in msg and msg["id"] == "exec-1":
                    break

            # Check that we got at least one status notification
            status_notifications = [m for m in messages if m.get("method") == "status"]
            assert len(status_notifications) > 0
            assert status_notifications[0]["params"]["phase"] == "receiving input"

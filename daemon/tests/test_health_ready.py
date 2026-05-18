"""Test health and ready JSON-RPC handlers."""

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
    Fixture that starts a PilotServer in the background with a clean temporary environment.
    """
    # Isolate the test environment using temporary directories
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    config_dir.mkdir()
    data_dir.mkdir()
    state_dir.mkdir()

    monkeypatch.setattr("pilot.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("pilot.config.CONFIG_FILE", config_dir / "config.toml")
    monkeypatch.setattr("pilot.config.RESTRICTIONS_FILE", config_dir / "restrictions.toml")
    monkeypatch.setattr("pilot.config.DATA_DIR", data_dir)
    monkeypatch.setattr("pilot.config.STATE_DIR", state_dir)
    monkeypatch.setattr("pilot.config.DB_FILE", data_dir / "pilot.db")
    monkeypatch.setattr("pilot.config.LOG_FILE", state_dir / "pilot.log")

    config = PilotConfig()
    config.server.host = "127.0.0.1"
    config.server.port = server_port

    # Mock heavy subsystems to avoid requiring a real LLM or OCR
    # Only patch classes/methods that are directly importable at module level
    with (
        patch("pilot.models.router.ModelRouter.initialize", new_callable=AsyncMock),
        patch("pilot.models.cache.LLMCache.initialize", new_callable=AsyncMock),
        patch("pilot.agents.screen_vision.ScreenVisionAgent.start", new_callable=AsyncMock),
        patch("pilot.cognitive.tribe_engine.TribeEngine.load_model", new_callable=AsyncMock),
        patch("pilot.models.budget_tracker.BudgetTracker.initialize", new_callable=AsyncMock),
        patch("pilot.agents.prompt_improver.PromptImprover.initialize", new_callable=AsyncMock),
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
async def test_health_handler(daemon_server, capsys):
    """Test the health JSON-RPC handler returns correct fields."""
    async with websockets.connect(daemon_server) as ws:
        request = {"jsonrpc": "2.0", "method": "health", "params": {}, "id": "health-test"}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        assert response["id"] == "health-test"
        assert "result" in response

        result = response["result"]

        # Verify all required fields are present
        assert "uptime" in result, "health response missing 'uptime' field"
        assert "memory_usage_mb" in result, "health response missing 'memory_usage_mb' field"
        assert "active_connections" in result, "health response missing 'active_connections' field"
        assert "loaded_agents" in result, "health response missing 'loaded_agents' field"

        # Verify field types
        assert isinstance(result["uptime"], (int, float)), "uptime should be a number"
        assert isinstance(result["memory_usage_mb"], (int, float)), "memory_usage_mb should be a number"
        assert isinstance(result["active_connections"], int), "active_connections should be an integer"
        assert isinstance(result["loaded_agents"], list), "loaded_agents should be a list"

        # Print results with PASS/FAIL
        print("\n--- HEALTH Handler Test Results ---")
        print(f"uptime: {result['uptime']} seconds - {'PASS' if result['uptime'] >= 0 else 'FAIL'}")
        print(
            f"memory_usage_mb: {result['memory_usage_mb']} MB - {'PASS' if result['memory_usage_mb'] > 0 else 'FAIL'}"
        )
        print(f"active_connections: {result['active_connections']} - PASS")
        print(f"loaded_agents: {result['loaded_agents']} - PASS")
        print("--- END HEALTH TEST ---\n")


@pytest.mark.asyncio
async def test_ready_handler(daemon_server, capsys):
    """Test the ready JSON-RPC handler returns correct fields."""
    async with websockets.connect(daemon_server) as ws:
        request = {"jsonrpc": "2.0", "method": "ready", "params": {}, "id": "ready-test"}
        await ws.send(json.dumps(request))

        response = json.loads(await ws.recv())
        assert response["id"] == "ready-test"
        assert "result" in response

        result = response["result"]

        # Verify required field is present
        assert "ready" in result, "ready response missing 'ready' field"

        # Verify field type
        assert isinstance(result["ready"], bool), "ready field should be a boolean"

        # Print results with PASS/FAIL
        print("\n--- READY Handler Test Results ---")
        print(f"ready: {result['ready']} - PASS")
        print("--- END READY TEST ---\n")

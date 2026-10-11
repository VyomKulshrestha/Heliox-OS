from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


class FakeBackground:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self._tasks = {}

    def stop(self, task_id: str) -> bool:
        self.stopped.append(task_id)
        return True


@pytest.mark.asyncio
async def test_factory_reset_reconciles_privileged_and_network_runtimes(monkeypatch) -> None:
    config = PilotConfig()
    config.air_handoff.enabled = True
    config.air_handoff.port = 9876
    config.air_handoff.max_transfer_mb = 75
    config.network.enabled = True
    config.self_healing.enabled = True
    config.self_healing.watched_metrics = ["cpu"]
    config.supervision.enabled = True
    config.supervision.keyboard_mouse_hook_enabled = True
    config.screen_vision.capture_interval_seconds = 9.0
    config.save = MagicMock()
    server = PilotServer(config)

    old_receiver = SimpleNamespace(running=True, stop=AsyncMock())
    old_manager = SimpleNamespace(clear_ephemeral=AsyncMock())
    mesh = SimpleNamespace(stop=AsyncMock())
    hook = SimpleNamespace(stop=MagicMock())
    background = FakeBackground()
    screen_vision = SimpleNamespace(set_interval=MagicMock())
    server._air_handoff_server = old_receiver
    server._air_handoff_manager = old_manager
    server._mesh = mesh
    server._supervision_hook = hook
    server._background = background
    server._screen_vision = screen_vision
    server._self_healing_started_monitors = {"monitor_cpu"}
    server._vault = SimpleNamespace(available=True)
    server._start_tts_warmup = MagicMock()

    created: dict[str, object] = {}

    class FakeManager:
        def __init__(self, vault, *, max_transfer_bytes):
            created["manager"] = self
            self.vault = vault
            self.max_transfer_bytes = max_transfer_bytes

    class FakeServer:
        def __init__(self, manager, *, host, port):
            created["server"] = self
            self.manager = manager
            self.host = host
            self.port = port
            self.running = False

    monkeypatch.setattr("pilot.air_handoff.AirHandoffManager", FakeManager)
    monkeypatch.setattr("pilot.air_handoff.AirHandoffServer", FakeServer)

    result = await server._handle_reset_config({}, MagicMock())

    assert result == {"status": "ok", "runtime_reconciled": True, "kept": ["server", "restrictions"]}
    old_receiver.stop.assert_awaited_once()
    old_manager.clear_ephemeral.assert_awaited_once()
    mesh.stop.assert_awaited_once()
    assert server._mesh is None
    hook.stop.assert_called_once_with()
    assert background.stopped == ["user_supervision", "monitor_cpu"]
    assert server._self_healing_started_monitors == set()
    assert config.air_handoff.enabled is False
    assert config.network.enabled is False
    assert config.self_healing.enabled is False
    assert config.supervision.enabled is False
    assert config.supervision.keyboard_mouse_hook_enabled is False
    assert created["manager"].max_transfer_bytes == 25 * 1024 * 1024
    assert created["server"].port == 8787
    assert server._air_handoff_manager is created["manager"]
    assert server._air_handoff_server is created["server"]
    screen_vision.set_interval.assert_called_once_with(3.0)
    server._start_tts_warmup.assert_called_once_with()
    config.save.assert_called_once_with()

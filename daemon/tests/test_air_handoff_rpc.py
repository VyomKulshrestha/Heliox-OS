from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.server import PilotServer


@pytest.fixture(autouse=True)
def _the_person_approves(monkeypatch):
    """The person says yes in the daemon's dialog; test_guard_changes.py drives that dialog. This file is about the
    work that follows."""

    async def approve(self, ws, found):
        return None

    async def yes(self, ws, lines):
        return True

    monkeypatch.setattr(PilotServer, "_approve_widenings", approve)
    monkeypatch.setattr(PilotServer, "_ask_about_data", yes)


class FakeConfig:
    def __init__(self, enabled: bool, *, fail_saves: int = 0) -> None:
        self.air_handoff = SimpleNamespace(enabled=enabled)
        self.fail_saves = fail_saves
        self.saved_values: list[bool] = []

    def save(self) -> None:
        self.saved_values.append(bool(self.air_handoff.enabled))
        if self.fail_saves:
            self.fail_saves -= 1
            raise OSError("config write failed")


class FakeReceiver:
    def __init__(self, running: bool, *, fail_start: bool = False) -> None:
        self.running = running
        self.fail_start = fail_start
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        if self.fail_start:
            raise OSError("receiver bind failed")
        self.running = True

    async def stop(self) -> None:
        self.stops += 1
        self.running = False


def make_server(config: FakeConfig, receiver: FakeReceiver) -> PilotServer:
    server = PilotServer.__new__(PilotServer)
    server.config = config
    server._vault = SimpleNamespace(available=True)
    server._air_handoff_manager = SimpleNamespace(clear_ephemeral=AsyncMock())
    server._air_handoff_server = receiver
    server._publish_air_handoff_state = AsyncMock(
        return_value={"enabled": config.air_handoff.enabled, "running": receiver.running}
    )
    return server


@pytest.mark.asyncio
async def test_enable_persists_before_starting_receiver() -> None:
    config = FakeConfig(False)
    receiver = FakeReceiver(False)
    server = make_server(config, receiver)

    result = await server._handle_air_handoff_set_enabled({"enabled": True}, None)

    assert result["status"] == "ok"
    assert config.saved_values == [True]
    assert config.air_handoff.enabled is True
    assert receiver.running is True


@pytest.mark.asyncio
async def test_enable_does_not_start_when_persistence_fails() -> None:
    config = FakeConfig(False, fail_saves=1)
    receiver = FakeReceiver(False)
    server = make_server(config, receiver)

    result = await server._handle_air_handoff_set_enabled({"enabled": True}, None)

    assert result == {"status": "error", "message": "config write failed"}
    assert config.saved_values == [True, False]
    assert config.air_handoff.enabled is False
    assert receiver.starts == 0
    assert receiver.running is False


@pytest.mark.asyncio
async def test_failed_receiver_start_restores_saved_setting() -> None:
    config = FakeConfig(False)
    receiver = FakeReceiver(False, fail_start=True)
    server = make_server(config, receiver)

    result = await server._handle_air_handoff_set_enabled({"enabled": True}, None)

    assert result == {"status": "error", "message": "receiver bind failed"}
    assert config.saved_values == [True, False]
    assert config.air_handoff.enabled is False
    assert receiver.running is False


@pytest.mark.asyncio
async def test_disable_persists_before_stopping_receiver() -> None:
    config = FakeConfig(True)
    receiver = FakeReceiver(True)
    server = make_server(config, receiver)

    result = await server._handle_air_handoff_set_enabled({"enabled": False}, None)

    assert result["status"] == "ok"
    assert config.saved_values == [False]
    server._air_handoff_manager.clear_ephemeral.assert_awaited_once()
    assert receiver.stops == 1
    assert receiver.running is False


@pytest.mark.asyncio
async def test_status_reports_uninitialized_service_as_error() -> None:
    server = PilotServer.__new__(PilotServer)
    server._air_handoff_manager = None
    server._air_handoff_server = None

    result = await server._handle_air_handoff_status({}, None)

    assert result == {
        "status": "error",
        "enabled": False,
        "running": False,
        "message": "Air Handoff is not initialized",
    }


@pytest.mark.asyncio
async def test_status_acknowledges_complete_receiver_state() -> None:
    server = PilotServer.__new__(PilotServer)
    server.config = SimpleNamespace(air_handoff=SimpleNamespace(enabled=True, port=8766))
    server._air_handoff_manager = SimpleNamespace(
        status=AsyncMock(
            return_value={
                "paired_devices": [],
                "pairing": None,
                "draft": None,
                "ready_transfers": 0,
                "recent": [],
                "secure_storage_available": True,
            }
        )
    )
    server._air_handoff_server = SimpleNamespace(running=True, base_url="http://192.0.2.1:8766")

    result = await server._handle_air_handoff_status({}, None)

    assert result["status"] == "ok"
    assert result["enabled"] is True
    assert result["running"] is True

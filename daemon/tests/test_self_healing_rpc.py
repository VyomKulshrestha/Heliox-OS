"""Tests for PilotServer's self_healing_status/self_healing_config_update
RPC handlers."""

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.mark.asyncio
async def test_status_reports_config_defaults_with_no_engine():
    server = PilotServer(PilotConfig())
    result = await server._handle_self_healing_status({}, ws=None)
    assert result["enabled"] is False
    assert result["attempts"] == []


@pytest.mark.asyncio
async def test_config_update_persists_enabled_toggle(tmp_path, monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update({"enabled": True}, ws=None)
    assert result["status"] == "ok"
    assert result["enabled"] is True
    assert server.config.self_healing.enabled is True


@pytest.mark.asyncio
async def test_config_update_sets_tiering_and_metrics(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update(
        {"auto_execute_max_tier": 2, "watched_metrics": ["cpu"]}, ws=None
    )
    assert result["status"] == "ok"
    assert result["auto_execute_max_tier"] == 2
    assert result["watched_metrics"] == ["cpu"]


@pytest.mark.asyncio
async def test_config_update_rejects_non_list_watched_metrics(monkeypatch):
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda: None)
    server = PilotServer(config)

    result = await server._handle_self_healing_config_update({"watched_metrics": "cpu"}, ws=None)
    assert result["status"] == "error"

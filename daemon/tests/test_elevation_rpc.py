from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.mark.asyncio
async def test_elevation_rpc_requires_root_policy(monkeypatch):
    server = PilotServer(PilotConfig())
    monkeypatch.setattr("pilot.server.sys.platform", "win32")

    result = await server._handle_restart_elevated({}, MagicMock())

    assert result["status"] == "blocked"
    assert "Enable Root Access" in result["message"]


@pytest.mark.asyncio
async def test_elevation_rpc_requests_windows_handoff(monkeypatch):
    config = PilotConfig()
    config.security.root_enabled = True
    server = PilotServer(config)
    restart = MagicMock(
        return_value={
            "status": "prompted",
            "message": "Administrator restart accepted.",
        }
    )
    monkeypatch.setattr("pilot.server.sys.platform", "win32")
    monkeypatch.setattr("pilot.security.privileges.has_elevated_privileges", lambda: False)
    monkeypatch.setattr("pilot.system.elevation.request_elevated_restart", restart)

    result = await server._handle_restart_elevated({}, MagicMock())

    assert result["status"] == "prompted"
    restart.assert_called_once_with()


@pytest.mark.asyncio
async def test_snapshot_retention_update_rejects_out_of_range_value():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "security",
            "values": {"snapshot_retention_count": 0},
        },
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.security.snapshot_retention_count == 10
    config.save.assert_not_called()

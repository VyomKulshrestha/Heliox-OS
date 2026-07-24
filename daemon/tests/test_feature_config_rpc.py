from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.mark.asyncio
async def test_preview_enabled_update_requires_boolean():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "preview", "values": {"enabled": "yes"}},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.preview.enabled is False
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_preview_enabled_update_is_applied_and_saved():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "preview", "values": {"enabled": True}},
        MagicMock(),
    )

    assert result["status"] == "ok"
    assert config.preview.enabled is True
    config.save.assert_called_once_with()

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"enabled": 1}, "enabled must be a boolean"),
        ({"sensitivity": 0}, "sensitivity must be from 0.1 to 3"),
        ({"prediction_ms": 251}, "prediction_ms must be from 0 to 250"),
        ({"blend": 1.1}, "blend must be from 0 to 1"),
    ],
)
async def test_gesture_cursor_update_rejects_invalid_values(values, message):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "gesture_cursor", "values": values},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert message in result["message"]
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_gesture_cursor_update_applies_runtime_tuning():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "gesture_cursor",
            "values": {"enabled": True, "sensitivity": 1.7, "blend": 0.45},
        },
        MagicMock(),
    )

    assert result["status"] == "ok"
    assert config.gesture_cursor.enabled is True
    assert config.gesture_cursor.sensitivity == 1.7
    assert config.gesture_cursor.blend == 0.45
    config.save.assert_called_once_with()


@pytest.mark.asyncio
async def test_gesture_calibration_update_requires_boolean():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "adaptive_calibration",
            "values": {"gesture_enabled": "enabled"},
        },
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.adaptive_calibration.gesture_enabled is True
    config.save.assert_not_called()


@pytest.mark.asyncio
async def test_voice_calibration_update_requires_boolean():
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {
            "section": "adaptive_calibration",
            "values": {"voice_wake_word_enabled": 1},
        },
        MagicMock(),
    )

    assert result["status"] == "error"
    assert config.adaptive_calibration.voice_wake_word_enabled is True
    config.save.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"tts_engine": "cloud"}, "tts_engine must be pocket_tts or os_native"),
        ({"tts_voice": "unknown"}, "tts_voice must be alba, giovanni, or lola"),
        ({"input_device": ""}, "input_device must be a valid microphone identifier"),
    ],
)
async def test_voice_output_update_rejects_unknown_options(values, message):
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    result = await server._handle_update_config(
        {"section": "voice", "values": values},
        MagicMock(),
    )

    assert result["status"] == "error"
    assert message in result["message"]
    config.save.assert_not_called()

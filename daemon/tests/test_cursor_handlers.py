"""Tests for the gesture-cursor bridge's browser/dev-mode fallback RPC handlers.

These handlers (_handle_cursor_move / _handle_cursor_click) exist as a
degraded fallback for testing the gesture-cursor bridge without a compiled
Tauri binary (the primary path is a native Rust command using enigo - see
tauri-app/src-tauri/src/commands.rs). They call pilot.system.input_control
directly, bypassing Planner/Executor/confirmation entirely (MOUSE_MOVE/
MOUSE_CLICK are Tier 1/USER_WRITE, never requiring confirmation).

Mocks input_control.mouse_move/mouse_click - these tests must never move the
real mouse or click anything on the machine running them.
"""

from unittest.mock import AsyncMock, patch

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A bare PilotServer instance - no .start(), no real WebSocket, no real
    subsystems. The handlers under test only touch pilot.system.input_control,
    so this is sufficient (matches the __init__-then-mock pattern already
    used for lightweight handler tests elsewhere in this suite)."""
    monkeypatch.setattr("pilot.config.CONFIG_DIR", tmp_path)
    config = PilotConfig()
    return PilotServer(config)


@pytest.mark.asyncio
async def test_cursor_move_calls_input_control_with_int_coords(server):
    with patch("pilot.system.input_control.mouse_move", new_callable=AsyncMock) as mock_move:
        mock_move.return_value = "Moved mouse to (100, 200) [absolute]"
        result = await server._handle_cursor_move({"x": 100, "y": 200}, None)

    mock_move.assert_awaited_once_with(100, 200, duration=0.0)
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_cursor_move_rejects_non_numeric_coords(server):
    with patch("pilot.system.input_control.mouse_move", new_callable=AsyncMock) as mock_move:
        result = await server._handle_cursor_move({"x": "not-a-number", "y": 200}, None)

    mock_move.assert_not_awaited()
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_cursor_click_calls_input_control_at_given_position(server):
    with patch("pilot.system.input_control.mouse_click", new_callable=AsyncMock) as mock_click:
        mock_click.return_value = "Clicked (left, 1x) at (50, 60)"
        result = await server._handle_cursor_click({"x": 50, "y": 60}, None)

    mock_click.assert_awaited_once_with(50, 60, button="left")
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_cursor_click_rejects_non_numeric_coords(server):
    with patch("pilot.system.input_control.mouse_click", new_callable=AsyncMock) as mock_click:
        result = await server._handle_cursor_click({"x": None, "y": 60}, None)

    mock_click.assert_not_awaited()
    assert result["status"] == "error"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from pilot.system import input_control


@pytest.mark.asyncio
async def test_keyboard_type_pastes_exact_case_and_restores_clipboard(monkeypatch):
    keyboard = MagicMock()
    clipboard = SimpleNamespace(paste=MagicMock(return_value="user clipboard"), copy=MagicMock())
    monkeypatch.setattr(input_control, "_ensure_pyautogui", lambda: keyboard)
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", clipboard)
    monkeypatch.setattr(input_control.time, "sleep", lambda _seconds: None)

    result = await input_control.keyboard_type("HELIOX Exact Test")

    keyboard.hotkey.assert_called_once_with(*input_control.paste_keys())  # command+V on macOS
    clipboard.copy.assert_has_calls([call("HELIOX Exact Test"), call("user clipboard")])
    assert result == "Typed exactly: HELIOX Exact Test"


@pytest.mark.asyncio
async def test_keyboard_type_restores_clipboard_when_paste_fails(monkeypatch):
    keyboard = MagicMock()
    keyboard.hotkey.side_effect = OSError("input unavailable")
    clipboard = SimpleNamespace(paste=MagicMock(return_value="user clipboard"), copy=MagicMock())
    monkeypatch.setattr(input_control, "_ensure_pyautogui", lambda: keyboard)
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", clipboard)

    with pytest.raises(OSError, match="input unavailable"):
        await input_control.keyboard_type("exact")

    assert clipboard.copy.call_args_list[-1].args == ("user clipboard",)


@pytest.mark.asyncio
async def test_keyboard_type_prefers_background_target_without_clipboard(monkeypatch):
    keyboard = MagicMock()
    set_text = AsyncMock(return_value="ok")
    monkeypatch.setattr(input_control, "_ensure_pyautogui", lambda: keyboard)
    monkeypatch.setattr("pilot.system.window_mgr.window_set_text", set_text)

    result = await input_control.keyboard_type("EXACT", target_window="Notepad")

    set_text.assert_awaited_once_with("EXACT", title="Notepad")
    keyboard.hotkey.assert_not_called()
    assert result == "Typed exactly in Notepad: EXACT"


@pytest.mark.asyncio
async def test_keyboard_type_fails_closed_when_target_cannot_be_reacquired(monkeypatch):
    set_text = AsyncMock(side_effect=RuntimeError("no UIA editor"))
    focus = AsyncMock(side_effect=RuntimeError("focus denied"))
    monkeypatch.setattr(input_control, "_ensure_pyautogui", MagicMock())
    monkeypatch.setattr("pilot.system.window_mgr.window_set_text", set_text)
    monkeypatch.setattr("pilot.system.window_mgr.window_focus", focus)

    with pytest.raises(RuntimeError, match="focus denied"):
        await input_control.keyboard_type("EXACT", target_window="Notepad")

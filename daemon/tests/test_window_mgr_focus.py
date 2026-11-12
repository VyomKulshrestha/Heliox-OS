from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

from pilot.system import window_mgr
from pilot.system.platform_detect import Platform


@pytest.mark.asyncio
async def test_windows_focus_uses_verified_app_activation(monkeypatch):
    run = AsyncMock(return_value=(0, "42", ""))
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", run)

    result = await window_mgr.window_focus(title="Owner's Notepad")

    script = run.await_args.args[0]
    assert run.await_args.kwargs["values"]["title"] == "Owner's Notepad" and "Owner" not in script
    assert "AppActivate([int]$p.Id)" in script
    assert "GetWindowThreadProcessId" in script
    assert "$foregroundPid -ne $p.Id" in script
    assert result == "Focused window: Owner's Notepad"


@pytest.mark.asyncio
async def test_windows_focus_rejects_unverified_activation(monkeypatch):
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(
        window_mgr,
        "run_powershell",
        AsyncMock(return_value=(1, "", "Window activation was rejected")),
    )

    with pytest.raises(RuntimeError, match="activation was rejected"):
        await window_mgr.window_focus(process_name="Notepad")


@pytest.mark.asyncio
async def test_windows_reads_target_text_without_foreground_focus(monkeypatch):
    payload = base64.b64encode(b"HELIOX EXACT").decode()
    run = AsyncMock(return_value=(0, payload, ""))
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", run)

    result = await window_mgr.window_read_text(title="Notepad")

    assert result == "HELIOX EXACT"
    script = run.await_args.args[0]
    assert "UIAutomationClient" in script
    assert "TextPattern" in script


@pytest.mark.asyncio
async def test_windows_sets_and_verifies_target_text_without_foreground_focus(monkeypatch):
    run = AsyncMock(return_value=(0, "ok", ""))
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", run)

    result = await window_mgr.window_set_text("HELIOX EXACT", title="Notepad")

    assert result == "Set exact text in window: Notepad"
    script = run.await_args.args[0]
    assert run.await_args.kwargs["values"]["text"] == "HELIOX EXACT" and "HELIOX EXACT" not in script
    assert "Text verification mismatch" in script

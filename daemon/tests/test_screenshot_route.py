"""The screenshot the Dashboard's "Take Screenshot" and the screenshot action take: through the daemon on every
system, the Screenshot portal first on Wayland, and a path PowerShell cannot mistake for code on Windows."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from pilot.system import portal, screen
from pilot.system.platform_detect import Platform


def _linux(monkeypatch, wayland: bool, png: bytes | None, answers: dict[str, int]):
    calls: list[list[str]] = []

    async def run_command(args, **_kwargs):
        calls.append(list(args))
        return answers.get(args[0], 1), "", f"{args[0]} failed"

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(screen, "run_command", run_command)
    monkeypatch.setattr(portal, "wayland_session", lambda env=None: wayland)
    monkeypatch.setattr(portal, "capture_png", lambda region=None: png)
    return calls


def test_on_wayland_the_portal_takes_it_and_no_x11_tool_runs(monkeypatch, tmp_path):
    calls = _linux(monkeypatch, wayland=True, png=b"\x89PNG portal", answers={})
    target = tmp_path / "shot.png"
    assert asyncio.run(screen.screenshot(str(target))) == f"Screenshot saved to {target}"
    assert target.read_bytes() == b"\x89PNG portal" and calls == []


def test_on_wayland_without_the_portal_gnome_screenshot_then_grim_are_tried(monkeypatch, tmp_path):
    calls = _linux(monkeypatch, wayland=True, png=None, answers={"grim": 0})
    asyncio.run(screen.screenshot(str(tmp_path / "shot.png")))
    assert [c[0] for c in calls] == ["gnome-screenshot", "grim"]


def test_on_x11_the_portal_is_not_asked(monkeypatch, tmp_path):
    asked = []
    calls = _linux(monkeypatch, wayland=False, png=None, answers={"scrot": 0})
    monkeypatch.setattr(portal, "capture_png", lambda region=None: asked.append(region))
    asyncio.run(screen.screenshot(str(tmp_path / "shot.png")))
    assert asked == [] and [c[0] for c in calls] == ["gnome-screenshot", "scrot"]


def test_a_quote_in_the_windows_path_cannot_end_the_powershell_string(monkeypatch, tmp_path):
    scripts: list[str] = []

    async def run_powershell(script, **_kwargs):
        scripts.append(script)
        return 0, "", ""

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(screen, "run_powershell", run_powershell)
    target = tmp_path / "O'Brien" / "shot.png"
    asyncio.run(screen.screenshot(str(target)))
    assert f"$bitmap.Save('{str(target).replace(chr(39), chr(39) * 2)}')" in scripts[0]
    assert f"Save('{target}')" not in scripts[0]


def test_the_take_screenshot_rpc_saves_where_the_system_keeps_screenshots(monkeypatch, tmp_path):
    from pilot.server import PilotServer

    taken: list[str] = []

    async def fake_screenshot(path=None, region=None):
        taken.append(path)
        return f"Screenshot saved to {path}"

    monkeypatch.setattr(screen, "default_screenshot_path", lambda: str(tmp_path / "shot.png"))
    monkeypatch.setattr(screen, "screenshot", fake_screenshot)
    result = asyncio.run(PilotServer._handle_take_screenshot(MagicMock(), {}, MagicMock()))
    assert result == {"status": "ok", "path": str(tmp_path / "shot.png")} and taken == [str(tmp_path / "shot.png")]

    async def failing(path=None, region=None):
        raise RuntimeError("Screenshot failed: no portal")

    monkeypatch.setattr(screen, "screenshot", failing)
    assert asyncio.run(PilotServer._handle_take_screenshot(MagicMock(), {}, MagicMock())) == {
        "status": "error",
        "message": "Screenshot failed: no portal",
    }


def test_the_default_place_follows_the_system(monkeypatch):
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    assert "Pictures" in screen.default_screenshot_path()
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.MACOS)
    assert "Desktop" in screen.default_screenshot_path()

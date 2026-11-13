"""The screenshot the Dashboard's "Take Screenshot" and the screenshot action take: through the daemon on every
system, the Screenshot portal first on Wayland, and a path PowerShell cannot mistake for code on Windows."""

from __future__ import annotations

import asyncio
from pathlib import Path
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
    """The path is a value the script reads, never part of its text, so no quote in it can end a string there."""
    calls: list[tuple[str, dict]] = []

    async def run_powershell(script, **kwargs):
        calls.append((script, kwargs))
        return 0, "", ""

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(screen, "run_powershell", run_powershell)
    for name in ("O'Brien", "O\u2019Brien", "a`b$(c)"):
        calls.clear()
        target = tmp_path / name / "shot.png"
        asyncio.run(screen.screenshot(str(target)))
        script, kwargs = calls[0]
        assert str(target) in kwargs["values"].values()
        assert name not in script and str(target) not in script


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


def _home(monkeypatch, tmp_path, own: bool):
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(screen, "_is_account_home", lambda _home: own)
    return home


def test_windows_uses_the_pictures_known_folder_wherever_it_is(monkeypatch, tmp_path):
    home = _home(monkeypatch, tmp_path, own=True)
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    onedrive = tmp_path / "OneDrive - Work" / "Bilder"
    monkeypatch.setattr(screen, "_windows_pictures", lambda: onedrive)
    path = screen.default_screenshot_path()
    assert path.startswith(str(onedrive)) and path.endswith(".png")
    monkeypatch.setattr(screen, "_windows_pictures", lambda: None)  # Windows did not say
    assert screen.screenshot_folder() == home / "Pictures"


def test_macos_uses_the_screenshot_apps_chosen_folder_else_the_desktop(monkeypatch, tmp_path):
    home = _home(monkeypatch, tmp_path, own=True)
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.MACOS)
    chosen = tmp_path / "Screenshots"
    monkeypatch.setattr(screen, "_mac_screencapture_location", lambda: chosen)
    assert screen.screenshot_folder() == chosen
    monkeypatch.setattr(screen, "_mac_screencapture_location", lambda: None)
    assert screen.screenshot_folder() == home / "Desktop"


def test_linux_uses_the_persons_pictures_folder_in_their_language(monkeypatch, tmp_path):
    home = _home(monkeypatch, tmp_path, own=True)
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.LINUX)
    config = home / ".config"
    config.mkdir()
    lines = [
        "# written by xdg-user-dirs-update",
        'XDG_DESKTOP_DIR="$HOME/Schreibtisch"',
        'XDG_PICTURES_DIR="$HOME/Bilder"',
    ]
    (config / "user-dirs.dirs").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert screen.screenshot_folder() == home / "Bilder"
    (config / "user-dirs.dirs").write_text('XDG_PICTURES_DIR="$HOME/"\n', encoding="utf-8")  # turned off
    assert screen.xdg_pictures_dir(home) is None
    (config / "user-dirs.dirs").write_text('XDG_PICTURES_DIR="/data/pics"\n', encoding="utf-8")
    assert screen.xdg_pictures_dir(home) == Path("/data/pics")
    (config / "user-dirs.dirs").unlink()
    assert screen.screenshot_folder() == home


def test_under_a_fake_home_the_real_accounts_folders_are_not_used(monkeypatch, tmp_path):
    home = _home(monkeypatch, tmp_path, own=False)
    asked = []
    monkeypatch.setattr(screen, "_windows_pictures", lambda: asked.append("windows") or tmp_path / "real")
    monkeypatch.setattr(screen, "_mac_screencapture_location", lambda: asked.append("mac") or tmp_path / "real")
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    assert screen.screenshot_folder() == home / "Pictures"
    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.MACOS)
    assert screen.screenshot_folder() == home / "Desktop"
    assert asked == []


def test_the_answer_says_where_the_file_went(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    async def run_command(args, **_kwargs):
        calls.append(args)
        Path(args[-1]).write_bytes(b"png")
        return 0, "", ""

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(screen, "run_command", run_command)
    target = tmp_path / "Bilder" / "shot.png"
    assert asyncio.run(screen.screenshot(str(target))) == f"Screenshot saved to {target}"
    assert calls == [["screencapture", str(target)]] and target.parent.is_dir()

"""macOS window actions through the Accessibility API, with a fake of the pyobjc calls (runs on every system)."""

from __future__ import annotations

import asyncio

import pytest

from pilot.system import macos_windows, platform_detect, window_mgr
from pilot.system.macos_windows import AccessibilityNeeded, MacWindows, RunningApp, ax_frame, match_apps

NOTES = RunningApp(501, "Notes", "Notes", "com.apple.Notes")
CODE = RunningApp(777, "Visual Studio Code", "Electron", "com.microsoft.VSCode")


class FakeAX:
    def __init__(self, trusted: bool = True) -> None:
        self.is_trusted = trusted
        self.apps = [NOTES, CODE]
        self.wins = {
            501: [{"AXTitle": "Groceries", "AXMinimized": False, "AXCloseButton": {"button": 1}}],
            777: [
                {"AXTitle": "server.py - Heliox", "AXMinimized": False, "AXCloseButton": {"button": 2}},
                {"AXTitle": "notes.md - Heliox", "AXMinimized": False, "AXCloseButton": {"button": 3}},
            ],
        }
        self.activated: list[int] = []
        self.raised: list[str] = []
        self.pressed: list[int] = []
        self.frames: list[tuple] = []

    def trusted(self):
        return self.is_trusted

    def running_apps(self):
        return self.apps

    def activate(self, pid):
        self.activated.append(pid)
        return True

    def windows(self, pid):
        return self.wins.get(pid, [])

    def get(self, element, name):
        return element.get(name)

    def set_bool(self, element, name, value):
        element[name] = value
        return True

    def press(self, element):
        self.pressed.append(element["button"])
        for windows in self.wins.values():
            windows[:] = [w for w in windows if w["AXCloseButton"] is not element]
        return True

    def raise_window(self, window):
        self.raised.append(window["AXTitle"])
        return True

    def visible_frame(self):
        return (0.0, 25.0, 1440.0, 800.0)

    def set_frame(self, window, frame):
        self.frames.append((window["AXTitle"], frame))
        return True


def test_app_matching_and_coordinates():
    assert match_apps("notes", [NOTES, CODE]) == [NOTES]
    assert match_apps("Electron", [NOTES, CODE]) == [CODE]
    assert match_apps("com.microsoft.vscode", [NOTES, CODE]) == [CODE]
    assert match_apps("Notes.app", [NOTES]) == [NOTES]
    assert match_apps("", [NOTES]) == [] and match_apps("Mail", [NOTES, CODE]) == []
    # A 1440x900 primary screen with a 25 pt menu bar and a 75 pt Dock: AppKit's visible frame starts at y=75.
    assert ax_frame((0, 75, 1440, 800), 900) == (0, 25, 1440, 800)


def test_focus_needs_no_permission_and_raises_the_titled_window():
    ax = FakeAX(trusted=False)
    assert MacWindows(ax).focus("Notes") == "Notes" and ax.activated == [501]
    with pytest.raises(AccessibilityNeeded, match="Accessibility"):
        MacWindows(ax).focus("Visual Studio Code", "notes.md")
    ax.is_trusted = True
    MacWindows(ax).focus("Visual Studio Code", "notes.md")
    assert ax.raised == ["notes.md - Heliox"]


def test_minimize_maximize_close_and_exists():
    ax = FakeAX()
    windows = MacWindows(ax)
    windows.minimize("Notes")
    assert ax.wins[501][0]["AXMinimized"] is True
    windows.maximize("Electron", "server.py")
    assert ax.frames == [("server.py - Heliox", (0.0, 25.0, 1440.0, 800.0))]
    assert windows.exists("Visual Studio Code", "notes.md")
    windows.close("Visual Studio Code", "notes.md")
    assert ax.pressed == [3] and not windows.exists("Visual Studio Code", "notes.md")
    assert windows.exists("Visual Studio Code")  # the other window is still open: the app was not quit
    assert windows.owner_of("Groceries") == "Notes"
    assert windows.list() == [("Notes", "Groceries"), ("Visual Studio Code", "server.py - Heliox")]
    with pytest.raises(RuntimeError, match="No running application"):
        windows.close("Mail")
    with pytest.raises(RuntimeError, match="no window titled"):
        windows.minimize("Notes", "Shopping")


def test_actions_other_than_focus_need_accessibility():
    windows = MacWindows(FakeAX(trusted=False))
    for action in (windows.minimize, windows.maximize, windows.close):
        with pytest.raises(AccessibilityNeeded):
            action("Notes")


def test_window_mgr_uses_accessibility_on_macos(monkeypatch):
    ax = FakeAX()
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", platform_detect.Platform.MACOS)
    monkeypatch.setattr(macos_windows, "available", lambda: MacWindows(ax))

    async def no_applescript(*_a, **_kw):
        raise AssertionError("AppleScript must not run when the Accessibility route is available")

    monkeypatch.setattr(window_mgr, "run_command", no_applescript)
    assert asyncio.run(window_mgr.window_focus(title="Groceries")) == "Focused window: Groceries"
    assert ax.activated == [501] and ax.raised == ["Groceries"]
    assert asyncio.run(window_mgr.window_close(process_name="Notes")) == "Closed window: Notes"
    assert asyncio.run(window_mgr.window_exists(process_name="Notes")) is False
    listing = asyncio.run(window_mgr.window_list())
    assert "Visual Studio Code - server.py - Heliox" in listing


def test_window_mgr_falls_back_to_applescript_without_pyobjc(monkeypatch):
    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", platform_detect.Platform.MACOS)
    monkeypatch.setattr(macos_windows, "available", lambda: None)
    calls = []

    async def run(args, **_kw):
        calls.append(args)
        return (0, "", "")

    monkeypatch.setattr(window_mgr, "run_command", run)
    asyncio.run(window_mgr.window_focus(process_name="Notes"))
    assert calls == [["osascript", "-e", 'tell application "Notes" to activate']]

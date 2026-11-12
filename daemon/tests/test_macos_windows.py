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
        self.front = 0
        self.honours_activation = True
        self.frontmost_set: list[int] = []
        self.honours_ax = True
        self.honours_launch_services = False
        self.opened: list[str] = []

    def trusted(self):
        return self.is_trusted

    def running_apps(self):
        return self.apps

    def activate(self, pid):
        self.activated.append(pid)
        if self.honours_activation:
            self.front = pid
        return self.honours_activation

    def frontmost_pid(self):
        return self.front

    def set_frontmost(self, pid):
        self.frontmost_set.append(pid)
        if self.honours_ax:
            self.front = pid
        return True

    def open_with_launch_services(self, bundle_id):
        self.opened.append(bundle_id)
        if self.honours_launch_services:
            self.front = next(app.pid for app in self.apps if app.bundle_id == bundle_id)
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
    # The name is an osascript argument ("x" + name), never part of the AppleScript text.
    ((program, flag, script, *args),) = calls
    assert (program, flag, args) == ("osascript", "-e", ["xNotes"])
    assert "tell application (item 1 of heliox_args) to activate" in script and "Notes" not in script


def test_focus_leaves_a_front_app_alone_and_uses_axfrontmost_when_activation_is_ignored():
    ax = FakeAX()
    ax.front = 501
    MacWindows(ax).focus("Notes", settle=0)
    assert ax.activated == []  # already in front: macOS would answer "not activated", which is not a failure
    ax.honours_activation = False  # macOS 14+: a background process's activation request is ignored
    MacWindows(ax).focus("Visual Studio Code", settle=0)
    assert ax.activated == [777] and ax.frontmost_set == [777] and ax.front == 777
    ax.front = 501
    ax.is_trusted = False
    with pytest.raises(RuntimeError, match="Accessibility for Heliox-OS"):
        MacWindows(ax).focus("Visual Studio Code", settle=0)


def test_launch_services_front_pid_parsing():
    from pilot.live import focused_text_macos as M

    assert M.parse_lsappinfo_pid('"pid"=2257\n') == 2257
    assert M.parse_lsappinfo_pid("") == 0 and M.parse_lsappinfo_pid('"pid"=[ NULL ]') == 0
    outputs = {"front": "ASN:0x0-0x1:\n", "info": '"pid"=42\n'}
    assert M.launch_services_front_pid(lambda args: outputs[args[1]]) == 42


def test_apps_sharing_a_name_resolve_to_the_one_meant():
    ax = FakeAX()
    checker = RunningApp(900, "Python", "Python", "org.python.python")
    fixture = RunningApp(901, "Python", "Python", "org.python.python")
    ax.apps = [checker, fixture]
    ax.wins[900] = []
    ax.wins[901] = [{"AXTitle": "Draft - edited", "AXMinimized": False, "AXCloseButton": {"button": 9}}]
    ax.front = 901
    assert MacWindows(ax).focus("Python", "Draft", settle=0) == "Python"
    assert ax.activated == []  # the titled window's app was already in front
    MacWindows(ax).minimize("Python", "Draft")
    assert ax.wins[901][0]["AXMinimized"] is True
    ax.front = 0
    assert MacWindows(ax)._app("Python").pid == 901  # no title, nothing in front: the one with a window


def test_focus_falls_back_to_launch_services_when_activation_and_axfrontmost_are_ignored():
    # CI on a real Mac, 12 November: activateWithOptions(Finder) returned True and Finder did not come forward.
    ax = FakeAX()
    ax.front = 501
    ax.honours_activation = False
    ax.honours_ax = False
    ax.honours_launch_services = True
    windows = MacWindows(ax)
    windows.focus("Visual Studio Code", settle=0)
    assert ax.activated == [777] and ax.frontmost_set == [777] and ax.opened == ["com.microsoft.VSCode"]
    assert windows.last_route == "launch-services" and ax.front == 777
    ax.front = 501
    ax.honours_ax = True
    windows.focus("Visual Studio Code", settle=0)
    assert windows.last_route == "ax-frontmost"
    windows.focus("Visual Studio Code", settle=0)
    assert windows.last_route == "already"


def test_without_accessibility_focus_skips_the_accessibility_route():
    ax = FakeAX(trusted=False)
    ax.front = 501
    ax.honours_activation = False
    ax.honours_launch_services = True
    windows = MacWindows(ax)
    windows.focus("Visual Studio Code", settle=0)
    assert ax.frontmost_set == [] and windows.last_route == "launch-services"


def test_launch_services_is_not_used_for_a_bundle_id_several_apps_share():
    ax = FakeAX()
    checker = RunningApp(900, "Python", "Python", "org.python.python")
    fixture = RunningApp(901, "Python", "Python", "org.python.python")
    ax.apps = [checker, fixture, NOTES]
    ax.wins[901] = [{"AXTitle": "Draft", "AXMinimized": False, "AXCloseButton": {"button": 9}}]
    ax.front = 501
    ax.honours_activation = False
    ax.honours_ax = False
    ax.honours_launch_services = True
    with pytest.raises(RuntimeError, match="did not bring Python"):
        MacWindows(ax).focus("Python", "Draft", settle=0)
    assert ax.opened == []  # "open -b org.python.python" would start another Python, not focus this one

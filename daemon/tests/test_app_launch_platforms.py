"""Launching apps on macOS and Linux: desktop entry Exec lines, matching names to installed apps, and what is
started. Pure parts run on every system; the launch itself is faked."""

from __future__ import annotations

import asyncio
import plistlib
from pathlib import Path

import pytest

from pilot.system import app_names, applications, platform_detect
from pilot.system.desktop_entry import ExecError, desktop_apps, expand_exec, split_exec

# ── the Exec line ──


def test_split_exec_follows_the_quoting_rules():
    assert split_exec("gedit %U") == ["gedit", "%U"]
    assert split_exec('"/opt/My App/run" --flag "a \\"b\\" \\$c \\\\ d"') == [
        "/opt/My App/run",
        "--flag",
        'a "b" $c \\ d',
    ]
    assert split_exec('prog ""') == ["prog", ""]
    assert split_exec("  env  FOO=1   tool  ") == ["env", "FOO=1", "tool"]
    with pytest.raises(ExecError):
        split_exec('prog "unterminated')


def test_expand_exec_puts_arguments_in_place_of_field_codes():
    entry = {"Name": "Text Editor", "Exec": "gnome-text-editor %U", "Icon": "org.gnome.TextEditor"}
    assert expand_exec(entry, ["/tmp/a.txt", "/tmp/b.txt"]) == ["gnome-text-editor", "/tmp/a.txt", "/tmp/b.txt"]
    assert expand_exec(entry) == ["gnome-text-editor"]
    single = {"Name": "Viewer", "Exec": "viewer %f"}
    assert expand_exec(single, ["one.png", "two.png"]) == ["viewer", "one.png"]
    icon = {"Name": "Files", "Exec": "nautilus --new-window %i %U %k", "Icon": "folder"}
    assert expand_exec(icon, ["/home"], path="/usr/share/applications/org.gnome.Nautilus.desktop") == [
        "nautilus",
        "--new-window",
        "--icon",
        "folder",
        "/home",
        "/usr/share/applications/org.gnome.Nautilus.desktop",
    ]
    assert expand_exec({"Name": "x", "Exec": "x %i"}) == ["x"]  # no icon: %i disappears
    named = {"Name": "Calc", "Exec": "calc --title=%c --percent=100%% %d"}
    assert expand_exec(named) == ["calc", "--title=Calc", "--percent=100%"]
    # No file field code: arguments are appended, as on Windows and with open --args.
    assert expand_exec({"Name": "Code", "Exec": "/usr/share/code/code --unity-launch"}, ["--new-window"]) == [
        "/usr/share/code/code",
        "--unity-launch",
        "--new-window",
    ]
    flatpak = {"Name": "App", "Exec": "/usr/bin/flatpak run --branch=stable org.x.App @@u %U @@"}
    assert expand_exec(flatpak, ["https://example.com"])[-3:] == ["@@u", "https://example.com", "@@"]
    with pytest.raises(ExecError):
        expand_exec({"Name": "Empty", "Exec": ""})


ENTRIES = [
    ("org.gnome.Nautilus", "[Desktop Entry]\nName=Files\nExec=nautilus --new-window %U\n"),
    ("code", "[Desktop Entry]\nName=Visual Studio Code\nExec=/usr/share/code/code %F\nTryExec=/usr/share/code/code\n"),
    ("org.gnome.TextEditor", "[Desktop Entry]\nName=Text Editor\nExec=gnome-text-editor %U\n"),
    ("org.kde.kate", "[Desktop Entry]\nName=Kate\nGenericName=Text Editor\nExec=kate -b %U\n"),
    ("removed", "[Desktop Entry]\nName=Removed\nExec=removed\nHidden=true\n"),
    ("helper", "[Desktop Entry]\nName=Background Helper\nExec=helper\nNoDisplay=true\n"),
    ("htop", "[Desktop Entry]\nName=Htop\nExec=htop\nTerminal=true\n"),
    ("site", "[Desktop Entry]\nType=Link\nName=Site\nURL=https://example.com\n"),
]


def test_desktop_apps_and_name_matching():
    apps = desktop_apps(ENTRIES, which=lambda program: None if program == "/usr/share/code/code" else "/usr/bin/x")
    ids = {app.desktop_id for app in apps}
    assert "code" not in ids  # TryExec missing
    assert "removed" not in ids and "site" not in ids
    find = applications.resolve_linux_desktop_app
    assert find("Files", apps).desktop_id == "org.gnome.Nautilus"
    assert find("nautilus", apps).desktop_id == "org.gnome.Nautilus"
    assert find("text editor", apps).desktop_id == "org.gnome.TextEditor"
    assert find("kate", apps).desktop_id == "org.kde.kate"
    assert find("helper", apps).desktop_id == "helper"  # hidden from menus: only its exact id
    assert find("Background Helper", apps) is None
    assert find("spreadsheet", apps) is None


def test_two_equally_good_desktop_matches_fail_closed():
    entries = [
        ("a", "[Desktop Entry]\nName=Notes\nExec=notes-a\n"),
        ("b", "[Desktop Entry]\nName=Notes\nExec=notes-b\n"),
    ]
    with pytest.raises(applications.ApplicationResolutionError, match="ambiguous"):
        applications.resolve_linux_desktop_app("notes", desktop_apps(entries))


def _linux(monkeypatch, entries, which):
    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform_detect.Platform.LINUX)
    monkeypatch.setattr(app_names, "_desktop_entries", lambda: tuple(entries))
    monkeypatch.setattr(applications.shutil, "which", which)
    started: list[list[str]] = []

    async def start(command, display_name):
        started.append(command)
        return 1

    monkeypatch.setattr(applications, "_start_and_check", start)
    return started


def test_linux_open_by_shown_name_with_a_file(monkeypatch):
    started = _linux(monkeypatch, ENTRIES, lambda p: {"gnome-text-editor": "/usr/bin/gnome-text-editor"}.get(p))
    result = asyncio.run(applications.launch_application("Text Editor", ["/home/sam/notes.txt"]))
    assert started == [["gnome-text-editor", "/home/sam/notes.txt"]]
    assert "Text Editor" in result and "org.gnome.TextEditor" in result


def test_linux_prefers_a_program_on_path(monkeypatch):
    started = _linux(monkeypatch, ENTRIES, lambda p: "/usr/bin/firefox" if p == "firefox" else None)
    assert "PATH" in asyncio.run(applications.launch_application("firefox", ["--private-window"]))
    assert started == [["/usr/bin/firefox", "--private-window"]]


def test_linux_terminal_apps_and_missing_programs(monkeypatch):
    started = _linux(
        monkeypatch,
        ENTRIES,
        lambda p: {"htop": "/usr/bin/htop", "x-terminal-emulator": "/usr/bin/x-terminal-emulator"}.get(p),
    )
    asyncio.run(applications.launch_application("Htop"))
    assert started[-1] == ["/usr/bin/x-terminal-emulator", "-e", "htop"]
    _linux(monkeypatch, ENTRIES, lambda p: None)
    with pytest.raises(applications.ApplicationLaunchError, match="not installed"):
        asyncio.run(applications.launch_application("Files"))


# ── macOS ──


def test_mac_bundles_match_by_file_and_shown_name(tmp_path):
    apps = tmp_path / "Applications"
    for stem, shown in (("Visual Studio Code", "Code"), ("Safari", None), ("Safari Technology Preview", None)):
        contents = apps / f"{stem}.app" / "Contents"
        contents.mkdir(parents=True)
        info = {"CFBundleName": shown or stem}
        (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    bundles = applications.mac_application_bundles([apps, tmp_path / "missing"])
    find = applications.resolve_mac_application
    assert Path(find("visual studio code", bundles)).name == "Visual Studio Code.app"
    assert Path(find("Code", bundles)).name == "Visual Studio Code.app"
    assert Path(find("safari", bundles)).name == "Safari.app"  # exact beats partial
    assert find("Mail", bundles) is None


def test_mac_launch_opens_the_matched_bundle_with_arguments(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    async def run(command, **_kw):
        calls.append(command)
        return (0, "", "")

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform_detect.Platform.MACOS)
    monkeypatch.setattr(platform_detect, "run_command", run)
    bundle = str(tmp_path / "Slack.app")
    monkeypatch.setattr(applications, "mac_application_bundles", lambda: [applications._NamedTarget("Slack", bundle)])
    result = asyncio.run(applications.launch_application("slack", ["--background"]))
    assert calls == [["open", "-a", bundle, "--args", "--background"]]
    assert "Slack" in result

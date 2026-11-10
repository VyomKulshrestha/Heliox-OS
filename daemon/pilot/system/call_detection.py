"""Is the person in a call? Then Heliox stays quiet and does not take what is said as addressed to it.

Windows only for now; on macOS and Linux ``current_call`` always answers "" (not in a call), so nothing changes
there.

Three signals are read, none needing a permission:
- Windows' record of which apps are using the microphone (the capability-access store: an app with a start time
  and no stop time is using it now). Heliox's own process is excluded. On the machine this was built on the
  store had no entries at all for a day of microphone use, so it is treated as a bonus, not relied on.
- Meeting processes that exist only during a call (Zoom's ``CptHost.exe``).
- Window titles of call apps and browser meetings ("Meet - …", "Zoom Meeting", a Teams meeting or call window,
  a WhatsApp voice or video call, a Slack huddle).

``decide`` is pure, so the rules are tested on plain lists.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass

logger = logging.getLogger("pilot.system.call_detection")

# Processes that run only while a meeting is in progress.
MEETING_PROCESSES = {"cpthost.exe": "Zoom"}
# (pattern on a window title, name of the call app). Matched case-insensitively.
CALL_TITLES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^meet\s*[-–]\s|google meet", re.I), "Google Meet"),
    (re.compile(r"^zoom meeting$|zoom meeting\b|zoom webinar", re.I), "Zoom"),
    (
        re.compile(
            r"\b(?:meeting|call) (?:with|in)\b.*\bmicrosoft teams\b|\bmicrosoft teams\b.*\b(?:meeting|call)\b", re.I
        ),
        "Microsoft Teams",
    ),  # fmt: skip
    (re.compile(r"\bwhatsapp\b.*\b(?:voice|video) call\b|\b(?:voice|video) call\b.*\bwhatsapp\b", re.I), "WhatsApp"),
    (re.compile(r"\bhuddle\b.*\bslack\b|\bslack\b.*\bhuddle\b", re.I), "Slack"),
    (re.compile(r"\bwebex\b.*\bmeeting\b|\bcisco webex meetings?\b", re.I), "Webex"),
    (re.compile(r"\bdiscord\b.*\bvoice connected\b", re.I), "Discord"),
)


@dataclass(frozen=True)
class MicUser:
    app: str  # a package name, or an executable path written with "#" for "\\" as Windows stores it
    in_use: bool


def _exe_name(app: str) -> str:
    return re.split(r"[#\\/]", app)[-1].casefold()


def decide(
    mic_users: Iterable[MicUser],
    processes: Iterable[str],
    titles: Iterable[str],
    own_executables: Iterable[str] = (),
) -> str:
    """The call app the person is in a call with, or "" when they are not in one."""
    own = {_exe_name(path) for path in own_executables if path}
    own_paths = {path.casefold().replace("\\", "#") for path in own_executables if path}
    for user in mic_users:
        if not user.in_use:
            continue
        if user.app.casefold() in own_paths or _exe_name(user.app) in own:
            continue  # Heliox's own listening
        name = re.split(r"[#\\/]", user.app)[-1]
        return name[:-4] if name.casefold().endswith(".exe") else name or user.app
    for name in processes:
        app = MEETING_PROCESSES.get(name.casefold())
        if app:
            return app
    for title in titles:
        for pattern, app in CALL_TITLES:
            if pattern.search(title.strip()):
                return app
    return ""


# ── reading the signals on Windows ──


def _mic_users() -> list[MicUser]:
    import winreg

    users: list[MicUser] = []
    root = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"

    def visit(key: winreg.HKEYType, name: str) -> None:
        try:
            start, _ = winreg.QueryValueEx(key, "LastUsedTimeStart")
            stop, _ = winreg.QueryValueEx(key, "LastUsedTimeStop")
            users.append(MicUser(app=name, in_use=bool(start) and not stop))
        except OSError:
            pass
        index = 0
        while True:
            try:
                child = winreg.EnumKey(key, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(key, child) as sub:
                    visit(sub, child)
            except OSError:
                continue

    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, root) as key:
                visit(key, "")
        except OSError:
            continue
    return users


def _processes() -> list[str]:
    import psutil

    names = []
    for proc in psutil.process_iter(["name"]):
        name = proc.info.get("name")
        if name:
            names.append(str(name))
    return names


def _window_titles() -> list[str]:
    import win32gui

    titles: list[str] = []

    def collect(hwnd: int, _extra: object) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                titles.append(title)
        return True

    win32gui.EnumWindows(collect, None)
    return titles


def current_call() -> str:
    """The call app the person is in a call with right now, or "". Always "" off Windows."""
    if sys.platform != "win32":
        return ""
    readers = (("microphone users", _mic_users), ("processes", _processes), ("window titles", _window_titles))
    values: dict[str, list] = {}
    for label, reader in readers:
        try:
            values[label] = reader()
        except Exception:
            logger.debug("Could not read %s for call detection", label, exc_info=True)
            values[label] = []
    own = [sys.executable, os.path.realpath(sys.executable), getattr(sys, "_base_executable", "")]
    return decide(values["microphone users"], values["processes"], values["window titles"], own)

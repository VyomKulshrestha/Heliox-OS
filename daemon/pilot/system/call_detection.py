"""Is the person in a call? Then Heliox stays quiet and does not take what is said as addressed to it.

``current_call()`` is the one entry point on every system (the listener and Live both call it). Each system has
its own readers; a reader that cannot tell answers None and is simply left out, so a missing tool or an older
system means "not known to be in a call", never an error.

macOS (no permission needed):
- CoreAudio's list of processes that are recording (macOS 14 and later: the audio process objects, each with its
  PID and whether its input is running). Heliox's own process is excluded, so its own listening never counts.
- On older macOS: whether the default input device is running somewhere, which counts only when Heliox is not
  listening itself (it cannot say which process), and then together with a call app being open.
Linux:
- PulseAudio's or PipeWire's (through pipewire-pulse) list of recording streams, ``pactl list source-outputs``:
  the application and its PID for every stream reading a real input (monitors of outputs are not microphones).
- Meeting processes as on Windows.

Windows, three signals, none needing a permission:
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
    pid: int | None = None  # macOS and Linux: the recording process


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


# ── macOS: CoreAudio, through ctypes (no extra package, no permission) ──

CALL_APPS_MACOS = {
    "zoom.us": "Zoom",
    "Microsoft Teams": "Microsoft Teams",
    "MSTeams": "Microsoft Teams",
    "FaceTime": "FaceTime",
    "Webex": "Webex",
    "Slack": "Slack",
    "Discord": "Discord",
    "WhatsApp": "WhatsApp",
    "Skype": "Skype",
}


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


def _coreaudio_property(object_id: int, selector: str, kind: type, many: bool = False):
    """Read one CoreAudio property: a value of ``kind``, or a list of them; None when it cannot be read."""
    import ctypes

    class Address(ctypes.Structure):
        _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]

    audio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    address = Address(_fourcc(selector), _fourcc("glob"), 0)
    size = ctypes.c_uint32(0)
    if audio.AudioObjectGetPropertyDataSize(object_id, ctypes.byref(address), 0, None, ctypes.byref(size)) != 0:
        return None
    count = max(1, size.value // ctypes.sizeof(kind)) if many else 1
    buffer = (kind * count)()
    size = ctypes.c_uint32(ctypes.sizeof(buffer))
    if audio.AudioObjectGetPropertyData(object_id, ctypes.byref(address), 0, None, ctypes.byref(size), buffer) != 0:
        return None
    values = list(buffer)[: size.value // ctypes.sizeof(kind)]
    return values if many else (values[0] if values else None)


def _recording_pids_macos() -> list[int] | None:
    """PIDs of the processes recording right now (macOS 14+), or None when this macOS cannot tell."""
    import ctypes

    objects = _coreaudio_property(1, "prs#", ctypes.c_uint32, many=True)  # kAudioHardwarePropertyProcessObjectList
    if objects is None:
        return None
    pids = []
    for process in objects:
        running = _coreaudio_property(process, "piri", ctypes.c_uint32)  # kAudioProcessPropertyIsRunningInput
        pid = _coreaudio_property(process, "ppid", ctypes.c_int32)  # kAudioProcessPropertyPID
        if running and pid:
            pids.append(int(pid))
    return pids


def _default_input_busy_macos() -> bool | None:
    """Whether any process is using the default input device (older macOS), or None when it cannot be read."""
    import ctypes

    device = _coreaudio_property(1, "dIn ", ctypes.c_uint32)  # kAudioHardwarePropertyDefaultInputDevice
    if not device:
        return None
    running = _coreaudio_property(device, "gone", ctypes.c_uint32)  # kAudioDevicePropertyDeviceIsRunningSomewhere
    return None if running is None else bool(running)


def mac_mic_users(pids: list[int] | None, own_pids: set[int], name_of) -> list[MicUser] | None:
    """Recording processes other than Heliox, as MicUser (pure: names come from ``name_of``)."""
    if pids is None:
        return None
    return [MicUser(name_of(pid) or "an app", True, pid) for pid in pids if pid not in own_pids]


def mac_call_from_busy_input(busy: bool | None, heliox_listening: bool, processes: list[str]) -> str:
    """Older macOS: the input is in use but CoreAudio cannot say by whom. Only when Heliox is not listening itself,
    and a call app is open, is that taken as a call."""
    if not busy or heliox_listening:
        return ""
    for name in processes:
        app = CALL_APPS_MACOS.get(name)
        if app:
            return app
    return ""


# ── Linux: PulseAudio or PipeWire recording streams ──


def parse_pactl_source_outputs(text: str, monitor_sources: set[str]) -> list[MicUser]:
    """``pactl list source-outputs`` as MicUser: one per stream reading a real input (pure)."""
    users = []
    for block in re.split(r"\n(?=Source Output #)", text):
        if not block.strip().startswith("Source Output #"):
            continue
        source = re.search(r"^\s*Source:\s*(\S+)", block, re.M)
        if source and source.group(1) in monitor_sources:
            continue  # recording what is played (a screen recorder), not a microphone
        name = re.search(r'application\.name = "([^"]*)"', block)
        binary = re.search(r'application\.process\.binary = "([^"]*)"', block)
        pid = re.search(r'application\.process\.id = "(\d+)"', block)
        corked = re.search(r"^\s*Corked:\s*yes", block, re.M)
        label = (name.group(1) if name else "") or (binary.group(1) if binary else "") or "an application"
        users.append(MicUser(label, in_use=not corked, pid=int(pid.group(1)) if pid else None))
    return users


def parse_pactl_monitor_sources(text: str) -> set[str]:
    """Indexes and names of monitor sources from ``pactl list short sources``."""
    monitors = set()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].endswith(".monitor"):
            monitors.update({parts[0], parts[1]})
    return monitors


def _mic_users_linux(own_pids: set[int], run=None) -> list[MicUser] | None:
    import shutil
    import subprocess

    run = run or (lambda args: subprocess.run(args, capture_output=True, text=True, timeout=3))
    if run is None or not shutil.which("pactl"):
        return None
    try:
        sources = run(["pactl", "list", "short", "sources"])
        outputs = run(["pactl", "list", "source-outputs"])
    except Exception:
        return None
    if outputs.returncode != 0:
        return None
    users = parse_pactl_source_outputs(outputs.stdout, parse_pactl_monitor_sources(sources.stdout or ""))
    return [user for user in users if user.pid not in own_pids]


def _own_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        import psutil

        pids.update(child.pid for child in psutil.Process().children(recursive=True))
    except Exception:
        pass
    return pids


def _process_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:
        return ""


def _read(label: str, reader):
    try:
        return reader()
    except Exception:
        logger.debug("Could not read %s for call detection", label, exc_info=True)
        return None


def current_call(heliox_listening: bool = True) -> str:
    """The call app the person is in a call with right now, or "" (not in one, or it cannot be told).

    ``heliox_listening``: Heliox's own listener has the microphone open. Only older macOS needs it (CoreAudio there
    says the input is busy but not by whom); everywhere else Heliox's own process is excluded by name or PID.
    """
    if sys.platform == "darwin":
        own = _own_pids()
        pids = _read("CoreAudio recording processes", _recording_pids_macos)
        users = mac_mic_users(pids, own, _process_name) if pids is not None else None
        processes = _read("processes", _processes) or []
        if users is not None:
            return decide(users, processes, [])
        busy = _read("CoreAudio default input", _default_input_busy_macos)
        return mac_call_from_busy_input(busy, heliox_listening, processes) or decide([], processes, [])
    if sys.platform.startswith("linux"):
        users = _read("PulseAudio recording streams", lambda: _mic_users_linux(_own_pids()))
        processes = _read("processes", _processes) or []
        return decide(users or [], processes, [])
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

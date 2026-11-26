"""Power management — shutdown, restart, sleep, lock, logout.

Cross-platform: Windows (shutdown.exe/rundll32), Linux (systemctl/loginctl),
macOS (pmset/osascript).
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.power")


async def power_transition_observed(transition: str) -> bool:
    """Observe a host-reported shutdown, restart, or logout transition.

    A missing or unavailable host signal is a failed observation; callers must
    not convert the executor's successful return value into verification.
    """
    if transition not in {"shutdown", "restart", "logout"}:
        raise ValueError(f"Unsupported power transition: {transition}")

    if CURRENT_PLATFORM == Platform.WINDOWS:
        event_id = 4647 if transition == "logout" else 1074
        code, out, err = await run_powershell(
            "$event = Get-WinEvent -FilterHashtable "
            f"@{{LogName='{('Security' if transition == 'logout' else 'System')}'; Id={event_id}; "
            "StartTime=(Get-Date).AddSeconds(-30)}} -ErrorAction SilentlyContinue | "
            "Select-Object -First 1; if ($event) { $event.Message }"
        )
        if code != 0:
            raise RuntimeError(f"Power transition observation failed: {err.strip()}")
        message = out.casefold()
        if transition == "logout":
            return bool(message.strip())
        expected = ("restart", "reboot") if transition == "restart" else ("shutdown", "shut down", "power off")
        return any(marker in message for marker in expected)

    if CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(
            [
                "log",
                "show",
                "--last",
                "30s",
                "--style",
                "compact",
                "--predicate",
                'process == "loginwindow" OR process == "powerd"',
            ]
        )
        if code != 0:
            raise RuntimeError(f"Power transition observation failed: {err.strip()}")
        message = out.casefold()
        markers = {
            "shutdown": ("shutdown", "shut down", "power off"),
            "restart": ("restart", "reboot"),
            "logout": ("logout", "log out", "session exit"),
        }[transition]
        return any(marker in message for marker in markers)

    if transition == "logout":
        code, out, err = await run_command(["loginctl", "show-session", "self", "--property=State", "--value"])
        if code != 0:
            error = f"{out}\n{err}".casefold()
            if "no session" in error or "not known" in error or "not found" in error:
                return True
            raise RuntimeError(f"Power transition observation failed: {err.strip() or out.strip()}")
        return out.strip().casefold() in {"closing", "offline"}

    target = "reboot.target" if transition == "restart" else "poweroff.target"
    code, out, err = await run_command(["systemctl", "list-jobs", "--no-legend", "--plain"])
    if code != 0:
        raise RuntimeError(f"Power transition observation failed: {err.strip()}")
    return any(target in line and "cancel" not in line.casefold() for line in out.splitlines())


async def shutdown(delay_seconds: int = 0, force: bool = False) -> str:
    """Shut down the system."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        args = ["shutdown", "/s"]
        if force:
            args.append("/f")
        args.extend(["/t", str(delay_seconds)])
        code, out, err = await run_command(args)
    elif CURRENT_PLATFORM == Platform.MACOS:
        if delay_seconds > 0:
            minutes = max(1, delay_seconds // 60)
            code, out, err = await run_command(["sudo", "shutdown", "-h", f"+{minutes}"])
        else:
            code, out, err = await run_command(["osascript", "-e", 'tell app "System Events" to shut down'])
    else:
        if delay_seconds > 0:
            minutes = max(1, delay_seconds // 60)
            code, out, err = await run_command(["shutdown", "-h", f"+{minutes}"], root=True)
        else:
            code, out, err = await run_command(["systemctl", "poweroff"], root=True)

    if code != 0:
        raise RuntimeError(f"Shutdown failed: {err.strip()}")
    return f"System shutdown initiated (delay: {delay_seconds}s)"


async def restart(delay_seconds: int = 0, force: bool = False) -> str:
    """Restart the system."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        args = ["shutdown", "/r"]
        if force:
            args.append("/f")
        args.extend(["/t", str(delay_seconds)])
        code, out, err = await run_command(args)
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["osascript", "-e", 'tell app "System Events" to restart'])
    else:
        code, out, err = await run_command(["systemctl", "reboot"], root=True)

    if code != 0:
        raise RuntimeError(f"Restart failed: {err.strip()}")
    return "System restart initiated"


async def sleep() -> str:
    """Put the system to sleep."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.Application]::SetSuspendState("
            "[System.Windows.Forms.PowerState]::Suspend, $false, $false)"
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["pmset", "sleepnow"])
    else:
        code, out, err = await run_command(["systemctl", "suspend"])

    if code != 0:
        raise RuntimeError(f"Sleep failed: {err.strip()}")
    return "System sleep initiated"


async def lock_screen() -> str:
    """Lock the screen."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["rundll32.exe", "user32.dll,LockWorkStation"])
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(
            ["osascript", "-e", 'tell application "System Events" to keystroke "q" using {command down, control down}']
        )
    else:
        code, out, err = await run_command(["loginctl", "lock-session"])

    if code != 0:
        raise RuntimeError(f"Lock failed: {err.strip()}")
    return "Screen locked"


# WTSQuerySessionInformationW(WTSSessionInfoEx) answers a WTSINFOEXW: Level (4 bytes, then padding to 8), then
# WTSINFOEX_LEVEL1_W: SessionId, SessionState, SessionFlags. SessionFlags is WTS_SESSIONSTATE_LOCK (0) or
# WTS_SESSIONSTATE_UNLOCK (1) from Windows 8 on (Windows 7 swapped the two; it is not supported).
WTS_SESSION_INFO_EX = 25
WTS_CURRENT_SESSION = 0xFFFFFFFF
WTS_SESSIONSTATE_LOCK = 0
WTS_SESSIONSTATE_UNLOCK = 1


class LockReading(NamedTuple):
    """Whether the session's screen is locked, as the session manager reports it.

    ``locked`` is None when this platform gave no answer that can be trusted; ``source`` names what was read (or
    why nothing could be), in words for the verification result."""

    locked: bool | None
    source: str


def parse_wts_session_flags(raw: bytes) -> bool | None:
    """Locked (True), unlocked (False) or unknown (None) from a WTSINFOEXW buffer."""
    if len(raw) < 20 or int.from_bytes(raw[0:4], "little") != 1:
        return None
    flags = int.from_bytes(raw[16:20], "little", signed=True)
    if flags == WTS_SESSIONSTATE_LOCK:
        return True
    if flags == WTS_SESSIONSTATE_UNLOCK:
        return False
    return None


def _windows_session_info() -> bytes:
    """This process's session record from the Terminal Services API: a read that needs no elevation."""
    import ctypes

    wts = ctypes.WinDLL("wtsapi32", use_last_error=True)
    wts.WTSQuerySessionInformationW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    wts.WTSFreeMemory.argtypes = [ctypes.c_void_p]
    buffer = ctypes.c_void_p()
    size = ctypes.c_uint32()
    if not wts.WTSQuerySessionInformationW(
        None, WTS_CURRENT_SESSION, WTS_SESSION_INFO_EX, ctypes.byref(buffer), ctypes.byref(size)
    ):
        raise OSError(ctypes.get_last_error(), "WTSQuerySessionInformationW failed")
    try:
        return ctypes.string_at(buffer, size.value)
    finally:
        wts.WTSFreeMemory(buffer)


def parse_mac_console_lock(plist_bytes: bytes, uid: int) -> bool | None:
    """Locked, unlocked or unknown for ``uid`` from ``ioreg -n Root -d1 -a`` (the IOConsoleUsers list that
    CGSessionCopyCurrentDictionary also reads). The lock key is present only while the screen is locked."""
    import plistlib

    try:
        root = plistlib.loads(plist_bytes)
    except Exception:
        return None
    if isinstance(root, list):
        root = root[0] if root and isinstance(root[0], dict) else {}
    if not isinstance(root, dict):
        return None
    users = root.get("IOConsoleUsers")
    if not isinstance(users, list):
        return None
    for user in users:
        if isinstance(user, dict) and user.get("kCGSSessionUserIDKey") == uid:
            return bool(user.get("CGSSessionScreenIsLocked", False))
    return None


async def screen_locked() -> LockReading:
    """Read whether this session is locked through the session manager, never through the command that locked it.

    Windows: the session's lock flag from WTSQuerySessionInformationW. macOS: CGSSessionScreenIsLocked in the
    console users' record (``ioreg``). Linux: logind's LockedHint for this session, which a screen locker sets when
    it has locked; ``loginctl lock-session`` only sends logind's Lock signal and exits 0 whether or not a locker
    acted on it. Raises when the reading itself fails."""
    import asyncio

    if CURRENT_PLATFORM == Platform.WINDOWS:
        raw = await asyncio.to_thread(_windows_session_info)
        locked = parse_wts_session_flags(raw)
        if locked is None:
            return LockReading(None, "Windows reported no lock state for this session")
        return LockReading(locked, "the Windows session's lock flag")

    if CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["ioreg", "-n", "Root", "-d1", "-a"], timeout=10)
        if code != 0:
            raise RuntimeError(f"Lock state read failed: {err.strip() or 'ioreg exited ' + str(code)}")
        locked = parse_mac_console_lock(out.encode("utf-8"), os.getuid())
        if locked is None:
            return LockReading(None, "macOS listed no console session for this user")
        return LockReading(locked, "macOS's console session record")

    session = os.environ.get("XDG_SESSION_ID") or "self"
    code, out, err = await run_command(
        ["loginctl", "show-session", session, "--property=LockedHint", "--value"], timeout=10
    )
    if code == -1 and "not found" in err:
        return LockReading(None, "this system has no loginctl (no systemd-logind) to ask")
    if code != 0:
        raise RuntimeError(f"Lock state read failed: {err.strip() or out.strip()}")
    value = out.strip().casefold()
    if value not in {"yes", "no"}:
        return LockReading(None, "logind gave no LockedHint for this session")
    return LockReading(value == "yes", "logind's LockedHint for this session")


async def logout() -> str:
    """Log out the current user."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["shutdown", "/l"])
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["osascript", "-e", 'tell app "System Events" to log out'])
    else:
        code, out, err = await run_command(["loginctl", "terminate-session", "self"])

    if code != 0:
        raise RuntimeError(f"Logout failed: {err.strip()}")
    return "User logout initiated"

"""Power management — shutdown, restart, sleep, lock, logout.

Cross-platform: Windows (shutdown.exe/rundll32), Linux (systemctl/loginctl),
macOS (pmset/osascript).
"""

from __future__ import annotations

import logging

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.power")


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

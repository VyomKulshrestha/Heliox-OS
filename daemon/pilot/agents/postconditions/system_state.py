"""Post-condition verifiers for system state: the lock screen, file modes, the registry, services, mounts, Wi-Fi
and package updates.

Each check asks the system afresh through a different call than the handler made (the session manager, the file
system, a new PowerShell process, systemctl, findmnt, the WLAN API, the package database) and says plainly when a
platform gives it nothing to read.
"""

from __future__ import annotations

import asyncio
import os

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform

# Locking, joining a network and mounting finish a moment after the command returns: the check reads the state
# again a fixed number of times, so a test counts readings instead of waiting on a clock.
LOCK_READINGS = 10
WIFI_READINGS = 10
READING_INTERVAL_SECONDS = 0.5


async def _pause() -> None:
    await asyncio.sleep(READING_INTERVAL_SECONDS)


@register(ActionType.POWER_LOCK, method="session_locked_postcondition", timeout=20.0)
async def session_locked(result: ActionResult) -> Outcome:
    """The session is locked, read from the session manager (Windows session flag, macOS console record, logind)."""
    from pilot.system.power import screen_locked

    reading = await screen_locked()
    for _ in range(LOCK_READINGS - 1):
        if reading.locked is not False:
            break
        await _pause()
        reading = await screen_locked()
    if reading.locked is None:
        return unobserved(f"Could not read the lock state: {reading.source}")
    if reading.locked:
        return verified(f"The session is locked ({reading.source})")
    seconds = LOCK_READINGS * READING_INTERVAL_SECONDS
    hint = (
        " On Linux the lock command only asks a screen locker to act; none set the session's lock flag."
        if CURRENT_PLATFORM == Platform.LINUX
        else ""
    )
    return mismatch(
        f"Expected the screen to be locked; {reading.source} still says unlocked after {seconds:g} s.{hint}"
    )


@register(ActionType.FILE_PERMISSIONS, method="conditional_file_mode_postcondition", timeout=10.0)
async def file_mode_matches(result: ActionResult) -> Outcome:
    """The path's mode bits are exactly the octal mode asked for, read from the file system with stat."""
    from pilot.actions import FileParams

    params: FileParams = result.action.parameters  # type: ignore[assignment]
    if params.permissions is None:
        return unobserved("A permissions read changes nothing, so there is no new state to observe")
    if CURRENT_PLATFORM == Platform.WINDOWS:
        # Windows keeps access in the file's ACL and the handler sets no mode there. A handler that set an ACL
        # would be checked by reading the ACL back (Get-Acl or icacls in a new process) and comparing its entries.
        return unobserved("Windows has no POSIX file mode to read back; access lives in the file's ACL")
    if not params.path:
        raise ValueError("File permissions change has no path to observe")
    expected = int(params.permissions, 8) & 0o7777
    try:
        found = (await asyncio.to_thread(os.stat, params.path)).st_mode & 0o7777
    except FileNotFoundError:
        return mismatch(f"Expected mode {expected:04o} on {params.path}; the path no longer exists")
    if found != expected:
        return mismatch(f"Expected mode {expected:04o} on {params.path}; found {found:04o}")
    return verified(f"{params.path} has mode {found:04o}")

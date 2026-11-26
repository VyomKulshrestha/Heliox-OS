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


_REGISTRY_KIND_NAMES = {
    "String": "REG_SZ",
    "ExpandString": "REG_EXPAND_SZ",
    "Binary": "REG_BINARY",
    "DWord": "REG_DWORD",
    "MultiString": "REG_MULTI_SZ",
    "QWord": "REG_QWORD",
}
_REGISTRY_NUMBER_MASKS = {"DWord": 0xFFFFFFFF, "QWord": 0xFFFFFFFFFFFFFFFF}


@register(ActionType.REGISTRY_WRITE, method="registry_value_postcondition", timeout=25.0)
async def registry_value_matches(result: ActionResult) -> Outcome:
    """The value exists with the requested data and kind, read back in a new PowerShell process."""
    from pilot.actions import RegistryParams
    from pilot.agents.executor import REGISTRY_KINDS
    from pilot.system.registry_readback import read_registry_value

    if CURRENT_PLATFORM != Platform.WINDOWS:
        return unobserved("Only Windows has a registry to read back")
    params: RegistryParams = result.action.parameters  # type: ignore[assignment]
    kind = REGISTRY_KINDS.get(str(params.value_type or "").strip().upper())
    if not params.key_path or not params.value_name or params.value_data is None or kind is None:
        raise ValueError("Registry write has no key, value name, data or known type to observe")
    where = f"{params.key_path}\\{params.value_name}"
    wanted_kind = _REGISTRY_KIND_NAMES[kind]
    found = await read_registry_value(params.key_path, params.value_name, kind, params.value_data)
    if not found.key_exists:
        return mismatch(f"Expected {where} as {wanted_kind}; the key does not exist")
    if not found.present:
        return mismatch(f"Expected {where} as {wanted_kind}; the key has no value of that name")
    found_kind = _REGISTRY_KIND_NAMES.get(found.kind or "", found.kind or "an unknown kind")
    if found.kind != kind:
        return mismatch(f"Expected {where} as {wanted_kind}; it is {found_kind}")
    data = found.data or []
    if kind in {"String", "ExpandString"}:
        expected: list = [params.value_data]
    elif found.expected is None:
        return unobserved(
            f"{where} exists as {wanted_kind}, but the requested data could not be read as {wanted_kind}, "
            "so the data was not compared"
        )
    else:
        expected = found.expected
    mask = _REGISTRY_NUMBER_MASKS.get(kind)
    if mask is not None:
        data = [int(item) & mask for item in data]
        expected = [int(item) & mask for item in expected]
    if data != expected:
        if kind in {"String", "ExpandString", "MultiString"}:
            # Text may be a secret someone stored: the sentence gives its size, never its content.
            return mismatch(
                f"Expected {where} to hold the requested {wanted_kind} text ({_text_size(expected)}); "
                f"it holds different text ({_text_size(data)})"
            )
        return mismatch(f"Expected {where} = {_shown(expected)} ({wanted_kind}); found {_shown(data)}")
    return verified(f"{where} holds the requested {wanted_kind} data")


def _text_size(values: list) -> str:
    characters = sum(len(str(item)) for item in values)
    return f"{characters} characters" + (f" in {len(values)} strings" if len(values) != 1 else "")


def _shown(values: list) -> str:
    """Numbers or bytes for a sentence: the value alone when there is one."""
    text = str(values[0]) if len(values) == 1 else "[" + ", ".join(str(item) for item in values[:16]) + "]"
    return text if len(text) <= 120 else text[:117] + "..."

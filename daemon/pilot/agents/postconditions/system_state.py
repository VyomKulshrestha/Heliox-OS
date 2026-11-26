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
    if CURRENT_PLATFORM != Platform.LINUX:
        return mismatch(
            f"Expected the screen to be locked; the session does not report itself as locked ({reading.source}, "
            f"after {seconds:g} s)"
        )
    # logind's hint is set by the screen locker. Some lockers lock without setting it, so ask the desktop's own
    # screensaver interface before calling it a failure.
    from pilot.system.power import linux_screensaver_active

    active = await linux_screensaver_active()
    if active:
        return unobserved(
            "Could not verify the lock: logind's LockedHint says unlocked, but the desktop's screensaver reports "
            "itself active (a screen locker that does not set the hint may have locked the screen)"
        )
    second = (
        "the desktop's screensaver also reports itself inactive"
        if active is False
        else "no desktop screensaver interface answered"
    )
    return mismatch(
        f"Expected the screen to be locked; the session does not report itself as locked (LockedHint is still no "
        f"after {seconds:g} s, and {second}). The lock command only asks a screen locker to act; none did, or none "
        "is running."
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


# What `systemctl is-enabled` prints after a successful enable or disable, and why other states fail the action.
_STARTS_AT_BOOT = {"enabled"}
_KEPT_FROM_BOOT = {"disabled", "masked", "masked-runtime"}
_SERVICE_STATE_MEANING = {
    "enabled-runtime": "enabled only until the next reboot",
    "linked": "linked from outside the unit search path, not enabled",
    "linked-runtime": "linked only until the next reboot",
    "alias": "an alias of another unit, whose own state decides",
    "static": "static: it has no [Install] section, so enabling or disabling it changes nothing; another unit pulls it in",
    "indirect": "indirect: another unit enables it",
    "generated": "generated at boot by a generator, so enable and disable do not apply",
    "transient": "transient: it exists only for this boot",
    "bad": "a unit file systemd cannot read",
    "not-found": "not found",
}


async def _service_boot_state(result: ActionResult, wanted: set[str], expectation: str) -> Outcome:
    from pilot.actions import ServiceParams
    from pilot.system.systemctl import is_enabled

    if CURRENT_PLATFORM != Platform.LINUX:
        # The service actions drive systemctl only. On Windows and macOS systemctl does not exist, the handler
        # raises, and the action fails before this check runs; a Windows check would read the service's start type
        # (Get-Service), a macOS one launchctl print-disabled, once handlers for those platforms exist.
        return unobserved("Services are managed through systemd here, which only Linux has")
    params: ServiceParams = result.action.parameters  # type: ignore[assignment]
    if not params.name:
        raise ValueError("Service action has no unit name to observe")
    state = await is_enabled(params.name, user_scope=params.user_scope)
    if state in wanted:
        return verified(f"systemctl is-enabled says {params.name} is {state}")
    meaning = _SERVICE_STATE_MEANING.get(state, state)
    return mismatch(f"Expected {params.name} {expectation}; systemctl is-enabled says it is {meaning}")


@register(ActionType.SERVICE_ENABLE, method="service_enabled_postcondition", timeout=20.0)
async def service_enabled(result: ActionResult) -> Outcome:
    """The unit starts at boot: `systemctl is-enabled` prints enabled."""
    return await _service_boot_state(result, _STARTS_AT_BOOT, "to start at boot (enabled)")


@register(ActionType.SERVICE_DISABLE, method="service_disabled_postcondition", timeout=20.0)
async def service_disabled(result: ActionResult) -> Outcome:
    """The unit will not start at boot: `systemctl is-enabled` prints disabled or masked."""
    return await _service_boot_state(result, _KEPT_FROM_BOOT, "not to start at boot (disabled or masked)")


@register(ActionType.DISK_MOUNT, method="mount_presence_postcondition", timeout=20.0)
async def disk_mounted(result: ActionResult) -> Outcome:
    """The requested device is mounted at the requested path, read from the mount table (findmnt, mount)."""
    from pilot.actions import DiskManageParams
    from pilot.system.disks import device_matches, mounted_source

    if CURRENT_PLATFORM == Platform.WINDOWS:
        # The handler refuses on Windows (drive letters are assigned, not mounted at a path here).
        return unobserved("Heliox does not mount disks at a path on Windows")
    params: DiskManageParams = result.action.parameters  # type: ignore[assignment]
    if not params.device or not params.mount_point:
        raise ValueError("Disk mount has no device or mount point to observe")
    source = await mounted_source(params.mount_point)
    if source is None:
        return mismatch(f"Expected {params.device} mounted at {params.mount_point}; nothing is mounted there")
    if not await device_matches(params.device, source):
        return mismatch(f"Expected {params.device} mounted at {params.mount_point}; {source} is mounted there")
    return verified(f"{source} is mounted at {params.mount_point}")


@register(ActionType.WIFI_CONNECT, method="wifi_network_postcondition", timeout=20.0)
async def wifi_joined(result: ActionResult) -> Outcome:
    """The requested SSID is among the networks joined now (WLAN API, NetworkManager, networksetup or ipconfig)."""
    from pilot.actions import WifiParams
    from pilot.system.network import connected_ssids

    params: WifiParams = result.action.parameters  # type: ignore[assignment]
    if not params.ssid:
        raise ValueError("Wi-Fi connect has no network name to observe")
    # Joining finishes after the command returns (netsh only starts it): read up to WIFI_READINGS times.
    reading = await connected_ssids(params.interface)
    for _ in range(WIFI_READINGS - 1):
        if reading.ssids is None or params.ssid in reading.ssids:
            break
        await _pause()
        reading = await connected_ssids(params.interface)
    if reading.ssids is None:
        return unobserved(f"Could not read which network is joined: {reading.reason}")
    if params.ssid in reading.ssids:
        return verified(f"Joined to {params.ssid}")
    found = ", ".join(reading.ssids) if reading.ssids else "no network"
    seconds = WIFI_READINGS * READING_INTERVAL_SECONDS
    return mismatch(f"Expected to be joined to {params.ssid}; after {seconds:g} s the Wi-Fi is on {found}")


@register(ActionType.PACKAGE_UPDATE, method="packages_current_postcondition", timeout=200.0)
async def packages_up_to_date(result: ActionResult) -> Outcome:
    """Nothing is left to upgrade, asked of the package manager afresh (winget, choco, brew, apt, dnf, pacman)."""
    from pilot.system.package_mgr import pending_upgrades

    pending = await pending_upgrades()
    if pending is None:
        return unobserved("This system's package manager has no read-only list of pending upgrades Heliox can ask")
    if not pending:
        return verified("The package manager lists no upgrades left to install")
    shown = ", ".join(pending[:5]) + (f" and {len(pending) - 5} more" if len(pending) > 5 else "")
    return mismatch(f"Expected every installed package to be up to date; {len(pending)} still have upgrades: {shown}")


@register(ActionType.WIFI_DISCONNECT, method="wifi_disconnected_postcondition", timeout=20.0)
async def wifi_left(result: ActionResult) -> Outcome:
    """No Wi-Fi network is joined any more (on the named interface when one was given)."""
    from pilot.actions import WifiParams
    from pilot.system.network import connected_ssids

    params: WifiParams = result.action.parameters  # type: ignore[assignment]
    reading = await connected_ssids(params.interface)
    for _ in range(WIFI_READINGS - 1):
        if reading.ssids is None or not reading.ssids:
            break
        await _pause()
        reading = await connected_ssids(params.interface)
    if reading.ssids is None:
        return unobserved(f"Could not read which network is joined: {reading.reason}")
    if reading.ssids:
        return mismatch(f"Expected no Wi-Fi network; still joined to {', '.join(reading.ssids)}")
    return verified("No Wi-Fi network is joined")


@register(ActionType.ENV_SET, method="conditional_environment_value_postcondition", timeout=25.0)
async def environment_value_stored(result: ActionResult) -> Outcome:
    """A persistent variable is stored with the requested value where new programs read it (the user's registry
    environment on Windows, the shell profile elsewhere). The value never appears in the sentence."""
    from pilot.actions import EnvParams
    from pilot.system.environment import persistent_value

    params: EnvParams = result.action.parameters  # type: ignore[assignment]
    if not params.persistent:
        return unobserved("A session-only variable lives in Heliox's own process; nothing outside it to read")
    if not params.name:
        raise ValueError("Environment change has no variable name to observe")
    found = await persistent_value(params.name)
    wanted = params.value or ""
    where = "the user environment" if CURRENT_PLATFORM == Platform.WINDOWS else "the shell profile"
    if found is None:
        return mismatch(f"Expected {params.name} in {where}; it is not there")
    if found != wanted:
        return mismatch(
            f"Expected {params.name} in {where} with the requested value ({len(wanted)} characters); "
            f"it holds a different one ({len(found)} characters)"
        )
    return verified(f"{params.name} is stored in {where} with the requested value")


@register(ActionType.VOLUME_MUTE, method="conditional_mute_state_postcondition", timeout=20.0)
async def mute_state_matches(result: ActionResult) -> Outcome:
    """The output is muted (or unmuted) as asked, read from the system's mixer afresh."""
    from pilot.actions import VolumeParams
    from pilot.system.volume import volume_state

    params: VolumeParams = result.action.parameters  # type: ignore[assignment]
    wanted = params.mute if params.mute is not None else True
    state = await volume_state()
    if state.muted is None:
        return unobserved("This system's mixer did not say whether the sound is muted")
    if state.muted != wanted:
        return mismatch(f"Expected the sound {'muted' if wanted else 'on'}; it is {'muted' if state.muted else 'on'}")
    return verified(f"The sound is {'muted' if wanted else 'on'}")


# Mixers round a percentage to their own steps (pactl, Core Audio's scalar, osascript): one point either way.
VOLUME_TOLERANCE = 1
# Backlights have coarse steps (WMI brightness levels, brightnessctl's raw range).
BRIGHTNESS_TOLERANCE = 5


@register(ActionType.VOLUME_SET, method="conditional_volume_level_postcondition", timeout=20.0)
async def volume_level_matches(result: ActionResult) -> Outcome:
    """The output level is the one asked for, read from the system's mixer afresh. A relative step is not
    observed: its target depends on the level before, which nothing recorded."""
    from pilot.actions import VolumeParams
    from pilot.system.volume import volume_state

    params: VolumeParams = result.action.parameters  # type: ignore[assignment]
    if params.step is not None or params.level is None:
        return unobserved("A volume step has no fixed target to compare with")
    wanted = max(0, min(100, int(params.level)))
    state = await volume_state()
    if state.level is None:
        return unobserved("This system's mixer did not report the volume")
    if abs(state.level - wanted) > VOLUME_TOLERANCE:
        return mismatch(f"Expected the volume at {wanted}; it is at {state.level}")
    return verified(f"The volume is at {state.level}")


@register(ActionType.BRIGHTNESS_SET, method="conditional_brightness_level_postcondition", timeout=25.0)
async def brightness_level_matches(result: ActionResult) -> Outcome:
    """The built-in display's brightness is the level asked for, within the backlight's steps."""
    from pilot.actions import BrightnessParams
    from pilot.system.screen import brightness_percent

    params: BrightnessParams = result.action.parameters  # type: ignore[assignment]
    if params.level is None:
        raise ValueError("Brightness change has no level to observe")
    wanted = max(0, min(100, int(params.level)))
    found = await brightness_percent()
    if found is None:
        return unobserved("This display's brightness cannot be read here (an external monitor, or no reader)")
    if abs(found - wanted) > BRIGHTNESS_TOLERANCE:
        return mismatch(f"Expected brightness {wanted}%; it is {found}%")
    return verified(f"Brightness is {found}%")

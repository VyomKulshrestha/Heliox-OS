"""Read-only disk and mount observations used by post-condition checks."""

from __future__ import annotations

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell


async def mount_exists(*, device: str | None = None, mount_point: str | None = None) -> bool:
    """Observe whether the exact device or mount point is currently mounted."""
    if not device and not mount_point:
        raise ValueError("Device or mount point is required")

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "$target = ([string]$HelioxArgs['target']).TrimEnd('\\'); "
            "$match = Get-CimInstance Win32_LogicalDisk | Where-Object { "
            "$_.DeviceID -eq $target -or $_.ProviderName -eq $target }; "
            "if ($match) { 'present' }",
            values={"target": mount_point or device or ""},
        )
        if code != 0:
            raise RuntimeError(f"Mount observation failed: {err.strip()}")
        return out.strip() == "present"

    if CURRENT_PLATFORM == Platform.LINUX:
        selector = ["--mountpoint", mount_point] if mount_point else ["--source", str(device)]
        code, out, err = await run_command(["findmnt", "--noheadings", "--output", "TARGET", *selector])
        if code not in {0, 1}:
            raise RuntimeError(f"Mount observation failed: {err.strip()}")
        return code == 0 and bool(out.strip())

    code, out, err = await run_command(["mount"])
    if code != 0:
        raise RuntimeError(f"Mount observation failed: {err.strip()}")
    target = mount_point or device or ""
    return any(
        (mount_point is not None and f" on {target} " in line)
        or (device is not None and line.startswith(f"{target} on "))
        for line in out.splitlines()
    )


async def mounted_source(mount_point: str) -> str | None:
    """What is mounted at ``mount_point`` now (its source device), or None when nothing is. Linux and macOS.

    Linux asks findmnt for the exact mount point. macOS reads ``mount``'s lines ("<device> on <path> (<options>)"),
    also under the path's real location, since macOS shows /tmp as /private/tmp."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        raise ValueError("Windows has no mount points to read here")

    if CURRENT_PLATFORM == Platform.LINUX:
        code, out, err = await run_command(
            ["findmnt", "--noheadings", "--first-only", "--output", "SOURCE", "--mountpoint", mount_point], timeout=10
        )
        if code not in {0, 1}:
            raise RuntimeError(f"Mount observation failed: {err.strip()}")
        return out.strip().splitlines()[0].strip() if code == 0 and out.strip() else None

    import os

    code, out, err = await run_command(["mount"], timeout=10)
    if code != 0:
        raise RuntimeError(f"Mount observation failed: {err.strip()}")
    places = {mount_point.rstrip("/") or "/", os.path.realpath(mount_point).rstrip("/") or "/"}
    for line in out.splitlines():
        source, separator, rest = line.partition(" on ")
        if separator and rest.rpartition(" (")[0] in places:
            return source
    return None


async def device_matches(device: str, source: str) -> bool:
    """Whether the mounted ``source`` is the ``device`` asked for. The device may be a path (/dev/sdb1, or a link
    to it such as /dev/disk/by-label/x) or, on Linux, a tag (UUID=, LABEL=, PARTUUID=, PARTLABEL=) that findmnt's
    own --source resolves."""
    import os

    if device == source:
        return True
    if not device.startswith("/") or not source.startswith("/"):
        if CURRENT_PLATFORM != Platform.LINUX:
            return False
        code, out, _ = await run_command(
            ["findmnt", "--noheadings", "--output", "SOURCE", "--source", device], timeout=10
        )
        return code == 0 and source in {line.strip() for line in out.splitlines()}
    return os.path.realpath(device) == os.path.realpath(source)

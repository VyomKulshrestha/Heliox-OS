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

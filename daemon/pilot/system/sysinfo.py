"""System information — CPU, memory, disk, network, battery, OS.

Cross-platform module using psutil (preferred) with OS command fallbacks.
"""

from __future__ import annotations

import asyncio
import logging
import platform

from pilot.system.platform_detect import (
    CURRENT_PLATFORM, Platform, get_platform_info,
    run_command, run_powershell,
)

logger = logging.getLogger("pilot.system.sysinfo")


async def system_info(categories: list[str] | None = None) -> str:
    """Get comprehensive system information."""
    if not categories:
        categories = ["os", "cpu", "memory", "disk", "network", "time"]

    sections: list[str] = []

    if "time" in categories:
        sections.append(await _time_info())

    if "os" in categories:
        info = get_platform_info()
        lines = ["=== Operating System ==="]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        sections.append("\n".join(lines))

    if "cpu" in categories:
        sections.append(await _cpu_info())

    if "memory" in categories:
        sections.append(await _memory_info())

    if "disk" in categories:
        sections.append(await _disk_info())

    if "network" in categories:
        sections.append(await _network_info())

    if "battery" in categories:
        sections.append(await _battery_info())

    return "\n\n".join(sections)


async def _time_info() -> str:
    import datetime
    now = datetime.datetime.now()
    return f"=== System Time ===\n  Current Local Time: {now.strftime('%A, %B %d, %Y %I:%M:%S %p')}\n  Timezone: {now.astimezone().tzname()}"


async def _cpu_info() -> str:
    try:
        import psutil
        count = psutil.cpu_count()
        count_logical = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        percent = psutil.cpu_percent(interval=0.5, percpu=True)
        lines = [
            "=== CPU ===",
            f"  Physical cores: {count}",
            f"  Logical cores: {count_logical}",
        ]
        if freq:
            lines.append(f"  Frequency: {freq.current:.0f} MHz (max {freq.max:.0f} MHz)")
        lines.append(f"  Usage per core: {', '.join(f'{p:.1f}%' for p in percent)}")
        lines.append(f"  Average usage: {sum(percent)/len(percent):.1f}%")
        return "\n".join(lines)
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_powershell(
            "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, "
            "NumberOfLogicalProcessors, CurrentClockSpeed, LoadPercentage | Format-List"
        )
        return f"=== CPU ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["lscpu"])
        return f"=== CPU ===\n{out.strip()}"


async def _memory_info() -> str:
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return (
            "=== Memory ===\n"
            f"  Total: {mem.total / (1024**3):.1f} GB\n"
            f"  Used:  {mem.used / (1024**3):.1f} GB ({mem.percent}%)\n"
            f"  Free:  {mem.available / (1024**3):.1f} GB\n"
            f"  Swap Total: {swap.total / (1024**3):.1f} GB\n"
            f"  Swap Used:  {swap.used / (1024**3):.1f} GB ({swap.percent}%)"
        )
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_powershell(
            "$os = Get-CimInstance Win32_OperatingSystem; "
            "$total = [math]::Round($os.TotalVisibleMemorySize/1MB, 1); "
            "$free = [math]::Round($os.FreePhysicalMemory/1MB, 1); "
            "$used = $total - $free; "
            "\"Total: ${total} GB`nUsed: ${used} GB`nFree: ${free} GB\""
        )
        return f"=== Memory ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["free", "-h"])
        return f"=== Memory ===\n{out.strip()}"


async def memory_usage() -> str:
    return await _memory_info()


async def cpu_usage() -> str:
    return await _cpu_info()


async def _disk_info() -> str:
    try:
        import psutil
        lines = ["=== Disk ==="]
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                lines.append(
                    f"  {part.device} ({part.mountpoint}) — "
                    f"{usage.total / (1024**3):.1f} GB total, "
                    f"{usage.used / (1024**3):.1f} GB used ({usage.percent}%), "
                    f"fstype={part.fstype}"
                )
            except PermissionError:
                lines.append(f"  {part.device} ({part.mountpoint}) — access denied")
        return "\n".join(lines)
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_powershell(
            "Get-PSDrive -PSProvider FileSystem | "
            "Select-Object Name, @{N='Used(GB)';E={[math]::Round($_.Used/1GB,1)}}, "
            "@{N='Free(GB)';E={[math]::Round($_.Free/1GB,1)}} | Format-Table"
        )
        return f"=== Disk ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["df", "-h"])
        return f"=== Disk ===\n{out.strip()}"


async def disk_usage() -> str:
    return await _disk_info()


async def _network_info() -> str:
    try:
        import psutil
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        lines = ["=== Network ==="]
        for iface, addr_list in addrs.items():
            stat = stats.get(iface)
            status = "UP" if stat and stat.isup else "DOWN"
            lines.append(f"  {iface} ({status}):")
            for addr in addr_list:
                if addr.family.name == "AF_INET":
                    lines.append(f"    IPv4: {addr.address}")
                elif addr.family.name == "AF_INET6":
                    lines.append(f"    IPv6: {addr.address}")
        return "\n".join(lines)
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_powershell("Get-NetIPAddress | Format-Table -AutoSize")
        return f"=== Network ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["ip", "addr"])
        return f"=== Network ===\n{out.strip()}"


async def network_info() -> str:
    return await _network_info()


async def _battery_info() -> str:
    try:
        import psutil
        batt = psutil.sensors_battery()
        if batt is None:
            return "=== Battery ===\n  No battery detected"
        plugged = "Plugged in" if batt.power_plugged else "On battery"
        secs = batt.secsleft
        time_left = f"{secs // 3600}h {(secs % 3600) // 60}m" if secs > 0 else "N/A"
        return (
            f"=== Battery ===\n"
            f"  Charge: {batt.percent}%\n"
            f"  Status: {plugged}\n"
            f"  Time remaining: {time_left}"
        )
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_powershell(
            "$b = Get-CimInstance Win32_Battery; "
            "if ($b) { \"Charge: $($b.EstimatedChargeRemaining)%`n"
            "Status: $($b.BatteryStatus)\" } else { 'No battery detected' }"
        )
        return f"=== Battery ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT0"])
        return f"=== Battery ===\n{out.strip()}"


async def battery_info() -> str:
    return await _battery_info()

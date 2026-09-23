"""System information — CPU, memory, disk, network, battery, OS.

Cross-platform module using psutil (preferred) with OS command fallbacks.
"""

from __future__ import annotations

import asyncio
import logging

from pilot.system.platform_detect import (
    CURRENT_PLATFORM,
    Platform,
    get_platform_info,
    run_command,
    run_powershell,
)

logger = logging.getLogger("pilot.system.sysinfo")

# A half-second blocking sample made simple status requests feel like model
# latency. These short windows retain a fresh psutil reading while keeping the
# interactive path responsive; comprehensive system info uses the steadier
# window and the dedicated CPU action uses the faster one.
SYSTEM_INFO_CPU_SAMPLE_SECONDS = 0.05
CPU_USAGE_SAMPLE_SECONDS = 0.02


def _prime_psutil_cpu() -> None:
    """Load and prime CPU probes before the daemon advertises readiness."""
    import psutil

    psutil.cpu_count()
    psutil.cpu_count(logical=True)
    try:
        psutil.cpu_freq()
    except (AttributeError, NotImplementedError):
        pass
    # The non-blocking call initializes psutil's per-thread CPU baseline. The
    # interactive action still performs its own fresh interval sample.
    psutil.cpu_percent(interval=None, percpu=True)


async def prepare_system_probes() -> None:
    """Warm common psutil/thread-pool paths during daemon setup."""
    try:
        await asyncio.to_thread(_prime_psutil_cpu)
    except ImportError:
        logger.debug("psutil is unavailable; system probes will use platform fallbacks")


async def system_info(categories: list[str] | None = None) -> str:
    """Get comprehensive system information."""
    if not categories:
        categories = ["os", "cpu", "memory", "disk", "network", "time"]
    elif "all" in categories:
        categories = ["time", "os", "cpu", "memory", "disk", "network", "battery"]

    section_values: dict[str, str] = {}
    if "os" in categories:
        info = get_platform_info()
        lines = ["=== Operating System ==="]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        section_values["os"] = "\n".join(lines)

    probes = {
        "time": _time_info,
        "cpu": _cpu_info,
        "memory": _memory_info,
        "disk": _disk_info,
        "network": _network_info,
        "battery": _battery_info,
    }
    requested = [(name, probe()) for name, probe in probes.items() if name in categories]
    if requested:
        results = await asyncio.gather(*(coroutine for _, coroutine in requested))
        section_values.update({name: result for (name, _), result in zip(requested, results, strict=True)})

    display_order = ("time", "os", "cpu", "memory", "disk", "network", "battery")
    return "\n\n".join(section_values[name] for name in display_order if name in section_values)


async def _time_info() -> str:
    import datetime

    now = datetime.datetime.now()
    return f"=== System Time ===\n  Current Local Time: {now.strftime('%A, %B %d, %Y %I:%M:%S %p')}\n  Timezone: {now.astimezone().tzname()}"


def _collect_cpu_info(sample_interval: float | None = SYSTEM_INFO_CPU_SAMPLE_SECONDS) -> str:
    import psutil

    count = psutil.cpu_count()
    count_logical = psutil.cpu_count(logical=True)
    try:
        # Not implemented on some platforms (e.g. Apple Silicon macOS),
        # where psutil raises AttributeError/NotImplementedError instead
        # of returning None -- frequency is optional info either way.
        freq = psutil.cpu_freq()
    except (AttributeError, NotImplementedError):
        freq = None
    percent = psutil.cpu_percent(interval=sample_interval, percpu=True)
    lines = [
        "=== CPU ===",
        f"  Physical cores: {count}",
        f"  Logical cores: {count_logical}",
    ]
    if freq:
        lines.append(f"  Frequency: {freq.current:.0f} MHz (max {freq.max:.0f} MHz)")
    if percent:
        lines.append(f"  Usage per core: {', '.join(f'{p:.1f}%' for p in percent)}")
        lines.append(f"  Average usage: {sum(percent) / len(percent):.1f}%")
    return "\n".join(lines)


async def _cpu_info(sample_interval: float | None = SYSTEM_INFO_CPU_SAMPLE_SECONDS) -> str:
    try:
        return await asyncio.to_thread(_collect_cpu_info, sample_interval)
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
            '"Total: ${total} GB`nUsed: ${used} GB`nFree: ${free} GB"'
        )
        return f"=== Memory ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["free", "-h"])
        return f"=== Memory ===\n{out.strip()}"


async def memory_usage() -> str:
    return await _memory_info()


async def cpu_usage() -> str:
    return await _cpu_info(sample_interval=CPU_USAGE_SAMPLE_SECONDS)


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
        return f"=== Battery ===\n  Charge: {batt.percent}%\n  Status: {plugged}\n  Time remaining: {time_left}"
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_powershell(
            "$b = Get-CimInstance Win32_Battery; "
            'if ($b) { "Charge: $($b.EstimatedChargeRemaining)%`n'
            "Status: $($b.BatteryStatus)\" } else { 'No battery detected' }"
        )
        return f"=== Battery ===\n{out.strip()}"
    else:
        code, out, _ = await run_command(["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT0"])
        return f"=== Battery ===\n{out.strip()}"


async def battery_info() -> str:
    return await _battery_info()


def _collect_system_health_snapshot() -> dict:
    """Collect one coherent, read-only psutil snapshot for health reporting."""
    import psutil

    cpu_percent = float(psutil.cpu_percent(interval=SYSTEM_INFO_CPU_SAMPLE_SECONDS))
    memory = psutil.virtual_memory()
    disks: list[dict[str, object]] = []
    seen_mounts: set[str] = set()
    for part in psutil.disk_partitions():
        if part.mountpoint in seen_mounts:
            continue
        seen_mounts.add(part.mountpoint)
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        disks.append(
            {
                "device": part.device or part.mountpoint,
                "mountpoint": part.mountpoint,
                "percent": float(usage.percent),
                "free_gb": usage.free / (1024**3),
            }
        )

    battery = psutil.sensors_battery()
    processes: list[dict[str, object]] = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            memory_info = info.get("memory_info")
            processes.append(
                {
                    "pid": int(info["pid"]),
                    "name": str(info.get("name") or "unknown"),
                    "memory_mb": float(memory_info.rss / (1024**2)) if memory_info else 0.0,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            continue
    processes.sort(key=lambda item: float(item["memory_mb"]), reverse=True)

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": float(memory.percent),
        "memory_available_gb": memory.available / (1024**3),
        "disks": disks,
        "battery_percent": float(battery.percent) if battery else None,
        "battery_plugged": bool(battery.power_plugged) if battery else None,
        "process_count": len(processes),
        "top_processes": processes[:5],
    }


def _format_system_health_review(snapshot: dict) -> str:
    """Turn measured evidence into two deterministic observations and advice."""
    cpu = float(snapshot["cpu_percent"])
    memory = float(snapshot["memory_percent"])
    available = float(snapshot["memory_available_gb"])
    disks = list(snapshot["disks"])
    battery_percent = snapshot["battery_percent"]
    battery_plugged = snapshot["battery_plugged"]
    process_count = int(snapshot["process_count"])
    top_processes = list(snapshot["top_processes"])

    disk = max(disks, key=lambda item: float(item["percent"]), default=None)
    disk_percent = float(disk["percent"]) if disk else 0.0
    disk_label = str(disk["device"]) if disk else "No accessible disk"
    disk_free = float(disk["free_gb"]) if disk else 0.0

    candidates: list[tuple[int, str, str]] = [
        (
            100 if memory >= 90 else 75 if memory >= 80 else 35,
            f"Memory is {memory:.1f}% used with {available:.1f} GB available.",
            "Close or pause the largest unneeded applications before starting another memory-heavy workload."
            if memory >= 80
            else "No immediate memory action is required; recheck if responsiveness degrades.",
        ),
        (
            95 if disk_percent >= 95 else 70 if disk_percent >= 85 else 30,
            f"The fullest accessible disk is {disk_label} at {disk_percent:.1f}% used ({disk_free:.1f} GB free)."
            if disk
            else "No accessible filesystem usage could be measured.",
            "Review large files and temporary data soon, without deleting anything automatically."
            if disk_percent >= 85
            else "Disk headroom is currently acceptable.",
        ),
        (
            85 if cpu >= 90 else 60 if cpu >= 75 else 25,
            f"CPU utilization is {cpu:.1f}% in the current sample.",
            "Inspect sustained high-CPU processes before deciding whether any intervention is needed."
            if cpu >= 75
            else "CPU load does not currently require intervention.",
        ),
    ]
    if battery_percent is not None:
        candidates.append(
            (
                80 if float(battery_percent) <= 15 and not battery_plugged else 20,
                f"Battery is {float(battery_percent):.0f}% and "
                f"{'plugged in' if battery_plugged else 'running on battery'}.",
                "Connect power before a long-running task."
                if float(battery_percent) <= 15 and not battery_plugged
                else "No battery action is needed.",
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    primary = candidates[:2]
    process_text = (
        ", ".join(f"{item['name']} (PID {item['pid']}, {float(item['memory_mb']):.0f} MB)" for item in top_processes)
        or "none available"
    )

    lines = [
        "READ-ONLY SYSTEM HEALTH REVIEW",
        "Evidence:",
        f"- CPU: {cpu:.1f}%",
        f"- Memory: {memory:.1f}% used; {available:.1f} GB available",
        (
            f"- Fullest disk: {disk_label}; {disk_percent:.1f}% used; {disk_free:.1f} GB free"
            if disk
            else "- Disk: no accessible filesystem measurement"
        ),
        (
            f"- Battery: {float(battery_percent):.0f}%; {'plugged in' if battery_plugged else 'on battery'}"
            if battery_percent is not None
            else "- Battery: no battery detected"
        ),
        f"- Running processes: {process_count}; largest by working set: {process_text}",
        "",
        "Two most important observations:",
        f"1. {primary[0][1]}",
        f"2. {primary[1][1]}",
        "",
        "Prioritized recommendation:",
        f"1. {primary[0][2]}",
        f"2. {primary[1][2]}",
        "No processes or files were changed.",
    ]
    return "\n".join(lines)


async def system_health_review() -> str:
    """Collect and summarize live health evidence without modifying state."""
    try:
        snapshot = await asyncio.to_thread(_collect_system_health_snapshot)
    except ImportError as exc:
        raise RuntimeError("System health review requires psutil") from exc
    return _format_system_health_review(snapshot)

"""Process management — list, kill, and inspect running processes.

Cross-platform implementation using psutil when available,
with fallback to OS-specific commands.
"""

from __future__ import annotations

import logging
import os
import signal as signal_module

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.processes")


async def process_list(filter_name: str | None = None) -> str:
    """List running processes with PID, name, CPU%, memory%."""
    try:
        import psutil

        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = proc.info
                if filter_name and filter_name.lower() not in info["name"].lower():
                    continue
                procs.append(
                    f"PID={info['pid']:>6}  CPU={info['cpu_percent']:>5.1f}%  "
                    f"MEM={info['memory_percent']:>5.1f}%  "
                    f"STATUS={info['status']:<12}  {info['name']}"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return "\n".join(procs[:100]) if procs else "No matching processes found"
    except ImportError:
        pass

    # Fallback to OS commands
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "Get-Process | Select-Object Id, ProcessName, CPU, WorkingSet64 "
            "| Format-Table -AutoSize | Out-String -Width 200"
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["ps", "aux"])
    else:
        code, out, err = await run_command(["ps", "aux", "--sort=-%mem"])

    if code != 0:
        raise RuntimeError(f"Process list failed: {err.strip()}")

    lines = out.strip().split("\n")
    if filter_name:
        header = lines[0] if lines else ""
        filtered = [l for l in lines[1:] if filter_name.lower() in l.lower()]
        return header + "\n" + "\n".join(filtered[:100])
    return "\n".join(lines[:100])


async def process_kill(pid: int | None = None, name: str | None = None, sig: str = "SIGTERM") -> str:
    """Kill a process by PID or by name."""
    if pid is not None:
        return await _kill_by_pid(pid, sig)
    elif name is not None:
        return await _kill_by_name(name, sig)
    else:
        raise ValueError("Either pid or name must be provided")


async def _kill_by_pid(pid: int, sig: str) -> str:
    try:
        import psutil

        proc = psutil.Process(pid)
        proc_name = proc.name()
        if sig == "SIGKILL" or sig == "9":
            proc.kill()
        else:
            proc.terminate()
        return f"Sent {sig} to process {proc_name} (PID {pid})"
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["taskkill", "/PID", str(pid), "/F"])
    else:
        sig_num = getattr(signal_module, sig, signal_module.SIGTERM)
        os.kill(pid, sig_num)
        return f"Sent {sig} to PID {pid}"

    if code != 0:
        raise RuntimeError(f"Kill failed: {err.strip()}")
    return f"Killed PID {pid}"


async def _kill_by_name(name: str, sig: str) -> str:
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["taskkill", "/IM", name, "/F"])
    else:
        code, out, err = await run_command(["pkill", f"-{sig}", name])

    if code != 0:
        raise RuntimeError(f"Kill by name failed: {err.strip()}")
    return f"Killed processes matching '{name}'"


async def process_info(pid: int) -> str:
    """Get detailed info about a process."""
    try:
        import psutil

        proc = psutil.Process(pid)
        info = proc.as_dict(
            attrs=[
                "pid",
                "name",
                "exe",
                "cmdline",
                "status",
                "cpu_percent",
                "memory_percent",
                "memory_info",
                "create_time",
                "num_threads",
                "username",
            ]
        )
        lines = [f"Process info for PID {pid}:"]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(f"Get-Process -Id {pid} | Format-List *")
    else:
        code, out, err = await run_command(
            ["ps", "-p", str(pid), "-o", "pid,user,%cpu,%mem,vsz,rss,tty,stat,start,time,command"]
        )

    if code != 0:
        raise RuntimeError(f"Process info failed: {err.strip()}")
    return out.strip()

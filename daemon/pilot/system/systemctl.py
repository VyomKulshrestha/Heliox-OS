"""Service management via systemctl.

Supports both system-level and user-level systemd units.
System-level operations requiring root use pkexec for privilege escalation.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("pilot.system.systemctl")


async def _run_systemctl(action: str, unit: str, *, user_scope: bool = False) -> tuple[int, str, str]:
    args = ["systemctl"]
    if user_scope:
        args.append("--user")
    args.extend([action, unit])

    if not user_scope and action in ("start", "stop", "restart", "enable", "disable"):
        args = ["pkexec"] + args

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def service_start(name: str, user_scope: bool = False) -> str:
    code, out, err = await _run_systemctl("start", name, user_scope=user_scope)
    if code != 0:
        raise RuntimeError(f"Failed to start {name}: {err.strip()}")
    return f"Started service: {name}"


async def service_stop(name: str, user_scope: bool = False) -> str:
    code, out, err = await _run_systemctl("stop", name, user_scope=user_scope)
    if code != 0:
        raise RuntimeError(f"Failed to stop {name}: {err.strip()}")
    return f"Stopped service: {name}"


async def service_restart(name: str, user_scope: bool = False) -> str:
    code, out, err = await _run_systemctl("restart", name, user_scope=user_scope)
    if code != 0:
        raise RuntimeError(f"Failed to restart {name}: {err.strip()}")
    return f"Restarted service: {name}"


async def service_enable(name: str, user_scope: bool = False) -> str:
    code, out, err = await _run_systemctl("enable", name, user_scope=user_scope)
    if code != 0:
        raise RuntimeError(f"Failed to enable {name}: {err.strip()}")
    return f"Enabled service: {name}"


async def service_disable(name: str, user_scope: bool = False) -> str:
    code, out, err = await _run_systemctl("disable", name, user_scope=user_scope)
    if code != 0:
        raise RuntimeError(f"Failed to disable {name}: {err.strip()}")
    return f"Disabled service: {name}"


async def service_status(name: str, user_scope: bool = False) -> str:
    code, out, err = await _run_systemctl("status", name, user_scope=user_scope)
    return out.strip() or err.strip()


async def is_active(name: str, user_scope: bool = False) -> bool:
    code, out, _ = await _run_systemctl("is-active", name, user_scope=user_scope)
    return out.strip() == "active"

"""Service management via systemctl.

Supports both system-level and user-level systemd units.
System-level operations requiring root use pkexec for privilege escalation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("pilot.system.systemctl")


# A read (is-active, status) answers at once; a change may first wait for the person at pkexec's password prompt
# and then for the unit to start or stop, so it gets longer. Past either limit the command is killed and the call
# reports a failure instead of hanging the task.
READ_TIMEOUT_SECONDS = 15
CHANGE_TIMEOUT_SECONDS = 120
_CHANGES = ("start", "stop", "restart", "enable", "disable")


async def _run_systemctl(action: str, unit: str, *, user_scope: bool = False) -> tuple[int, str, str]:
    from pilot.system.platform_detect import run_command

    args = ["systemctl"]
    if user_scope:
        args.append("--user")
    args.extend([action, unit])

    if not user_scope and action in _CHANGES:
        args = ["pkexec"] + args

    # run_command refuses a service change under the test suite (pilot.system.system_guard).
    timeout = CHANGE_TIMEOUT_SECONDS if action in _CHANGES else READ_TIMEOUT_SECONDS
    return await run_command(args, timeout=timeout)


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


async def is_enabled(name: str, user_scope: bool = False) -> str:
    """The unit's boot state as ``systemctl is-enabled`` names it: enabled, enabled-runtime, linked, alias, static,
    indirect, generated, transient, disabled, masked, masked-runtime, bad or not-found. These words are systemd's
    own and are not translated. A read: no pkexec, no password prompt, at most 15 seconds. Raises when systemctl
    gives no state (not installed, or not running systemd)."""
    from pilot.system.platform_detect import run_command

    args = ["systemctl", *(["--user"] if user_scope else []), "is-enabled", "--", name]
    code, out, err = await run_command(args, timeout=15)
    state = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not state:
        raise RuntimeError(f"systemctl gave no enabled state for {name}: {err.strip() or 'exit ' + str(code)}")
    return state

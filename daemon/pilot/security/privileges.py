"""Report the daemon's real operating-system privilege state."""

from __future__ import annotations

import os
import sys
from typing import Any


def has_elevated_privileges() -> bool:
    """Return whether the current daemon process has admin/root privileges."""
    if sys.platform == "win32":
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError):
            return False

    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid is not None and geteuid() == 0)


def security_runtime_status(
    root_policy_enabled: bool,
    *,
    platform_name: str | None = None,
    process_elevated: bool | None = None,
) -> dict[str, Any]:
    """Build the user-facing distinction between policy and OS privileges."""
    platform_value = platform_name or sys.platform
    elevated = has_elevated_privileges() if process_elevated is None else process_elevated

    if not root_policy_enabled:
        detail = "Root-tier actions are blocked by Heliox policy."
    elif elevated:
        detail = "Root-tier actions are allowed by policy and the daemon has elevated OS privileges."
    elif platform_value == "win32":
        detail = (
            "Root-tier actions are allowed by policy, but the daemon is not running as Administrator. "
            "Windows may deny protected operations."
        )
    else:
        detail = (
            "Root-tier actions are allowed by policy. The OS may still require an elevation prompt "
            "for protected operations."
        )

    return {
        "root_policy_enabled": root_policy_enabled,
        "process_elevated": elevated,
        "platform": platform_value,
        "detail": detail,
    }

"""Cross-platform detection and helpers.

Provides OS-aware constants and subprocess helpers used by all system modules.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
from enum import StrEnum

logger = logging.getLogger("pilot.system.platform")


class Platform(StrEnum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


def detect_platform() -> Platform:
    if sys.platform == "win32":
        return Platform.WINDOWS
    elif sys.platform == "darwin":
        return Platform.MACOS
    elif sys.platform.startswith("linux"):
        return Platform.LINUX
    return Platform.UNKNOWN


CURRENT_PLATFORM = detect_platform()


def get_platform_info() -> dict[str, str]:
    """Return detailed platform information."""
    return {
        "platform": CURRENT_PLATFORM.value,
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }


async def run_command(
    args: list[str],
    *,
    root: bool = False,
    timeout: int = 30,
    cwd: str | None = None,
    input_data: bytes | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess command cross-platform.

    On Linux, root commands use pkexec.
    On Windows, elevated commands use runas conceptually (but we just run them).
    """
    if root and CURRENT_PLATFORM == Platform.LINUX:
        args = ["pkexec"] + args

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE if input_data else None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(input_data), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except TimeoutError:
        if proc:
            proc.kill()
        return (-1, "", "Command timed out")
    except FileNotFoundError:
        return (-1, "", f"Command not found: {args[0]}")
    except Exception as e:
        return (-1, "", f"Command error: {e}")


async def run_powershell(script: str, *, timeout: int = 30, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a PowerShell script (Windows)."""
    return await run_command(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        timeout=timeout,
        cwd=cwd,
    )


async def run_shell_script(
    script: str,
    *,
    interpreter: str | None = None,
    timeout: int = 60,
    cwd: str | None = None,
    elevated: bool = False,
) -> tuple[int, str, str]:
    """Run a multi-line script with the appropriate interpreter."""
    if interpreter is None:
        if CURRENT_PLATFORM == Platform.WINDOWS:
            interpreter = "powershell"
        else:
            interpreter = "bash"

    if interpreter in ("powershell", "pwsh"):
        args = [interpreter, "-NoProfile", "-NonInteractive", "-Command", script]
    elif interpreter in ("bash", "sh", "zsh"):
        args = [interpreter, "-c", script]
    elif interpreter == "python":
        args = [sys.executable, "-c", script]
    elif interpreter == "cmd":
        args = ["cmd", "/c", script]
    else:
        args = [interpreter, "-c", script]

    return await run_command(args, root=elevated, timeout=timeout, cwd=cwd)

"""Safe, explicit Windows elevation handoff for the Heliox daemon."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


class ElevationError(RuntimeError):
    """Raised when Windows refuses or cannot start the elevated daemon."""


def _shell_execute_runas(executable: str, parameters: str, working_directory: str) -> int:
    """Launch a process through Windows ShellExecute's native ``runas`` verb."""
    import ctypes

    shell_execute = ctypes.windll.shell32.ShellExecuteW
    shell_execute.restype = ctypes.c_void_p
    result = shell_execute(
        None,
        "runas",
        executable,
        parameters,
        working_directory,
        0,  # SW_HIDE: keep the daemon console hidden; the UAC prompt is still visible.
    )
    return int(result or 0)


def request_elevated_restart(
    *,
    replace_pid: int | None = None,
    executable: str | None = None,
    working_directory: str | Path | None = None,
    platform_name: str | None = None,
    launcher: Callable[[str, str, str], int] | None = None,
) -> dict[str, str]:
    """Ask Windows to start a replacement daemon with Administrator privileges.

    The existing daemon stays alive until the elevated replacement has started
    and validated that the target PID really is another Pilot daemon.
    """
    platform_value = platform_name or sys.platform
    if platform_value != "win32":
        raise ElevationError("Administrator restart is only available on Windows.")

    target_pid = replace_pid or os.getpid()
    if target_pid <= 0:
        raise ElevationError("Cannot restart an invalid daemon process.")

    python_executable = str(Path(executable or sys.executable).resolve())
    daemon_directory = str(
        Path(working_directory or Path(__file__).resolve().parents[2]).resolve()
    )
    parameters = subprocess.list2cmdline(
        ["-m", "pilot.server", "--replace-pid", str(target_pid)]
    )
    launch = launcher or _shell_execute_runas
    result = launch(python_executable, parameters, daemon_directory)

    if result <= 32:
        if result == 5:
            raise ElevationError(
                "Windows did not grant Administrator access. Approve the UAC prompt to continue."
            )
        raise ElevationError(f"Windows could not start the elevated daemon (ShellExecute {result}).")

    return {
        "status": "prompted",
        "message": (
            "Administrator restart accepted. Heliox will reconnect automatically "
            "when the elevated daemon is ready."
        ),
    }


def replace_existing_daemon(
    replace_pid: int,
    *,
    platform_name: str | None = None,
    response_grace_seconds: float = 1.5,
) -> None:
    """Stop the non-elevated daemon after validating its identity.

    A short grace period lets the old daemon return the successful RPC response
    before its elevated replacement takes over the WebSocket port.
    """
    platform_value = platform_name or sys.platform
    if platform_value != "win32":
        raise ElevationError("Daemon replacement is only supported on Windows.")
    if replace_pid <= 0 or replace_pid == os.getpid():
        raise ElevationError("Refusing to replace an invalid daemon process.")

    import psutil

    try:
        process = psutil.Process(replace_pid)
        command_line = [part.casefold() for part in process.cmdline()]
    except psutil.NoSuchProcess:
        return
    except (psutil.AccessDenied, OSError) as error:
        raise ElevationError(f"Could not inspect the existing daemon: {error}") from error

    is_pilot_daemon = any(
        part == "pilot.server"
        or part.endswith("\\pilot\\server.py")
        or part.endswith("/pilot/server.py")
        for part in command_line
    )
    if not is_pilot_daemon:
        raise ElevationError("Refusing to stop a process that is not the Heliox daemon.")

    if response_grace_seconds > 0:
        time.sleep(response_grace_seconds)

    try:
        process.terminate()
        process.wait(timeout=8)
    except psutil.NoSuchProcess:
        return
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

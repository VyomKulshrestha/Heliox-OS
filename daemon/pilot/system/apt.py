"""Package management via apt — install, remove, update, search.

All operations use subprocess with shell=False for security.
Root operations use pkexec for PolicyKit-based privilege escalation.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("pilot.system.apt")


async def _run(args: list[str], *, root: bool = False) -> tuple[int, str, str]:
    """Run a command, optionally with pkexec for root."""
    cmd = ["pkexec"] + args if root else args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def package_install(name: str, version: str | None = None) -> str:
    """Install a package via apt."""
    pkg = f"{name}={version}" if version else name
    code, out, err = await _run(["apt-get", "install", "-y", pkg], root=True)
    if code != 0:
        raise RuntimeError(f"apt install failed (exit {code}): {err.strip()}")
    return f"Installed package: {name}" + (f" (version {version})" if version else "")


async def package_remove(name: str) -> str:
    """Remove a package via apt."""
    code, out, err = await _run(["apt-get", "remove", "-y", name], root=True)
    if code != 0:
        raise RuntimeError(f"apt remove failed (exit {code}): {err.strip()}")
    return f"Removed package: {name}"


async def package_update() -> str:
    """Update package lists."""
    code, out, err = await _run(["apt-get", "update"], root=True)
    if code != 0:
        raise RuntimeError(f"apt update failed (exit {code}): {err.strip()}")
    return "Package lists updated"


async def package_search(name: str) -> str:
    """Search for packages matching a name."""
    code, out, err = await _run(["apt-cache", "search", name])
    if code != 0:
        raise RuntimeError(f"apt search failed (exit {code}): {err.strip()}")
    lines = out.strip().split("\n")[:20]
    return "\n".join(lines) if lines[0] else f"No packages found matching '{name}'"


async def is_installed(name: str) -> bool:
    """Check if a package is installed."""
    code, out, _ = await _run(["dpkg", "-s", name])
    return code == 0 and "Status: install ok installed" in out

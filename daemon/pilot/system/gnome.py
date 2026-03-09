"""GNOME settings management via gsettings / dconf.

Uses subprocess with argument lists (never shell=True).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("pilot.system.gnome")


async def _run_gsettings(args: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "gsettings",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


async def get_setting(schema: str, key: str) -> str:
    """Read a GNOME setting."""
    code, out, err = await _run_gsettings(["get", schema, key])
    if code != 0:
        raise RuntimeError(f"gsettings get failed: {err}")
    return out


async def set_setting(schema: str, key: str, value: str) -> str:
    """Write a GNOME setting."""
    code, out, err = await _run_gsettings(["set", schema, key, value])
    if code != 0:
        raise RuntimeError(f"gsettings set failed: {err}")
    return f"Set {schema} {key} = {value}"


async def reset_setting(schema: str, key: str) -> str:
    """Reset a GNOME setting to its default value."""
    code, out, err = await _run_gsettings(["reset", schema, key])
    if code != 0:
        raise RuntimeError(f"gsettings reset failed: {err}")
    return f"Reset {schema} {key} to default"


async def list_schemas() -> list[str]:
    """List all available gsettings schemas."""
    code, out, err = await _run_gsettings(["list-schemas"])
    if code != 0:
        raise RuntimeError(f"gsettings list-schemas failed: {err}")
    return out.split("\n") if out else []


async def list_keys(schema: str) -> list[str]:
    """List all keys in a gsettings schema."""
    code, out, err = await _run_gsettings(["list-keys", schema])
    if code != 0:
        raise RuntimeError(f"gsettings list-keys failed: {err}")
    return out.split("\n") if out else []

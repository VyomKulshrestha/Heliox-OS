"""Async DBus client for system service interaction.

Provides access to NetworkManager, UPower, and other system services
through the DBus message bus.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("pilot.system.dbus")


async def call_dbus_method(
    bus_type: str,
    service: str,
    object_path: str,
    interface: str,
    method: str,
    args: list[str] | None = None,
) -> str:
    """Call a DBus method using dbus-send (subprocess-based, no shell=True).

    For full async DBus, use dbus-next directly. This subprocess approach
    is simpler and safer for the initial implementation.
    """
    cmd = [
        "dbus-send",
        f"--{bus_type}",
        "--print-reply",
        f"--dest={service}",
        object_path,
        f"{interface}.{method}",
    ]
    if args:
        cmd.extend(args)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"DBus call failed: {error_msg}")

    return stdout.decode("utf-8", errors="replace").strip()


async def get_system_info() -> dict[str, str]:
    """Gather basic system information via DBus."""
    info: dict[str, str] = {}

    try:
        hostname = await call_dbus_method(
            "system",
            "org.freedesktop.hostname1",
            "/org/freedesktop/hostname1",
            "org.freedesktop.DBus.Properties",
            "Get",
            ["string:org.freedesktop.hostname1", "string:Hostname"],
        )
        info["hostname"] = hostname
    except Exception:
        pass

    try:
        os_release = await call_dbus_method(
            "system",
            "org.freedesktop.hostname1",
            "/org/freedesktop/hostname1",
            "org.freedesktop.DBus.Properties",
            "Get",
            ["string:org.freedesktop.hostname1", "string:OperatingSystemPrettyName"],
        )
        info["os"] = os_release
    except Exception:
        pass

    return info


async def list_network_connections() -> str:
    """List network connections via NetworkManager DBus API."""
    return await call_dbus_method(
        "system",
        "org.freedesktop.NetworkManager",
        "/org/freedesktop/NetworkManager",
        "org.freedesktop.DBus.Properties",
        "Get",
        ["string:org.freedesktop.NetworkManager", "string:ActiveConnections"],
    )

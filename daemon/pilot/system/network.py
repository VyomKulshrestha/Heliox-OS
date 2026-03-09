"""Network management — WiFi list, connect, disconnect.

Cross-platform: Windows (netsh), Linux (nmcli), macOS (networksetup/airport).
"""

from __future__ import annotations

import logging

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

logger = logging.getLogger("pilot.system.network")


async def wifi_list() -> str:
    """List available WiFi networks."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["netsh", "wlan", "show", "networks", "mode=bssid"])
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(
            ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-s"]
        )
    else:  # Linux
        code, out, err = await run_command(["nmcli", "device", "wifi", "list"])

    if code != 0:
        raise RuntimeError(f"WiFi scan failed: {err.strip()}")
    return out.strip()


async def wifi_connect(ssid: str, password: str | None = None, interface: str | None = None) -> str:
    """Connect to a WiFi network."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        # First try to connect to a known profile
        code, out, err = await run_command(["netsh", "wlan", "connect", f"name={ssid}"])
        if code != 0 and password:
            # Create a profile and connect
            profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM><security>
        <authEncryption><authentication>WPA2PSK</authentication>
        <encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
        <sharedKey><keyType>passPhrase</keyType><protected>false</protected>
        <keyMaterial>{password}</keyMaterial></sharedKey>
    </security></MSM>
</WLANProfile>"""
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
                f.write(profile_xml)
                profile_path = f.name
            try:
                await run_command(["netsh", "wlan", "add", "profile", f"filename={profile_path}"])
                code, out, err = await run_command(["netsh", "wlan", "connect", f"name={ssid}"])
            finally:
                os.unlink(profile_path)

    elif CURRENT_PLATFORM == Platform.MACOS:
        args = ["networksetup", "-setairportnetwork"]
        iface = interface or "en0"
        args.extend([iface, ssid])
        if password:
            args.append(password)
        code, out, err = await run_command(args)

    else:  # Linux
        args = ["nmcli", "device", "wifi", "connect", ssid]
        if password:
            args.extend(["password", password])
        if interface:
            args.extend(["ifname", interface])
        code, out, err = await run_command(args)

    if code != 0:
        raise RuntimeError(f"WiFi connect failed: {err.strip()}")
    return f"Connected to WiFi: {ssid}"


async def wifi_disconnect(interface: str | None = None) -> str:
    """Disconnect from the current WiFi network."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["netsh", "wlan", "disconnect"])
    elif CURRENT_PLATFORM == Platform.MACOS:
        iface = interface or "en0"
        code, out, err = await run_command(["networksetup", "-setairportpower", iface, "off"])
        # Re-enable but disconnected
        await run_command(["networksetup", "-setairportpower", iface, "on"])
    else:  # Linux
        iface = interface or "wlan0"
        code, out, err = await run_command(["nmcli", "device", "disconnect", iface])

    if code != 0:
        raise RuntimeError(f"WiFi disconnect failed: {err.strip()}")
    return "Disconnected from WiFi"

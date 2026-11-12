"""Network management — WiFi list, connect, disconnect.

Cross-platform: Windows (netsh), Linux (nmcli), macOS (networksetup/airport).
"""

from __future__ import annotations

import logging
import os

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
            code, out, err = await _connect_with_profile(ssid, password)

    elif CURRENT_PLATFORM == Platform.MACOS:
        # A network the Mac already knows joins with the password in the keychain; none is passed.
        code, out, err = await run_command(["networksetup", "-setairportnetwork", interface or "en0", ssid])
        if code != 0 and password:
            # networksetup takes a password only as an argument, which every user of the Mac can read in the
            # process list while it runs, and macOS offers no other way to a command-line tool. So Heliox does not
            # type a Wi-Fi password on macOS: it opens Wi-Fi settings for the person to pick the network there.
            await run_command(["open", MAC_WIFI_SETTINGS])
            raise RuntimeError(MAC_PASSWORD_DECLINED.format(ssid=ssid))

    else:  # Linux
        args = ["nmcli", "device", "wifi", "connect", ssid]
        if interface:
            args.extend(["ifname", interface])
        # A known or open network connects with no password given.
        code, out, err = await run_command(args)
        if code != 0 and password:
            # The password goes to NetworkManager over D-Bus, never as an nmcli argument (which any user of the
            # machine can read in the process list while it runs) and never in a file.
            import asyncio

            try:
                await asyncio.to_thread(nm_connect_with_password, ssid, password, interface)
            except NetworkManagerUnavailable as error:
                raise RuntimeError(str(error)) from error
            code, err = 0, ""

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


def wlan_profile_xml(ssid: str, password: str) -> str:
    """The WLAN profile netsh adds. The network name and the password are XML text, escaped: the name is chosen by
    whoever runs the access point, and a name holding "<" or "&" would otherwise rewrite the profile."""
    from xml.sax.saxutils import escape

    name, key = escape(ssid), escape(password)
    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{name}</name>
    <SSIDConfig><SSID><name>{name}</name></SSID></SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM><security>
        <authEncryption><authentication>WPA2PSK</authentication>
        <encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
        <sharedKey><keyType>passPhrase</keyType><protected>false</protected>
        <keyMaterial>{key}</keyMaterial></sharedKey>
    </security></MSM>
</WLANProfile>"""


async def _connect_with_profile(ssid: str, password: str) -> tuple[int, str, str]:
    """Add a WLAN profile from a file only the person can read, connect, and remove the file and its folder on
    every path, also when writing it or adding it fails half-way. The folder is new and private (mkdtemp: 0700 on
    macOS and Linux; on Windows inside the person's own temp folder, which other users cannot open)."""
    import shutil
    import tempfile

    folder = tempfile.mkdtemp(prefix="heliox-wlan-")
    try:
        profile_path = os.path.join(folder, "profile.xml")
        fd = os.open(profile_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(wlan_profile_xml(ssid, password))
        await run_command(["netsh", "wlan", "add", "profile", f"filename={profile_path}"])
        return await run_command(["netsh", "wlan", "connect", f"name={ssid}"])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


# ── Wi-Fi passwords on macOS and Linux: never in a command's arguments ──

MAC_WIFI_SETTINGS = "x-apple.systempreferences:com.apple.wifi-settings-extension"
MAC_PASSWORD_DECLINED = (
    "Heliox did not join {ssid}: on a Mac it would have to put the Wi-Fi password in a command's arguments, which "
    "other users of this Mac can read while it runs. Wi-Fi settings are open; choose {ssid} there and type the "
    "password into macOS's own box."
)
NM = "org.freedesktop.NetworkManager"
NM_PATH = "/org/freedesktop/NetworkManager"
NM_DEVICE_TYPE_WIFI = 2
NM_ACTIVE_ACTIVATED = 2
NM_ACTIVE_DEACTIVATED = 4


class NetworkManagerUnavailable(RuntimeError):
    """NetworkManager could not be asked over D-Bus; the message is the sentence the person sees."""


def nm_wifi_settings(ssid: str, password: str) -> dict:
    """The connection NetworkManager adds: a WPA-PSK network named ``ssid``. The password is a D-Bus value."""
    return {
        "connection": {"type": ("s", "802-11-wireless"), "id": ("s", ssid)},
        "802-11-wireless": {"ssid": ("ay", ssid.encode("utf-8")), "mode": ("s", "infrastructure")},
        "802-11-wireless-security": {"key-mgmt": ("s", "wpa-psk"), "psk": ("s", password)},
    }


def nm_connect_with_password(
    ssid: str, password: str, interface: str | None = None, *, bus: object | None = None, timeout: float = 30.0
) -> str:
    """Add and activate the network through NetworkManager's D-Bus API (AddAndActivateConnection), then wait until
    it is up. ``bus`` is a jeepney blocking connection to the system bus (made here when not given). The active
    connection's path, or NetworkManagerUnavailable / RuntimeError with the reason."""
    import time

    try:
        from jeepney import DBusAddress, Properties, new_method_call
        from jeepney.io.blocking import open_dbus_connection
        from jeepney.wrappers import unwrap_msg
    except ImportError as error:
        raise NetworkManagerUnavailable(
            "Heliox can only give NetworkManager a Wi-Fi password over D-Bus, which needs the jeepney package "
            "(pip install 'pilot-daemon[linux]'); it does not pass passwords as command arguments."
        ) from error

    def ask(connection, address, method, signature=None, body=()):
        return unwrap_msg(connection.send_and_get_reply(new_method_call(address, method, signature, body), timeout=10))

    def prop(connection, path, interface, name):
        return unwrap_msg(
            connection.send_and_get_reply(Properties(DBusAddress(path, bus_name=NM, interface=interface)).get(name))
        )[0][1]

    def run(connection) -> str:
        manager = DBusAddress(NM_PATH, bus_name=NM, interface=NM)
        if interface:
            device = ask(connection, manager, "GetDeviceByIpIface", "s", (interface,))[0]
        else:
            devices = ask(connection, manager, "GetDevices")[0]
            wifi = [d for d in devices if prop(connection, d, NM + ".Device", "DeviceType") == NM_DEVICE_TYPE_WIFI]
            if not wifi:
                raise RuntimeError("WiFi connect failed: NetworkManager has no Wi-Fi device")
            device = wifi[0]
        _path, active = ask(
            connection,
            manager,
            "AddAndActivateConnection",
            "a{sa{sv}}oo",
            (nm_wifi_settings(ssid, password), device, "/"),
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = prop(connection, active, NM + ".Connection.Active", "State")
            except Exception:
                state = NM_ACTIVE_DEACTIVATED
            if state == NM_ACTIVE_ACTIVATED:
                return str(active)
            if state == NM_ACTIVE_DEACTIVATED:
                raise RuntimeError(f"WiFi connect failed: NetworkManager could not join {ssid} (wrong password?)")
            time.sleep(0.5)
        raise RuntimeError(f"WiFi connect failed: {ssid} was not up after {timeout:.0f} seconds")

    if bus is not None:
        return run(bus)
    try:
        connection = open_dbus_connection(bus="SYSTEM")
    except Exception as error:
        raise NetworkManagerUnavailable(
            "Heliox gives a Wi-Fi password to NetworkManager over D-Bus, and could not reach it here "
            f"({type(error).__name__}); it does not pass passwords as command arguments. Join the network from the "
            "desktop's network menu."
        ) from error
    with connection:
        return run(connection)

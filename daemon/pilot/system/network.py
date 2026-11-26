"""Network management — WiFi list, connect, disconnect.

Cross-platform: Windows (netsh), Linux (nmcli), macOS (networksetup/airport).
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

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
        # macOS has no command-line "leave this network" that stays left: turning Wi-Fi back on rejoins the
        # preferred network at once. So disconnecting turns Wi-Fi off, and says so.
        iface = interface or "en0"
        code, out, err = await run_command(["networksetup", "-setairportpower", iface, "off"])
        if code == 0:
            return "Disconnected from WiFi (Wi-Fi is turned off; turn it on to join a network again)"
    else:  # Linux
        iface = interface or "wlan0"
        code, out, err = await run_command(["nmcli", "device", "disconnect", iface])

    if code != 0:
        raise RuntimeError(f"WiFi disconnect failed: {err.strip()}")
    return "Disconnected from WiFi"


# ── Which network is joined now: read-only, for the wifi_connect post-condition ──

WLAN_INTERFACE_STATE_CONNECTED = 1
WLAN_INTF_OPCODE_CURRENT_CONNECTION = 7
WLAN_INTERFACE_INFO_SIZE = 532  # GUID (16) + WCHAR[256] description (512) + state (4)
WLAN_SSID_OFFSET = 520  # in WLAN_CONNECTION_ATTRIBUTES: state, mode, WCHAR[256] profile name, then DOT11_SSID
ERROR_ACCESS_DENIED = 5


class WifiReading(NamedTuple):
    """The networks the Wi-Fi interfaces are joined to now. ``ssids`` is None when the platform withheld the
    answer, with ``reason`` saying why."""

    ssids: list[str] | None
    reason: str = ""


def parse_nmcli_active_ssids(text: str) -> list[str]:
    """SSIDs marked active in ``nmcli -t -f ACTIVE,SSID dev wifi list`` (terse: "yes:Name", with ":" and "\\"
    inside the name escaped by a backslash)."""
    found: list[str] = []
    for line in text.splitlines():
        fields: list[str] = []
        current: list[str] = []
        escaped = False
        for char in line:
            if escaped:
                current.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == ":" and not fields:
                fields.append("".join(current))
                current = []
            else:
                current.append(char)
        fields.append("".join(current))
        if len(fields) == 2 and fields[0] == "yes" and fields[1]:
            found.append(fields[1])
    return found


def parse_mac_airport_network(text: str) -> str | None:
    """The network from ``networksetup -getairportnetwork``: "Current Wi-Fi Network: Name" (older systems say
    "AirPort"). None for "You are not associated with an AirPort network." or anything else."""
    import re

    match = re.search(r"^Current (?:Wi-Fi|AirPort) Network: (.+)$", text.strip(), re.MULTILINE)
    return match.group(1) if match else None


def parse_mac_ipconfig_ssid(text: str) -> str | None:
    """The "SSID : Name" line of ``ipconfig getsummary <interface>``; "<redacted>" when macOS hides it."""
    import re

    match = re.search(r"^\s*SSID : (.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _windows_connected_ssids() -> WifiReading:
    """Ask the native WLAN API (wlanapi.dll) which interfaces are connected and to which SSID. No netsh text, so
    nothing depends on the language Windows speaks. Windows 11 24H2 and later withhold the SSID from apps that do
    not have location access; that comes back as access denied, reported as such."""
    import ctypes
    from ctypes import wintypes

    wlan = ctypes.WinDLL("wlanapi")
    wlan.WlanOpenHandle.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.HANDLE),
    ]
    wlan.WlanEnumInterfaces.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    wlan.WlanQueryInterface.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    wlan.WlanFreeMemory.argtypes = [ctypes.c_void_p]
    wlan.WlanCloseHandle.argtypes = [wintypes.HANDLE, ctypes.c_void_p]

    handle = wintypes.HANDLE()
    version = wintypes.DWORD()
    code = wlan.WlanOpenHandle(2, None, ctypes.byref(version), ctypes.byref(handle))
    if code != 0:
        # No WLAN service (a desktop without Wi-Fi hardware, or the service stopped).
        return WifiReading(None, f"the Windows WLAN service did not answer (error {code})")
    interfaces = ctypes.c_void_p()
    try:
        code = wlan.WlanEnumInterfaces(handle, None, ctypes.byref(interfaces))
        if code != 0:
            raise OSError(code, "WlanEnumInterfaces failed")
        try:
            count = ctypes.c_uint32.from_address(interfaces.value).value
            ssids: list[str] = []
            for index in range(count):
                base = interfaces.value + 8 + index * WLAN_INTERFACE_INFO_SIZE
                state = ctypes.c_uint32.from_address(base + 16 + 512).value
                if state != WLAN_INTERFACE_STATE_CONNECTED:
                    continue
                size = wintypes.DWORD()
                data = ctypes.c_void_p()
                code = wlan.WlanQueryInterface(
                    handle,
                    base,
                    WLAN_INTF_OPCODE_CURRENT_CONNECTION,
                    None,
                    ctypes.byref(size),
                    ctypes.byref(data),
                    None,
                )
                if code == ERROR_ACCESS_DENIED:
                    return WifiReading(None, "Windows withholds the network name from apps without location access")
                if code != 0:
                    raise OSError(code, "WlanQueryInterface failed")
                try:
                    raw = ctypes.string_at(data, size.value)
                finally:
                    wlan.WlanFreeMemory(data)
                length = min(int.from_bytes(raw[WLAN_SSID_OFFSET : WLAN_SSID_OFFSET + 4], "little"), 32)
                start = WLAN_SSID_OFFSET + 4
                ssids.append(raw[start : start + length].decode("utf-8", errors="replace"))
            return WifiReading(ssids)
        finally:
            wlan.WlanFreeMemory(interfaces)
    finally:
        wlan.WlanCloseHandle(handle, None)


async def connected_ssids(interface: str | None = None) -> WifiReading:
    """The SSIDs joined now, read through a different path than the one that joined: the WLAN API on Windows,
    NetworkManager's device list on Linux, networksetup (then ipconfig) on macOS. Raises when the read fails."""
    import asyncio

    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await asyncio.to_thread(_windows_connected_ssids)

    if CURRENT_PLATFORM == Platform.MACOS:
        iface = interface or "en0"
        code, out, err = await run_command(["networksetup", "-getairportnetwork", iface], timeout=10)
        if code != 0:
            raise RuntimeError(f"Wi-Fi read failed: {err.strip() or out.strip()}")
        name = parse_mac_airport_network(out)
        if name is not None:
            return WifiReading([name])
        # macOS 15 answers "not associated" here even when joined; ipconfig still knows, unless it redacts the name.
        code, out, err = await run_command(["ipconfig", "getsummary", iface], timeout=10)
        if code != 0:
            raise RuntimeError(f"Wi-Fi read failed: {err.strip() or out.strip()}")
        ssid = parse_mac_ipconfig_ssid(out)
        if ssid == "<redacted>":
            return WifiReading(None, "macOS hides the network name from apps without Location Services access")
        return WifiReading([ssid] if ssid else [])

    # --rescan no: read what NetworkManager knows now, never start a scan. LC_ALL=C keeps "yes" untranslated.
    args = ["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi", "list"]
    if interface:
        args.extend(["ifname", interface])
    args.extend(["--rescan", "no"])
    code, out, err = await run_command(args, timeout=10, env={**os.environ, "LC_ALL": "C"})
    if code != 0:
        raise RuntimeError(f"Wi-Fi read failed: {err.strip() or out.strip()}")
    return WifiReading(parse_nmcli_active_ssids(out))


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

"""A Wi-Fi password is never a command's argument, on any system.

Arguments of a running command are visible to every user of the machine (the process list), so the password goes
to the system some other way, or Heliox declines and hands the person to the system's own dialog:
- Windows: inside the WLAN profile file, owner-only and removed afterwards (tests/test_script_values_sites_more.py);
- Linux: to NetworkManager over D-Bus (AddAndActivateConnection);
- macOS: networksetup only takes it as an argument, so Heliox opens Wi-Fi settings and says so.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from pilot.system import network
from pilot.system.platform_detect import Platform

PASSWORD = "c0rrect-h0rse-b4ttery"


def _recorder(fail_first: bool = True):
    calls: list[list[str]] = []

    async def run(args, **_kwargs):
        calls.append([str(a) for a in args])
        if fail_first and len(calls) == 1:
            return 1, "", "Secrets were required, but not provided"
        return 0, "", ""

    return calls, run


def _no_password_in(calls: list[list[str]]) -> None:
    for args in calls:
        assert all(PASSWORD not in arg for arg in args), args


def test_windows_puts_the_password_in_the_profile_file_only(monkeypatch):
    calls, run = _recorder()
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(network, "run_command", run)
    asyncio.run(network.wifi_connect("Cafe", PASSWORD))
    _no_password_in(calls)
    assert any(args[2] == "add" for args in calls)


def test_macos_declines_and_opens_wifi_settings(monkeypatch):
    calls, run = _recorder()
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(network, "run_command", run)
    with pytest.raises(RuntimeError, match="Heliox did not join Cafe: on a Mac") as refusal:
        asyncio.run(network.wifi_connect("Cafe", PASSWORD))
    assert "Wi-Fi settings are open" in str(refusal.value) and PASSWORD not in str(refusal.value)
    _no_password_in(calls)
    assert calls == [["networksetup", "-setairportnetwork", "en0", "Cafe"], ["open", network.MAC_WIFI_SETTINGS]]


def test_macos_joins_a_known_network_without_a_password(monkeypatch):
    calls, run = _recorder(fail_first=False)
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(network, "run_command", run)
    assert asyncio.run(network.wifi_connect("Home", PASSWORD)) == "Connected to WiFi: Home"
    _no_password_in(calls)


def test_linux_gives_the_password_to_networkmanager_not_nmcli(monkeypatch):
    calls, run = _recorder()
    given: list[tuple] = []
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)
    monkeypatch.setattr(network, "nm_connect_with_password", lambda *args: given.append(args) or "/active/1")
    assert asyncio.run(network.wifi_connect("Cafe", PASSWORD, "wlan0")) == "Connected to WiFi: Cafe"
    _no_password_in(calls)
    assert calls == [["nmcli", "device", "wifi", "connect", "Cafe", "ifname", "wlan0"]]
    assert given == [("Cafe", PASSWORD, "wlan0")]


def test_linux_without_jeepney_declines_with_a_sentence(monkeypatch):
    calls, run = _recorder()
    monkeypatch.setattr(network, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(network, "run_command", run)
    monkeypatch.setitem(sys.modules, "jeepney", None)
    with pytest.raises(RuntimeError, match="does not pass passwords as command arguments"):
        asyncio.run(network.wifi_connect("Cafe", PASSWORD))
    _no_password_in(calls)


class FakeSystemBus:
    """NetworkManager on the system bus: one Wi-Fi device, and an active connection that comes up."""

    def __init__(self, states=(1, 2)) -> None:
        self.sent: list = []
        self.states = list(states)

    def send_and_get_reply(self, message, timeout=None):
        self.sent.append(message)
        return message


def _fake_nm(monkeypatch, bus: FakeSystemBus):
    pytest.importorskip("jeepney")
    import jeepney.wrappers

    def unwrap(message):
        fields = message.header.fields
        from jeepney import HeaderFields

        member = fields.get(HeaderFields.member)
        body = message.body
        if member == "GetDevices":
            return (["/org/freedesktop/NetworkManager/Devices/1", "/org/freedesktop/NetworkManager/Devices/2"],)
        if member == "Get":
            name = body[1]
            if name == "DeviceType":
                path = fields.get(HeaderFields.path)
                return (("u", 2 if path.endswith("/2") else 1),)
            if name == "State":
                return (("u", bus.states.pop(0)),)
        if member == "AddAndActivateConnection":
            return ("/org/freedesktop/NetworkManager/Settings/9", "/org/freedesktop/NetworkManager/ActiveConnection/4")
        raise AssertionError(member)

    monkeypatch.setattr(jeepney.wrappers, "unwrap_msg", unwrap)
    monkeypatch.setattr(network.time if hasattr(network, "time") else __import__("time"), "sleep", lambda _s: None)


def test_networkmanager_receives_the_password_as_a_dbus_value(monkeypatch):
    bus = FakeSystemBus()
    _fake_nm(monkeypatch, bus)
    active = network.nm_connect_with_password("Cafe", PASSWORD, bus=bus, timeout=5)
    assert active.endswith("/ActiveConnection/4")
    from jeepney import HeaderFields

    add = next(m for m in bus.sent if m.header.fields.get(HeaderFields.member) == "AddAndActivateConnection")
    settings, device, ap = add.body
    assert settings["802-11-wireless-security"]["psk"] == ("s", PASSWORD)
    assert settings["802-11-wireless"]["ssid"] == ("ay", b"Cafe")
    assert device.endswith("/Devices/2") and ap == "/"


def test_networkmanager_failing_to_join_is_said_plainly(monkeypatch):
    bus = FakeSystemBus(states=(1, 4))
    _fake_nm(monkeypatch, bus)
    with pytest.raises(RuntimeError, match="could not join Cafe"):
        network.nm_connect_with_password("Cafe", PASSWORD, bus=bus, timeout=5)

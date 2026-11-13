"""A test media player on the session bus for mpris_check.py: org.mpris.MediaPlayer2.<name>, with Play, Pause,
PlayPause, Next and Previous, and PlaybackStatus, Metadata and Identity as a real player reports them.

Run with the system Python, which has PyGObject (python3-gi): ``python3 mpris_fixture_player.py <name> <status>``.
"""

from __future__ import annotations

import sys

from gi.repository import Gio, GLib

XML = """<node>
  <interface name="org.mpris.MediaPlayer2">
    <property name="Identity" type="s" access="read"/>
  </interface>
  <interface name="org.mpris.MediaPlayer2.Player">
    <method name="Play"/><method name="Pause"/><method name="PlayPause"/>
    <method name="Next"/><method name="Previous"/>
    <property name="PlaybackStatus" type="s" access="read"/>
    <property name="Metadata" type="a{sv}" access="read"/>
  </interface>
</node>"""
TRACKS = ["Blue", "River", "Carey"]
name = sys.argv[1] if len(sys.argv) > 1 else "heliox_test"
state = {"status": sys.argv[2] if len(sys.argv) > 2 else "Playing", "track": 0}


def on_call(_conn, _sender, _path, _iface, method, _params, invocation):
    if method == "Play":
        state["status"] = "Playing"
    elif method == "Pause":
        state["status"] = "Paused"
    elif method == "PlayPause":
        state["status"] = "Paused" if state["status"] == "Playing" else "Playing"
    elif method == "Next":
        state["track"] = (state["track"] + 1) % len(TRACKS)
    elif method == "Previous":
        state["track"] = (state["track"] - 1) % len(TRACKS)
    invocation.return_value(None)


def on_get(_conn, _sender, _path, _iface, prop):
    if prop == "Identity":
        return GLib.Variant("s", f"Heliox Test Player {name}")
    if prop == "PlaybackStatus":
        return GLib.Variant("s", state["status"])
    if prop == "Metadata":
        return GLib.Variant(
            "a{sv}",
            {
                "xesam:title": GLib.Variant("s", TRACKS[state["track"]]),
                "xesam:artist": GLib.Variant("as", ["Joni Mitchell"]),
            },
        )
    return None


node = Gio.DBusNodeInfo.new_for_xml(XML)


def on_bus(connection, _name):
    for interface in node.interfaces:
        connection.register_object("/org/mpris/MediaPlayer2", interface, on_call, on_get, None)


Gio.bus_own_name(Gio.BusType.SESSION, "org.mpris.MediaPlayer2." + name, Gio.BusNameOwnerFlags.NONE, on_bus, None, None)
GLib.MainLoop().run()

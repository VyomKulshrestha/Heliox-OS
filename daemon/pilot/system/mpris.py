"""Next, previous, play and pause on Linux through MPRIS: the media player itself, not a key press.

Every Linux media player that takes part in the desktop's media controls (Spotify, Firefox and Chromium, VLC, mpv
with its plugin, Rhythmbox, ...) answers on the session bus as ``org.mpris.MediaPlayer2.<name>``. Heliox calls the
player's own method and then reads back what it reports, so the sentence is what is true afterwards ("Paused
Spotify.", "Now playing "Blue" by Joni Mitchell."). The connection is jeepney's, already a dependency on Linux; no
other package and no script text.

Which player: the one that is playing; if none is playing, the only one that is paused; if none is paused, the only
player there is. No player: Heliox says so and presses nothing. Two or more playing (or, with none playing, two or
more paused): Heliox names them and changes none, since it cannot tell which one the person means. MPRIS keeps no
"last active" time, so there is no fairer pick without guessing.

``media_command`` returns None when MPRIS cannot be reached at all (not Linux, no session bus, no jeepney): only then
does the caller fall back to the key press.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol

PLAYER = "org.mpris.MediaPlayer2.Player"
ROOT = "org.mpris.MediaPlayer2"
PATH = "/org/mpris/MediaPlayer2"
PREFIX = "org.mpris.MediaPlayer2."
# What each request calls on the player.
METHODS = {"pause": "Pause", "play": "Play", "playpause": "PlayPause", "nexttrack": "Next", "prevtrack": "Previous"}
SETTLE_SECONDS = 0.6  # how long to wait for the player to report its new state


class Bus(Protocol):
    def names(self) -> list[str]: ...
    def get(self, name: str, interface: str, prop: str) -> Any: ...
    def call(self, name: str, method: str) -> None: ...
    def close(self) -> None: ...


class JeepneyBus:
    """The session bus through jeepney (blocking; callers run it in a thread)."""

    def __init__(self) -> None:
        from jeepney.io.blocking import open_dbus_connection

        address = "SESSION" if os.environ.get("DBUS_SESSION_BUS_ADDRESS") else None
        if address is None:
            runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"  # type: ignore[attr-defined]
            address = f"unix:path={runtime}/bus"
        self._connection = open_dbus_connection(address)

    def _send(self, message: Any) -> tuple:
        from jeepney.wrappers import unwrap_msg

        return tuple(unwrap_msg(self._connection.send_and_get_reply(message, timeout=2.0)))

    def names(self) -> list[str]:
        from jeepney import DBusAddress, new_method_call

        bus = DBusAddress("/org/freedesktop/DBus", bus_name="org.freedesktop.DBus", interface="org.freedesktop.DBus")
        return [str(name) for name in self._send(new_method_call(bus, "ListNames"))[0]]

    def get(self, name: str, interface: str, prop: str) -> Any:
        from jeepney import DBusAddress, Properties

        return self._send(Properties(DBusAddress(PATH, bus_name=name, interface=interface)).get(prop))[0][1]

    def call(self, name: str, method: str) -> None:
        from jeepney import DBusAddress, new_method_call

        self._send(new_method_call(DBusAddress(PATH, bus_name=name, interface=PLAYER), method))

    def close(self) -> None:
        self._connection.close()


def _text(value: Any) -> str:
    """A metadata value as text: a string, or the first of a list of strings (xesam:artist is a list)."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str) and len(value[0]) <= 4:
        value = value[1]  # a (signature, value) variant
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "")


class Player:
    def __init__(self, bus: Bus, name: str) -> None:
        self.bus, self.name = bus, name

    def status(self) -> str:
        try:
            return str(self.bus.get(self.name, PLAYER, "PlaybackStatus"))
        except Exception:
            return ""

    def label(self) -> str:
        try:
            identity = str(self.bus.get(self.name, ROOT, "Identity"))
        except Exception:
            identity = ""
        return identity or self.name[len(PREFIX) :].split(".")[0].capitalize()

    def track(self) -> str:
        """'"Blue" by Joni Mitchell', '"Blue"', or "" when the player reports no title."""
        try:
            metadata = self.bus.get(self.name, PLAYER, "Metadata") or {}
        except Exception:
            return ""
        title, artist = _text(metadata.get("xesam:title")), _text(metadata.get("xesam:artist"))
        if not title:
            return ""
        return f'"{title}"' + (f" by {artist}" if artist else "")


def _joined(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def choose(players: list[Player]) -> tuple[Player | None, str]:
    """(the player to act on, "") or (None, the sentence to say instead). The rule is in the module's docstring."""
    if not players:
        return None, "No media player is running."
    statuses = {player.name: player.status() for player in players}
    for wanted in ("Playing", "Paused"):
        found = [player for player in players if statuses[player.name] == wanted]
        if len(found) == 1:
            return found[0], ""
        if len(found) > 1:
            names = _joined(sorted(player.label() for player in found))
            state = "playing" if wanted == "Playing" else "paused"
            return None, f"{names} are all {state}, so I didn't change any of them. Say which one."
    if len(players) == 1:
        return players[0], ""
    names = _joined(sorted(player.label() for player in players))
    return None, f"{names} are open but none is playing, so I didn't change any of them. Say which one."


def said_after(player: Player, request: str, before_track: str) -> str:
    """What is true after the call, from what the player reports."""
    status, track, label = player.status(), player.track(), player.label()
    if request in {"nexttrack", "prevtrack"}:
        if track and track != before_track:
            return f"Now playing {track}."
        return (
            f"{label} didn't change track."
            if track
            else f"{'Next' if request == 'nexttrack' else 'Previous'} track in {label}."
        )
    if status == "Paused":
        return f"Paused {label}."
    if status == "Playing":
        return f"Playing {track} in {label}." if track else f"{label} is playing."
    if status == "Stopped":
        return f"{label} is stopped."
    return f"I asked {label}, but it didn't say what it is doing."


def goes_to_player(request: str, platform: str | None = None, reachable: bool | None = None) -> bool:
    """Whether a media command goes to the player itself through MPRIS rather than as a key press. The one place
    this is decided, from two inputs: ``platform``, the system the command is for (sys.platform's form, "linux",
    "win32", "darwin"; None for this computer), and ``reachable``, whether the session bus answers (None: asked
    here, and only on Linux). Windows and macOS have no such call here, so there it is always the key press,
    whatever a bus would say."""
    if request not in METHODS:
        return False
    if not (sys.platform if platform is None else platform).casefold().startswith("linux"):
        return False
    return media_call_available() if reachable is None else bool(reachable)


def media_command(
    request: str,
    *,
    platform: str | None = None,
    bus_factory: Callable[[], Bus] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Do ``request`` ("pause", "play", "playpause", "nexttrack", "prevtrack") through MPRIS and say what is true.
    None when MPRIS cannot be reached (not Linux, no session bus, no jeepney): the caller may press the key.
    ``platform`` as for goes_to_player (None: this computer)."""
    if request not in METHODS:
        return None
    if bus_factory is None:
        if not goes_to_player(request, platform, reachable=True):
            return None
        bus_factory = JeepneyBus
    try:
        bus = bus_factory()
    except Exception:
        return None
    try:
        players = [Player(bus, name) for name in bus.names() if name.startswith(PREFIX)]
        player, instead = choose(players)
        if player is None:
            return instead
        before_track = player.track()
        before_status = player.status()
        if request == "pause" and before_status != "Playing":
            return (
                f"{player.label()} is already paused."
                if before_status == "Paused"
                else f"{player.label()} isn't playing."
            )
        if request == "play" and before_status == "Playing":
            return f"{player.label()} is already playing."
        bus.call(player.name, METHODS[request])
        waited = 0.0
        while waited < SETTLE_SECONDS:  # players report their new state a moment after the call
            sleep(0.1)
            waited += 0.1
            if player.status() != before_status or player.track() != before_track:
                break
        return said_after(player, request, before_track)
    except Exception as error:
        return f"The media player didn't answer ({type(error).__name__}), so nothing changed."
    finally:
        try:
            bus.close()
        except Exception:
            pass


def media_call_available(bus_factory: Callable[[], Bus] | None = None) -> bool:
    """Whether MPRIS can be reached on this computer (Linux with a session bus and jeepney)."""
    if bus_factory is None:
        if not goes_to_player("playpause", reachable=True):
            return False
        bus_factory = JeepneyBus
    try:
        bus_factory().close()
        return True
    except Exception:
        return False

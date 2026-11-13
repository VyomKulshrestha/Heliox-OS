"""Next, previous, play and pause through MPRIS for real, on the Linux runner (``platform-readers``), against test
players on a private session bus (``dbus-run-session``; mpris_fixture_player.py, started with the system Python).

Checks: no player is said; the one playing player is paused, played and moved to the next and previous track, each
sentence from what the player reports; with two players playing, neither is changed and both are named.

It starts processes that own names on the session bus, so it runs only on a runner (ci_only). Each check prints one
``CHECK`` line; the exit code is the number of failed checks.
Usage: dbus-run-session -- python mpris_check.py --python /usr/bin/python3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_only import require_runner  # noqa: E402

require_runner("starts test media players on the session bus")

from pilot.system import mpris  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "mpris_fixture_player.py"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def wait_for(name: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bus = mpris.JeepneyBus()
        try:
            if mpris.PREFIX + name in bus.names():
                return True
        finally:
            bus.close()
        time.sleep(0.2)
    return False


def status(name: str) -> str:
    bus = mpris.JeepneyBus()
    try:
        return str(bus.get(mpris.PREFIX + name, mpris.PLAYER, "PlaybackStatus"))
    finally:
        bus.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/usr/bin/python3", help="a Python with PyGObject for the test players")
    args = parser.parse_args()
    check("mpris/session-bus-reached", mpris.media_call_available())
    said = mpris.media_command("pause")
    check("mpris/no-player-is-said", said == "No media player is running.", repr(said))
    players = []
    try:
        players.append(subprocess.Popen([args.python, str(FIXTURE), "first", "Playing"]))
        check("mpris/test-player-on-the-bus", wait_for("first"))
        said = mpris.media_command("pause")
        check("mpris/pause", said == "Paused Heliox Test Player first." and status("first") == "Paused", repr(said))
        said = mpris.media_command("play")
        check("mpris/play", said == 'Playing "Blue" by Joni Mitchell in Heliox Test Player first.', repr(said))
        said = mpris.media_command("nexttrack")
        check("mpris/next", said == 'Now playing "River" by Joni Mitchell.', repr(said))
        said = mpris.media_command("prevtrack")
        check("mpris/previous", said == 'Now playing "Blue" by Joni Mitchell.', repr(said))
        players.append(subprocess.Popen([args.python, str(FIXTURE), "second", "Playing"]))
        check("mpris/second-player-on-the-bus", wait_for("second"))
        said = mpris.media_command("pause")
        both_still_playing = status("first") == "Playing" and status("second") == "Playing"
        check(
            "mpris/two-playing-changes-neither",
            said is not None and "are all playing" in said and both_still_playing,
            repr(said),
        )
    finally:
        for player in players:
            player.terminate()
            try:
                player.wait(timeout=5)
            except subprocess.TimeoutExpired:
                player.kill()
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

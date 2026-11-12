"""Window actions on Linux for real: X11 (Xvfb) with a window manager (openbox) and two GTK windows.

Run inside an X session with openbox running; the GTK fixture runs with the system Python (``--python``):

    python linux_windows_check.py --python /usr/bin/python3

Checks, each through ``pilot.system.window_mgr`` as the planner calls it:

- the list names both windows; ``window_exists`` finds one by its title;
- focus brings each window to the front (the window manager's active window says so);
- the timeline's foreground falls back to xdotool when the accessibility bus cannot be reached: the reader is run in
  a child process whose session bus address points nowhere, and it must still name the front window;
- minimise hides the window (``_NET_WM_STATE_HIDDEN``), maximise sets both maximised states;
- close closes that window only: the other stays listed.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import one_loop  # noqa: E402  (next to this script)
from linux_atspi_check import Fixture  # noqa: E402

from pilot.system import window_mgr  # noqa: E402

FIRST = "Heliox AT-SPI fixture"
SECOND = "Heliox second window"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=10, check=False).stdout.strip()


def active_title() -> str:
    window = run(["xdotool", "getactivewindow"])
    return run(["xdotool", "getwindowname", window]) if window else ""


def window_state(title: str) -> str:
    ids = run(["xdotool", "search", "--name", f"^{title}$"]).split()
    return run(["xprop", "-id", ids[0], "_NET_WM_STATE"]) if ids else "no window"


def call(coroutine) -> str:
    try:
        return one_loop.run(coroutine)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


FALLBACK = """
import json
from pilot.agents.screen_vision import _get_active_window_linux
from pilot.live import focused_text_linux
print(json.dumps({"atspi": focused_text_linux.foreground_window(), "found": _get_active_window_linux()}))
"""


def foreground_without_the_bus() -> dict:
    env = {key: value for key, value in os.environ.items() if key != "AT_SPI_BUS_ADDRESS"}
    env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/nonexistent/heliox-no-bus"
    done = subprocess.run(
        [sys.executable, "-c", FALLBACK],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
        check=False,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else {"error": done.stderr[-400:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    first, second = Fixture(args.python), Fixture(args.python)
    try:
        ready = first.wait_for("ready") and second.wait_for("ready")
        if not check("windows-x11/fixtures-started", ready):
            return failures
        second.send(f"title {SECOND}")
        time.sleep(1.0)

        listing = call(window_mgr.window_list())
        check("windows-x11/list-names-both", FIRST in listing and SECOND in listing, listing.replace("\n", " | "))
        exists = call(window_mgr.window_exists(title=SECOND))
        check("windows-x11/exists-by-title", exists is True, repr(exists))

        call(window_mgr.window_focus(title=FIRST))
        time.sleep(0.8)
        check("windows-x11/focus-first", active_title() == FIRST, active_title())
        call(window_mgr.window_focus(title=SECOND))
        time.sleep(0.8)
        check("windows-x11/focus-second", active_title() == SECOND, active_title())

        found = foreground_without_the_bus()
        title = (found.get("found") or ["", ""])[1]
        check(
            "windows-x11/foreground-falls-back-to-xdotool",
            found.get("atspi") is None and title == SECOND,
            json.dumps(found),
        )

        call(window_mgr.window_maximize(title=FIRST))
        time.sleep(0.8)
        state = window_state(FIRST)
        check(
            "windows-x11/maximize",
            "_NET_WM_STATE_MAXIMIZED_VERT" in state and "_NET_WM_STATE_MAXIMIZED_HORZ" in state,
            state,
        )
        call(window_mgr.window_minimize(title=FIRST))
        time.sleep(0.8)
        state = window_state(FIRST)
        check("windows-x11/minimize", "_NET_WM_STATE_HIDDEN" in state, state)

        output = call(window_mgr.window_close(title=SECOND))
        time.sleep(1.5)
        listing = call(window_mgr.window_list())
        check(
            "windows-x11/close-closes-that-window-only",
            SECOND not in listing and FIRST in listing,
            f"{output}; now: {listing.replace(chr(10), ' | ')}",
        )
    finally:
        for fixture in (first, second):
            try:
                fixture.proc.kill()
            except Exception:
                pass
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens windows on this desktop and moves, focuses and closes them")
    sys.exit(main())

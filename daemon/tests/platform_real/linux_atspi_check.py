"""A real run of the Linux focused-text reader against a GTK window, for the ``platform-readers`` CI job.

Expects a session bus, a display (Xvfb, or a Wayland compositor with ``--wayland``) and AT-SPI installed. Starts
``gtk_fixture.py`` with the system Python, types into its entry (xdotool on X11; from inside the app on Wayland,
where no tool may inject keys), and checks what the reader returns. Every check prints one ``CHECK`` line;
the exit code is the number of failed checks.

    python tests/platform_real/linux_atspi_check.py [--wayland] [--python /usr/bin/python3]
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.live import focused_text_linux as L  # noqa: E402
from pilot.live.focused_text_policy import same_field  # noqa: E402

HERE = Path(__file__).resolve().parent
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


class Fixture:
    def __init__(self, python: str) -> None:
        env = {key: value for key, value in os.environ.items() if key != "NO_AT_BRIDGE"}
        self.proc = subprocess.Popen(
            [python, str(HERE / "gtk_fixture.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )

    def wait_for(self, prefix: str, timeout: float = 30.0) -> bool:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                return False
            if line.startswith(prefix):
                return True
        return False

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        self.wait_for(f"ok {command}", 10.0)
        time.sleep(0.3)

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def read_until(predicate, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = L.read_focused()
        if last is not None and predicate(last):
            return last
        time.sleep(0.3)
    return last


def typer(wayland: bool, fixture: Fixture):
    """Real keystrokes: xdotool on X11, wtype on Wayland; inside-the-app insertion where neither can send keys."""
    if not wayland and shutil.which("xdotool"):
        found = subprocess.run(
            ["xdotool", "search", "--sync", "--name", "Heliox AT-SPI fixture"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        window = (found.stdout.split() or [""])[0]
        if window:
            subprocess.run(["xdotool", "windowfocus", "--sync", window], timeout=10, check=False)

            def type_keys(text: str) -> None:
                subprocess.run(["xdotool", "type", "--delay", "20", text], timeout=30, check=False)
                time.sleep(0.4)

            return type_keys, "xdotool"
    if wayland and shutil.which("wtype"):
        # wtype sends keys through the compositor's virtual-keyboard protocol (sway supports it).

        def type_wayland(text: str) -> None:
            subprocess.run(["wtype", "-d", "20", text], timeout=30, check=False)
            time.sleep(0.4)

        return type_wayland, "wtype"
    return (lambda text: fixture.send(f"insert {text}")), "in-app insert"


STATE_NAMES = {
    L.STATE_ACTIVE: "active",
    L.STATE_DEFUNCT: "defunct",
    L.STATE_EDITABLE: "editable",
    11: "focusable",
    L.STATE_FOCUSED: "focused",
    L.STATE_SHOWING: "showing",
    30: "visible",
    L.STATE_READ_ONLY: "read-only",
}
events_seen: list = []


def count_events() -> None:
    """Record every focus event the reader receives, to tell "no event" from "event but not focused"."""
    client = L.SHARED.get()
    if client is None:
        return
    original = client._t.events

    def counted():
        found = original()
        events_seen.extend(found)
        return found

    client._t.events = counted


def describe_tree(limit: int = 40) -> None:
    """Print each application's windows and the fixture's widgets with their roles and states."""
    client = L.SHARED.get()
    if client is None:
        print(f"diagnosis: no AT-SPI client ({L.SHARED.error})")
        return
    print(f"diagnosis: focus events received so far: {len(events_seen)} {events_seen[-3:]}")
    try:
        apps = client._children((L.REGISTRY, L.ROOT_PATH))
    except Exception as exc:
        print(f"diagnosis: registry children failed: {exc}")
        return
    for app in apps:
        try:
            name = client._get(app, L.ACCESSIBLE, "Name")
        except Exception:
            name = "?"
        print(f"diagnosis: app {app[0]} {name!r} pid {client.pid(app[0])}")
        stack = [(child, 1) for child in client._children(app)]
        shown = 0
        while stack and shown < limit:
            ref, depth = stack.pop(0)
            shown += 1
            try:
                state = client._state(ref)
                names = sorted(STATE_NAMES.get(bit, str(bit)) for bit in state if bit in STATE_NAMES)
                print(f"diagnosis: {'  ' * depth}{client._role(ref)} {ref[1]} {names}")
                stack[:0] = [(child, depth + 1) for child in client._children(ref)]
            except Exception as exc:
                print(f"diagnosis: {'  ' * depth}{ref[1]} error {exc}")


def sway_state() -> None:
    """What sway says about seats and the focused window (Wayland only)."""
    import glob

    socks = glob.glob(os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "sway-ipc.*.sock"))
    if not socks or not shutil.which("swaymsg"):
        return
    env = dict(os.environ, SWAYSOCK=socks[0])
    for kind in ("get_seats", "get_tree"):
        done = subprocess.run(["swaymsg", "-t", kind, "-r"], capture_output=True, text=True, timeout=10, env=env)
        text = done.stdout
        if kind == "get_tree":
            import json

            def focused(node):
                if node.get("focused"):
                    return node
                for child in node.get("nodes", []) + node.get("floating_nodes", []):
                    hit = focused(child)
                    if hit:
                        return hit
                return None

            try:
                node = focused(json.loads(text)) or {}
                text = f"focused node: {node.get('type')} {node.get('app_id')!r} {node.get('name')!r}"
            except ValueError:
                pass
        else:
            text = " ".join(text.split())[:600]
        print(f"diagnosis: sway {kind}: {text}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wayland", action="store_true")
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    session = "wayland" if args.wayland else "x11"
    print(
        f"session={session} DISPLAY={os.environ.get('DISPLAY', '')} WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}"
    )

    fixture = Fixture(args.python)
    try:
        if not check(f"{session}/fixture-started", fixture.wait_for("ready"), "GTK window shown"):
            return failures
        type_keys, how = typer(args.wayland, fixture)
        print(f"typing with: {how}")

        if args.wayland and shutil.which("wtype"):
            # A headless compositor has no keyboard, so no window would get keyboard focus and GTK would mark nothing
            # focused (CI on 12 November: 0 focus events, seat0 with no capabilities). A virtual keyboard held open
            # for the whole run gives the seat one, as every real machine has.
            keeper = subprocess.Popen(["wtype", "-s", "600000"])
            atexit.register(keeper.kill)
            print("holding a virtual keyboard open for the run (wtype -s 600000)")
            time.sleep(1.0)
        count_events()
        first = read_until(lambda r: r.editable)
        status = L.LinuxFocusedTextReader().status()
        print(f"reader status: {status.as_dict()}; focus events: {len(events_seen)}")
        found = first is not None and first.editable
        if not found:
            describe_tree()
            sway_state()
        if not check(f"{session}/focused-entry-found", found, repr(first)):
            return failures

        type_keys("hello from ci")
        typed = read_until(lambda r: r.text == "hello from ci")
        check(f"{session}/reads-typed-text", typed is not None and typed.text == "hello from ci", repr(typed))
        check(f"{session}/kind-is-field", typed is not None and typed.kind == "field" and typed.pid > 0, repr(typed))

        fixture.send("title Draft - edited 1")
        type_keys(" more")
        later = read_until(lambda r: r.text.endswith("more"))
        check(
            f"{session}/identity-stable-across-title-change",
            same_field(typed, later),
            f"{typed.field if typed else None} -> {later.field if later else None}",
        )
        foreground = L.foreground_window()
        check(
            f"{session}/foreground-title-follows",
            foreground is not None and foreground[1] == "Draft - edited 1",
            repr(foreground),
        )
        check(
            f"{session}/foreground-app-name",
            foreground is not None and foreground[0].startswith("python"),
            repr(foreground),
        )

        fixture.send("focus password")
        type_keys("s3cret")
        secret = read_until(lambda r: r.password)
        check(
            f"{session}/password-refused",
            secret is not None and secret.password and secret.text == "" and not secret.editable,
            repr(secret),
        )

        fixture.send("focus entry")
        back = read_until(lambda r: r.editable and not r.password)
        check(f"{session}/same-field-after-refocus", same_field(later, back), repr(back))

        timings = []
        for _ in range(40):
            start = time.perf_counter()
            L.read_focused()
            timings.append((time.perf_counter() - start) * 1000)
        median = statistics.median(timings)
        check(f"{session}/read-cost", median < 50.0, f"median {median:.1f} ms, max {max(timings):.1f} ms over 40 reads")
    finally:
        fixture.close()
    return failures


if __name__ == "__main__":
    sys.exit(main())

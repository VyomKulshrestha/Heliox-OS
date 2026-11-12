"""A real run of the Linux focused-text reader against a GTK window, for the ``platform-readers`` CI job.

Expects a session bus, a display (Xvfb, or a Wayland compositor with ``--wayland``) and AT-SPI installed. Starts
``gtk_fixture.py`` with the system Python, types into its entry (xdotool on X11; from inside the app on Wayland,
where no tool may inject keys), and checks what the reader returns. Every check prints one ``CHECK`` line;
the exit code is the number of failed checks.

    python tests/platform_real/linux_atspi_check.py [--wayland] [--python /usr/bin/python3]
"""

from __future__ import annotations

import argparse
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

        first = read_until(lambda r: r.editable)
        status = L.LinuxFocusedTextReader().status()
        print(f"reader status: {status.as_dict()}")
        if not check(f"{session}/focused-entry-found", first is not None and first.editable, repr(first)):
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

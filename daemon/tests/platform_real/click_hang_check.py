"""Two real checks on a control and a window that do not cooperate, on macOS and on Linux (X11), for CI.

    python click_hang_check.py                          # macOS, in the platform-readers job
    python click_hang_check.py --python /usr/bin/python3  # Linux, in an X session with AT-SPI

The centre click: the fixture's "Click me" label offers no press action (no AXPress, no AT-SPI action), so a click
on its snapshot ref through ``Executor._exec_mouse_click`` must fall back to a real mouse click at its centre, and
the label's own mouse handler must count it.

The hung app: the fixture's main thread stops answering for 6 s. Reading the focused field during that time must
come back within the bound (``HUNG_READ_BOUND`` seconds per read: the reader's per-call timeouts, 0.25 s on macOS
and 0.5 s on Linux, times the calls one read makes, with room), without raising, so Live is never stalled by it.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import one_loop  # noqa: E402  (next to this script)
from press_checks import _click  # noqa: E402

HERE = Path(__file__).resolve().parent
HUNG_READ_BOUND = 3.0
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


class Fixture:
    def __init__(self, command: list[str]) -> None:
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.lines: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))

    def wait_line(self, prefix: str, timeout: float = 15.0, start: int = 0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in self.lines[start:]:
                if line.startswith(prefix):
                    return line
            time.sleep(0.1)
        return ""

    def send(self, command: str) -> int:
        assert self.proc.stdin is not None
        mark = len(self.lines)
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        return mark

    def clicks(self) -> int:
        line = self.wait_line("clicks ", 10.0, self.send("clicks"))
        return int(line.split()[1]) if line else -1


def reader():
    if sys.platform == "darwin":
        from pilot.live import focused_text_macos as M

        shared = M.shared()
        return lambda: shared.read() if shared else None
    from pilot.live import focused_text_linux as L

    return L.read_focused


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if sys.platform == "darwin":
        import mac_front

        mac_front.diag("at the start of click_hang_check")
        fixture = Fixture([sys.executable, str(HERE / "click_hang_fixture_macos.py")])
        prefix = "macos"
    else:
        fixture = Fixture([args.python, str(HERE / "click_hang_fixture_gtk.py")])
        prefix = "x11"
    try:
        if not check(f"{prefix}/click-fixture-started", bool(fixture.wait_line("ready"))):
            return failures
        time.sleep(1.0)
        from pilot.system import accessibility

        # The window must be in front first: a step before this one can leave another app there (CI on e51aead and
        # 105b573: "No window is in front to read"; on macOS the fixture asking for the front was ignored), so the
        # check brings it forward the way Heliox's window focus does (mac_front), and asks until the snapshot is of
        # its window, for up to 15 s.
        if sys.platform == "darwin":
            route = mac_front.bring_to_front(fixture.proc.pid)
            check("macos/click-fixture-in-front", bool(route), f"route {route or 'none'}")
        snapshot, target, deadline = None, None, time.monotonic() + 15.0
        while target is None and time.monotonic() < deadline:
            fixture.send("front")
            time.sleep(0.8)
            try:
                snapshot = one_loop.run(accessibility.capture_snapshot(max_elements=60))
                target = next((e for e in snapshot.elements if e.name == "Click me"), None)
            except Exception as exc:
                print(f"snapshot: {type(exc).__name__}: {exc}")
        if check(f"{prefix}/snapshot-finds-the-label", target is not None):
            before = fixture.clicks()
            try:
                output = _click(snapshot.snapshot_id, target.ref)
            except Exception as exc:
                output = f"{type(exc).__name__}: {exc}"
            deadline = time.monotonic() + 5.0
            after = fixture.clicks()
            while after < before + 1 and time.monotonic() < deadline:
                time.sleep(0.3)
                after = fixture.clicks()
            check(
                f"{prefix}/centre-click-reaches-a-control-without-a-press-action",
                after == before + 1 and "cursor did not move" not in output,
                f"clicks {before} -> {after}; {output}",
            )

        read = reader()
        if sys.platform == "darwin":
            fixture.send("insert hello from the hang check")
        else:
            subprocess.run(["xdotool", "type", "--delay", "20", "hello from the hang check"], timeout=30, check=False)
        time.sleep(1.0)
        seen = read()
        print(f"before the hang: {seen!r}")
        # Without the field read before the hang, a fast read during it proves nothing (CI on e51aead read None
        # throughout and passed in 0.0 s): the check needs the reader to be talking to the fixture first.
        if not check(
            f"{prefix}/hang-check-reads-the-field-first",
            seen is not None and "hello from the hang check" in (getattr(seen, "text", "") or ""),
            repr(seen),
        ):
            return failures
        fixture.wait_line("ok hang", 10.0, fixture.send("hang 6"))
        time.sleep(0.3)
        timings, errors = [], []
        for _ in range(3):
            start = time.perf_counter()
            try:
                read()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            timings.append(time.perf_counter() - start)
        check(
            f"{prefix}/read-of-a-hung-app-returns-within-{HUNG_READ_BOUND:.0f}s",
            not errors and max(timings) <= HUNG_READ_BOUND,
            f"reads took {[round(t, 2) for t in timings]} s; errors {errors}",
        )
    finally:
        time.sleep(6.0)  # let a hang still in progress end, so quit is heard
        if sys.platform == "darwin":
            import mac_front

            mac_front.end(fixture.proc)
            mac_front.diag("at the end of click_hang_check")
        else:
            fixture.send("quit")
            time.sleep(0.5)
            fixture.proc.kill()
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens a window on this desktop and clicks and types into it")
    sys.exit(main())

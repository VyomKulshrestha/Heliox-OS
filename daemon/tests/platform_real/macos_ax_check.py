"""A real run of the macOS readers and permission checks, for the ``platform-readers`` CI job.

Hosted macOS runners decide whether a process may use the Accessibility API; nothing here prompts. The script
reports what the runner allows and checks the furthest honest thing:

- always: pyobjc imports, every permission reads without a prompt, the frontmost app is named from the Quartz
  window list (no permission needed), and the reader's status is consistent with the Accessibility state;
- without Accessibility: reads return nothing quickly, the status says what to grant and where, and no prompt
  was asked;
- with Accessibility: an AppKit window with a text field and a secure text field (``macos_fixture.py``) is read:
  typed text, the secure field refused, identity kept across a window title change.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.live import focused_text_macos as M  # noqa: E402
from pilot.live.focused_text_policy import same_field  # noqa: E402
from pilot.system import macos_permissions as mp  # noqa: E402

HERE = Path(__file__).resolve().parent
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def main() -> int:
    try:
        import AppKit  # noqa: F401
        import ApplicationServices  # noqa: F401
        import Quartz  # noqa: F401

        check("macos/pyobjc-imports", True)
    except ImportError as exc:
        check("macos/pyobjc-imports", False, str(exc))
        return failures

    start = time.perf_counter()
    reports = mp.snapshot()
    took = (time.perf_counter() - start) * 1000
    for report in reports:
        print(f"permission {report.name}: {report.state.value} | {report.message}")
    check("macos/permissions-read-without-prompt", len(reports) == 4, f"{took:.0f} ms")

    owner = M.frontmost_app_name()
    check("macos/frontmost-app-from-window-list", bool(owner), repr(owner))

    reader = M.MacFocusedTextReader()
    status = reader.status()
    trusted = mp.accessibility_state() == mp.PermissionState.GRANTED
    print(f"accessibility trusted: {trusted}; reader status: {status.as_dict()}")
    shared = M.shared()
    if not trusted:
        check(
            "macos/status-asks-for-accessibility",
            status.state == "needs_permission" and "Accessibility" in status.message,
        )
        start = time.perf_counter()
        results = [shared.read() if shared else None for _ in range(20)]
        took = (time.perf_counter() - start) * 1000
        check("macos/no-reads-without-permission", all(r is None for r in results), f"20 reads in {took:.1f} ms")
        check("macos/never-prompted", shared is not None and not shared.gate.prompted)
        print("RESULT: this runner does not grant Accessibility to the job; field reading was checked with fakes only.")
        return failures

    fixture = subprocess.Popen(
        [sys.executable, str(HERE / "macos_fixture.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert fixture.stdin is not None and fixture.stdout is not None

    def send(command: str) -> None:
        fixture.stdin.write(command + "\n")
        fixture.stdin.flush()
        time.sleep(0.6)

    def read_until(predicate, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = shared.read() if shared else None
            if last is not None and predicate(last):
                return last
            time.sleep(0.3)
        return last

    try:
        check("macos/fixture-started", fixture.stdout.readline().startswith("ready"))
        send("insert hello from ci")
        typed = read_until(lambda r: r.text == "hello from ci")
        check("macos/reads-typed-text", typed is not None and typed.text == "hello from ci", repr(typed))
        send("title Draft - edited")
        send("insert  more")
        later = read_until(lambda r: r.text.endswith("more"))
        check("macos/identity-stable-across-title-change", same_field(typed, later), repr(later))
        print(f"foreground: {M.foreground_window()!r}")
        send("focus password")
        send("insert s3cret")
        secret = read_until(lambda r: r.password)
        check("macos/secure-field-refused", secret is not None and secret.password and secret.text == "", repr(secret))
        timings = []
        for _ in range(30):
            start = time.perf_counter()
            shared.read()
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        check("macos/read-cost", timings[len(timings) // 2] < 50.0, f"median {timings[len(timings) // 2]:.1f} ms")
    finally:
        send("quit")
        fixture.kill()
    return failures


if __name__ == "__main__":
    sys.exit(main())

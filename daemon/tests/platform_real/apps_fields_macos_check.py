"""Fields where people type on a Mac: Safari, Chrome and an Electron app, read by Heliox's macOS reader.

    python apps_fields_macos_check.py safari-default chrome-default chrome-manual chrome-enhanced chrome-forced \\
        edge-default edge-enhanced electron-default electron-manual electron-enhanced electron-forced

Each target shows field_fixture.html: a text area and a password input. For each:

- the window is brought to the front (mac_front), the text typed into the text area must be read back through the
  reader Live uses (``<target>/reads-typed-text``);
- only after that first read, every read for five seconds after typing into the password input must be refused,
  and none may carry the typed text (``<target>/password-refused``);
- DIAG lines dump the focused element and its two parents: AXRole, AXSubrole, AXRoleDescription, the label
  attributes and the names of all attributes (never AXValue).

The variants answer "does a person's default setup give Heliox the field":
- ``*-default``: the app as a person has it, read by Heliox's reader as it is (which turns AXManualAccessibility
  on in an Electron app, and AXEnhancedUserInterface in Chrome and Edge, itself). The read is an ANSWER line, yes or no; the password check still counts when the
  field was read;
- ``*-manual``: the check sets AXManualAccessibility on the app, the attribute Electron documents for assistive
  apps (Chrome rejected it in CI on 052f815, so for the browsers it is an ANSWER line);
- ``*-enhanced``: the check sets AXEnhancedUserInterface on the app, the attribute VoiceOver sets, and puts the
  value it found back afterwards. An ANSWER line;
- ``*-forced``: the app's own switch, ``--force-renderer-accessibility``. Manual and forced must read the field.

For Chrome and Electron an ANSWER line also says how the window takes a move through the Accessibility API (read
back at once and after a second), which AXEnhancedUserInterface is said to disturb.

Typing goes through pyautogui (CGEvents), which the runner's Accessibility grant allows. Everything the check
starts it ends: its Chrome and Electron processes, and Safari (asked to quit, as it was not running before).
CI-only (``ci_only.require_runner``).
Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent
TITLE = "Heliox field fixture"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Chromium browsers the check runs when the runner has them; Heliox wakes each with AXEnhancedUserInterface.
BROWSERS = {"chrome": CHROME, "edge": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"}
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def ax() -> Any:
    import ApplicationServices

    return ApplicationServices


def attr(element: Any, name: str) -> Any:
    err, value = ax().AXUIElementCopyAttributeValue(element, name, None)
    return value if err == 0 else None


def diag_focused(label: str) -> None:
    """DIAG: the focused element and its two parents, never a field's value."""
    a = ax()
    system = a.AXUIElementCreateSystemWide()
    element = attr(system, "AXFocusedUIElement")
    for level in range(3):
        if element is None:
            print(f"DIAG {label} level {level}: no element", flush=True)
            return
        err, names = a.AXUIElementCopyAttributeNames(element, None)
        facts = {
            name: str(attr(element, name))
            for name in ("AXRole", "AXSubrole", "AXRoleDescription", "AXTitle", "AXDescription", "AXPlaceholderValue")
        }
        facts["attributes"] = sorted(str(n) for n in (names or [])) if err == 0 else f"error {err}"
        print(f"DIAG {label} level {level}: {facts}", flush=True)
        element = attr(element, "AXParent")


def set_manual_accessibility(pid: int) -> str:
    """AXManualAccessibility on the app, as an assistive app turns on web accessibility in Chrome and Electron."""
    a = ax()
    app = a.AXUIElementCreateApplication(pid)
    err = a.AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
    return f"AXManualAccessibility set: error {err}" if err else "AXManualAccessibility set"


def app_bool(pid: int, name: str) -> bool | None:
    a = ax()
    err, value = a.AXUIElementCopyAttributeValue(a.AXUIElementCreateApplication(pid), name, None)
    return bool(value) if err == 0 and value is not None else None


def set_app_bool(pid: int, name: str, value: bool) -> int:
    a = ax()
    return int(a.AXUIElementSetAttributeValue(a.AXUIElementCreateApplication(pid), name, value))


def window_move(target: str, pid: int) -> None:
    """ANSWER: how the fixture's window takes a move through the Accessibility API, as window tools (and Heliox's own
    window actions) move it. AXEnhancedUserInterface is said to make apps animate such moves, so a tool reading the
    position back sees the old one; this is measured, with the app as it is and with the attribute on."""
    from pilot.system.macos_windows import Backend

    a = ax()
    backend = Backend()
    windows = [w for w in backend.windows(pid) if TITLE in str(backend.get(w, "AXTitle") or "")]
    if not windows:
        print(f"ANSWER {target}: window move not measured, no fixture window", flush=True)
        return
    window = windows[0]

    def position() -> tuple[int, int] | None:
        value = backend.get(window, "AXPosition")
        if value is None:
            return None
        ok, point = a.AXValueGetValue(value, a.kAXValueCGPointType, None)
        return (round(point.x), round(point.y)) if ok else None

    def move_to(x: int, y: int) -> int:
        import Quartz

        value = a.AXValueCreate(a.kAXValueCGPointType, Quartz.CGPoint(x, y))
        return int(a.AXUIElementSetAttributeValue(window, "AXPosition", value))

    start = position()
    if start is None:
        print(f"ANSWER {target}: window move not measured, its position cannot be read", flush=True)
        return
    wanted = (start[0] + 60, start[1] + 40)
    try:
        err = move_to(*wanted)
        at_once = position()
        time.sleep(1.0)
        later = position()
        print(
            f"ANSWER {target}: window move to {wanted} (error {err}): read back at once {at_once}, after 1 s {later}; "
            f"took effect at once: {'yes' if at_once == wanted else 'no'}",
            flush=True,
        )
    finally:
        move_to(*start)


def running_pid(bundle_id: str, timeout: float = 30.0) -> int:
    import AppKit

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        apps = AppKit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
        if apps and len(apps):
            return int(apps[0].processIdentifier())
        time.sleep(0.5)
    return 0


def window_titled(pid: int, timeout: float = 60.0) -> bool:
    from pilot.system.macos_windows import Backend

    backend = Backend()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(TITLE in str(backend.get(w, "AXTitle") or "") for w in backend.windows(pid)):
            return True
        time.sleep(0.5)
    return False


def type_text(text: str) -> None:
    import pyautogui

    pyautogui.write(text, interval=0.03)
    time.sleep(0.6)


def run(target: str, workdir: Path) -> None:
    import pyautogui
    from mac_front import bring_to_front, diag

    from pilot.live import focused_text_macos as M

    kind, _, variant = target.partition("-")
    page = workdir / "field_fixture.html"
    process: subprocess.Popen | None = None
    pid = 0
    enhanced_before: bool | None = None
    safari_was_running = running_pid("com.apple.Safari", timeout=0.1) != 0
    diag(f"at the start of {target}")
    try:
        if kind == "safari":
            subprocess.run(["/usr/bin/open", "-a", "Safari", str(page)], capture_output=True, timeout=30, check=False)
            pid = running_pid("com.apple.Safari")
        elif kind in BROWSERS:
            program = BROWSERS[kind]
            if not Path(program).exists():
                print(f"ANSWER {target}: not run, {Path(program).name} is not on this runner", flush=True)
                return
            args = [
                program,
                f"--user-data-dir={workdir / f'chrome-{target}'}",  # a profile per target: no handing off
                "--no-first-run",
                "--no-default-browser-check",
            ]
            if variant == "forced":
                args.append("--force-renderer-accessibility")
            process = subprocess.Popen([*args, page.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid = process.pid
        else:
            electron = os.environ.get("HELIOX_ELECTRON") or shutil.which("electron")
            if not electron:
                print(f"ANSWER {target}: not run, Electron is not installed here", flush=True)
                return
            args = [electron, str(HERE / "electron_fixture.js")]
            if variant == "forced":
                args.append("--force-renderer-accessibility")
            env = {**os.environ, "HELIOX_FIXTURE_PAGE": str(page)}
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            pid = 0
            deadline = time.monotonic() + 30
            while not pid and time.monotonic() < deadline:  # Electron's window belongs to its app process
                pid = next((p for p in _pids_named("Electron") if p), 0)
                time.sleep(0.5)
        if not check(f"{target}/window-shown", bool(pid) and window_titled(pid), f"pid {pid}"):
            return
        route = bring_to_front(pid)
        check(f"{target}/in-front", bool(route), f"route {route or 'none'}")
        if variant == "manual":
            print(f"DIAG {target}: {set_manual_accessibility(pid)}", flush=True)
        if variant == "enhanced":
            enhanced_before = app_bool(pid, "AXEnhancedUserInterface")  # written down, put back in finally
            err = set_app_bool(pid, "AXEnhancedUserInterface", True)
            print(f"DIAG {target}: AXEnhancedUserInterface was {enhanced_before}, set: error {err}", flush=True)
        if kind in (*BROWSERS, "electron"):
            window_move(target, pid)
        time.sleep(2.0)
        reader = M.shared()
        if not check(f"{target}/reader", reader is not None, M._error):
            return
        word = f"hello {target}"
        type_text(word)
        seen = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            seen = reader.read()
            if seen is not None and (seen.text or "").endswith(word):
                break
            time.sleep(0.3)
        ok = seen is not None and (seen.text or "").endswith(word)
        diag_focused(f"{target} text field")
        # Chrome refuses AXManualAccessibility, so its manual target is an answer, not a check.
        if variant in ("default", "enhanced") or (kind in BROWSERS and variant == "manual"):
            print(f"ANSWER {target}: Heliox reads the field with this setup: {'yes' if ok else 'no'}", flush=True)
            if kind != "safari":
                print(f"DIAG {target}: what the reader turned on: {reader.woken!r}", flush=True)
        else:
            check(f"{target}/reads-typed-text", ok, repr(seen))
        if not ok:
            return
        # Only after a first successful read: each password input must be refused, and nothing typed may leak. The
        # second has no label and a neutral id: the refusal must not depend on what the page calls the field.
        for name, secret in (("password-refused", "s3cret"), ("unnamed-password-refused", "n3utral")):
            pyautogui.press("tab")
            time.sleep(0.6)
            type_text(secret)
            diag_focused(f"{target} {name} field")
            # The reader refuses by either mark; which one the app gives is the answer to "refused by which mark".
            focused = attr(ax().AXUIElementCreateSystemWide(), "AXFocusedUIElement")
            subrole = str(attr(focused, "AXSubrole") or "") if focused is not None else ""
            print(
                f"ANSWER {target}: {name} field marks: subrole {subrole or 'none'}, "
                f"secure keyboard entry {'on' if M.secure_input_enabled() else 'off'}",
                flush=True,
            )
            reads = []
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                reads.append(reader.read())
                time.sleep(0.3)
            leaked = [r for r in reads if r is not None and secret in (r.text or "")]
            refused = [r for r in reads if r is not None and r.password]
            check(
                f"{target}/{name}",
                not leaked and bool(refused),
                f"{len(refused)} of {len(reads)} reads refused, leaked {leaked[:1]!r}",
            )
    finally:
        if variant == "enhanced" and pid:
            set_app_bool(pid, "AXEnhancedUserInterface", bool(enhanced_before))
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        if kind == "safari" and not safari_was_running:
            quit_app("com.apple.Safari")
        for leftover in _pids_named("Electron") if kind == "electron" else []:
            subprocess.run(["/bin/kill", str(leftover)], capture_output=True, check=False)
        time.sleep(1.5)
        diag(f"at the end of {target}")


def quit_app(bundle_id: str, timeout: float = 10.0) -> None:
    """Quit an app through NSRunningApplication, then force it. Not through osascript: CI on 7f553b9 timed out on
    'quit app "Safari"' after 15 s (sending Apple Events to Safari needs an Automation grant the runner lacks), and
    that stopped every target after Safari."""
    import AppKit

    def running() -> list:
        return list(AppKit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id) or [])

    for app in running():
        app.terminate()
    deadline = time.monotonic() + timeout
    while running() and time.monotonic() < deadline:
        time.sleep(0.5)
    for app in running():
        print(f"DIAG {bundle_id} did not quit when asked; forcing pid {app.processIdentifier()}", flush=True)
        app.forceTerminate()


def _pids_named(name: str) -> list[int]:
    from pilot.system.macos_windows import Backend

    return [app.pid for app in Backend().running_apps() if app.name == name]


def main(argv: list[str]) -> int:
    workdir = Path(tempfile.mkdtemp(prefix="heliox-fields-"))
    shutil.copyfile(HERE / "field_fixture.html", workdir / "field_fixture.html")
    try:
        for target in argv:
            try:
                run(target, workdir)
            except Exception as exc:  # one target failing must not stop the others
                check(f"{target}/ran", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens Safari, Chrome and Electron windows and types into them")
    sys.exit(main(sys.argv[1:]))

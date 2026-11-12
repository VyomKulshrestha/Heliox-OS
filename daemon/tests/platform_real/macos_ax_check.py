"""A real run of the macOS readers and permission checks, for the ``platform-readers`` CI job.

Hosted macOS runners decide whether a process may use the Accessibility API; nothing here prompts. The script
reports what the runner allows and checks the furthest honest thing:

- always: pyobjc imports, every permission reads without a prompt, and the reader's status is consistent with
  the Accessibility state; what the window list and LaunchServices say before any test window opens is printed;
- with a window of ours in front: the frontmost app is named without any permission, by LaunchServices and by
  the Quartz window list separately;
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

from pilot.live import focused_text  # noqa: E402
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


OCR_TEXT = "Heliox reads this line"


def rendered_text_png(text: str) -> bytes:
    """A PNG of ``text`` in black Helvetica on white, drawn with AppKit."""
    import AppKit

    width, height = 900, 160
    image = AppKit.NSImage.alloc().initWithSize_((width, height))
    image.lockFocus()
    AppKit.NSColor.whiteColor().set()
    AppKit.NSRectFill(((0, 0), (width, height)))
    attributes = {
        AppKit.NSFontAttributeName: AppKit.NSFont.fontWithName_size_("Helvetica", 56),
        AppKit.NSForegroundColorAttributeName: AppKit.NSColor.blackColor(),
    }
    AppKit.NSString.stringWithString_(text).drawAtPoint_withAttributes_((30, 50), attributes)
    image.unlockFocus()
    rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
    return bytes(rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {}))


def ocr_checks() -> None:
    import asyncio

    from pilot.system import vision
    from pilot.system.macos_vision_ocr import recognize_text

    png = rendered_text_png(OCR_TEXT)
    start = time.perf_counter()
    found = recognize_text(png)
    took = (time.perf_counter() - start) * 1000
    normal = " ".join(found.split()).casefold()
    check("macos/ocr-vision-in-process", OCR_TEXT.casefold() in normal, f"{found!r} in {took:.0f} ms")
    start = time.perf_counter()
    through = asyncio.run(vision._ocr_macos_native(png))
    took = (time.perf_counter() - start) * 1000
    check(
        "macos/ocr-through-vision-module", OCR_TEXT.casefold() in " ".join(through.split()).casefold(), f"{took:.0f} ms"
    )


def window_checks(pid: int) -> None:
    """Window actions through the Accessibility API on the fixture's window (the AppleScript route is not used)."""
    from pilot.system.macos_windows import MacWindows, match_apps

    windows = MacWindows()
    apps = [app for app in windows.b.running_apps() if app.pid == pid]
    if not check("macos/windows-fixture-is-a-regular-app", bool(apps), repr(apps)):
        return
    name = apps[0].name
    print(
        f"fixture app as macOS names it: {apps[0]!r}; matches by name: {len(match_apps(name, windows.b.running_apps()))}"
    )
    listing = windows.list()
    check(
        "macos/windows-list-has-fixture", any(title == "Draft - edited" for _app, title in listing), repr(listing[:10])
    )
    b = windows.b
    print(
        f"focus: front before: pid {b.frontmost_pid()} (fixture {pid}); LaunchServices {M.launch_services_front_pid()}"
    )
    windows.focus(name, "Draft")
    check("macos/windows-focus-already-front", b.frontmost_pid() == pid, f"front pid {b.frontmost_pid()}")
    finder = [app for app in b.running_apps() if app.bundle_id == "com.apple.finder"]
    if finder:
        accepted = b.activate(finder[0].pid)
        time.sleep(0.5)
        print(f"focus: activateWithOptions(Finder) returned {accepted}; front now pid {b.frontmost_pid()}")
        try:
            windows.focus("Finder")
            check(
                "macos/windows-focus-another-app",
                b.frontmost_pid() == finder[0].pid,
                f"front {b.frontmost_pid()}, route {windows.last_route}",
            )
        except Exception as exc:
            check("macos/windows-focus-another-app", False, f"{type(exc).__name__}: {exc}")
        accepted = b.activate(pid)
        time.sleep(0.5)
        print(f"focus: activateWithOptions(fixture) returned {accepted}; front now pid {b.frontmost_pid()}")
        try:
            windows.focus(name, "Draft")
            check(
                "macos/windows-focus-back-to-fixture",
                b.frontmost_pid() == pid,
                f"front {b.frontmost_pid()}, route {windows.last_route}",
            )
        except Exception as exc:
            check("macos/windows-focus-back-to-fixture", False, f"{type(exc).__name__}: {exc}")
    else:
        print("focus: Finder is not running; the switch-and-back check was skipped")
    windows.maximize(name, "Draft")
    time.sleep(0.5)
    window = windows._window(apps[0], "Draft")
    size = windows.b.get(window, "AXSize")
    frame = windows.b.visible_frame()
    print(f"after maximise: AXSize {size!r}, visible frame {frame!r}")
    check(
        "macos/windows-maximize",
        size is not None and "w:" + str(int(frame[2])) in str(size).replace(" ", ""),
        str(size),
    )
    windows.minimize(name, "Draft")
    time.sleep(0.8)
    check("macos/windows-minimize", bool(windows.b.get(window, "AXMinimized")))
    windows.b.set_bool(window, "AXMinimized", False)
    time.sleep(0.5)
    windows.close(name, "Draft")
    time.sleep(0.8)
    check("macos/windows-close-closes-the-window-only", not windows.exists(name, "Draft") and fixture_alive(pid))


def fixture_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def snapshot_checks(prefix: str, check) -> None:
    """Accessibility snapshot of the window in front: the button by name, the fields by label, never a value."""
    import asyncio

    from pilot.system import accessibility

    try:
        snapshot = asyncio.run(accessibility.capture_snapshot(max_elements=60))
    except Exception as exc:
        check(f"{prefix}/snapshot", False, f"{type(exc).__name__}: {exc}")
        return
    labels = [element.label for element in snapshot.elements]
    print(f"snapshot {snapshot.snapshot_id} of {snapshot.title!r}: {labels[:12]}")
    button = next((e for e in snapshot.elements if e.role == "Button" and e.name == "Press me"), None)
    check(f"{prefix}/snapshot-finds-the-button", button is not None, repr(labels[:12]))
    # Fields are named by their placeholder when they have no title or description; never by what is in them.
    check(
        f"{prefix}/snapshot-names-fields-by-placeholder",
        'Edit "message"' in labels and 'Edit "password"' in labels,
        repr(labels[:12]),
    )
    check(
        f"{prefix}/snapshot-never-has-values",
        not any("hello from ci" in label or "s3cret" in label for label in labels),
    )
    if button is not None:
        try:
            element = accessibility.default_store().resolve(snapshot.snapshot_id, button.ref)
            check(
                f"{prefix}/snapshot-ref-resolves-in-the-same-window",
                element.ref == button.ref,
                f"center {element.center}",
            )
        except Exception as exc:
            check(f"{prefix}/snapshot-ref-resolves-in-the-same-window", False, f"{type(exc).__name__}: {exc}")


def window_detail(field: str) -> str:
    """What the element and its window answer, for the window-gone check."""
    element = M._ELEMENTS.get(field)
    reader = M._reader
    if element is None or reader is None:
        return "element not remembered"
    try:
        return f"element answers {reader.backend.element_valid(element)!r}, window open {reader.backend.window_open(element)!r}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    try:
        import AppKit  # noqa: F401
        import ApplicationServices  # noqa: F401
        import Quartz  # noqa: F401

        check("macos/pyobjc-imports", True)
    except ImportError as exc:
        check("macos/pyobjc-imports", False, str(exc))
        return failures

    try:
        ocr_checks()
    except Exception as exc:
        check("macos/ocr-vision-in-process", False, f"{type(exc).__name__}: {exc}")

    start = time.perf_counter()
    reports = mp.snapshot()
    took = (time.perf_counter() - start) * 1000
    for report in reports:
        print(f"permission {report.name}: {report.state.value} | {report.message}")
    check("macos/permissions-read-without-prompt", len(reports) == 4, f"{took:.0f} ms")

    windows = M.window_list()
    normal = [(w.get("kCGWindowOwnerName"), w.get("kCGWindowAlpha")) for w in windows if w.get("kCGWindowLayer") == 0]
    print(f"before any test window: {len(windows)} on-screen windows, layer 0: {normal[:8]}")
    print(f"before any test window: LaunchServices front app {M.launch_services_front_app()!r}")

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

    def presses() -> int:
        """How many times the button's own action has run, or -1 when the fixture does not say."""
        fixture.stdin.write("presses\n")
        fixture.stdin.flush()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = fixture.stdout.readline()
            if not line:
                return -1
            if line.startswith("presses "):
                return int(line.split()[1])
        return -1

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
        time.sleep(1.0)
        front = M.launch_services_front_app()
        check("macos/frontmost-app-from-launch-services", front.casefold().startswith("python"), repr(front))
        owner = M.front_window_owner(M.window_list())
        check("macos/frontmost-app-from-window-list", owner.casefold().startswith("python"), repr(owner))
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
        snapshot_checks("macos", check)
        # The real press through AXPress, and nothing pressed or clicked once the window is a refused one.
        from press_checks import press_checks

        press_checks("macos", check, presses, lambda title: send(f"title {title}"), "Draft - edited")

        # Live drops a card when its field's window has gone; window_checks ends by closing the window.
        handle = focused_text.window_handle(later.field) if later is not None and later.field else None
        check("macos/window-exists-while-open", handle is not None and focused_text.window_exists(handle))
        try:
            window_checks(fixture.pid)
        except Exception as exc:
            check("macos/windows-actions", False, f"{type(exc).__name__}: {exc}")
        if handle is not None:
            time.sleep(0.5)
            check(
                "macos/window-gone-after-close",
                not focused_text.window_exists(handle),
                f"field_exists says {M.field_exists(later.field)!r}; {window_detail(later.field)}",
            )
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

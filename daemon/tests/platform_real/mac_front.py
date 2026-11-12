"""Bringing a check's own fixture window to the front on a macOS runner, and saying what is in front.

Since macOS 14, activation is cooperative: a process asking to come forward while another app is active can be
accepted and ignored. CI on 105b573: after the first macOS check, every later fixture stayed behind ("No window is
in front to read"), while the first check, run with nothing else active, had its window in front. This helper:

- ``diag(label)`` prints one DIAG line with what is in front: the frontmost app (pid, name, bundle id), its focused
  window's title, how many windows it has, and the on-screen windows front to back (owner, pid, title);
- ``bring_to_front(pid)`` tries the routes Heliox's own window focus uses (``pilot.system.macos_windows``), in its
  order: NSRunningApplication activation, then AXFrontmost with AXRaise and AXMain on the window, then
  LaunchServices (``open -b``) when the bundle id names that app alone, then System Events by unix id. After each it
  waits until the Accessibility API's focused application is ``pid`` and has a focused window, which is what the
  readers need. Returns the route that worked, or "" with a DIAG line when none did;
- ``end(process)`` ends a fixture process (quit, then kill), so nothing a check started stays in front of the next.

Checks call ``diag`` at their start and end.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any


def _ax() -> Any:
    import ApplicationServices as ax  # noqa: N813

    return ax


def focused_app_pid() -> int:
    ax = _ax()
    system = ax.AXUIElementCreateSystemWide()
    err, app = ax.AXUIElementCopyAttributeValue(system, "AXFocusedApplication", None)
    if err != 0 or app is None:
        return 0
    err, pid = ax.AXUIElementGetPid(app, None)
    return int(pid) if err == 0 else 0


def focused_window_title(pid: int) -> str | None:
    """The title of ``pid``'s focused window; None when it has no focused window."""
    ax = _ax()
    app = ax.AXUIElementCreateApplication(pid)
    err, window = ax.AXUIElementCopyAttributeValue(app, "AXFocusedWindow", None)
    if err != 0 or window is None:
        return None
    err, title = ax.AXUIElementCopyAttributeValue(window, "AXTitle", None)
    return str(title or "") if err == 0 else ""


def state() -> dict:
    import AppKit

    from pilot.live import focused_text_macos as M

    facts: dict[str, Any] = {}
    try:
        pid = focused_app_pid()
        running = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if pid else None
        facts["front"] = {
            "pid": pid,
            "name": str(running.localizedName() or "") if running is not None else "",
            "bundle": str(running.bundleIdentifier() or "") if running is not None else "",
            "focused_window": focused_window_title(pid) if pid else None,
        }
        from pilot.system.macos_windows import Backend

        facts["front"]["windows"] = len(Backend().windows(pid)) if pid else 0
    except Exception as exc:
        facts["front"] = f"{type(exc).__name__}: {exc}"
    try:
        facts["launch_services_front_pid"] = M.launch_services_front_pid()
    except Exception as exc:
        facts["launch_services_front_pid"] = f"{type(exc).__name__}: {exc}"
    try:
        windows = M.window_list()
        facts["on_screen"] = [
            (w.get("kCGWindowOwnerName"), w.get("kCGWindowOwnerPID"), w.get("kCGWindowName") or "")
            for w in windows
            if w.get("kCGWindowLayer") == 0
        ][:8]
    except Exception as exc:
        facts["on_screen"] = f"{type(exc).__name__}: {exc}"
    return facts


def diag(label: str) -> None:
    try:
        print(f"DIAG front {label}: {state()}", flush=True)
    except Exception as exc:
        print(f"DIAG front {label}: {type(exc).__name__}: {exc}", flush=True)


def _in_front(pid: int, settle: float) -> bool:
    deadline = time.monotonic() + settle
    while True:
        if focused_app_pid() == pid and focused_window_title(pid) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _ax_frontmost(pid: int) -> None:
    ax = _ax()
    app = ax.AXUIElementCreateApplication(pid)
    ax.AXUIElementSetAttributeValue(app, "AXFrontmost", True)
    err, windows = ax.AXUIElementCopyAttributeValue(app, "AXWindows", None)
    if err == 0 and windows:
        window = windows[0]
        ax.AXUIElementPerformAction(window, "AXRaise")
        ax.AXUIElementSetAttributeValue(window, "AXMain", True)
        ax.AXUIElementSetAttributeValue(window, "AXFocused", True)


def _launch_services(pid: int) -> None:
    import AppKit

    from pilot.system.macos_windows import Backend

    running = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    bundle = str(running.bundleIdentifier() or "") if running is not None else ""
    same = [app for app in Backend().running_apps() if app.bundle_id == bundle]
    if not bundle or len(same) != 1:
        raise RuntimeError(f"bundle id {bundle!r} is shared by {len(same)} running apps; open -b would start another")
    subprocess.run(["/usr/bin/open", "-b", bundle], capture_output=True, timeout=5.0, check=False)


def _system_events(pid: int) -> None:
    script = f'tell application "System Events" to set frontmost of (first process whose unix id is {int(pid)}) to true'
    subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, timeout=5.0, check=False)


def bring_to_front(pid: int, settle: float = 2.0) -> str:
    """Bring process ``pid`` (a fixture with a window) to the front; the route that worked, or ""."""
    if _in_front(pid, 0.3):
        return "already"
    from pilot.system.macos_windows import Backend

    routes = [
        ("activate", lambda: Backend().activate(pid)),
        ("ax-frontmost", lambda: _ax_frontmost(pid)),
        ("launch-services", lambda: _launch_services(pid)),
        ("system-events", lambda: _system_events(pid)),
    ]
    tried = []
    for name, route in routes:
        try:
            route()
        except Exception as exc:
            tried.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if _in_front(pid, settle):
            return name
        tried.append(f"{name}: no")
    print(f"DIAG front: pid {pid} did not come to the front; tried {tried}", flush=True)
    diag(f"after failing to bring pid {pid} forward")
    return ""


def end(process: subprocess.Popen, quit_command: str | None = "quit") -> None:
    """End a fixture process: ask it to quit, then kill it, and wait, so it stays in front of nothing."""
    try:
        if quit_command and process.stdin is not None and process.poll() is None:
            process.stdin.write(quit_command + "\n")
            process.stdin.flush()
            process.wait(timeout=3)
    except Exception:
        pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except Exception:
        pass

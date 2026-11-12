"""Window actions on macOS through the Accessibility API instead of AppleScript and System Events.

The AppleScript route needed two permissions (Automation of System Events, then Accessibility inside it), asked
for them at unpredictable moments, cost a process start per action, and its "close" quit the whole application.
Here, in the daemon's process through pyobjc:

- list: every regular application's windows (``AXWindows``) with their titles;
- focus: ``NSRunningApplication.activateWithOptions_`` (needs no permission), then ``AXRaise`` on the window
  whose title matches, when one was named;
- minimise: ``AXMinimized`` set to true; maximise: the window's position and size set to the screen's visible
  frame (the area without the menu bar and Dock);
- close: the window's close button (``AXCloseButton``) is pressed, which closes that window only and lets the
  app ask about unsaved work, as a click would;
- exists: whether the application still has a window (with that title).

Everything except focus needs the Accessibility permission that Live's reader already uses. ``window_mgr`` keeps
the AppleScript route for when pyobjc is not installed. Matching an application name and converting the
screen's coordinates are pure functions tested on every system.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class AccessibilityNeeded(RuntimeError):
    """The action needs the Accessibility permission, which is not granted."""


@dataclass(frozen=True)
class RunningApp:
    pid: int
    name: str  # localizedName
    executable: str
    bundle_id: str


def match_apps(target: str, apps: Iterable[RunningApp]) -> list[RunningApp]:
    """Applications whose shown name, executable or bundle id is ``target`` (case-insensitive, ".app" ignored)."""
    wanted = target.strip().casefold().removesuffix(".app")
    if not wanted:
        return []
    return [app for app in apps if wanted in (app.name.casefold(), app.executable.casefold(), app.bundle_id.casefold())]


def title_matches(title: str, wanted: str | None) -> bool:
    return not wanted or wanted.casefold() in (title or "").casefold()


def ax_frame(visible: tuple[float, float, float, float], primary_height: float) -> tuple[float, float, float, float]:
    """A screen rectangle from AppKit (origin bottom-left of the primary screen) in Accessibility coordinates
    (origin top-left of the primary screen): (x, y, width, height)."""
    x, y, width, height = visible
    return (x, primary_height - (y + height), width, height)


class Backend:
    """The pyobjc calls. Raises ImportError when pyobjc is not installed."""

    def __init__(self) -> None:
        import AppKit
        import ApplicationServices as ax  # noqa: N813

        self.ax = ax
        self.appkit = AppKit

    def trusted(self) -> bool:
        return bool(self.ax.AXIsProcessTrusted())

    def running_apps(self) -> list[RunningApp]:
        import psutil

        apps = []
        for proc in psutil.process_iter(["pid"]):
            running = self.appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(proc.info["pid"])
            if running is None or running.activationPolicy() != self.appkit.NSApplicationActivationPolicyRegular:
                continue
            url = running.executableURL()
            apps.append(
                RunningApp(
                    pid=int(running.processIdentifier()),
                    name=str(running.localizedName() or ""),
                    executable=str(url.lastPathComponent()) if url is not None else "",
                    bundle_id=str(running.bundleIdentifier() or ""),
                )
            )
        return apps

    def activate(self, pid: int) -> bool:
        running = self.appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return bool(
            running is not None and running.activateWithOptions_(self.appkit.NSApplicationActivateIgnoringOtherApps)
        )

    def windows(self, pid: int) -> list[Any]:
        app = self.ax.AXUIElementCreateApplication(pid)
        self.ax.AXUIElementSetMessagingTimeout(app, 0.5)
        err, windows = self.ax.AXUIElementCopyAttributeValue(app, "AXWindows", None)
        return list(windows or []) if err == 0 else []

    def get(self, element: Any, name: str) -> Any:
        err, value = self.ax.AXUIElementCopyAttributeValue(element, name, None)
        return value if err == 0 else None

    def set_bool(self, element: Any, name: str, value: bool) -> bool:
        return self.ax.AXUIElementSetAttributeValue(element, name, value) == 0

    def press(self, element: Any) -> bool:
        return self.ax.AXUIElementPerformAction(element, "AXPress") == 0

    def raise_window(self, window: Any) -> bool:
        return self.ax.AXUIElementPerformAction(window, "AXRaise") == 0

    def visible_frame(self) -> tuple[float, float, float, float]:
        screens = self.appkit.NSScreen.screens()
        main = self.appkit.NSScreen.mainScreen() or screens[0]
        frame = main.visibleFrame()
        primary = screens[0].frame()
        return ax_frame((frame.origin.x, frame.origin.y, frame.size.width, frame.size.height), primary.size.height)

    def set_frame(self, window: Any, frame: tuple[float, float, float, float]) -> bool:
        import Quartz

        x, y, width, height = frame
        position = self.ax.AXValueCreate(self.ax.kAXValueCGPointType, Quartz.CGPoint(x, y))
        size = self.ax.AXValueCreate(self.ax.kAXValueCGSizeType, Quartz.CGSize(width, height))
        moved = self.ax.AXUIElementSetAttributeValue(window, "AXPosition", position) == 0
        sized = self.ax.AXUIElementSetAttributeValue(window, "AXSize", size) == 0
        return moved and sized


class MacWindows:
    def __init__(self, backend: Any | None = None) -> None:
        self.b = backend or Backend()

    def _app(self, target: str) -> RunningApp:
        found = match_apps(target, self.b.running_apps())
        if not found:
            raise RuntimeError(f"No running application is called {target!r}.")
        return found[0]

    def _need_access(self) -> None:
        if not self.b.trusted():
            raise AccessibilityNeeded(
                "Heliox needs Accessibility access for window actions: System Settings > Privacy & Security > "
                "Accessibility, turn on Heliox-OS."
            )

    def _window(self, app: RunningApp, title: str | None) -> Any:
        for window in self.b.windows(app.pid):
            if title_matches(str(self.b.get(window, "AXTitle") or ""), title):
                return window
        raise RuntimeError(f"{app.name} has no window" + (f" titled {title!r}." if title else "."))

    def owner_of(self, title: str) -> str:
        """The application that has a window whose title contains ``title``."""
        self._need_access()
        for app in self.b.running_apps():
            for window in self.b.windows(app.pid):
                if title_matches(str(self.b.get(window, "AXTitle") or ""), title):
                    return app.name
        raise RuntimeError(f"No window titled {title!r} is open.")

    def list(self) -> list[tuple[str, str]]:
        self._need_access()
        rows = []
        for app in self.b.running_apps():
            for window in self.b.windows(app.pid):
                rows.append((app.name, str(self.b.get(window, "AXTitle") or "")))
        return rows

    def focus(self, target: str, title: str | None = None) -> str:
        app = self._app(target)
        if not self.b.activate(app.pid):
            raise RuntimeError(f"macOS did not bring {app.name} to the front.")
        if title:
            self._need_access()
            self.b.raise_window(self._window(app, title))
        return app.name

    def minimize(self, target: str, title: str | None = None) -> str:
        self._need_access()
        app = self._app(target)
        if not self.b.set_bool(self._window(app, title), "AXMinimized", True):
            raise RuntimeError(f"{app.name}'s window could not be minimised.")
        return app.name

    def maximize(self, target: str, title: str | None = None) -> str:
        self._need_access()
        app = self._app(target)
        if not self.b.set_frame(self._window(app, title), self.b.visible_frame()):
            raise RuntimeError(f"{app.name}'s window could not be resized.")
        return app.name

    def close(self, target: str, title: str | None = None) -> str:
        self._need_access()
        app = self._app(target)
        button = self.b.get(self._window(app, title), "AXCloseButton")
        if button is None or not self.b.press(button):
            raise RuntimeError(f"{app.name}'s window has no close button Heliox can press.")
        return app.name

    def exists(self, target: str, title: str | None = None) -> bool:
        self._need_access()
        found = match_apps(target, self.b.running_apps())
        return any(
            title_matches(str(self.b.get(window, "AXTitle") or ""), title)
            for app in found
            for window in self.b.windows(app.pid)
        )


def available() -> MacWindows | None:
    """The Accessibility route, or None when pyobjc is not installed (window_mgr then uses AppleScript)."""
    try:
        return MacWindows()
    except ImportError:
        return None

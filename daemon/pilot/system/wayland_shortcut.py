"""Heliox's global shortcut on Wayland, through the desktop's GlobalShortcuts portal.

On Wayland an app cannot grab keys for itself, so the desktop app's own shortcut (Tauri's global-shortcut plugin,
X11 only) does nothing there. When the daemon runs in a Wayland session it binds "toggle" through the portal once:
the desktop shows its own dialog the first time, where the person confirms or changes the keys, and remembers the
choice. Each activation is sent to the app as a ``global_shortcut`` notification, and the app shows or hides its
window. Where the desktop has no GlobalShortcuts portal (GNOME before 48, sway, Xfce), the reason is kept in
``status`` for Settings, and nothing else happens.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from pilot.system import portal

logger = logging.getLogger("pilot.system.wayland_shortcut")

DESCRIPTION = "Show or hide Heliox"
DEFAULT_TRIGGER = "CTRL+space"


class WaylandShortcut:
    def __init__(self, notify: Callable[[str, dict], None], trigger: str = DEFAULT_TRIGGER) -> None:
        self._notify = notify
        self._trigger = trigger
        self._stop = threading.Event()
        self.status = "starting"
        self.message = ""
        self.keys = ""
        self._thread = threading.Thread(target=self._run, name="wayland-shortcut", daemon=True)

    def start(self) -> WaylandShortcut:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            session = portal.bind_shortcuts({"toggle": (DESCRIPTION, self._trigger)})
        except portal.PortalError as error:
            self.status, self.message = "unavailable", str(error)
            logger.info("Wayland global shortcut unavailable: %s", error)
            return
        except Exception as error:  # no session bus, no jeepney
            self.status, self.message = "unavailable", f"The desktop's portal could not be reached: {error}"
            return
        self.status, self.keys = "ready", session.shortcuts.get("toggle", "")
        self._notify("global_shortcut_status", {"status": self.status, "keys": self.keys})
        try:
            while not self._stop.is_set():
                if session.wait(1.0) == "toggle":
                    self._notify("global_shortcut", {"id": "toggle"})
        finally:
            session.close()


def start_if_wayland(notify: Callable[[str, dict], None]) -> WaylandShortcut | None:
    """Start the portal shortcut in a Wayland session on Linux; None everywhere else."""
    import sys

    if not sys.platform.startswith("linux") or not portal.wayland_session():
        return None
    return WaylandShortcut(notify).start()

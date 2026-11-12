"""XDG desktop portals on Linux (Wayland): taking a screenshot and global shortcuts, over D-Bus with jeepney.

Under Wayland an application may not read the screen or grab keys on its own; it asks the desktop through
``org.freedesktop.portal.Desktop`` on the session bus, and the desktop asks the person.

- Screenshot (``org.freedesktop.portal.Screenshot``): one image of the screen as a file URI. GNOME and KDE show a
  consent dialog the first time (GNOME 46 and later can remember it); wlroots desktops (sway, Hyprland through
  xdg-desktop-portal-wlr or -hyprland) answer without a dialog unless configured to ask.
- Global shortcuts (``org.freedesktop.portal.GlobalShortcuts``, portal version 1 and later): a session in which
  Heliox binds "toggle" with a preferred trigger; the desktop shows its own dialog where the person confirms or
  changes the keys, and ``Activated`` signals arrive while the session lives. Implemented by GNOME 48 and later,
  KDE Plasma 5.27 and later and Hyprland; not by xdg-desktop-portal-wlr (sway), GNOME before 48 or Xfce.

Portal calls return a request object path; the answer is a ``Response`` signal on it (0 success, 1 cancelled by
the person, 2 other failure). The request path is predictable from the caller's unique name and a handle token,
so the signal subscription is made before the call, as the portal documentation asks. The decisions (paths,
response codes, the person-facing message) are pure and tested on every system; ``Transport`` is the D-Bus part,
faked in tests and run for real under sway in CI.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
REQUEST = "org.freedesktop.portal.Request"
SCREENSHOT = "org.freedesktop.portal.Screenshot"
SHORTCUTS = "org.freedesktop.portal.GlobalShortcuts"

RESPONSES = {0: "ok", 1: "cancelled", 2: "failed"}


class PortalError(RuntimeError):
    """The desktop has no such portal, the person said no, or the call failed. The message says which."""


def wayland_session(env: dict[str, str] | None = None) -> bool:
    env = dict(os.environ) if env is None else env
    return bool(env.get("WAYLAND_DISPLAY")) or env.get("XDG_SESSION_TYPE", "").casefold() == "wayland"


def token() -> str:
    return "heliox_" + secrets.token_hex(6)


def request_path(unique_name: str, handle_token: str) -> str:
    """The object path the portal will use for a request: /org/freedesktop/portal/desktop/request/SENDER/TOKEN,
    SENDER being the caller's unique name without its leading ':' and with '.' replaced by '_'."""
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"{PORTAL_PATH}/request/{sender}/{handle_token}"


def session_path(unique_name: str, handle_token: str) -> str:
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"{PORTAL_PATH}/session/{sender}/{handle_token}"


def explain(code: int, what: str) -> str:
    if code == 1:
        return f"The desktop asked whether Heliox may {what}, and the answer was no."
    return f"The desktop could not let Heliox {what} (portal response {code})."


def file_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise PortalError(f"The desktop returned a screenshot Heliox cannot open: {uri!r}")
    return unquote(parsed.path)


# xdg-desktop-portal (1.18) exports its Screenshot front end only when some backend also implements
# org.freedesktop.impl.portal.Access, the consent dialog; xdg-desktop-portal-wlr and -hyprland implement Screenshot
# but not Access, so on sway or Hyprland with only them installed there is no Screenshot portal at all.
ACCESS_HINT = (
    " On sway, Hyprland and other wlroots desktops also install xdg-desktop-portal-gtk: it provides the consent "
    "dialog the screenshot portal needs, and without it the screenshot portal is not offered."
)


def no_portal(what: str, error: str) -> str:
    """What to say when the portal itself is missing."""
    if "UnknownMethod" in error or "UnknownInterface" in error or "ServiceUnknown" in error:
        hint = ACCESS_HINT if "screenshot" in what else ""
        return (
            f"This desktop does not offer the portal Heliox needs to {what}. Install xdg-desktop-portal and the one "
            f"for your desktop (xdg-desktop-portal-gnome, -kde, -wlr or -hyprland), then log in again.{hint}"
        )
    return f"Heliox could not ask the desktop to {what}: {error}"


def missing_access_backend(names: list[str], interfaces: list[str]) -> bool:
    """Whether the screenshot portal is absent only for want of an Access backend: a backend that can take
    screenshots is on the bus, but the front end exports no Screenshot. Pure (for diagnosis)."""
    screenshot_backends = ("impl.portal.desktop.wlr", "impl.portal.desktop.hyprland")
    has_backend = any(any(b in n for b in screenshot_backends) for n in names)
    return has_backend and SCREENSHOT not in interfaces


class Transport:
    """What the portal client needs from the session bus."""

    unique_name = ""

    def call(self, interface: str, method: str, signature: str, body: tuple, path: str = PORTAL_PATH) -> tuple:
        raise NotImplementedError

    def watch(self, interface: str, member: str, path: str) -> None:
        """Subscribe to a signal before the call that will cause it."""
        raise NotImplementedError

    def next_signal(self, interface: str, member: str, path: str | None, timeout: float) -> tuple | None:
        """The body of the next matching signal, or None after ``timeout`` seconds."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class JeepneyTransport(Transport):
    def __init__(self) -> None:
        from collections import deque

        from jeepney.io.blocking import open_dbus_connection

        from pilot.live.focused_text_linux import _session_bus

        self._conn = open_dbus_connection(_session_bus())
        self.unique_name = self._conn.unique_name
        self._queue: deque = deque(maxlen=256)
        self._filters: list[Any] = []

    def call(self, interface: str, method: str, signature: str, body: tuple, path: str = PORTAL_PATH) -> tuple:
        from jeepney import DBusAddress, new_method_call
        from jeepney.wrappers import DBusErrorResponse, unwrap_msg

        message = new_method_call(DBusAddress(path, bus_name=PORTAL, interface=interface), method, signature, body)
        try:
            return tuple(unwrap_msg(self._conn.send_and_get_reply(message, timeout=10.0)))
        except DBusErrorResponse as error:
            raise PortalError(f"{error.name}: {error.data}") from error

    def watch(self, interface: str, member: str, path: str) -> None:
        from jeepney import MatchRule, message_bus

        rule = MatchRule(type="signal", interface=interface, member=member, path=path)
        self._filters.append(self._conn.filter(rule, queue=self._queue))
        self._conn.send_and_get_reply(message_bus.AddMatch(rule), timeout=5.0)

    def next_signal(self, interface: str, member: str, path: str | None, timeout: float) -> tuple | None:
        from jeepney import HeaderFields

        deadline = time.monotonic() + timeout
        while True:
            for message in list(self._queue):
                fields = message.header.fields
                if fields.get(HeaderFields.member) == member and (
                    path is None or fields.get(HeaderFields.path) == path
                ):
                    self._queue.remove(message)
                    return tuple(message.body)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                self._conn.recv_messages(timeout=remaining)
            except TimeoutError:
                return None

    def close(self) -> None:
        for handle in self._filters:
            handle.close()
        self._conn.close()


def _request(
    t: Transport, interface: str, method: str, signature: str, body: tuple, handle: str, what: str, timeout: float
) -> dict:
    path = request_path(t.unique_name, handle)
    t.watch(REQUEST, "Response", path)
    try:
        t.call(interface, method, signature, body)
    except PortalError as error:
        raise PortalError(no_portal(what, str(error))) from error
    answer = t.next_signal(REQUEST, "Response", path, timeout)
    if answer is None:
        raise PortalError(f"The desktop did not answer in {timeout:.0f} s when Heliox asked to {what}.")
    code, results = int(answer[0]), answer[1] if len(answer) > 1 else {}
    if code != 0:
        raise PortalError(explain(code, what))
    return {key: value[1] if isinstance(value, tuple) and len(value) == 2 else value for key, value in results.items()}


def screenshot(transport: Transport | None = None, *, interactive: bool = False, timeout: float = 120.0) -> str:
    """Ask the desktop for a screenshot; returns the image file's path. The desktop may show a consent dialog,
    which is why the timeout is long."""
    t = transport or JeepneyTransport()
    try:
        handle = token()
        options = {"handle_token": ("s", handle), "interactive": ("b", interactive)}
        results = _request(t, SCREENSHOT, "Screenshot", "sa{sv}", ("", options), handle, "take a screenshot", timeout)
        return file_from_uri(str(results.get("uri", "")))
    finally:
        if transport is None:
            t.close()


@dataclass
class ShortcutSession:
    transport: Transport
    session: str
    shortcuts: dict[str, str]  # id -> the trigger the desktop chose ("" when it does not say)

    def wait(self, timeout: float) -> str | None:
        """The id of the next activated shortcut, or None after ``timeout`` seconds."""
        body = self.transport.next_signal(SHORTCUTS, "Activated", None, timeout)
        if body is None or str(body[0]) != self.session:
            return None
        return str(body[1])

    def close(self) -> None:
        try:
            self.transport.call("org.freedesktop.portal.Session", "Close", "", (), path=self.session)
        except PortalError:
            pass


def bind_shortcuts(
    wanted: dict[str, tuple[str, str]],
    transport: Transport | None = None,
    *,
    timeout: float = 300.0,
    on_dialog: Callable[[], None] | None = None,
) -> ShortcutSession:
    """Create a GlobalShortcuts session and bind ``wanted`` ({id: (description, preferred trigger)}).

    The desktop shows its own dialog in which the person confirms or changes each trigger; the timeout covers
    the time they take. ``preferred_trigger`` uses the shortcuts XDG specification ("CTRL+space").
    """
    t = transport or JeepneyTransport()
    session_handle = token()
    handle = token()
    created = _request(
        t,
        SHORTCUTS,
        "CreateSession",
        "a{sv}",
        ({"handle_token": ("s", handle), "session_handle_token": ("s", session_handle)},),
        handle,
        "use a global shortcut",
        30.0,
    )
    session = str(created.get("session_handle") or session_path(t.unique_name, session_handle))
    t.watch(SHORTCUTS, "Activated", PORTAL_PATH)
    shortcuts = [
        (key, {"description": ("s", description), "preferred_trigger": ("s", trigger)})
        for key, (description, trigger) in wanted.items()
    ]
    handle = token()
    if on_dialog:
        on_dialog()
    bound = _request(
        t,
        SHORTCUTS,
        "BindShortcuts",
        "oa(sa{sv})sa{sv}",
        (session, shortcuts, "", {"handle_token": ("s", handle)}),
        handle,
        "use a global shortcut",
        timeout,
    )
    chosen: dict[str, str] = {}
    for key, props in bound.get("shortcuts", []) or []:
        trigger = props.get("trigger_description", ("s", ""))
        chosen[str(key)] = str(trigger[1] if isinstance(trigger, tuple) else trigger)
    return ShortcutSession(t, session, chosen)


def tauri_to_xdg_trigger(shortcut: str) -> str:
    """A Tauri shortcut ("Ctrl+Space", "Alt+Shift+H") in the shortcuts XDG specification's form
    ("CTRL+space", "ALT+SHIFT+h"), for ``preferred_trigger``."""
    names = {
        "ctrl": "CTRL",
        "control": "CTRL",
        "alt": "ALT",
        "option": "ALT",
        "shift": "SHIFT",
        "super": "LOGO",
        "cmd": "LOGO",
        "command": "LOGO",
        "meta": "LOGO",
        "win": "LOGO",
    }
    parts = [p.strip() for p in shortcut.split("+") if p.strip()]
    out = [names.get(p.casefold(), "") for p in parts[:-1]]
    key = parts[-1] if parts else ""
    key = "space" if key.casefold() == "space" else key.lower() if len(key) == 1 else key
    return "+".join([m for m in out if m] + ([key] if key else []))


_screenshot_refused = ""


def capture_png(region: tuple[int, int, int, int] | None = None) -> bytes | None:
    """The screen (or a region of it) as PNG bytes through the Screenshot portal, or None when the portal is not
    there or the person said no. After a "no", the portal is not asked again during this run, so a periodic
    reader cannot bring the desktop's dialog back again and again. Blocking: call from a worker thread."""
    global _screenshot_refused
    if _screenshot_refused:
        return None
    try:
        path = screenshot()
    except PortalError as error:
        if "answer was no" in str(error):
            _screenshot_refused = str(error)
        import logging

        logging.getLogger("pilot.system.portal").warning("Wayland screenshot unavailable: %s", error)
        return None
    except Exception:
        return None
    try:
        data = open(path, "rb").read()  # noqa: SIM115
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if region is None:
        return data
    try:
        from io import BytesIO

        from PIL import Image

        left, top, width, height = region
        image = Image.open(BytesIO(data)).crop((left, top, left + width, top + height))
        out = BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return data

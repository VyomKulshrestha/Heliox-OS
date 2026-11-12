"""Read the focused text field on Linux through AT-SPI, the desktop accessibility bus, over D-Bus.

AT-SPI is a D-Bus protocol, so it works the same under X11 and Wayland: what matters is whether the application
exposes itself on the accessibility bus. GTK 3 and 4 applications always do. Qt, Chromium, Electron and Firefox do
when accessibility is switched on for the session (GNOME: ``gsettings set org.gnome.desktop.interface
toolkit-accessibility true``, which sets org.a11y.Status IsEnabled; the app reads it when it starts). In CI on
7f553b9 ``--force-renderer-accessibility`` alone did not put Chrome or Electron on the bus. Terminals
built on VTE (GNOME Terminal, Tilix, Terminator, Xfce Terminal) expose their text with the "terminal" role and
are read like the Windows console; Konsole exposes some of it; GPU terminals (Alacritty, kitty, WezTerm, foot)
and xterm expose none, so nothing is read there.

The connection uses jeepney, a pure-Python D-Bus library (optional, ``pilot-daemon[linux]``): no compiler and no
GObject introspection. Focus is followed by subscribing to ``object:state-changed:focused`` events, so a read is
four or five D-Bus round trips to the focused application (about 1 to 5 ms). When no event has arrived yet (the
reader just started, or the application sends none), the focused element is looked for at most every few
seconds: first by asking each active window's application for it in one call (``Collection.GetMatches`` with the
FOCUSED state, which the app answers from its own tree), then by a bounded walk of the windows for toolkits that do
not answer that call.

Field identity is the application's bus name plus the object path of the element: it names the element itself,
so it does not change while the person types or when the window title changes. Every element goes through
``focused_text_policy.admit``, the same decision as on Windows and macOS; a "password text" element is refused
before its text is asked for.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot.live.focused_text_policy import (
    FIELD,
    MAX_CHARS,
    TERMINAL,
    ElementFacts,
    FocusedText,
    ReaderStatus,
    admit,
)

logger = logging.getLogger("pilot.live.focused_text_linux")

REGISTRY = "org.a11y.atspi.Registry"
ROOT_PATH = "/org/a11y/atspi/accessible/root"
ACCESSIBLE = "org.a11y.atspi.Accessible"
TEXT = "org.a11y.atspi.Text"
PROPS = "org.freedesktop.DBus.Properties"
COLLECTION = "org.a11y.atspi.Collection"
FOCUS_EVENT = "object:state-changed:focused"

# AtspiStateType bit numbers (at-spi2-core atspi-constants.h).
STATE_ACTIVE = 1
STATE_DEFUNCT = 6
STATE_EDITABLE = 7
STATE_FOCUSED = 12
STATE_SHOWING = 25
STATE_MANAGES_DESCENDANTS = 31
STATE_READ_ONLY = 43

PASSWORD_ROLES = frozenset({"password text"})
# AtspiRole ATSPI_ROLE_PASSWORD_TEXT (at-spi2-core atspi-constants.h). GTK 4 names its roles after ARIA in
# GetRoleName, so a password entry's role name is "text box" like any entry's, and marks the password only in the
# numeric role (GetRole). CI on 008f1a0 read a GTK 4 password entry as a "text box"; the dump from CI on 8d513fd
# shows the mark on the focused object itself (role name "text box", numeric role 40), its parent a plain
# "generic" (39). The numeric role is asked of the field and, for toolkits or versions that mark only an
# enclosing entry, of its two parents too.
ROLE_PASSWORD_TEXT = 40
PASSWORD_PARENT_DEPTH = 2
TERMINAL_ROLES = frozenset({"terminal"})
WINDOW_ROLES = frozenset({"frame", "window", "dialog", "alert", "file chooser"})
# The same by numeric role (ATSPI_ROLE_ALERT, DIALOG, FILE_CHOOSER, FRAME, WINDOW). GTK 4 names its window
# "application" in GetRoleName (CI on 8d513fd: the window holding the focused entry, titled and active, had role name
# "application" and numeric role 23, a frame), so a window is recognised by either.
WINDOW_ROLE_NUMBERS = frozenset({2, 16, 19, 23, 69})
# Roles that are never somewhere the person writes, even if a toolkit marks them editable.
NOT_WRITING_ROLES = frozenset({"label", "link", "push button", "menu item", "list item", "heading", "static", "image"})

CALL_TIMEOUT = 0.5
SEARCH_BUDGET = 250  # accessible objects visited by one search for the focused element
# Collection.GetMatches(rule, sortby, count, traverse) -> a(so) (at-spi2-core xml/Collection.xml). The rule is
# (states as 32-bit words, match type, attributes, match type, roles, match type, interfaces, match type, invert);
# MATCH_ALL with an empty set matches everything, so only the FOCUSED state narrows it.
MATCH_SIGNATURE = "(aiia{ss}iaiiasib)uib"
MATCH_ALL = 1  # ATSPI_Collection_MATCH_ALL
SORT_CANONICAL = 1  # ATSPI_Collection_SORT_ORDER_CANONICAL
FOCUSED_STATE_BITS = [1 << 12, 0]  # STATE_FOCUSED in the first of two 32-bit words
SEARCH_EVERY_SECONDS = 3.0
RESTART_BACKOFF_SECONDS = (2.0, 5.0, 15.0, 60.0)

TOOLKIT_HINT = (
    "GTK apps can be read. Qt, Chromium, Electron and Firefox apps share their text only when accessibility is on: "
    "run 'gsettings set org.gnome.desktop.interface toolkit-accessibility true' and restart those apps."
)
NO_BUS = (
    "The desktop accessibility bus (AT-SPI) is not running, so Heliox cannot see the field you are typing in. "
    "Install at-spi2-core (most desktops include it) and log in again."
)
NO_BINDING = "Reading the focused field on Linux needs the jeepney package: pip install 'pilot-daemon[linux]'."

Ref = tuple[str, str]  # (bus name, object path)


# ── pure decisions ──


def states(bits: Iterable[int]) -> set[int]:
    """AT-SPI sends the state set as 32-bit words, lowest first."""
    found: set[int] = set()
    for word_index, word in enumerate(bits):
        for bit in range(32):
            if int(word) >> bit & 1:
                found.add(word_index * 32 + bit)
    return found


def classify(role: str, state: set[int], interfaces: Iterable[str]) -> tuple[bool, str]:
    """(secure, kind) for an element, from its role name, states and interfaces, before any text is read."""
    role = role.casefold()
    if role in PASSWORD_ROLES:
        return True, ""
    if TEXT not in set(interfaces) or STATE_DEFUNCT in state:
        return False, ""
    if role in TERMINAL_ROLES:
        return False, TERMINAL
    if role in NOT_WRITING_ROLES:
        return False, ""
    if STATE_EDITABLE in state and STATE_READ_ONLY not in state:
        return False, FIELD
    return False, ""


def password_by_role(role_names: Iterable[str], role_numbers: Iterable[int | None]) -> bool:
    """Whether the field, or a parent it sits in, is a password field, from the role names and numeric roles of the
    field and its parents (field first). A numeric role that could not be read counts as a password: Heliox does not
    read a field it cannot tell is not one. Pure."""
    if any(str(name).casefold() in PASSWORD_ROLES for name in role_names):
        return True
    return any(number is None or number == ROLE_PASSWORD_TEXT for number in role_numbers)


def field_id(ref: Ref) -> str:
    return f"atspi:{ref[0]}{ref[1]}"


def ref_from_field(field: str) -> Ref | None:
    """The (bus name, object path) a field id was made from. Pure."""
    rest = field.removeprefix("atspi:")
    if rest == field or "/" not in rest:
        return None
    bus, _, path = rest.partition("/")
    return (bus, "/" + path) if bus else None


GONE_ERRORS = ("UnknownObject", "ServiceUnknown", "UnknownMethod", "NameHasNoOwner", "Disconnected")


def gone_from_error(text: str) -> bool:
    """Whether a D-Bus error says the object or its application no longer exists. Pure."""
    return any(name in text for name in GONE_ERRORS)


def focus_from_event(body: tuple, sender: str, path: str) -> Ref | None:
    """The element that just took focus, from an ``object:state-changed`` signal, or None."""
    if len(body) >= 2 and body[0] == "focused" and int(body[1]) == 1 and sender and path:
        return (sender, path)
    return None


def comm_name(pid: int, proc: Path = Path("/proc")) -> str:
    try:
        return (proc / str(pid) / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


# ── the D-Bus side ──


Call = Callable[[str, str, str, str, str, tuple], tuple]


class Transport:
    """The two things the client needs from D-Bus: a method call and the focus events received so far."""

    unique_name = ""

    def call(self, bus: str, path: str, interface: str, method: str, signature: str = "", body: tuple = ()) -> tuple:
        raise NotImplementedError

    def events(self) -> list[Ref]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def accessibility_bus_address() -> str:
    """The AT-SPI bus address: $AT_SPI_BUS_ADDRESS, else asked of the session bus (org.a11y.Bus)."""
    address = os.environ.get("AT_SPI_BUS_ADDRESS", "")
    if address:
        return address
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection

    with open_dbus_connection(_session_bus()) as session:
        reply = session.send_and_get_reply(
            new_method_call(
                DBusAddress("/org/a11y/bus", bus_name="org.a11y.Bus", interface="org.a11y.Bus"), "GetAddress"
            ),
            timeout=2.0,
        )
        return str(_unwrap(reply)[0])


def toolkit_accessibility_enabled() -> bool | None:
    """org.a11y.Status IsEnabled: whether Qt, Chromium and similar toolkits are told to expose themselves."""
    try:
        from jeepney import DBusAddress, Properties
        from jeepney.io.blocking import open_dbus_connection

        with open_dbus_connection(_session_bus()) as session:
            status = DBusAddress("/org/a11y/bus", bus_name="org.a11y.Bus", interface="org.a11y.Status")
            reply = session.send_and_get_reply(Properties(status).get("IsEnabled"), timeout=2.0)
            return bool(_unwrap(reply)[0][1])
    except Exception:
        return None


def _session_bus() -> str:
    if os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return "SESSION"
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"  # type: ignore[attr-defined]
    return f"unix:path={runtime}/bus"


def _unwrap(msg: Any) -> tuple:
    from jeepney.wrappers import unwrap_msg

    return tuple(unwrap_msg(msg))


class JeepneyTransport(Transport):
    """A blocking connection to the accessibility bus, subscribed to focus events."""

    def __init__(self, address: str | None = None) -> None:
        from jeepney import HeaderFields, MatchRule, message_bus
        from jeepney.io.blocking import open_dbus_connection

        self._fields = HeaderFields
        self._conn = open_dbus_connection(address or accessibility_bus_address())
        self.unique_name = self._conn.unique_name
        rule = MatchRule(type="signal", interface="org.a11y.atspi.Event.Object", member="StateChanged")
        rule.add_arg_condition(0, "focused")
        self._queue: deque = deque(maxlen=64)
        self._filter = self._conn.filter(rule, queue=self._queue)
        self._conn.send_and_get_reply(message_bus.AddMatch(rule), timeout=2.0)
        self._register()

    def _register(self) -> None:
        # Toolkits send events only for kinds some client registered. The method's arguments grew over at-spi2-core
        # releases; the first form the registry accepts is used.
        for signature, body in (("sass", (FOCUS_EVENT, [], "")), ("sas", (FOCUS_EVENT, [])), ("s", (FOCUS_EVENT,))):
            try:
                self.call(
                    REGISTRY, "/org/a11y/atspi/registry", "org.a11y.atspi.Registry", "RegisterEvent", signature, body
                )
                return
            except Exception:
                continue
        logger.debug("The AT-SPI registry accepted no RegisterEvent form; focus is found by searching")

    def call(self, bus: str, path: str, interface: str, method: str, signature: str = "", body: tuple = ()) -> tuple:
        from jeepney import DBusAddress, new_method_call

        msg = new_method_call(DBusAddress(path, bus_name=bus, interface=interface), method, signature or None, body)
        return _unwrap(self._conn.send_and_get_reply(msg, timeout=CALL_TIMEOUT))

    def events(self) -> list[Ref]:
        for _ in range(500):
            try:
                self._conn.recv_messages(timeout=0)
            except TimeoutError:
                break
        found: list[Ref] = []
        while self._queue:
            msg = self._queue.popleft()
            ref = focus_from_event(
                tuple(msg.body),
                str(msg.header.fields.get(self._fields.sender, "")),
                str(msg.header.fields.get(self._fields.path, "")),
            )
            if ref is not None:
                found.append(ref)
        return found

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._filter.close()
        with contextlib.suppress(Exception):
            self._conn.close()


@dataclass
class _Focus:
    ref: Ref
    window: Ref | None = None


class AtspiClient:
    """Follows the focused accessible element and describes it. Blocking: call from one thread at a time."""

    def __init__(self, transport: Transport, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._t = transport
        self._clock = clock
        self._focus: _Focus | None = None
        self._searched_at = -1e9
        self._pids: dict[str, int] = {}

    def close(self) -> None:
        self._t.close()

    # small wrappers

    def _get(self, ref: Ref, interface: str, prop: str) -> Any:
        return self._t.call(ref[0], ref[1], PROPS, "Get", "ss", (interface, prop))[0][1]

    def _state(self, ref: Ref) -> set[int]:
        return states(self._t.call(ref[0], ref[1], ACCESSIBLE, "GetState")[0])

    def _children(self, ref: Ref) -> list[Ref]:
        return [(str(bus), str(path)) for bus, path in self._t.call(ref[0], ref[1], ACCESSIBLE, "GetChildren")[0]]

    def _role(self, ref: Ref) -> str:
        return str(self._t.call(ref[0], ref[1], ACCESSIBLE, "GetRoleName")[0])

    def _role_number(self, ref: Ref) -> int | None:
        try:
            return int(self._t.call(ref[0], ref[1], ACCESSIBLE, "GetRole")[0])
        except Exception:
            return None

    def is_window(self, ref: Ref) -> bool:
        """Whether ``ref`` is a window, by role name or by numeric role (GTK 4 names its windows "application")."""
        if self._role(ref) in WINDOW_ROLES:
            return True
        return self._role_number(ref) in WINDOW_ROLE_NUMBERS

    def is_password(self, ref: Ref, role: str | None = None) -> bool:
        """Whether ``ref`` is a password field, or sits inside one, by role name or numeric role (GTK 4 marks it
        only in the numeric role).

        Only a walk that ended normally can say "not a password": it reached the root or a null parent, or went
        PASSWORD_PARENT_DEPTH levels up with every role read. Any call that fails on the way (the parent, its role
        name, its numeric role) means Heliox cannot tell, and cannot telling is a refusal: GTK 4 marks the focused
        object itself (CI on 8d513fd), but a toolkit or version that marks only an enclosing entry is caught only
        through its parents."""
        try:
            names = [role if role is not None else self._role(ref)]
        except Exception:
            return True
        numbers: list[int | None] = [self._role_number(ref)]
        current = ref
        for _ in range(PASSWORD_PARENT_DEPTH):
            try:
                bus, path = self._get(current, ACCESSIBLE, "Parent")
            except Exception:
                return True
            current = (str(bus), str(path))
            if not current[1] or current[1] == ROOT_PATH or current[1].endswith("/null"):
                break
            try:
                names.append(self._role(current))
            except Exception:
                return True
            numbers.append(self._role_number(current))  # None, when it cannot be read, counts as a password
        return password_by_role(names, numbers)

    def pid(self, bus: str) -> int:
        if bus not in self._pids:
            if len(self._pids) > 200:
                self._pids.clear()
            try:
                self._pids[bus] = int(
                    self._t.call(
                        "org.freedesktop.DBus",
                        "/org/freedesktop/DBus",
                        "org.freedesktop.DBus",
                        "GetConnectionUnixProcessID",
                        "s",
                        (bus,),
                    )[0]
                )
            except Exception:
                self._pids[bus] = 0
        return self._pids[bus]

    # following focus

    def focused(self) -> Ref | None:
        for ref in self._t.events():
            if ref[0] != self._t.unique_name:
                self._focus = _Focus(ref)
        if self._focus is not None:
            try:
                state = self._state(self._focus.ref)
                if STATE_FOCUSED in state and STATE_DEFUNCT not in state:
                    return self._focus.ref
            except Exception:
                pass
            self._focus = None
        if self._clock() - self._searched_at < SEARCH_EVERY_SECONDS:
            return None
        self._searched_at = self._clock()
        ref = self._search()
        if ref is not None:
            self._focus = _Focus(ref)
        return ref

    def focused_in(self, window: Ref) -> Ref | None:
        """The focused element inside ``window``, asked of its application in one call (Collection.GetMatches with
        the FOCUSED state). None when the app has none focused there, does not implement Collection, or does not
        answer within the call limit; the caller then walks the tree itself. In canonical order an ancestor comes
        before what it holds, so the last match still focused is taken: if a toolkit marks the page as focused as well
        as the field in it, the field is the one read.

        A browser's tree is deep and wide (the window's own toolbars and tabs come before the page), so a walk from
        outside within ``SEARCH_BUDGET`` objects may never reach the page; the app walks its own tree in process."""
        rule = (FOCUSED_STATE_BITS, MATCH_ALL, {}, MATCH_ALL, [], MATCH_ALL, [], MATCH_ALL, False)
        try:
            found = self._t.call(
                window[0], window[1], COLLECTION, "GetMatches", MATCH_SIGNATURE, (rule, SORT_CANONICAL, 8, True)
            )[0]
        except Exception:
            return None
        for bus, path in reversed(list(found)):
            ref = (str(bus or window[0]), str(path))
            try:
                state = self._state(ref)
            except Exception:
                continue
            if STATE_FOCUSED in state and STATE_DEFUNCT not in state:
                return ref
        return None

    def _search(self) -> Ref | None:
        """Look for the focused element in active windows first: by asking each active window's application
        (``focused_in``), then by a walk visiting at most ``SEARCH_BUDGET`` objects."""
        budget = SEARCH_BUDGET
        try:
            apps = self._children((REGISTRY, ROOT_PATH))
        except Exception:
            return None
        windows: list[tuple[int, Ref]] = []
        for app in apps:
            if app[0] == self._t.unique_name:
                continue
            try:
                for window in self._children(app):
                    budget -= 1
                    windows.append((0 if STATE_ACTIVE in self._state(window) else 1, window))
            except Exception:
                continue
        for _rank, window in (item for item in windows if item[0] == 0):
            ref = self.focused_in(window)
            if ref is not None:
                return ref
        for _rank, window in sorted(windows, key=lambda item: item[0]):
            stack = [window]
            while stack and budget > 0:
                ref = stack.pop()
                budget -= 1
                try:
                    state = self._state(ref)
                except Exception:
                    continue
                if STATE_FOCUSED in state:
                    return ref
                if STATE_MANAGES_DESCENDANTS in state or (ref != window and STATE_SHOWING not in state):
                    continue
                with contextlib.suppress(Exception):
                    stack.extend(reversed(self._children(ref)))
        return None

    # describing it

    def describe(self, ref: Ref) -> FocusedText:
        role = self._role(ref)
        state = self._state(ref)
        try:
            interfaces = [str(name) for name in self._t.call(ref[0], ref[1], ACCESSIBLE, "GetInterfaces")[0]]
        except Exception:
            interfaces = []
        secure, kind = classify(role, state, interfaces)
        if kind and not secure and self.is_password(ref, role):
            secure, kind = True, ""
        text = None
        if kind:
            count = int(self._get(ref, TEXT, "CharacterCount") or 0)
            text = str(self._t.call(ref[0], ref[1], TEXT, "GetText", "ii", (max(0, count - MAX_CHARS), count))[0])
        return admit(
            ElementFacts(role=role, pid=self.pid(ref[0]), secure=secure, kind=kind, field=field_id(ref), text=text)
        )

    def window_title(self) -> str:
        """The title of the window holding the focused element ("" when unknown)."""
        focus = self._focus
        if focus is None:
            return ""
        if focus.window is None:
            ref: Ref = focus.ref
            for _ in range(40):
                try:
                    if self.is_window(ref):
                        focus.window = ref
                        break
                    bus, path = self._get(ref, ACCESSIBLE, "Parent")
                except Exception:
                    return ""
                if (str(bus), str(path)) == ref or str(path) in ("/org/a11y/atspi/null", ROOT_PATH):
                    return ""
                ref = (str(bus), str(path))
            else:
                return ""
        try:
            return str(self._get(focus.window, ACCESSIBLE, "Name") or "")
        except Exception:
            return ""

    def foreground(self) -> tuple[str, str] | None:
        """(process name, window title) of the application that has focus, or None when AT-SPI cannot tell."""
        ref = self.focused()
        if ref is None:
            return None
        app = comm_name(self.pid(ref[0]))
        return (app, self.window_title()) if app else None


# ── one shared connection for the reader and the foreground query ──


class _Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.client: AtspiClient | None = None
        self.error = ""
        self.retry_at = 0.0
        self.failures = 0
        self.make: Callable[[], Transport] = JeepneyTransport

    def get(self) -> AtspiClient | None:
        """The client, connecting on first use; None (with ``error`` set) while the bus cannot be reached."""
        if self.client is not None:
            return self.client
        if time.monotonic() < self.retry_at:
            return None
        try:
            self.client = AtspiClient(self.make())
            self.error = ""
            self.failures = 0
        except ImportError:
            self.error = NO_BINDING
            self.retry_at = float("inf")
        except Exception as exc:
            logger.debug("AT-SPI connection failed: %s", exc)
            self.error = NO_BUS
            self.failures += 1
            self.retry_at = (
                time.monotonic() + RESTART_BACKOFF_SECONDS[min(self.failures - 1, len(RESTART_BACKOFF_SECONDS) - 1)]
            )
        return self.client

    def drop(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()


SHARED = _Shared()


def read_focused() -> FocusedText | None:
    with SHARED.lock:
        client = SHARED.get()
        if client is None:
            return None
        try:
            ref = client.focused()
            return client.describe(ref) if ref is not None else None
        except (ConnectionError, EOFError):
            logger.debug("AT-SPI connection lost", exc_info=True)
            SHARED.drop()
            return None
        except Exception:
            # One application answered with an error or too slowly: the connection is fine, try again next look.
            logger.debug("AT-SPI read failed", exc_info=True)
            return None


def field_exists(field: str) -> bool | None:
    """Whether the accessible behind a field id still exists: False once its window closed or its app quit."""
    ref = ref_from_field(field)
    if ref is None or not SHARED.lock.acquire(timeout=1.0):
        return None
    try:
        client = SHARED.get()
        if client is None:
            return None
        try:
            return STATE_DEFUNCT not in client._state(ref)
        except (ConnectionError, EOFError):
            SHARED.drop()
            return None
        except Exception as error:
            return False if gone_from_error(f"{type(error).__name__} {error}") else None
    finally:
        SHARED.lock.release()


def foreground_window() -> tuple[str, str] | None:
    """(process name, window title) through AT-SPI, for the timeline. Works under X11 and Wayland alike."""
    if not SHARED.lock.acquire(timeout=1.0):
        return None
    try:
        client = SHARED.get()
        if client is None:
            return None
        try:
            return client.foreground()
        except (ConnectionError, EOFError):
            SHARED.drop()
            return None
        except Exception:
            return None
    finally:
        SHARED.lock.release()


class LinuxFocusedTextReader:
    """``FocusedTextSource`` for Linux. Safe to construct anywhere; it connects on the first read."""

    def __init__(self) -> None:
        self.available = True
        self._toolkit: bool | None = None

    async def read(self) -> FocusedText | None:
        return await asyncio.to_thread(read_focused)

    def status(self) -> ReaderStatus:
        if SHARED.error:
            return ReaderStatus("unavailable", SHARED.error)
        if SHARED.client is None:
            return ReaderStatus("starting")
        if self._toolkit is None:
            self._toolkit = toolkit_accessibility_enabled()
        return ReaderStatus("ready", "" if self._toolkit else TOOLKIT_HINT)

    async def close(self) -> None:
        with SHARED.lock:
            SHARED.drop()

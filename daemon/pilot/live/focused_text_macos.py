"""Read the focused text field on macOS through the Accessibility API.

The reader asks the system-wide accessibility element for the focused element (``AXFocusedUIElement``), then for
its role and subrole, whether its value can be set (that is what "editable" means to the Accessibility API), and
only then for its text. A secure text field (subrole ``AXSecureTextField``) is refused before its value is asked
for, and so is any element while the system's secure keyboard entry is on (a password field has focus somewhere).
Terminal.app and iTerm2 expose their buffer as a text area and are read as terminals; other terminals expose
nothing.

It runs in the daemon's own process through pyobjc (``pyobjc-framework-ApplicationServices`` and
``pyobjc-framework-Cocoa``, the optional ``pilot-daemon[macos]`` extra: prebuilt wheels, no compiler), on a
worker thread, with a 0.25 s messaging timeout so a hung application cannot stall Live. A read is five to seven
attribute calls to the focused application, typically 2 to 10 ms. The Accessibility API also offers observers
for focus changes, but they need a run loop on a dedicated thread; at a 0.6 s poll the plain read is cheap enough.

It needs the Accessibility permission. ``PermissionGate`` checks it without a prompt, shows the system prompt at
most once per run (only when the person turns Live's field reading on), and re-checks every 10 s, so turning it on
in System Settings takes effect without a restart. ``status`` tells the app what to grant and where.

Field identity is the application's process id plus ``CFHash`` of the element: the Accessibility API hashes an
element by what it refers to, not by the reference object, so it holds while the person types and the window
title changes.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
from typing import Any

from pilot.live.focused_text_policy import (
    FIELD,
    MAX_CHARS,
    TERMINAL,
    ElementFacts,
    FocusedText,
    Permission,
    PermissionGate,
    ReaderStatus,
    admit,
)

logger = logging.getLogger("pilot.live.focused_text_macos")

TEXT_ROLES = frozenset({"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"})
SECURE_SUBROLES = frozenset({"AXSecureTextField"})
TERMINAL_BUNDLES = frozenset({"com.apple.Terminal", "com.googlecode.iterm2"})
MESSAGING_TIMEOUT = 0.25

NEEDS_PERMISSION = (
    "Heliox needs Accessibility access to read the field you are typing in. Open System Settings > Privacy & "
    "Security > Accessibility and turn on Heliox-OS (or the terminal that started the Heliox daemon)."
)
NO_BINDING = (
    "Reading the focused field on macOS needs pyobjc: pip install 'pilot-daemon[macos]' "
    "(pyobjc-framework-ApplicationServices and pyobjc-framework-Cocoa)."
)
SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"


def classify(role: str, subrole: str, value_settable: bool, bundle_id: str, secure_input: bool) -> tuple[bool, str]:
    """(secure, kind) for an element, from what the Accessibility API says before any text is read."""
    if subrole in SECURE_SUBROLES or role in SECURE_SUBROLES or secure_input:
        return True, ""
    if bundle_id in TERMINAL_BUNDLES and role == "AXTextArea":
        return False, TERMINAL
    if role in TEXT_ROLES and value_settable:
        return False, FIELD
    return False, ""


def field_id(pid: int, element_hash: int) -> str:
    return f"ax:{pid}:{element_hash & 0xFFFFFFFFFFFFFFFF:x}"


def secure_input_enabled() -> bool:
    """Whether secure keyboard entry is on (a password field has focus somewhere). False when it cannot be told."""
    try:
        carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        fn = carbon.IsSecureEventInputEnabled
        fn.restype = ctypes.c_bool
        fn.argtypes = []
        return bool(fn())
    except Exception:
        return False


class AXBackend:
    """The pyobjc calls, kept small so tests can replace them. Raises ImportError when pyobjc is missing."""

    def __init__(self) -> None:
        import AppKit
        import ApplicationServices as ax  # noqa: N813
        import CoreFoundation

        self._ax = ax
        self._appkit = AppKit
        self._cf = CoreFoundation
        self._system = ax.AXUIElementCreateSystemWide()
        ax.AXUIElementSetMessagingTimeout(self._system, MESSAGING_TIMEOUT)

    def trusted(self) -> Permission:
        return Permission.GRANTED if self._ax.AXIsProcessTrusted() else Permission.NOT_ASKED

    def prompt(self) -> None:
        self._ax.AXIsProcessTrustedWithOptions({self._ax.kAXTrustedCheckOptionPrompt: True})

    def attr(self, element: Any, name: str) -> Any:
        err, value = self._ax.AXUIElementCopyAttributeValue(element, name, None)
        return value if err == 0 else None

    def settable(self, element: Any, name: str) -> bool:
        err, value = self._ax.AXUIElementIsAttributeSettable(element, name, None)
        return bool(value) if err == 0 else False

    def pid(self, element: Any) -> int:
        err, pid = self._ax.AXUIElementGetPid(element, None)
        return int(pid) if err == 0 else 0

    def element_hash(self, element: Any) -> int:
        return int(self._cf.CFHash(element))

    def focused(self) -> Any:
        return self.attr(self._system, "AXFocusedUIElement")

    def focused_app(self) -> Any:
        return self.attr(self._system, "AXFocusedApplication")

    def bundle_id(self, pid: int) -> str:
        app = self._appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(app.bundleIdentifier() or "") if app is not None else ""

    def app_name(self, pid: int) -> str:
        app = self._appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(app.localizedName() or "") if app is not None else ""

    def text_tail(self, element: Any) -> str:
        count = self.attr(element, "AXNumberOfCharacters")
        if isinstance(count, int) and count > MAX_CHARS:
            try:
                rng = self._ax.AXValueCreate(self._ax.kAXValueCFRangeType, (count - MAX_CHARS, MAX_CHARS))
                err, value = self._ax.AXUIElementCopyParameterizedAttributeValue(element, "AXStringForRange", rng, None)
                if err == 0 and value is not None:
                    return str(value)
            except Exception:
                logger.debug("AXStringForRange failed; reading the whole value", exc_info=True)
        value = self.attr(element, "AXValue")
        return str(value) if isinstance(value, str) else ""


class MacReader:
    """Blocking reads through an ``AXBackend``; ``secure_input`` is injectable for tests."""

    def __init__(self, backend: Any, secure_input: Any = secure_input_enabled) -> None:
        self.backend = backend
        self.gate = PermissionGate(backend.trusted, backend.prompt)
        self._secure_input = secure_input

    def read(self) -> FocusedText | None:
        if not self.gate.allowed():
            return None
        b = self.backend
        element = b.focused()
        if element is None:
            return None
        role = str(b.attr(element, "AXRole") or "")
        subrole = str(b.attr(element, "AXSubrole") or "")
        pid = b.pid(element)
        settable = b.settable(element, "AXValue")
        secure, kind = classify(role, subrole, settable, b.bundle_id(pid) if pid else "", self._secure_input())
        text = b.text_tail(element) if kind and not secure else None
        return admit(
            ElementFacts(
                role=role, pid=pid, secure=secure, kind=kind, field=field_id(pid, b.element_hash(element)), text=text
            )
        )

    def foreground(self) -> tuple[str, str] | None:
        """(app name, window title) of the frontmost application. The title needs Accessibility; "" without it."""
        b = self.backend
        if not self.gate.allowed():
            return None
        app = b.focused_app()
        if app is None:
            return None
        name = b.app_name(b.pid(app))
        window = b.attr(app, "AXFocusedWindow")
        title = str(b.attr(window, "AXTitle") or "") if window is not None else ""
        return (name, title) if name else None


_lock = threading.Lock()
_reader: MacReader | None = None
_error = ""


def shared() -> MacReader | None:
    """The process-wide reader, or None (with the reason in ``_error``) when pyobjc is not installed."""
    global _reader, _error
    if _reader is None and not _error:
        try:
            _reader = MacReader(AXBackend())
        except ImportError:
            _error = NO_BINDING
        except Exception as exc:
            logger.debug("Accessibility setup failed: %s", exc)
            _error = f"The Accessibility API could not be used: {exc}"
    return _reader


def front_window_owner(windows: list[dict]) -> str:
    """The app owning the frontmost normal window, from CGWindowListCopyWindowInfo's front-to-back list. Pure."""
    for info in windows:
        if int(info.get("kCGWindowLayer", 1) or 0) == 0 and info.get("kCGWindowOwnerName"):
            return str(info["kCGWindowOwnerName"])
    return ""


def frontmost_app_name() -> str:
    """The frontmost app's name without any permission (Quartz window list); "" when it cannot be told.

    NSWorkspace's frontmostApplication is not used: in a process without a running AppKit event loop, such as
    the daemon, it keeps the value from when it was first asked.
    """
    try:
        import Quartz

        options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
        return front_window_owner([dict(info) for info in windows])
    except Exception:
        logger.debug("Quartz window list failed", exc_info=True)
        return ""


def foreground_window() -> tuple[str, str] | None:
    """(app name, window title) on macOS, or None when pyobjc is not installed (the caller falls back).

    With Accessibility granted both come from the Accessibility API; without it the app name comes from the
    Quartz window list and the title is "" (macOS gives window titles only to apps holding a permission).
    """
    with _lock:
        reader = shared()
        if reader is None:
            return None
        try:
            found = reader.foreground()
        except Exception:
            logger.debug("Accessibility foreground read failed", exc_info=True)
            found = None
    if found is not None:
        return found
    name = frontmost_app_name()
    return (name, "") if name else None


class MacFocusedTextReader:
    """``FocusedTextSource`` for macOS."""

    def __init__(self) -> None:
        self.available = True

    def _read(self) -> FocusedText | None:
        with _lock:
            reader = shared()
            if reader is None:
                return None
            try:
                return reader.read()
            except Exception:
                logger.debug("Accessibility read failed", exc_info=True)
                return None

    async def read(self) -> FocusedText | None:
        return await asyncio.to_thread(self._read)

    def request_permission(self) -> bool:
        """Show the system's Accessibility prompt, at most once per run. Call only on the person's action."""
        reader = shared()
        return bool(reader is not None and reader.gate.request())

    def status(self) -> ReaderStatus:
        reader = shared()
        if reader is None:
            return ReaderStatus("unavailable", _error)
        if not reader.gate.allowed():
            return ReaderStatus("needs_permission", NEEDS_PERMISSION, SETTINGS_URL)
        return ReaderStatus("ready")

    async def close(self) -> None:
        return None

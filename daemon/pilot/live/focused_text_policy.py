"""What Live may read, decided the same way on Windows, macOS and Linux. Pure: no platform calls, tested everywhere.

Each platform reader turns what its accessibility interface says about the focused element into ``ElementFacts``
(UI Automation on Windows, the Accessibility API on macOS, AT-SPI on Linux) and hands it to ``admit``, which is the
only place that decides whether text is passed on. ``never_read`` is the window-level rule applied before any
reader runs. ``PermissionGate`` is the state machine for a permission the person grants once (macOS Accessibility):
it never prompts more than once per run and re-checks quietly so a grant takes effect without a restart.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

MAX_CHARS = 6000

FIELD = "field"
TERMINAL = "terminal"

# Password managers by name, matched against the app and the window title (the list Live has always used).
PASSWORD_MANAGERS = ("1password", "bitwarden", "keepass", "lastpass", "dashlane", "keeper", "enpass", "nordpass")
# Credential stores that ship with an operating system, matched against the whole app name only, so a window
# title such as "Change your passwords" is still read: macOS Keychain Access and Passwords, GNOME Passwords and
# Keys (Seahorse), KDE Wallet Manager.
SYSTEM_CREDENTIAL_APPS = frozenset(
    {
        "keychain access",
        "passwords",
        "seahorse",
        "org.gnome.seahorse.application",
        "kwalletmanager",
        "kwalletmanager5",
        "kwalletmanager6",
    }
)


@dataclass(frozen=True)
class FocusedText:
    text: str
    control_type: str
    pid: int
    password: bool
    # True only when the focused element is a writable text field, document or terminal; text is empty otherwise.
    editable: bool = False
    kind: str = ""  # "field", "terminal", or "" when not editable
    # Which field this is, stable while the person types in it (it must not change with the window title):
    # Windows: the UI Automation runtime id; macOS: "ax:<pid>:<element hash>"; Linux: "atspi:<bus name><path>".
    field: str = ""


@dataclass(frozen=True)
class ElementFacts:
    """What a platform says about the focused element, before any of its text is used."""

    role: str  # the platform's own role name ("Edit", "AXTextArea", "entry"), for display only
    pid: int = 0
    secure: bool = False  # a password field, by the platform's own marking
    kind: str = ""  # FIELD or TERMINAL when the platform reader found a place the person writes, else ""
    field: str = ""
    text: str | None = None  # read only after the reader decided ``kind``; never trusted on its own


def admit(facts: ElementFacts) -> FocusedText:
    """The one decision every platform goes through: text passes only from a writable, non-secure field."""
    editable = not facts.secure and facts.kind in (FIELD, TERMINAL)
    text = (facts.text or "") if editable else ""
    if len(text) > MAX_CHARS:
        text = text[-MAX_CHARS:]
    return FocusedText(
        text=text,
        control_type=facts.role,
        pid=int(facts.pid or 0),
        password=bool(facts.secure),
        editable=editable,
        kind=facts.kind if editable else "",
        field=facts.field if editable else "",
    )


def never_read(app: str, title: str, excluded: Iterable[str] = (), *, is_heliox: bool = False) -> bool:
    """Whether a window is off limits before anything in it is read: Heliox's own window, password managers,
    the operating system's credential stores, and apps the person excluded from the timeline."""
    where = f"{app} {title}".casefold()
    bare = app.strip().casefold().removesuffix(".exe").removesuffix(".app")
    return (
        is_heliox
        or any(name in where for name in PASSWORD_MANAGERS)
        or bare in SYSTEM_CREDENTIAL_APPS
        or any(name.strip().casefold() in where for name in excluded if name.strip())
    )


HELIOX_BUNDLE_ID = "com.helioxos.app"  # tauri.conf.json "identifier"


def _excluded_none() -> Iterable[str]:
    return ()


# Where the readers get the person's excluded apps when they act on an app before anything in it is read (the
# daemon points it at its live settings, so a change applies at once). Until then: none excluded beyond the rules.
_excluded_source: Callable[[], Iterable[str]] = _excluded_none


def set_excluded_apps(source: Callable[[], Iterable[str]]) -> None:
    """Tell the readers where the person's excluded apps are (a callable, read on every check)."""
    global _excluded_source
    _excluded_source = source


def _bare_name(text: str) -> str:
    return " ".join(text.casefold().replace("-", " ").replace("_", " ").split())


def may_touch(app: str, title: str = "") -> bool:
    """Whether a reader may act on an app it has not read yet, such as asking it to share its page with assistive
    tools: ``never_read`` with the person's excluded apps, and Heliox's own window by name or title. ``title`` can be
    a window title or, on macOS, the bundle id. False when the excluded list cannot be had: Heliox does not touch an
    app it cannot tell is allowed."""
    try:
        excluded = [str(name) for name in _excluded_source()]
    except Exception:
        return False
    is_heliox = (
        _bare_name(app) in ("heliox os", "heliox")
        or _bare_name(title) == "heliox os"
        or title.strip().casefold() == HELIOX_BUNDLE_ID
    )
    return not never_read(app, title, excluded, is_heliox=is_heliox)


def same_field(previous: FocusedText | None, current: FocusedText | None) -> bool:
    """Whether two reads are the same field (the person kept typing there), whatever happened to the title."""
    return bool(previous and current and previous.field and previous.field == current.field)


# ── a permission the person grants once ──


class Permission(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"  # asked, or set off in System Settings
    NOT_ASKED = "not_asked"
    UNKNOWN = "cannot_tell"


@dataclass(frozen=True)
class ReaderStatus:
    """What the app shows about the reader: ``state`` is "ready", "needs_permission", "unavailable" or "off"."""

    state: str
    message: str = ""
    settings_url: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"state": self.state, "message": self.message, "settings_url": self.settings_url}


class PermissionGate:
    """Tracks one permission without prompting in a loop.

    ``check`` reads the state without a prompt. ``request`` asks the system to prompt, at most once per run, and only
    when the state is not already granted. While the permission is missing, ``allowed`` re-checks at most every
    ``recheck_seconds`` so a grant made in System Settings takes effect without a restart.
    """

    def __init__(
        self,
        check: Callable[[], Permission],
        prompt: Callable[[], None] | None = None,
        *,
        recheck_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._check = check
        self._prompt = prompt
        self._recheck = recheck_seconds
        self._clock = clock
        self._state: Permission | None = None
        # Never checked. Only read after a check sets it, but 0.0 would read as "just checked" on a machine booted
        # less than ``recheck_seconds`` ago (a daemon started at login), so it must never mean "never".
        self._checked_at = float("-inf")
        self.prompted = False

    @property
    def state(self) -> Permission:
        if self._state is None:
            self._refresh()
        assert self._state is not None
        return self._state

    def _refresh(self) -> None:
        try:
            self._state = self._check()
        except Exception:
            self._state = Permission.UNKNOWN
        self._checked_at = self._clock()

    def allowed(self) -> bool:
        if self._state is None or (
            self._state != Permission.GRANTED and self._clock() - self._checked_at >= self._recheck
        ):
            self._refresh()
        return self._state == Permission.GRANTED

    def request(self) -> bool:
        """Show the system prompt once per run if the permission is missing. Returns whether a prompt was asked."""
        if self.allowed() or self.prompted or self._prompt is None:
            return False
        self.prompted = True
        try:
            self._prompt()
        except Exception:
            return False
        return True

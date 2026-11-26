"""Read what the desktop shows, for the post-condition checks of desktop actions.

Everything here only reads. Nothing focuses a window, moves the pointer, presses or releases a key, writes the
clipboard, starts a program or asks for a permission: a read that would need one of those is "not observable" with
the reason, never a pass. Each read goes through a different call than the action that changed the state:

==================  =====================================  =========================================  ===========================
What                Windows                                macOS                                      Linux
==================  =====================================  =========================================  ===========================
window in front     GetForegroundWindow (in process)       AXFocusedApplication with Accessibility,   X11: xprop (xdotool when
                                                           else LaunchServices (lsappinfo), no title  xprop is missing); Wayland:
                                                                                                      AT-SPI, else not observable
pointer, buttons    GetCursorPos, GetAsyncKeyState          Quartz CGEvent location and button state   X11: python-xlib
                                                                                                      query_pointer; Wayland: no
keys held down      GetAsyncKeyState                       Quartz modifier flags (other keys through  X11: python-xlib
                                                           pyautogui's key table)                     query_keymap; Wayland: no
clipboard text      Win32 GetClipboardData (own call,      pbpaste (a new process)                    X11: xclip or xsel; Wayland:
                    not pyperclip or Set-Clipboard)                                                   not observable
focused field       UI Automation through Live's running   the Accessibility API, without waking      AT-SPI
                    helper only                            any app
window state        IsIconic / IsZoomed                    AXMinimized, AXPosition/AXSize             X11: xprop _NET_WM_STATE
window list         EnumWindows                            Quartz CGWindowListCopyWindowInfo          X11: wmctrl -lp
==================  =====================================  =========================================  ===========================

Why the reading is made inside the step and kept, and must stay that way: a reading taken at verify time cannot tell
"never focused" from "focused, then moved on". What makes a check independent is that the reading comes from a
different call than the one that acted, taken at the moment it is true; that is what is kept here. Reading the same
state again after the whole plan, when later steps or the person have changed it, is not a better check but a
worse one.

Desktop state is fleeting: the plan's next step, or the person, moves the focus and the pointer again within a
second, and the verifier runs after the whole plan. So the checks that matter for the next keystroke run right after
the step that changed the state, while the executor is still inside that step (``step`` marks it), and are kept
here for the verifier (``recalled``). A wrong window, a pointer that is not where the click was sent, or a clipboard
that does not hold what was copied fails that step at once, so the executor skips the steps after it: no more keys go
to a window the plan did not mean. A read that cannot be made never fails a step. Only what shows input going to
another window stops the run, and then without a retry (``input_went_elsewhere``, ``RECONCILE_NOT_RETRY_ACTIONS``):
the window in front changed while typing, or a key or button is still held. A field that does not show the text, a
pointer moved after the event, an editor that reformatted the text: those are "not verified", and the run goes on.
Where the action names its window and another one is in front before the first key, no key is sent at all
(``refuse_unless_in_front``); a click whose pointer is not at its target when the button would go down is not made.

Typed text and clipboard text are compared in memory and dropped; only lengths, hashes and yes/no answers are kept,
logged or put in a message. A password field is never read.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform
from pilot.timeouts import bounded

logger = logging.getLogger("pilot.system.desktop_observe")

READ_SECONDS = 2.0  # one read of the foreground, the pointer, the keys or the clipboard
FOCUS_SETTLE_SECONDS = 0.6  # how long a focus request may take to show (window managers act after wmctrl returns)
SETTLE_STEP_SECONDS = 0.05
POINTER_TOLERANCE = 2  # pixels: rounding between logical and physical coordinates
SUBPROCESS_SECONDS = 1.5

WAYLAND_FOREGROUND = (
    "Wayland does not let a program ask which window has the focus, and AT-SPI (the accessibility bus) did not "
    "answer: accessibility may be off in this session, or the window's app is not on the bus"
)
WAYLAND_POINTER = (
    "Wayland does not tell a program where the pointer is, or which buttons are down, outside its own windows"
)
WAYLAND_KEYS = "Wayland does not let a program read which keys are held down"
WAYLAND_CLIPBOARD = (
    "Wayland lets only the window with the focus read the clipboard (GNOME has no clipboard-manager protocol, and "
    "wl-paste would open a window to take the focus), so it is not read"
)
NEVER_READ_WINDOW = (
    "the window in front is one Heliox never reads (a password manager, the system's credential store, or an app "
    "excluded from the timeline)"
)


class Unobservable(Exception):
    """The state cannot be read here; ``str(error)`` is the reason, one sentence a person can read."""


class DesktopStepMismatch(RuntimeError):
    """A desktop step's effect was read right after it and is not there; the step fails so later steps do not run."""


# ── what was read ──


@dataclass(frozen=True)
class Foreground:
    """The window with the keyboard focus. ``title`` None: the title cannot be read here (macOS without
    Accessibility); "" is a window without a title. ``window_id`` is the platform's handle where it has one."""

    names: tuple[str, ...]
    title: str | None
    pid: int = 0
    window_id: int | None = None
    source: str = ""

    def describe(self) -> str:
        """How a result names this window: by its application, never by its title (titles carry mail subjects,
        document names and chat names)."""
        name = next((n for n in self.names if n), "")
        if name:
            return name.removesuffix(".exe").removesuffix(".EXE")
        if self.title:
            return "a window"
        return "no window" if self.window_id == 0 else "an unnamed window"

    def app(self) -> str:
        """The application, by name only: window titles can show what the person is writing."""
        name = next((n for n in self.names if n), "")
        if name:
            return name.removesuffix(".exe").removesuffix(".EXE")
        return "no window" if self.window_id == 0 else "an unnamed window"

    def same_window(self, other: Foreground) -> bool:
        if self.window_id is not None and other.window_id is not None:
            return self.window_id == other.window_id
        if self.pid and other.pid and self.pid != other.pid:
            return False
        return self.names == other.names and self.title == other.title


@dataclass(frozen=True)
class FieldReading:
    """The focused element, by the same rules as Live: text only from a writable, non-password field."""

    editable: bool
    secure: bool
    text: str | None  # None when not read (not editable, a password field)
    source: str


@dataclass(frozen=True)
class WindowRow:
    pid: int
    name: str
    title: str
    window_id: int = 0  # the platform's window handle or number; 0 when the list gives none


@dataclass
class StepRecord:
    """What was read right after a desktop step. ``facts`` never holds typed or copied text, only lengths, hashes
    and yes/no answers."""

    action: Any
    outcome: str  # "match", "mismatch", "unobservable" or "partial"
    detail: str
    facts: dict[str, Any] = field(default_factory=dict)


# ── the step the executor is running, and what was read after it ──

_CURRENT_STEP: contextvars.ContextVar[Any] = contextvars.ContextVar("heliox_desktop_step", default=None)
_RECORDS: OrderedDict[int, StepRecord] = OrderedDict()
_MAX_RECORDS = 128


@contextlib.contextmanager
def step(action: Any) -> Iterator[None]:
    """Mark ``action`` as the step being run, so the real input functions read the result right after it."""
    token = _CURRENT_STEP.set(action)
    try:
        yield
    finally:
        _CURRENT_STEP.reset(token)


def current_step() -> Any:
    return _CURRENT_STEP.get()


def remember(record: StepRecord) -> None:
    key = id(record.action)
    _RECORDS[key] = record
    _RECORDS.move_to_end(key)
    while len(_RECORDS) > _MAX_RECORDS:
        _RECORDS.popitem(last=False)


def recalled(action: Any) -> StepRecord | None:
    """The record made right after ``action`` ran (the same object), or None."""
    record = _RECORDS.get(id(action))
    return record if record is not None and record.action is action else None


def forget_all() -> None:
    _RECORDS.clear()


# ── small pure helpers ──


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def bare_name(name: str) -> str:
    text = (name or "").strip().casefold()
    for suffix in (".exe", ".app"):
        text = text.removesuffix(suffix)
    return text


def parse_window_id(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip(), 0)
    except ValueError:
        return None


def focus_matches(
    found: Foreground,
    *,
    window_id: str | None = None,
    title: str | None = None,
    process_name: str | None = None,
    title_pids: frozenset[int] = frozenset(),
) -> bool | None:
    """Whether the window in front is the one asked for, matched the way the focus handlers match it: a window id
    exactly, a process name exactly (".exe" and ".app" ignored, any case), a title as a substring in any case. On
    Windows a title also matches a window of a process that has a window with that title (a dialog of the app in
    front). None: the title was asked for and cannot be read here."""
    wanted_id = parse_window_id(window_id)
    if wanted_id is not None:
        if found.window_id is None:
            return None
        return found.window_id == wanted_id
    if process_name:
        wanted = bare_name(process_name)
        names = {bare_name(name) for name in found.names if name}
        # A name of exactly 15 characters may be the kernel's cut form of a longer one.
        return wanted in names or any(len(name) == 15 and wanted[:15] == name for name in names)
    if title:
        if found.pid and found.pid in title_pids:
            return True
        if found.title is None:
            return None
        return title.casefold() in found.title.casefold()
    return None


def pointer_at(found: tuple[int, int], target: tuple[int, int], tolerance: int = POINTER_TOLERANCE) -> bool:
    return abs(found[0] - target[0]) <= tolerance and abs(found[1] - target[1]) <= tolerance


def parse_xprop_active(output: str) -> int | None:
    """The id in ``xprop -root _NET_ACTIVE_WINDOW`` ("_NET_ACTIVE_WINDOW(WINDOW): window id # 0x3a00007")."""
    match = re.search(r"#\s*(0x[0-9a-fA-F]+|\d+)", output)
    return int(match.group(1), 0) if match else None


def parse_xprop_window(output: str) -> tuple[int, str]:
    """(pid, title) from ``xprop -id <id> _NET_WM_PID _NET_WM_NAME WM_NAME``; 0 and "" when absent."""
    pid = 0
    title = ""
    fallback = ""
    for line in output.splitlines():
        if line.startswith("_NET_WM_PID"):
            digits = re.search(r"=\s*(\d+)", line)
            pid = int(digits.group(1)) if digits else 0
        elif line.startswith("_NET_WM_NAME"):
            title = _xprop_string(line)
        elif line.startswith("WM_NAME"):
            fallback = _xprop_string(line)
    return pid, title or fallback


def _xprop_string(line: str) -> str:
    _, sep, value = line.partition("=")
    value = value.strip()
    if not sep or not value.startswith('"'):
        return ""
    try:
        import ast

        return str(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return value.strip('"')


def parse_xprop_states(output: str) -> set[str]:
    """The atoms in ``xprop -id <id> _NET_WM_STATE`` ("_NET_WM_STATE(ATOM) = _NET_WM_STATE_HIDDEN, ...")."""
    _, sep, value = output.partition("=")
    if not sep:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_wmctrl_lp(output: str) -> list[tuple[int, int, str]]:
    """(window id, pid, title) rows from ``wmctrl -lp`` ("0x03a00007  0 4242 host Title words")."""
    rows = []
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        window = parse_window_id(parts[0])
        try:
            pid = int(parts[2])
        except ValueError:
            continue
        if window is None:
            continue
        rows.append((window, pid, parts[4] if len(parts) == 5 else ""))
    return rows


def parse_xdotool_location(output: str) -> tuple[int, int] | None:
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    try:
        return int(values["X"]), int(values["Y"])
    except (KeyError, ValueError):
        return None


# ── platform plumbing ──


def wayland() -> bool:
    from pilot.system.portal import wayland_session

    return wayland_session()


def _run(args: list[str], *, utf8_locale: bool = False) -> str | None:
    """stdout of a short read-only command as UTF-8, or None when it is missing, fails or takes too long.
    ``utf8_locale`` runs it under a UTF-8 locale (pbpaste writes text in the locale's encoding)."""
    if not shutil.which(args[0]):
        return None
    env = {**os.environ, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"} if utf8_locale else None
    try:
        done = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUBPROCESS_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def process_name(pid: int) -> str:
    if not pid:
        return ""
    try:
        import psutil

        return str(psutil.Process(pid).name())
    except Exception:
        return ""


async def _threaded(read: Callable[[], Any], seconds: float = READ_SECONDS) -> Any:
    try:
        return await bounded(asyncio.to_thread(read), timeout=seconds)
    except TimeoutError as error:
        raise Unobservable(f"the read did not answer within {seconds:g} s") from error


# ── the window in front ──


def _win_foreground() -> Foreground:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetForegroundWindow.restype = wintypes.HWND
    handle = user32.GetForegroundWindow()
    if not handle:
        raise Unobservable(
            "no window has the keyboard focus that Heliox can see (the lock screen or an elevation prompt may be "
            "showing)"
        )
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    length = int(user32.GetWindowTextLengthW(handle))
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return Foreground((process_name(int(pid.value)),), buffer.value, int(pid.value), int(handle), "GetForegroundWindow")


def _win_windows() -> list[tuple[int, int, str, bool, bool]]:
    """Top-level windows with a title: (handle, pid, title, minimised, maximised), front to back."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    rows: list[tuple[int, int, str, bool, bool]] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(handle: Any, _param: Any) -> bool:
        if not user32.IsWindowVisible(handle):
            return True
        length = int(user32.GetWindowTextLengthW(handle))
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        rows.append(
            (int(handle), int(pid.value), buffer.value, bool(user32.IsIconic(handle)), bool(user32.IsZoomed(handle)))
        )
        return True

    user32.EnumWindows(callback_type(visit), 0)
    return rows


def _mac_names(pid: int) -> tuple[str, ...]:
    names = [process_name(pid)]
    try:
        import AppKit

        running = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if running is not None:
            names += [str(running.localizedName() or ""), str(running.bundleIdentifier() or "")]
    except Exception:
        pass
    return tuple(name for name in names if name)


def _mac_foreground() -> Foreground:
    try:
        import ApplicationServices as ax  # noqa: N813
    except ImportError:
        ax = None
    if ax is not None and ax.AXIsProcessTrusted():
        system = ax.AXUIElementCreateSystemWide()
        ax.AXUIElementSetMessagingTimeout(system, 0.25)
        err, app = ax.AXUIElementCopyAttributeValue(system, "AXFocusedApplication", None)
        if err == 0 and app is not None:
            err, pid = ax.AXUIElementGetPid(app, None)
            if err == 0 and pid:
                title = ""
                got, window = ax.AXUIElementCopyAttributeValue(app, "AXFocusedWindow", None)
                if got == 0 and window is not None:
                    got, value = ax.AXUIElementCopyAttributeValue(window, "AXTitle", None)
                    title = str(value or "") if got == 0 else ""
                return Foreground(_mac_names(int(pid)), title, int(pid), None, "the Accessibility API")
    from pilot.live.focused_text_macos import launch_services_front_app, launch_services_front_pid

    pid = launch_services_front_pid()
    if not pid:
        raise Unobservable("macOS did not say which application is in front (lsappinfo gave no answer)")
    names = tuple(name for name in (launch_services_front_app(), process_name(pid)) if name)
    return Foreground(names, None, pid, None, "LaunchServices (window titles need Accessibility)")


def _x11_foreground() -> Foreground:
    active = _run(["xprop", "-root", "_NET_ACTIVE_WINDOW"])
    if active is not None:
        window = parse_xprop_active(active)
        if window is None:
            raise Unobservable("the window manager does not publish the active window (_NET_ACTIVE_WINDOW)")
        if window == 0:
            return Foreground((), "", 0, 0, "xprop _NET_ACTIVE_WINDOW")
        details = _run(["xprop", "-id", hex(window), "_NET_WM_PID", "_NET_WM_NAME", "WM_NAME"]) or ""
        pid, title = parse_xprop_window(details)
        return Foreground((_full_name(pid),) if pid else (), title, pid, window, "xprop _NET_ACTIVE_WINDOW")
    window_text = _run(["xdotool", "getactivewindow"])
    if window_text is None:
        raise Unobservable("reading the active X11 window needs xprop (x11-utils) or xdotool, and neither answered")
    window = parse_window_id(window_text.strip()) or 0
    pid_text = _run(["xdotool", "getwindowpid", str(window)]) or ""
    title = (_run(["xdotool", "getwindowname", str(window)]) or "").rstrip("\n")
    pid = int(pid_text.strip()) if pid_text.strip().isdigit() else 0
    return Foreground((_full_name(pid),) if pid else (), title, pid, window, "xdotool getactivewindow")


def _full_name(pid: int) -> str:
    """The process name, whole: psutil reads it from the command line when /proc/<pid>/comm is cut at 15
    characters ("gnome-terminal-" for gnome-terminal-server)."""
    return process_name(pid) or _comm(pid)


def _comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return process_name(pid)


def _atspi_foreground() -> Foreground | None:
    from pilot.live.focused_text_linux import foreground_window

    found = foreground_window()
    if not found:
        return None
    name, title = found
    return Foreground((name,), title, 0, None, "AT-SPI")


def read_foreground_now() -> Foreground:
    """The window in front, read now (blocking; call through ``foreground``)."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return _win_foreground()
    if CURRENT_PLATFORM == Platform.MACOS:
        return _mac_foreground()
    if CURRENT_PLATFORM == Platform.LINUX:
        if wayland():
            found = _atspi_foreground()
            if found is None:
                raise Unobservable(WAYLAND_FOREGROUND)
            return found
        if not os.environ.get("DISPLAY"):
            raise Unobservable("there is no graphical session (DISPLAY is not set)")
        return _x11_foreground()
    raise Unobservable(f"the window in front cannot be read on {sys.platform}")


async def foreground() -> Foreground:
    return await _threaded(read_foreground_now)


def title_pids_now(title: str) -> frozenset[int]:
    """Processes with a visible window whose title contains ``title`` (Windows only; empty elsewhere)."""
    if CURRENT_PLATFORM != Platform.WINDOWS or not title:
        return frozenset()
    wanted = title.casefold()
    return frozenset(pid for _h, pid, text, _m, _x in _win_windows() if wanted in text.casefold())


# ── pointer, buttons and keys ──

_WIN_BUTTONS = {"left": 0x01, "right": 0x02, "middle": 0x04}
_WIN_KEYS = {
    "shift": 0x10,
    "shiftleft": 0xA0,
    "shiftright": 0xA1,
    "ctrl": 0x11,
    "control": 0x11,
    "ctrlleft": 0xA2,
    "ctrlright": 0xA3,
    "alt": 0x12,
    "option": 0x12,
    "altleft": 0xA4,
    "altright": 0xA5,
    "win": 0x5B,
    "winleft": 0x5B,
    "winright": 0x5C,
    "command": 0x5B,
    "cmd": 0x5B,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "del": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "capslock": 0x14,
}


def windows_virtual_key(name: str) -> int | None:
    """The Windows virtual-key code of a pyautogui key name, or None when this table does not know it. Pure."""
    key = name.strip().casefold()
    if key in _WIN_KEYS:
        return _WIN_KEYS[key]
    if len(key) == 1 and (key.isascii() and key.isalnum()):
        return ord(key.upper())
    match = re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", key)
    if match:
        return 0x70 + int(match.group(1)) - 1
    return None


_MAC_MODIFIER_FLAGS = {
    "shift": 0x20000,
    "shiftleft": 0x20000,
    "shiftright": 0x20000,
    "ctrl": 0x40000,
    "control": 0x40000,
    "ctrlleft": 0x40000,
    "ctrlright": 0x40000,
    "alt": 0x80000,
    "option": 0x80000,
    "altleft": 0x80000,
    "altright": 0x80000,
    "command": 0x100000,
    "cmd": 0x100000,
    "win": 0x100000,
}
_X11_MODIFIER_KEYSYMS = {
    "shift": ("Shift_L", "Shift_R"),
    "shiftleft": ("Shift_L",),
    "shiftright": ("Shift_R",),
    "ctrl": ("Control_L", "Control_R"),
    "control": ("Control_L", "Control_R"),
    "ctrlleft": ("Control_L",),
    "ctrlright": ("Control_R",),
    "alt": ("Alt_L", "Alt_R"),
    "altleft": ("Alt_L",),
    "altright": ("Alt_R",),
    "win": ("Super_L", "Super_R"),
    "winleft": ("Super_L",),
    "winright": ("Super_R",),
    "command": ("Super_L", "Super_R"),
    "enter": ("Return",),
    "return": ("Return",),
    "esc": ("Escape",),
    "escape": ("Escape",),
    "space": ("space",),
    "backspace": ("BackSpace",),
    "delete": ("Delete",),
    "tab": ("Tab",),
    "pageup": ("Prior",),
    "pagedown": ("Next",),
}


def _win_pointer() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    point = wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):  # type: ignore[attr-defined]
        raise Unobservable("Windows did not give the pointer position (GetCursorPos failed)")
    return int(point.x), int(point.y)


def _win_down(codes: dict[str, int]) -> set[str]:
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    return {name for name, code in codes.items() if user32.GetAsyncKeyState(code) & 0x8000}


def _mac_pointer() -> tuple[int, int]:
    try:
        import Quartz
    except ImportError as error:
        raise Unobservable("reading the pointer on macOS needs pyobjc's Quartz framework") from error
    location = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return round(location.x), round(location.y)


def _mac_buttons_down(buttons: list[str]) -> set[str]:
    try:
        import Quartz
    except ImportError as error:
        raise Unobservable("reading the mouse buttons on macOS needs pyobjc's Quartz framework") from error
    numbers = {"left": 0, "right": 1, "middle": 2}
    state = Quartz.kCGEventSourceStateCombinedSessionState
    return {name for name in buttons if Quartz.CGEventSourceButtonState(state, numbers.get(name, 0))}


def _mac_keys_down(keys: list[str]) -> tuple[set[str], list[str]]:
    try:
        import Quartz
    except ImportError as error:
        raise Unobservable("reading held keys on macOS needs pyobjc's Quartz framework") from error
    state = Quartz.kCGEventSourceStateCombinedSessionState
    flags = int(Quartz.CGEventSourceFlagsState(state))
    down: set[str] = set()
    unknown: list[str] = []
    table: dict[str, int] = {}
    with contextlib.suppress(Exception):
        from pyautogui import _pyautogui_osx

        table = dict(_pyautogui_osx.keyboardMapping)
    for key in keys:
        name = key.strip().casefold()
        if name in _MAC_MODIFIER_FLAGS:
            if flags & _MAC_MODIFIER_FLAGS[name]:
                down.add(key)
        elif name in table and table[name] is not None:
            if Quartz.CGEventSourceKeyState(state, int(table[name])):
                down.add(key)
        else:
            unknown.append(key)
    return down, unknown


def _x11_display() -> Any:
    try:
        from Xlib import display as xdisplay
    except ImportError as error:
        raise Unobservable("reading the X11 pointer and keys needs python-xlib") from error
    try:
        return xdisplay.Display()
    except Exception as error:
        raise Unobservable(f"the X display could not be reached ({type(error).__name__})") from error


def _x11_pointer_and_buttons() -> tuple[tuple[int, int], set[str]]:
    display = _x11_display()
    try:
        reply = display.screen().root.query_pointer()
        masks = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}
        return (int(reply.root_x), int(reply.root_y)), {name for name, bit in masks.items() if reply.mask & bit}
    finally:
        display.close()


def _x11_keys_down(keys: list[str]) -> tuple[set[str], list[str]]:
    display = _x11_display()
    try:
        from Xlib import XK

        keymap = display.query_keymap()
        down: set[str] = set()
        unknown: list[str] = []
        for key in keys:
            name = key.strip()
            symbols = _X11_MODIFIER_KEYSYMS.get(name.casefold(), (name,))
            codes = [display.keysym_to_keycode(XK.string_to_keysym(symbol)) for symbol in symbols]
            codes = [code for code in codes if code]
            if not codes:
                unknown.append(key)
                continue
            if any(keymap[code // 8] & (1 << (code % 8)) for code in codes):
                down.add(key)
        return down, unknown
    finally:
        display.close()


def read_pointer_now() -> tuple[int, int]:
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return _win_pointer()
    if CURRENT_PLATFORM == Platform.MACOS:
        return _mac_pointer()
    if CURRENT_PLATFORM == Platform.LINUX:
        if wayland():
            raise Unobservable(WAYLAND_POINTER)
        location = parse_xdotool_location(_run(["xdotool", "getmouselocation", "--shell"]) or "")
        if location is not None:
            return location
        return _x11_pointer_and_buttons()[0]
    raise Unobservable(f"the pointer cannot be read on {sys.platform}")


def read_buttons_down_now(buttons: list[str]) -> set[str]:
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return _win_down({name: _WIN_BUTTONS.get(name, 0x01) for name in buttons})
    if CURRENT_PLATFORM == Platform.MACOS:
        return _mac_buttons_down(buttons)
    if CURRENT_PLATFORM == Platform.LINUX:
        if wayland():
            raise Unobservable(WAYLAND_POINTER)
        return _x11_pointer_and_buttons()[1] & set(buttons)
    raise Unobservable(f"the mouse buttons cannot be read on {sys.platform}")


def read_keys_down_now(keys: list[str]) -> tuple[set[str], list[str]]:
    """(keys of ``keys`` held down now, keys whose state this platform's table cannot read)."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        codes = {key: windows_virtual_key(key) for key in keys}
        known = {key: code for key, code in codes.items() if code is not None}
        return _win_down(known), [key for key, code in codes.items() if code is None]
    if CURRENT_PLATFORM == Platform.MACOS:
        return _mac_keys_down(keys)
    if CURRENT_PLATFORM == Platform.LINUX:
        if wayland():
            raise Unobservable(WAYLAND_KEYS)
        return _x11_keys_down(keys)
    raise Unobservable(f"held keys cannot be read on {sys.platform}")


async def pointer() -> tuple[int, int]:
    return await _threaded(read_pointer_now)


async def buttons_down(buttons: list[str]) -> set[str]:
    return await _threaded(lambda: read_buttons_down_now(buttons))


async def keys_down(keys: list[str]) -> tuple[set[str], list[str]]:
    return await _threaded(lambda: read_keys_down_now(keys))


# ── the clipboard ──


def _win_clipboard_text() -> str:
    import ctypes
    import time

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    for _attempt in range(10):  # another program may hold the clipboard open for a moment
        if user32.OpenClipboard(None):
            break
        time.sleep(0.02)
    else:
        raise Unobservable("another program kept the clipboard open, so it could not be read")
    try:
        if not user32.IsClipboardFormatAvailable(13):  # CF_UNICODETEXT
            return ""
        handle = user32.GetClipboardData(13)
        if not handle:
            return ""
        locked = kernel32.GlobalLock(handle)
        if not locked:
            raise Unobservable("the clipboard's text could not be locked for reading")
        try:
            return ctypes.wstring_at(locked)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def read_clipboard_now() -> tuple[str, str]:
    """(clipboard text, the tool that read it)."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return _win_clipboard_text(), "Win32 GetClipboardData"
    if CURRENT_PLATFORM == Platform.MACOS:
        text = _run(["pbpaste"], utf8_locale=True)
        if text is None:
            raise Unobservable("pbpaste did not answer")
        return text, "pbpaste"
    if CURRENT_PLATFORM == Platform.LINUX:
        if wayland():
            raise Unobservable(WAYLAND_CLIPBOARD)
        for args in (["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
            text = _run(args)
            if text is not None:
                return text, args[0]
        raise Unobservable("reading the X11 clipboard needs xclip or xsel, and neither answered")
    raise Unobservable(f"the clipboard cannot be read on {sys.platform}")


async def clipboard_text() -> tuple[str, str]:
    return await _threaded(read_clipboard_now)


def clipboard_holds(found: str, expected: str) -> str | None:
    """ "exact", "line endings" (equal once CR LF and a final newline are set aside) or None. Pure."""
    if found == expected:
        return "exact"
    if normalise_newlines(found).rstrip("\n") == normalise_newlines(expected).rstrip("\n"):
        return "line endings"
    return None


# ── the focused field ──


_FIELD_READER: Any = None

FIELD_NOT_READ = "the window was right; the field's content was not read (Live's text reader was not running)"


def set_field_reader(reader: Any) -> None:
    """Live's focused-text reader, when the daemon has one; its UI Automation helper is reused while it runs."""
    global _FIELD_READER
    _FIELD_READER = reader


def _running_helper(reader: Any) -> bool:
    process = getattr(reader, "_proc", None)
    return process is not None and getattr(process, "returncode", 0) is None


async def _win_field() -> FieldReading:
    """The focused field on Windows, only when it is cheap: through Live's UI Automation helper while it is running
    (a read costs milliseconds). Starting a helper for one read takes one to three seconds, which would hold up the
    next step, so without one the field is not read."""
    reader = _FIELD_READER
    if reader is None or not _running_helper(reader):
        raise Unobservable(FIELD_NOT_READ)
    found = await reader.read()
    if found is None:
        raise Unobservable("UI Automation did not describe the focused element")
    return FieldReading(found.editable, found.password, found.text if found.editable else None, "UI Automation")


def _mac_field() -> FieldReading:
    from pilot.live import focused_text_macos as mac

    with mac._lock:
        reader = mac.shared()
        if reader is None:
            raise Unobservable(mac._error or "the Accessibility API cannot be used")
        if not reader.gate.allowed():  # checks only; never prompts
            raise Unobservable("not verified: the Accessibility permission is not granted to Heliox")
        backend = reader.backend
        element = backend.focused()
        if element is None:
            raise Unobservable("the app in front does not expose its focused element to the Accessibility API")
        role = str(backend.attr(element, "AXRole") or "")
        subrole = str(backend.attr(element, "AXSubrole") or "")
        pid = backend.pid(element)
        settable = backend.settable(element, "AXValue")
        secure, kind = mac.classify(
            role, subrole, settable, backend.bundle_id(pid) if pid else "", mac.secure_input_enabled()
        )
        text = backend.text_tail(element) if kind and not secure else None
    return FieldReading(bool(kind) and not secure, secure, text, "the Accessibility API")


def _linux_field() -> FieldReading:
    from pilot.live.focused_text_linux import read_focused

    found = read_focused()
    if found is None:
        raise Unobservable("AT-SPI did not describe the focused element (accessibility may be off in this session)")
    return FieldReading(found.editable, found.password, found.text if found.editable else None, "AT-SPI")


async def focused_field(front: Foreground | None) -> FieldReading:
    """The focused element, read through the platform's accessibility interface. Refuses windows Live never reads."""
    if front is not None:
        from pilot.live.focused_text_policy import may_touch

        name = next((n for n in front.names if n), "")
        if not may_touch(name, front.title or ""):
            raise Unobservable(NEVER_READ_WINDOW)
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return await _win_field()
    if CURRENT_PLATFORM == Platform.MACOS:
        return await _threaded(_mac_field)
    if CURRENT_PLATFORM == Platform.LINUX:
        return await _threaded(_linux_field)
    raise Unobservable(f"the focused field cannot be read on {sys.platform}")


def field_holds(field_text: str, typed: str) -> dict[str, Any]:
    """Whether a field's text holds what was typed: yes/no answers and lengths only. Pure."""
    text = normalise_newlines(field_text)
    wanted = normalise_newlines(typed)
    if len(wanted) > len(text) and len(text) >= 1000:
        # The reader gives the tail of a long field; compare the end of what was typed with it.
        wanted = wanted[-(len(text) // 2) :]
    return {
        "field_length": len(field_text),
        "typed_length": len(typed),
        "contains": bool(wanted) and wanted in text,
        "ends_with": bool(wanted) and text.rstrip("\n").endswith(wanted.rstrip("\n")),
    }


# ── window state and the window list ──


def _mac_window_states(target: str) -> list[tuple[bool, bool]]:
    """(minimised, fills the screen's visible area) for each window of the app called ``target``."""
    from pilot.system import macos_windows

    windows = macos_windows.available()
    if windows is None:
        raise Unobservable("reading window state on macOS needs pyobjc")
    backend = windows.b
    if not backend.trusted():
        raise Unobservable("not verified: the Accessibility permission is not granted to Heliox")
    apps = macos_windows.match_apps(target, backend.running_apps())
    if not apps:
        owner = None
        with contextlib.suppress(Exception):
            owner = windows.owner_of(target)
        apps = macos_windows.match_apps(owner, backend.running_apps()) if owner else []
    visible = backend.visible_frame()
    states = []
    for app in apps:
        for window in backend.windows(app.pid):
            minimised = bool(backend.get(window, "AXMinimized"))
            frame = _mac_frame(backend, window)
            states.append((minimised, frame is not None and frames_match(frame, visible)))
    return states


def _mac_frame(backend: Any, window: Any) -> tuple[float, float, float, float] | None:
    ax = backend.ax
    position = backend.get(window, "AXPosition")
    size = backend.get(window, "AXSize")
    if position is None or size is None:
        return None
    ok_point, point = ax.AXValueGetValue(position, ax.kAXValueCGPointType, None)
    ok_size, extent = ax.AXValueGetValue(size, ax.kAXValueCGSizeType, None)
    if not (ok_point and ok_size):
        return None
    return (float(point.x), float(point.y), float(extent.width), float(extent.height))


def frames_match(frame: tuple[float, ...], visible: tuple[float, ...], tolerance: float = 24.0) -> bool:
    """A window fills the visible area when each edge is within ``tolerance`` points (apps snap to their own
    size steps, a terminal to whole character cells). Pure."""
    return all(abs(a - b) <= tolerance for a, b in zip(frame, visible, strict=True))


def window_states_now(target: str) -> list[tuple[bool, bool]]:
    """(minimised, maximised) for every window matching ``target`` by title, else by process name."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        rows = _win_windows()
        wanted = target.casefold()
        titled = [row for row in rows if wanted in row[2].casefold()]
        if not titled:
            titled = [row for row in rows if bare_name(process_name(row[1])) == bare_name(target)]
        return [(row[3], row[4]) for row in titled]
    if CURRENT_PLATFORM == Platform.MACOS:
        return _mac_window_states(target)
    if CURRENT_PLATFORM == Platform.LINUX:
        listing = _run(["wmctrl", "-lp"])
        if listing is None:
            reason = "native Wayland windows are not listed to other programs; " if wayland() else ""
            raise Unobservable(f"{reason}reading window state needs wmctrl, and it did not answer")
        wanted = target.casefold()
        states = []
        for window, _pid, title in parse_wmctrl_lp(listing):
            if wanted not in title.casefold():
                continue
            atoms = parse_xprop_states(_run(["xprop", "-id", hex(window), "_NET_WM_STATE"]) or "")
            states.append(
                (
                    "_NET_WM_STATE_HIDDEN" in atoms,
                    {"_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ"} <= atoms,
                )
            )
        if not states and wayland():
            raise Unobservable("native Wayland windows are not listed to other programs (only XWayland ones are)")
        return states
    raise Unobservable(f"window state cannot be read on {sys.platform}")


async def window_states(target: str) -> list[tuple[bool, bool]]:
    return await _threaded(lambda: window_states_now(target))


def window_rows_now() -> list[WindowRow]:
    """Every application window on screen with its process: (pid, process or app name, title)."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return [WindowRow(pid, process_name(pid), title, handle) for handle, pid, title, _m, _x in _win_windows()]
    if CURRENT_PLATFORM == Platform.MACOS:
        try:
            import Quartz
        except ImportError as error:
            raise Unobservable("listing windows on macOS needs pyobjc's Quartz framework") from error
        options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        rows = []
        for info in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
            if int(info.get("kCGWindowLayer", 1)) != 0:
                continue
            pid = int(info.get("kCGWindowOwnerPID", 0))
            rows.append(
                WindowRow(
                    pid,
                    str(info.get("kCGWindowOwnerName", "")),
                    str(info.get("kCGWindowName", "")),
                    int(info.get("kCGWindowNumber", 0)),
                )
            )
        return rows
    if CURRENT_PLATFORM == Platform.LINUX:
        listing = _run(["wmctrl", "-lp"])
        if listing is None:
            reason = "native Wayland windows are not listed to other programs, and " if wayland() else ""
            raise Unobservable(f"{reason}listing windows needs wmctrl, which did not answer")
        return [
            WindowRow(pid, _full_name(pid) if pid else "", title, window)
            for window, pid, title in parse_wmctrl_lp(listing)
        ]
    raise Unobservable(f"windows cannot be listed on {sys.platform}")


async def window_rows() -> list[WindowRow]:
    return await _threaded(window_rows_now)


# ── the checks the input functions run right after a real step ──


def _record(outcome: str, detail: str, **facts: Any) -> StepRecord | None:
    action = current_step()
    if action is None:
        return None
    record = StepRecord(action, outcome, detail, facts)
    remember(record)
    return record


def _stop(detail: str, sentence: str, **facts: Any) -> None:
    """Record the mismatch and fail the step with one plain sentence for the person. When the step is one whose input
    may now sit in another window (keys, a click, a drag), the record says so and the run stops instead of
    retrying."""
    from pilot.actions import RECONCILE_NOT_RETRY_ACTIONS

    action = current_step()
    elsewhere = getattr(action, "action_type", None) in RECONCILE_NOT_RETRY_ACTIONS
    _record("mismatch", detail, elsewhere=elsewhere, **facts)
    raise DesktopStepMismatch(sentence)


def input_went_elsewhere(results: list[Any]) -> str | None:
    """The sentence of the first failed step whose keys, click or drag may have gone to another window than the one
    meant, read from its own record; None when there is none. Such a step is not retried: the run stops and the
    person is told what was in front instead."""
    from pilot.actions import RECONCILE_NOT_RETRY_ACTIONS

    for result in results:
        if result.success or result.action.action_type not in RECONCILE_NOT_RETRY_ACTIONS:
            continue
        record = recalled(result.action)
        if record is not None and record.outcome == "mismatch" and record.facts.get("elsewhere"):
            return str(result.error or record.detail)
    return None


async def refuse_unless_in_front(target_window: str) -> Foreground | None:
    """Before the first key of a step that names its window: when another window is read in front, no key is sent
    and the step fails with that reason. Returns the reading (None when it cannot be made: then the keys go)."""
    if current_step() is None or not target_window:
        return None
    try:
        found = await foreground()
        pids = await _safe_title_pids(target_window)
    except Exception:
        return None
    by_title = focus_matches(found, title=target_window, title_pids=pids)
    by_name = focus_matches(found, process_name=target_window)
    if by_title is False and by_name is False:
        detail = f"Before any key: expected {target_window} in front; {found.describe()} is in front ({found.source})"
        _record("mismatch", detail, elsewhere=False, refused=True)
        raise DesktopStepMismatch(f"No key was sent: {found.app()} is in front, not {target_window}.")
    return found


async def _settle_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


_sleep: Callable[[float], Any] = _settle_sleep


async def confirm_focus(*, window_id: str | None, title: str | None, process_name: str | None) -> None:
    """After a focus request: read the window in front until it is the one asked for (or the settle time is
    over). A different window in front fails the step; a window that cannot be read is recorded, not failed."""
    if current_step() is None:
        return
    try:
        title_pids = await _threaded(lambda: title_pids_now(title or "")) if title and not process_name else frozenset()
        found = await foreground()
        waited = 0.0
        while (
            focus_matches(found, window_id=window_id, title=title, process_name=process_name, title_pids=title_pids)
            is False
            and waited < FOCUS_SETTLE_SECONDS
        ):
            await _sleep(SETTLE_STEP_SECONDS)
            waited += SETTLE_STEP_SECONDS
            found = await foreground()
    except Unobservable as reason:
        _record("unobservable", str(reason), kind="not_observable")
        return
    except Exception as error:  # an observer that broke never fails the step
        logger.debug("Focus read failed", exc_info=True)
        _record(
            "unobservable", f"the window in front could not be read ({type(error).__name__})", kind="could_not_look"
        )
        return
    verdict = focus_matches(found, window_id=window_id, title=title, process_name=process_name, title_pids=title_pids)
    wanted = title or process_name or window_id or ""
    if verdict is None:
        _record(
            "unobservable",
            f"{found.describe()} is in front; its {_unread_part(found)} cannot be read here",
            kind="not_observable",
        )
        return
    if verdict:
        _record("match", f"{found.describe()} is in front ({found.source})", source=found.source)
        return
    detail = f"Expected {wanted} in front; {found.describe()} is in front ({found.source})"
    _record("mismatch", detail, source=found.source)
    raise DesktopStepMismatch(
        f"The focus did not move to {wanted}: {found.app()} is in front. The steps after this one were not run, "
        "so no keys went to the wrong window."
    )


def _unread_part(found: Foreground) -> str:
    return "title" if found.title is None else "window id"


async def typing_started() -> Foreground | None:
    """The window in front before typing, or None when it cannot be read."""
    if current_step() is None:
        return None
    try:
        return await foreground()
    except Exception:
        return None


# The Windows shell's own windows: typing into the Start menu or the taskbar's search box moves the focus from
# StartMenuExperienceHost (or explorer, for the taskbar) to SearchHost as part of normal use, and the emoji panel and
# input methods show TextInputHost. A move among these is not "the window changed while typing".
SHELL_SEARCH_HOSTS = frozenset(
    {"searchhost", "searchapp", "searchui", "startmenuexperiencehost", "shellexperiencehost", "textinputhost"}
)


def same_place(before: Foreground, after: Foreground) -> bool:
    """Whether typing stayed where it started: the same window, a window of the same process (a dialog, a search
    results pop-up), or the shell's own search windows. Pure."""
    if before.same_window(after):
        return True
    if before.pid and before.pid == after.pid:
        return True
    names_before = {bare_name(name) for name in before.names if name}
    names_after = {bare_name(name) for name in after.names if name}
    if names_after & SHELL_SEARCH_HOSTS and names_before & (SHELL_SEARCH_HOSTS | {"explorer"}):
        return True
    return False


async def confirm_typing(text: str, before: Foreground | None, *, target_window: str | None, background: bool) -> None:
    """After typing: the window that had the focus still has it (or the named one, when given), and, where the
    focused field shows its text, the field holds what was typed. Only lengths and yes/no answers are kept.

    Only the window in front having changed stops the run. A field that does not show the text yet is read again a
    few times over about 300 ms (Chromium and Electron apply a paste after it returns), and if it still does not
    show it, the step is "not verified", never stopped: the run goes on to its Enter."""
    if current_step() is None:
        return
    if background and target_window:
        # The handler's own UI Automation call set the editor's text and compared it; reading it back in another
        # PowerShell process would add half a second to every such step, so it is not read again here.
        _record(
            "partial",
            f"The text was set in {target_window}'s editor through UI Automation, which compared it in the same call; "
            "it was not read back separately",
            typed_length=len(text),
            kind="not_observable",
        )
        return
    try:
        after = await foreground()
    except Exception as error:
        reason = (
            str(error)
            if isinstance(error, Unobservable)
            else f"the window in front could not be read ({type(error).__name__})"
        )
        _record(
            "unobservable",
            reason,
            typed_length=len(text),
            kind="not_observable" if isinstance(error, Unobservable) else "could_not_look",
        )
        return
    ends_with_enter = "\n" in text or "\r" in text
    window_ok = True
    expected = ""
    if target_window:
        window_ok = bool(focus_matches(after, title=target_window, title_pids=await _safe_title_pids(target_window)))
        window_ok = window_ok or bool(focus_matches(after, process_name=target_window))
        expected = target_window
    elif before is not None:
        window_ok = same_place(before, after)
        expected = before.app()
    if not window_ok and not ends_with_enter:
        detail = (
            f"The window in front changed while typing {len(text)} characters: expected {expected}, found {after.app()}"
        )
        meant = target_window or (before.app() if before is not None else expected)
        _stop(detail, f"The text may have been typed into {after.app()} instead of {meant}.", typed_length=len(text))
    if not window_ok:
        _record(
            "partial",
            f"After typing text that contains Enter, {after.app()} is in front (Enter often opens or closes a window)",
            typed_length=len(text),
            kind="changed_later",
        )
        return
    reading = None
    try:
        for attempt in range(FIELD_READS):
            if attempt:
                await _sleep(FIELD_REREAD_SECONDS)
            reading = await focused_field(after)
            if reading.secure or not reading.editable or reading.text is None:
                break
            if field_holds(reading.text, text)["contains"]:
                break
    except Unobservable as reason:
        _record(
            "partial",
            f"{after.app()} kept the focus; the field's text was not read: {reason}",
            typed_length=len(text),
            kind="not_observable",
        )
        return
    except Exception as error:
        logger.debug("Field read failed", exc_info=True)
        _record(
            "partial",
            f"{after.app()} kept the focus; the field could not be read ({type(error).__name__})",
            typed_length=len(text),
            kind="could_not_look",
        )
        return
    if reading is not None:
        _record_field(after, reading, text)


FIELD_READS = 4  # the first read and three more
FIELD_REREAD_SECONDS = 0.1


def _record_field(after: Foreground, reading: FieldReading, text: str) -> None:
    where = after.app()
    if reading.secure:
        _record(
            "partial",
            f"{where} kept the focus; the focused field is a password field, which is never read",
            typed_length=len(text),
            kind="not_observable",
        )
        return
    if not reading.editable or reading.text is None:
        _record(
            "partial",
            f"{where} kept the focus; the focused element does not expose its text through {reading.source}",
            typed_length=len(text),
            kind="not_observable",
        )
        return
    facts = field_holds(reading.text, text)
    if facts["contains"]:
        _record(
            "match", f"The focused field in {where} holds the {len(text)} typed characters ({reading.source})", **facts
        )
        return
    # Empty or different, with the same window in front: the paste may land later, a reader may see an unbuilt
    # accessibility tree, or the app reformatted the text. None of that is evidence the keys went elsewhere.
    _record(
        "partial",
        f"{where} kept the focus; its field ({facts['field_length']} characters) does not show the {len(text)} typed "
        f"characters as typed, which apps that paste late, complete or reformat text also cause ({reading.source})",
        kind="not_observable",
        **facts,
    )


async def _safe_title_pids(title: str) -> frozenset[int]:
    try:
        return await _threaded(lambda: title_pids_now(title))
    except Exception:
        return frozenset()


def pointer_off_target_now(target: tuple[int, int]) -> tuple[int, int] | None:
    """Right at a click, between moving the pointer and pressing the button (blocking): where the pointer is when
    that is not the target, else None (also when it cannot be read)."""
    if current_step() is None:
        return None
    try:
        found = read_pointer_now()
    except Exception:
        return None
    return None if pointer_at(found, target) else found


def refuse_click(found: tuple[int, int], target: tuple[int, int]) -> None:
    """The pointer was not at the target when the button would have been pressed: nothing was clicked."""
    _record("mismatch", f"Before the click: the pointer was at {found}, not at {target}", elsewhere=False, refused=True)
    raise DesktopStepMismatch(f"No click was made: the pointer was at {found}, not at {target}.")


async def confirm_pointer(target: tuple[int, int], *, released: tuple[str, ...] = ()) -> None:
    """After a pointer step: the pointer is where it was sent, and the buttons it pressed are up again. A pointer
    elsewhere after the event is "not verified" (the person may have moved the mouse; a click is refused before the
    button is pressed when the pointer is not at its target). A button still held stops the run."""
    if current_step() is None:
        return
    try:
        found = await pointer()
        held = await buttons_down(list(released)) if released else set()
    except Unobservable as reason:
        _record("unobservable", str(reason), target=list(target), kind="not_observable")
        return
    except Exception as error:
        logger.debug("Pointer read failed", exc_info=True)
        _record(
            "unobservable",
            f"the pointer could not be read ({type(error).__name__})",
            target=list(target),
            kind="could_not_look",
        )
        return
    facts = {"target": list(target), "found": list(found), "held": sorted(held)}
    if held:
        buttons = ", ".join(sorted(held))
        detail = f"The {buttons} mouse button is still held down after the step"
        _stop(detail, f"{detail}, so whatever the pointer touches next could be dragged.", **facts)
    if not pointer_at(found, target):
        _record(
            "partial",
            f"The pointer was at {found} after the step, not at {target}; the mouse may have been moved since",
            kind="changed_later",
            **facts,
        )
        return
    _record("match", f"The pointer is at {found}" + (", buttons released" if released else ""), **facts)


async def confirm_keys_released(keys: list[str]) -> None:
    """After a key step: none of its keys is still held down (a stuck modifier changes every key after it). Keys
    whose state no table here knows (media keys) are left out; a step of only such keys records nothing."""
    if current_step() is None:
        return
    try:
        held, unknown = await keys_down(keys)
    except Unobservable as reason:
        _record("unobservable", str(reason), keys=list(keys), kind="not_observable")
        return
    except Exception as error:
        logger.debug("Key state read failed", exc_info=True)
        _record(
            "unobservable",
            f"the key state could not be read ({type(error).__name__})",
            keys=list(keys),
            kind="could_not_look",
        )
        return
    facts = {"keys": list(keys), "held": sorted(held), "unknown": list(unknown)}
    if held:
        detail = f"{'+'.join(sorted(held))} is still held down after the step"
        _stop(detail, f"{detail}, so every key after it would change.", **facts)
    if unknown and len(unknown) == len(keys):
        return
    _record("match", f"{'+'.join(k for k in keys if k not in unknown)} released", **facts)


async def confirm_clipboard(content: str) -> None:
    """After a clipboard write: read the clipboard back through a different tool and compare by hash."""
    if current_step() is None:
        return
    facts: dict[str, Any] = {"expected_sha256": digest(content), "expected_length": len(content)}
    try:
        found, tool = await clipboard_text()
    except Unobservable as reason:
        _record("unobservable", str(reason), kind="not_observable", **facts)
        return
    except Exception as error:
        logger.debug("Clipboard read failed", exc_info=True)
        _record(
            "unobservable", f"the clipboard could not be read ({type(error).__name__})", kind="could_not_look", **facts
        )
        return
    holds = clipboard_holds(found, content)
    if holds is None:
        detail = f"The clipboard holds {len(found)} characters that differ from the {len(content)} copied ({tool})"
        _record("mismatch", detail, found_length=len(found), **facts)
        raise DesktopStepMismatch(f"{detail}; the steps after this one were not run, so nothing stale was pasted.")
    note = "" if holds == "exact" else ", once line endings are set aside"
    _record("match", f"The clipboard holds the {len(content)} copied characters ({tool}{note})", **facts)


# ── open_application: the window list before the launch, the program the launcher started ──


@dataclass
class _TaskLaunches:
    """One task's open_application steps: the steps that ran the launch check, the ones whose launch succeeded,
    and how many successful launches each app had."""

    seen: list[Any] = field(default_factory=list)
    launched: list[Any] = field(default_factory=list)
    made: dict[str, int] = field(default_factory=dict)


_TASK_LAUNCHES: OrderedDict[str, _TaskLaunches] = OrderedDict()
_MAX_TASKS = 32


def _merge_facts(**facts: Any) -> None:
    action = current_step()
    if action is None:
        return
    record = recalled(action)
    if record is None:
        record = StepRecord(action, "launch", "", {})
        remember(record)
    record.facts.update(facts)


async def windows_before_launch() -> None:
    """Before a launch: the ids of the windows on screen, so a new one is evidence whatever its title."""
    if current_step() is None:
        return
    try:
        rows = await window_rows()
    except Exception:
        return
    _merge_facts(windows_before=[row.window_id for row in rows if row.window_id])


def note_launch(program: str, pid: int = 0) -> None:
    """A program a launcher started (an executable, a Store app id, a bundle, a desktop entry) and its process id
    when it is the app's own, for the check. Called once or more per launch."""
    action = current_step()
    if action is None or not program:
        return
    record = recalled(action)
    programs = list(record.facts.get("programs", [])) if record is not None else []
    if program not in programs:
        programs.append(program)
    facts: dict[str, Any] = {"programs": programs}
    if pid:
        facts["pid"] = int(pid)
    _merge_facts(**facts)


def launch_allowed(task: str, action: Any, name: str, wanted: int) -> bool:
    """Whether this open_application step may launch, so a repair round never opens an app a second time while a
    plan that asks for two launches gets two.

    A step is the same step again when it is the same ``Action`` object (a partial repair round re-issues the very
    objects that failed): it launches only if it has not launched yet. A full re-plan makes new objects, and nothing
    identifies "the same step" across it, so a new step launches only while this task has made fewer successful
    launches of the app than the running plan asks for (``wanted``). Counted, never timed."""
    if not task:
        return True
    entry = _TASK_LAUNCHES.get(task)
    if entry is None:
        entry = _TaskLaunches()
        _TASK_LAUNCHES[task] = entry
        while len(_TASK_LAUNCHES) > _MAX_TASKS:
            _TASK_LAUNCHES.popitem(last=False)
    if any(step is action for step in entry.launched):
        return False
    if any(step is action for step in entry.seen):
        return True
    entry.seen.append(action)
    return entry.made.get(bare_name(name), 0) < max(1, wanted)


def launch_done(task: str, action: Any, name: str) -> None:
    """The step's launcher succeeded."""
    entry = _TASK_LAUNCHES.get(task)
    if not task or entry is None:
        return
    entry.launched.append(action)
    key = bare_name(name)
    entry.made[key] = entry.made.get(key, 0) + 1

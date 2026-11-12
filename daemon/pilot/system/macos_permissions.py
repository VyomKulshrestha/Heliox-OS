"""The macOS privacy permissions Heliox needs, read without ever showing a prompt.

Each check calls the system's own non-prompting query through ctypes, so it needs no extra package:

- microphone: ``AVCaptureDevice authorizationStatusForMediaType:`` (AVFoundation);
- accessibility: ``AXIsProcessTrusted`` (ApplicationServices), used to read the focused field and window titles;
- screen recording: ``CGPreflightScreenCaptureAccess`` (CoreGraphics, macOS 11 and later), used by screen reading;
- automation of System Events: ``AEDeterminePermissionToAutomateTarget`` with ``askUserIfNeeded`` false, used by
  the window actions that still go through AppleScript.

Which process holds them. The app starts the daemon as its child process, and macOS attributes a child's
requests to the app that is responsible for it, so every permission is granted to "Heliox-OS" in System Settings,
not to Python. A daemon started from a terminal during development is attributed to that terminal instead, and a
daemon the app found already running keeps whatever started it. Grants are tied to the app's code signature: a
rebuilt app with an ad-hoc signature may need to be granted again.

``snapshot`` gives the app one status to show. ``microphone_state`` is the small interface the audio side uses.
Off macOS every state is "cannot_tell" and nothing is loaded.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger("pilot.system.macos_permissions")


class PermissionState(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    NOT_ASKED = "not_asked"
    CANNOT_TELL = "cannot_tell"


SETTINGS = "x-apple.systempreferences:com.apple.preference.security?"
PANES = {
    "microphone": SETTINGS + "Privacy_Microphone",
    "accessibility": SETTINGS + "Privacy_Accessibility",
    "screen_recording": SETTINGS + "Privacy_ScreenCapture",
    "automation": SETTINGS + "Privacy_Automation",
}
WHAT_FOR = {
    "microphone": 'hear you say "Hey Heliox" and what you ask it',
    "accessibility": "read the field you are typing in and the name of the window in front",
    "screen_recording": "read text on your screen when you ask it to",
    "automation": "move, focus and close windows through System Events",
}
LABELS = {
    "microphone": "Microphone",
    "accessibility": "Accessibility",
    "screen_recording": "Screen & System Audio Recording",
    "automation": "Automation (System Events)",
}

# AVAuthorizationStatus
_AV = {0: PermissionState.NOT_ASKED, 1: PermissionState.DENIED, 2: PermissionState.DENIED, 3: PermissionState.GRANTED}
# OSStatus from AEDeterminePermissionToAutomateTarget
_AE = {0: PermissionState.GRANTED, -1743: PermissionState.DENIED, -1744: PermissionState.NOT_ASKED}


def av_status(code: int) -> PermissionState:
    return _AV.get(int(code), PermissionState.CANNOT_TELL)


def ae_status(code: int) -> PermissionState:
    """-600 (System Events is not running) and anything else unknown cannot be told without launching it."""
    return _AE.get(int(code), PermissionState.CANNOT_TELL)


def trusted_status(trusted: bool | None) -> PermissionState:
    """AXIsProcessTrusted and CGPreflightScreenCaptureAccess only say yes or no: "no" covers both never asked and
    turned off, so it is reported as not asked (the app then explains where to turn it on)."""
    if trusted is None:
        return PermissionState.CANNOT_TELL
    return PermissionState.GRANTED if trusted else PermissionState.NOT_ASKED


def _is_mac() -> bool:
    return sys.platform == "darwin"


def _framework(name: str) -> ctypes.CDLL:
    return ctypes.CDLL(f"/System/Library/Frameworks/{name}.framework/{name}")


def microphone_state() -> PermissionState:
    """Whether the microphone is granted to Heliox, without prompting. Off macOS: "cannot_tell"."""
    if not _is_mac():
        return PermissionState.CANNOT_TELL
    try:
        objc = ctypes.CDLL(ctypes.util.find_library("objc") or "/usr/lib/libobjc.A.dylib")
        avf = _framework("AVFoundation")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = objc.objc_msgSend
        send.restype = ctypes.c_long
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        media = ctypes.c_void_p.in_dll(avf, "AVMediaTypeAudio")
        cls = objc.objc_getClass(b"AVCaptureDevice")
        sel = objc.sel_registerName(b"authorizationStatusForMediaType:")
        if not cls or not sel or not media.value:
            return PermissionState.CANNOT_TELL
        return av_status(send(cls, sel, media))
    except Exception:
        logger.debug("Microphone permission check failed", exc_info=True)
        return PermissionState.CANNOT_TELL


def accessibility_trusted() -> bool | None:
    if not _is_mac():
        return None
    try:
        fn = _framework("ApplicationServices").AXIsProcessTrusted
        fn.restype = ctypes.c_bool
        fn.argtypes = []
        return bool(fn())
    except Exception:
        logger.debug("Accessibility permission check failed", exc_info=True)
        return None


def accessibility_state() -> PermissionState:
    return trusted_status(accessibility_trusted())


def screen_recording_state() -> PermissionState:
    if not _is_mac():
        return PermissionState.CANNOT_TELL
    try:
        fn = _framework("CoreGraphics").CGPreflightScreenCaptureAccess
        fn.restype = ctypes.c_bool
        fn.argtypes = []
        return trusted_status(bool(fn()))
    except Exception:  # missing before macOS 11
        return PermissionState.CANNOT_TELL


class _AEDesc(ctypes.Structure):
    _fields_ = [("descriptorType", ctypes.c_uint32), ("dataHandle", ctypes.c_void_p)]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


def automation_state(bundle_id: str = "com.apple.systemevents") -> PermissionState:
    """Whether Heliox may send Apple Events to ``bundle_id``, asked with ``askUserIfNeeded`` false."""
    if not _is_mac():
        return PermissionState.CANNOT_TELL
    try:
        core = _framework("CoreServices")
        desc = _AEDesc()
        data = bundle_id.encode("utf-8")
        core.AECreateDesc.restype = ctypes.c_int16
        core.AECreateDesc.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(_AEDesc)]
        if core.AECreateDesc(_fourcc("bund"), data, len(data), ctypes.byref(desc)) != 0:
            return PermissionState.CANNOT_TELL
        try:
            fn = core.AEDeterminePermissionToAutomateTarget
            fn.restype = ctypes.c_int32
            fn.argtypes = [ctypes.POINTER(_AEDesc), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
            wildcard = _fourcc("****")
            return ae_status(fn(ctypes.byref(desc), wildcard, wildcard, False))
        finally:
            core.AEDisposeDesc.argtypes = [ctypes.POINTER(_AEDesc)]
            core.AEDisposeDesc(ctypes.byref(desc))
    except Exception:
        logger.debug("Automation permission check failed", exc_info=True)
        return PermissionState.CANNOT_TELL


CHECKS = {
    "microphone": microphone_state,
    "accessibility": accessibility_state,
    "screen_recording": screen_recording_state,
    "automation": automation_state,
}


@dataclass(frozen=True)
class PermissionReport:
    name: str
    state: PermissionState
    label: str
    what_for: str
    settings_url: str

    @property
    def message(self) -> str:
        """One sentence the app can show, or "" when nothing is needed."""
        if self.state == PermissionState.GRANTED or self.state == PermissionState.CANNOT_TELL:
            return ""
        verb = "was turned off" if self.state == PermissionState.DENIED else "is not granted yet"
        return (
            f"{self.label} {verb}, so Heliox cannot {self.what_for}. Open System Settings > Privacy & Security > "
            f"{self.label} and turn on Heliox-OS."
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "state": self.state.value,
            "label": self.label,
            "message": self.message,
            "settings_url": self.settings_url,
        }


def report(name: str, state: PermissionState) -> PermissionReport:
    return PermissionReport(name, state, LABELS[name], WHAT_FOR[name], PANES[name])


def snapshot() -> list[PermissionReport]:
    """Every permission Heliox uses on macOS, read without a prompt. Empty off macOS."""
    if not _is_mac():
        return []
    return [report(name, check()) for name, check in CHECKS.items()]


def screen_reading_blocked() -> str:
    """The sentence to say instead of reading the screen when macOS only hands over the wallpaper, or "".

    Without Screen Recording, macOS still lets an app capture the screen but shows it only the desktop picture and
    its own windows, so OCR would silently find nothing. The first capture attempt is what makes macOS show its
    prompt, so callers capture first and check after.
    """
    state = screen_recording_state()
    if state in (PermissionState.GRANTED, PermissionState.CANNOT_TELL):
        return ""
    return report("screen_recording", state).message

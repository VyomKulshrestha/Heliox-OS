"""Keeping the microphone and the speakers working through what happens on a real desk.

On 10 November at 15:41:58 a call app took the headset, the capture stream failed, and the listener stayed deaf
without saying so. The watchdog here looks at what the capture backend delivers once a second and decides when to
reopen it: no frames at all (stalled or failed stream), a reported stream error, the system default microphone
changing while the setting is "auto", the configured microphone coming back after a fallback, and minutes of exact
digital zeros (a muted or captured device). Reopening backs off when it keeps failing, and every change is reported
so the voice bar can say what happened instead of going quiet.

The decision logic is pure (no audio packages) so it is tested everywhere; the device probes are Windows-only and
return "" elsewhere, where the watchdog then skips the default-device checks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("pilot.system.audio_health")

STALL_SECONDS = 3.0  # no frames at all for this long: the stream has stopped delivering
SILENT_SECONDS = 30.0  # exact zeros for this long: muted, or another app holds the device


def silence_message(platform: str | None = None) -> str:
    """What to tell the person when the microphone sends only zeros. On macOS that is also what a refused
    microphone permission looks like: the stream opens and every sample is zero."""
    import sys

    if (platform or sys.platform) == "darwin":
        return (
            "The microphone is sending only silence: it may be muted, in use by another app, or Heliox may not "
            "have microphone permission (System Settings, Privacy & Security, Microphone)."
        )
    return "The microphone is sending only silence (muted, or in use by another app)."


RETRY_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0)  # between failed reopen attempts


@dataclass(frozen=True)
class CaptureState:
    """What the capture backend shows at one check."""

    frames: int  # frames delivered since the listener started (keeps counting across reopens)
    error: str  # the backend's last error ("" when none)
    zero_seconds: float  # how long every frame has been exact digital zeros
    active: bool  # a stream is open
    default_name: str = ""  # the system default microphone now ("" when it cannot be read)
    # Its own identifier, which two microphones of the same model do not share ("" when the system gives none; the
    # name is compared then). See default_microphone_identity.
    default_key: str = ""
    auto: bool = True  # the setting follows the system default
    configured_available: bool = False  # the configured microphone is present again (after a fallback)
    using_fallback: bool = False


@dataclass(frozen=True)
class Decision:
    action: str  # "ok", "reopen" or "notice"
    reason: str = ""
    message: str = ""


class CaptureWatchdog:
    """Decides, from successive CaptureState checks, when the microphone needs reopening and what to tell the person.

    Call ``opened(default_name)`` after every successful open, ``check(state)`` about once a second, and
    ``reopen_failed()`` / ``reopen_succeeded()`` after acting on a "reopen".
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_frames = -1
        self._frames_changed_at = clock()
        self._opened_default = ""
        self._opened_key = ""
        self._error_at_open = ""
        self._failures = 0
        self._next_attempt = 0.0
        self._silent_reported = False
        self._last_silent_reopen = float("-inf")  # never (0.0 is a real time just after boot)
        self.problem = ""  # what is wrong now, for the status ("" when healthy)
        self._problem_frames = 0

    def opened(self, default_name: str = "", error: str = "", default_key: str = "") -> None:
        now = self._clock()
        self._opened_default = default_name
        self._opened_key = default_key
        self._error_at_open = error
        self._frames_changed_at = now
        self._last_silent_reopen = now

    def reopen_failed(self) -> None:
        delay = RETRY_SECONDS[min(self._failures, len(RETRY_SECONDS) - 1)]
        self._failures += 1
        self._next_attempt = self._clock() + delay

    def reopen_succeeded(self, default_name: str = "", error: str = "", default_key: str = "") -> None:
        """The stream opened again; the backoff is cleared only once sound actually flows (``recovered``)."""
        self.opened(default_name, error, default_key)

    def _default_changed(self, state: CaptureState) -> bool:
        """By identifier when both readings have one (two microphones of one model share a name), else by name."""
        if state.default_key and self._opened_key:
            return state.default_key != self._opened_key
        return bool(state.default_name and self._opened_default and state.default_name != self._opened_default)

    def recovered(self, state: CaptureState) -> bool:
        """True once, when sound flows again after a reported problem."""
        if not self.problem or not state.active or state.frames <= self._problem_frames + 2:
            return False
        if self.problem in ("silent", "permission") and state.zero_seconds > 0:
            return False
        self.problem = ""
        self._silent_reported = False
        self._failures = 0
        self._next_attempt = 0.0
        return True

    def check(self, state: CaptureState) -> Decision:
        now = self._clock()
        if state.frames != self._last_frames:
            self._last_frames = state.frames
            self._frames_changed_at = now
        if now < self._next_attempt:
            return Decision("ok")

        if not state.active:
            return self._reopen("closed", "The microphone stream closed; reopening it.")
        if state.error and state.error != self._error_at_open:
            return self._reopen("stream_failed", f"The microphone stopped ({state.error}); reopening it.")
        if now - self._frames_changed_at >= STALL_SECONDS:
            return self._reopen("no_audio", "The microphone stopped sending sound; reopening it.")
        if state.auto and self._default_changed(state):
            return self._reopen("default_changed", f"Switching to the new default microphone, {state.default_name}.")
        if state.using_fallback and state.configured_available:
            return self._reopen("device_returned", "The chosen microphone is back; switching to it.")
        if state.zero_seconds >= SILENT_SECONDS:
            if now - self._last_silent_reopen >= SILENT_SECONDS:
                # Another app may have taken the device exclusively: reopening is cheap and sometimes gets it back.
                self._last_silent_reopen = now
                return self._reopen("silent", silence_message())
            if not self._silent_reported:
                self._silent_reported = True
                self.problem = "silent"
                self._problem_frames = self._last_frames
                return Decision("notice", "silent", silence_message())
        return Decision("ok")

    def report_problem(self, reason: str) -> None:
        """A problem found outside these checks (a refused macOS permission): recovered once real sound flows."""
        self.problem = reason
        self._problem_frames = max(0, self._last_frames)

    def _reopen(self, reason: str, message: str) -> Decision:
        if self.problem and self.problem != "silent":
            self.reopen_failed()  # the last reopen brought no sound back: wait longer before the next one
        self.problem = reason
        self._problem_frames = self._last_frames
        return Decision("reopen", reason, message)


# ── device probes (Windows; "" elsewhere) ──


def default_microphone_name() -> str:
    """The system default microphone now, read live through WASAPI ("" when unavailable)."""
    try:
        import soundcard as sc

        microphone = sc.default_microphone()
        return str(getattr(microphone, "name", "") or "")
    except Exception:
        return ""


def default_microphone_key(platform: str | None = None) -> str:
    """A stable identifier of the system default microphone ("" when unavailable).

    soundcard's own documentation and source give: on PulseAudio/PipeWire (Linux) ``id`` is the source name
    ("bluez_input.XX_XX.headset-head-unit"), stable across restarts, and ``name`` its description, which two
    devices can share; on CoreAudio (macOS) ``id`` is the AudioObjectID, which changes when the device is plugged
    in again, and ``name`` is kAudioObjectPropertyName, which is also the name PortAudio's Core Audio host API
    gives the device; on WASAPI (Windows) ``name`` is the endpoint's friendly name, the one the listener opens.
    So: the source name on Linux, the name elsewhere.
    """
    import sys

    try:
        import soundcard as sc

        microphone = sc.default_microphone()
    except Exception:
        return ""
    if (platform or sys.platform).startswith("linux"):
        return str(getattr(microphone, "id", "") or "")
    return str(getattr(microphone, "name", "") or "")


def default_microphone_identity() -> tuple[str, str]:
    """(identifier, name) of the system default microphone now; ("", "") when it cannot be read.

    The identifier tells apart two microphones of the same model, which share a name (two USB microphones, a dock's
    and a headset's of one maker): soundcard's ``id`` is the source name on PulseAudio/PipeWire (Linux), the
    endpoint id on WASAPI (Windows, "{0.0.1.00000000}.{...}") and the AudioObjectID on CoreAudio (macOS). The macOS
    one changes when the device is plugged in again, which only means one reopen more, after a replug that needs one
    anyway; it is never stored (default_microphone_key is the stored one). No identifier: the name stands in.
    """
    try:
        import soundcard as sc

        microphone = sc.default_microphone()
    except Exception:
        return "", ""
    name = str(getattr(microphone, "name", "") or "")
    return str(getattr(microphone, "id", "") or "") or name, name


def pulse_sources() -> list[dict[str, str]]:
    """Linux: the PulseAudio/PipeWire inputs (monitors excluded) as list_audio_input_devices-like entries whose id
    is "pulse::<source name>"; [] elsewhere or when unavailable."""
    import sys

    if not sys.platform.startswith("linux"):
        return []
    try:
        import soundcard as sc

        return [
            {"id": f"pulse::{mic.id}", "name": str(mic.name), "hostapi": "PulseAudio"}
            for mic in sc.all_microphones(include_loopback=False)
        ]
    except Exception:
        return []


def microphone_present(name: str) -> bool:
    if not name:
        return False
    try:
        import soundcard as sc

        return any(mic.name.strip().casefold() == name.strip().casefold() for mic in sc.all_microphones())
    except Exception:
        return False


def default_speaker_name() -> str:
    """The system default output now, read live through WASAPI ("" when unavailable)."""
    try:
        import soundcard as sc

        speaker = sc.default_speaker()
        return str(getattr(speaker, "name", "") or "")
    except Exception:
        return ""

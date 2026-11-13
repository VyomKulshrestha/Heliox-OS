"""How many errors the graphics card's driver reported this week, from the system's own log.

Whisper runs inside the daemon, so a driver that aborts during a CUDA pass takes the daemon with it. A card whose
driver has reported errors recently does not get Whisper (the wake check and the final pass): it runs on the
processor for the session instead (voice._faster_whisper_runtime). A card with a clean record is used exactly as
before. The local voice runs in its own worker, so it keeps the card: a fault there costs at most one sentence.

The count is read once, in the background, bounded by READ_SECONDS, and only where an ordinary user can read it:
    Windows  the System log through wevtutil: nvlddmkm errors (driver faults, "Graphics SM Warp Exception", Xid-like
             153 and 14) and Display 4101/4107 (the driver stopped responding and was reset).
    Linux    the kernel's "NVRM: Xid" lines through journalctl, only when this user may read the kernel journal.
    macOS    nothing to read (no NVIDIA driver).
A read that fails, times out or is refused counts as no record: never as a fault.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

WINDOW_DAYS = 7
LIMIT = 3  # this many driver errors in WINDOW_DAYS (with the GPU arbiter's own faults) keeps Whisper off the card
READ_SECONDS = 5.0
WAIT_SECONDS = 3.0  # how long the first Whisper load waits for the read (it runs in the background at warm-up)

_WINDOWS_QUERY = (
    "*[System[((Provider[@Name='nvlddmkm'] and (Level=1 or Level=2)) or "
    "(Provider[@Name='Display'] and (EventID=4101 or EventID=4107))) and "
    f"TimeCreated[timediff(@SystemTime) <= {WINDOW_DAYS * 86_400_000}]]]"
)

_lock = threading.Lock()
_state: dict[str, Any] = {"thread": None, "count": None, "said": False}
_done = threading.Event()


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=READ_SECONDS, creationflags=creationflags, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def read_count(platform: str | None = None) -> int | None:
    """Driver errors in the last WINDOW_DAYS, or None when they cannot be read (no record)."""
    platform = platform or sys.platform
    if platform == "win32":
        if shutil.which("wevtutil") is None:
            return None
        done = _run(["wevtutil", "qe", "System", f"/q:{_WINDOWS_QUERY}", "/c:1000", "/f:xml"])
        if done is None or done.returncode != 0:
            return None
        return done.stdout.count("<Event ")
    if platform.startswith("linux"):
        if shutil.which("journalctl") is None:
            return None
        done = _run(
            ["journalctl", "-k", "--since", f"-{WINDOW_DAYS}d", "--no-pager", "-q", "-o", "cat", "-g", "NVRM: Xid"]
        )
        if done is None:
            return None
        refused = "insufficient permissions" in (done.stderr or "").lower()
        if refused or done.returncode not in (0, 1):  # 1: nothing matched
            return None
        return len([line for line in done.stdout.splitlines() if "NVRM: Xid" in line])
    return None


def start_reading() -> None:
    """Read the count once, in the background."""
    with _lock:
        if _state["thread"] is not None:
            return

        def read() -> None:
            try:
                _state["count"] = read_count()
            except Exception:
                logger.debug("Could not read the graphics driver's errors", exc_info=True)
                _state["count"] = None
            finally:
                _done.set()

        _state["thread"] = threading.Thread(target=read, name="heliox-gpu-driver-record", daemon=True)
        _state["thread"].start()


def errors_this_week(wait: float = 0.0) -> int | None:
    """The count, starting the read if it has not started; None while unread or unreadable."""
    start_reading()
    if wait > 0:
        _done.wait(wait)
    return _state["count"]


def first_time_said() -> bool:
    """True once: the person is told why speech recognition is on the processor only the first time."""
    with _lock:
        if _state["said"]:
            return False
        _state["said"] = True
        return True


def reset() -> None:
    """Forget the count (tests)."""
    with _lock:
        _state.update(thread=None, count=None, said=False)
        _done.clear()

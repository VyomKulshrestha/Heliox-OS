"""A full disk or a read-only data folder, said once in plain words; the daemon keeps running.

Every SQLite write goes through ``AsyncSqlitePool.write`` and every local answer that saves (reminders, to-dos, notes,
the profile) through the server's local-answer chain; both report a failed write here. ``report`` classifies it
("full": no space left; "read_only": the folder or database cannot be written; "other"), logs it once and tells the
window once per problem (the ``storage_problem`` notification) until a write succeeds again. The words are the same
on Windows, macOS and Linux.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.storage_health")

MESSAGES = {
    "full": "The disk is full, so Heliox could not save {what}. Free some space; Heliox keeps working meanwhile.",
    "read_only": "Heliox's data folder cannot be written to, so it could not save {what}. Check the folder's "
    "permissions or that the drive is not read-only; Heliox keeps working meanwhile.",
}
RECOVERED = "Heliox can save again."
_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}
_READ_ONLY_ERRNOS = {errno.EROFS, errno.EACCES, errno.EPERM}


def classify(error: BaseException) -> str | None:
    """ "full", "read_only", or None for an error that is not about the disk."""
    if isinstance(error, OSError) and not isinstance(error, (TimeoutError, ConnectionError)):
        if error.errno in _FULL_ERRNOS or getattr(error, "winerror", None) in {112, 39}:  # ERROR_DISK_FULL
            return "full"
        if error.errno in _READ_ONLY_ERRNOS or getattr(error, "winerror", None) in {5, 19}:  # access denied, protect
            return "read_only"
        return None
    if isinstance(error, sqlite3.Error):
        text = str(error).lower()
        if "disk is full" in text or "disk full" in text:
            return "full"
        if "readonly" in text or "read-only" in text or "unable to open database" in text:
            return "read_only"
    return None


def sentence(kind: str, what: str = "that") -> str:
    return MESSAGES[kind].format(what=what)


_FULL_WORDS = ("no space left", "errno 28", "not enough space on the disk", "disk is full", "disk full")
_READ_ONLY_WORDS = ("read-only file system", "readonly database", "errno 30", "errno 13", "permission denied",
                    "access is denied")  # fmt: skip


def plain_reason(text: str, what: str = "the model download") -> str:
    """A component's stored error text, rewritten in plain words when it is about the disk; unchanged otherwise.
    (Model downloads keep only the text of their error, e.g. "OSError: [Errno 28] No space left on device".)"""
    lowered = text.lower()
    if any(word in lowered for word in _FULL_WORDS):
        return sentence("full", what)
    if any(word in lowered for word in _READ_ONLY_WORDS):
        return sentence("read_only", what)
    return text


def preflight(folder: Path) -> BaseException | None:
    """Can Heliox write its data folder? Creates it, writes and removes a small file and a small SQLite database.

    Returns the error when the folder is full or read-only (``classify``), else None. Run before the stores open,
    so a daemon that cannot write says why instead of failing part-way through start-up.
    """
    probe = folder / ".heliox-write-check"
    database = folder / ".heliox-write-check.db"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with open(probe, "wb") as handle:
            handle.write(b"x" * 4096)
            handle.flush()
            os.fsync(handle.fileno())
        probe.unlink()
        connection = sqlite3.connect(str(database))
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS check_table (x INTEGER)")
            connection.execute("INSERT INTO check_table VALUES (1)")
            connection.commit()
        finally:
            connection.close()
        database.unlink()
    except (OSError, sqlite3.Error) as error:
        for leftover in (probe, database):
            with contextlib.suppress(OSError):
                leftover.unlink()
        return error if classify(error) is not None else None
    return None


class StorageHealth:
    """Remembers the current storage problem and tells the window about it once."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._problem: dict[str, Any] | None = None
        self.notify: Callable[[dict[str, Any]], None] | None = None

    def report(self, component: str, error: BaseException, what: str = "that") -> str | None:
        """Record a failed write; returns the plain sentence when it was about the disk, else None."""
        kind = classify(error)
        if kind is None:
            return None
        text = sentence(kind, what)
        with self._lock:
            first = self._problem is None or self._problem["kind"] != kind
            if first:
                self._problem = {"kind": kind, "component": component, "message": text, "since": self._clock()}
        if first:
            logger.warning("Storage problem (%s) in %s: %s", kind, component, error)
            if self.notify is not None:
                try:
                    self.notify({"kind": kind, "component": component, "message": sentence(kind, "some data")})
                except Exception:
                    logger.debug("Could not announce the storage problem", exc_info=True)
        return text

    def report_ok(self, *, announce: bool = True) -> None:
        """A write succeeded: the problem is over (said once: "Heliox can save again."); the next one is told again."""
        if self._problem is None:
            return
        with self._lock:
            had, self._problem = self._problem, None
        if announce and had is not None and self.notify is not None:
            try:
                self.notify({"kind": "", "component": had["component"], "message": RECOVERED})
            except Exception:
                logger.debug("Could not announce the storage recovery", exc_info=True)

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._problem) if self._problem else None


_shared = StorageHealth()


def storage() -> StorageHealth:
    return _shared

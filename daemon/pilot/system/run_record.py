"""Whether the daemon's last run ended cleanly, kept as one small file in the state folder (``run_state.json``,
beside the log): it is Heliox's own state, not the person's data, and changes on every start and stop.

At start (after the port is claimed and the data folder checked) the daemon reads what the last run left and writes a
marker for this one: its process id, version and start time. A clean end marks it so: stop() (a signal, daemon_stop,
the app quitting or updating, which asks first) and a replacement on purpose (``--replace-pid``). A marker without a
clean end means the last run did not end cleanly. That is not called a crash: sleep, a forced shutdown, a power cut
or ending the process in the task manager leave the same marker. What is said:

- the computer restarted since that run started (``psutil.boot_time``): nothing to the window, a line in the log;
- the desktop app's supervisor restarted Heliox and passed what it saw (``HELIOX_PREVIOUS_EXIT``): an exit code, a
  hang it ended, or a signal;
- otherwise "did not end cleanly".

The last HISTORY unclean ends are kept in the same file. Nothing here ever stops or slows a start: a data folder that
cannot be read or written means no record and no complaint (the storage sentence already covers the folder), and a
file that cannot be parsed counts as "unknown", is replaced and is not reported.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.run_record")

FILE_NAME = "run_state.json"
HISTORY = 20
PREVIOUS_EXIT_VARIABLE = "HELIOX_PREVIOUS_EXIT"  # set by the desktop app's supervisor when it restarts the daemon


def _boot_time() -> float | None:
    try:
        import psutil

        return float(psutil.boot_time())
    except Exception:
        return None


def _clock_text(at: float) -> str:
    return time.strftime("%H:%M", time.localtime(at))


class RunRecord:
    def __init__(
        self,
        path: Path,
        *,
        version: str,
        pid: int | None = None,
        clock: Callable[[], float] = time.time,
        boot_time: Callable[[], float | None] = _boot_time,
        previous_exit: str | None = None,
        legacy: Path | None = None,
    ) -> None:
        self._path = Path(path)
        # Where versions before 13 November 2026 kept it (the data folder): read once if there is no record in
        # the state folder yet, then removed, so an unclean end just before an update is still said.
        self._legacy = Path(legacy) if legacy is not None else None
        self._version = version
        self._pid = os.getpid() if pid is None else pid
        self._clock = clock
        self._boot_time = boot_time
        self._previous_exit = (
            os.environ.pop(PREVIOUS_EXIT_VARIABLE, None) if previous_exit is None else previous_exit
        )  # one restart's information, not inherited by anything this daemon starts
        self._started_at = 0.0
        self._writable = False
        self._history: list[dict[str, Any]] = []
        self.last_run: dict[str, Any] | None = None

    # ── start ──

    def begin(self, *, folder_writable: bool = True, replaced_pid: int | None = None) -> dict[str, Any] | None:
        """Read what the last run left, write this run's marker, and return what to say about the last run (None
        when it ended cleanly, or there was none). Never raises."""
        try:
            return self._begin(folder_writable, replaced_pid)
        except Exception:
            logger.debug("The run record could not be read or written", exc_info=True)
            self._writable = False
            return None

    def _begin(self, folder_writable: bool, replaced_pid: int | None) -> dict[str, Any] | None:
        self._started_at = self._clock()
        previous, readable = self._read()
        self._history = [entry for entry in (previous or {}).get("history", []) if isinstance(entry, dict)]
        self._history = self._history[-HISTORY:]
        self.last_run = self._judge(previous, readable, replaced_pid)
        if self.last_run is not None and self.last_run["ended"] != "unknown":
            self._history = [*self._history, self.last_run][-HISTORY:]
        if not folder_writable:
            return self.last_run
        self._writable = self._write(
            {
                "pid": self._pid,
                "version": self._version,
                "started_at": self._started_at,
                "ended": None,
                "history": self._history,
            }
        )
        if self.last_run is not None:
            log = logger.info if self.last_run["ended"] == "computer restarted" else logger.warning
            log("Heliox's last run (started %s) %s", self.last_run["started_at_text"], self.last_run["ended"])
        return self.last_run

    def _read(self) -> tuple[dict[str, Any] | None, bool]:
        source = self._path
        if not source.exists() and self._legacy is not None and self._legacy.exists():
            source = self._legacy
        try:
            text = source.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, True
        except OSError:
            return None, False
        try:
            data = json.loads(text)
        except ValueError:
            return None, False
        return (data, True) if isinstance(data, dict) else (None, False)

    def _judge(
        self, previous: dict[str, Any] | None, readable: bool, replaced_pid: int | None
    ) -> dict[str, Any] | None:
        if not readable:
            return {"ended": "unknown", "started_at": None, "started_at_text": "", "version": "", "notice": None}
        if previous is None or previous.get("ended"):
            return None
        if replaced_pid is not None and previous.get("pid") == replaced_pid:
            return None  # replaced on purpose by this run (an update or repair): a clean end
        started = previous.get("started_at")
        started_at = float(started) if isinstance(started, (int, float)) else None
        entry: dict[str, Any] = {
            "started_at": started_at,
            "started_at_text": _clock_text(started_at) if started_at else "",
            "version": str(previous.get("version") or ""),
            "noticed_at": self._started_at,
        }
        boot = self._boot_time()
        if started_at is not None and boot is not None and boot > started_at:
            return {**entry, "ended": "computer restarted", "notice": None}
        exit_seen = (self._previous_exit or "").strip()
        if exit_seen == "hung":
            ended, notice = "stopped answering", "Heliox's background service stopped answering and was restarted."
        elif exit_seen.lstrip("-").isdigit() and exit_seen != "0":
            ended = f"stopped with exit code {exit_seen}"
            notice = f"Heliox's background service stopped unexpectedly (exit code {exit_seen}) and was started again."
        else:
            ended = "did not end cleanly"
            when = f" (started {entry['started_at_text']})" if entry["started_at_text"] else ""
            notice = f"Heliox's last run{when} did not end cleanly. It is running again."
        return {**entry, "ended": ended, "notice": notice}

    def _write(self, data: dict[str, Any]) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(self._path.name + ".tmp")
            temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self._path)
        except OSError:
            return False
        if self._legacy is not None:
            try:
                self._legacy.unlink(missing_ok=True)  # read once above; the state folder holds it from now on
            except OSError:
                pass
        return True

    # ── end ──

    def end_cleanly(self, reason: str) -> None:
        """Mark this run as ended on purpose (stop() began, or the daemon is being replaced). Never raises."""
        if not self._writable:
            return
        try:
            self._write(
                {
                    "pid": self._pid,
                    "version": self._version,
                    "started_at": self._started_at,
                    "ended": "clean",
                    "ended_at": self._clock(),
                    "reason": reason[:80],
                    "history": self._history,
                }
            )
        except Exception:
            logger.debug("Could not mark the run as ended cleanly", exc_info=True)

    # ── answers ──

    def summary(self) -> dict[str, Any] | None:
        """For startup_status and health: the last run when it did not end cleanly, and today's count."""
        if self.last_run is None or self.last_run["ended"] == "unknown":
            return None
        today = time.strftime("%Y-%m-%d", time.localtime(self._started_at))
        same_day = [
            entry
            for entry in self._history
            if entry.get("ended") != "computer restarted"
            and time.strftime("%Y-%m-%d", time.localtime(float(entry.get("noticed_at") or 0))) == today
        ]
        return {
            "ended": self.last_run["ended"],
            "started_at": self.last_run["started_at"],
            "version": self.last_run["version"],
            "notice": self.last_run["notice"],
            "id": f"{self.last_run['started_at']}",
            "unclean_today": len(same_day),
        }

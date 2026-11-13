"""Heliox says when its last run did not end cleanly, and only then (CI runners: ubuntu, macOS, Windows).

A real daemon, three starts:

1. started, then killed (SIGKILL / TerminateProcess) once it is ready, as a crash or the task manager would end it;
2. started again: startup_status must carry last_run "did not end cleanly" with a notice, never the word crash;
   then stopped cleanly with daemon_stop, as the app does when it quits, updates or repairs;
3. started again: no last_run at all.

Run with the environment's Python. Heliox's folders go to a fresh folder through the XDG_* variables, the runtime
folder included. Prints CHECK lines; exit code 1 if one fails. Off a CI runner it changes nothing and exits with 3
(ci_only.py).

What this cannot show: a power cut, sleep, a forced shutdown or a real restart of the computer (the boot-time rule is
tested with a fake boot time), and the desktop app's supervisor passing an exit code (Rust unit tests).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ci_only import require_runner

require_runner("starts and kills real daemons in a folder it makes")

HERE = Path(__file__).resolve().parent
FOLDERS = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "heliox-run-record-check"
shutil.rmtree(FOLDERS, ignore_errors=True)
for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
    os.environ[variable] = str(FOLDERS / variable.lower())  # before pilot.config is imported, here and in the daemon
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
os.environ.pop("HELIOX_PREVIOUS_EXIT", None)  # this check is not the app's supervisor
sys.path.insert(0, str(HERE))

import real_daemon  # noqa: E402

real_daemon.time_limit("run-record", 300)  # no check may hang: a CHECK line and exit 1 instead

results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"CHECK run-record/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    results.append(ok)


async def started() -> tuple[object, real_daemon.Session, dict]:
    daemon = real_daemon.start()
    session = await real_daemon.connect(240)
    ready, _status = await session.wait_ready(240)
    status = await session.call("startup_status", timeout=30)
    print(f"INFO started: ready={ready}, last_run={status.get('last_run')}", flush=True)
    return daemon, session, status


async def run() -> int:
    real_daemon.write_config()

    daemon, session, _status = await started()
    await session.close()
    real_daemon.kill_serving(daemon)  # SIGKILL / TerminateProcess of the serving interpreter: no stop(), no mark

    daemon, session, status = await started()
    last = status.get("last_run") or {}
    notice = str(last.get("notice") or "")
    check(
        "killed-run-is-said",
        last.get("ended") == "did not end cleanly"
        and "did not end cleanly" in notice
        and "crash" not in notice.lower(),
        f"last_run {last}",
    )
    await real_daemon.stop(daemon, session)  # daemon_stop: the app quitting, updating or repairing

    daemon, session, status = await started()
    check("clean-stop-says-nothing", "last_run" not in status, f"last_run {status.get('last_run')}")
    await real_daemon.stop(daemon, session)
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run()))
    finally:
        shutil.rmtree(FOLDERS, ignore_errors=True)

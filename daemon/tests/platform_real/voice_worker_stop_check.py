"""A real daemon stopped, or killed, while its local voice worker runs (CI runner with Kokoro and its weights).

The local voice (Kokoro) runs in a worker process the daemon starts on the first sentence it speaks
(pilot.system.local_tts_worker). It ends when its stdin closes (stop() closes it on purpose; a hard kill of the
daemon closes it with the process), and it watches the daemon's process: a worker busy importing torch and Kokoro,
loading weights or synthesising reads its stdin only when it is done, so the watch ends it when the daemon is gone
(Windows: a wait on the daemon's process handle; macOS and Linux: its parent id, looked at every second). This check
measures how long the worker outlives the daemon in four cases:

1. stopped (daemon_stop) while the worker is still loading;
2. stopped while the worker is loaded and idle;
3. killed hard (SIGKILL / TerminateProcess) while the worker is loaded and idle;
4. killed hard while the worker is still loading.

CHECK lines say how long the daemon took to exit and how long the worker lived after it. Bounds: the stop's own 30 s
for the daemon to exit, and 10 s for the worker after that, loading or idle: the watch notices within a second, the
rest is room for a slow runner (as kill_checks.py's WORKER_BOUND for Gnomon's worker). Before the watch, a busy worker
was allowed 120 s. Every process it started is ended at the end, whatever happened.

    HELIOX_BENCH_ROOT=<root with the models> python daemon/tests/platform_real/voice_worker_stop_check.py

Runs in the conversation timing workflow (ubuntu, CPU only, Kokoro cached); off a runner it changes nothing (ci_only).
What it cannot show: the voice on a graphics card, and macOS and Windows (the workflow has Kokoro on ubuntu only).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from ci_only import require_runner

require_runner("starts and kills real daemons and their voice workers")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # the daemon folder, for pilot

from pilot.real_folders import use_scratch_folders  # noqa: E402

ROOT = Path(os.environ.get("HELIOX_BENCH_ROOT") or (Path(os.environ["RUNNER_TEMP"]) / "voice-worker-check"))
use_scratch_folders(ROOT)  # every folder, the daemon's too (it inherits the marked root)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

import psutil  # noqa: E402
import real_daemon  # noqa: E402

real_daemon.time_limit("voice-worker-stop", 480)  # no check may hang: a CHECK line and exit 1 instead

STOP_BOUND = 30.0
WORKER_BOUND = 10.0  # after the daemon is gone, loading or idle (the parent watch)
GIVE_UP = 150.0  # how long it looks before calling the worker left behind
WORKER_MARK = "pilot.system.local_tts_worker"
results: list[bool] = []
started: list[int] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"CHECK voice-worker/{name}: {'PASS' if ok else 'FAIL'} - {detail}", flush=True)
    results.append(ok)


def workers(daemon_pid: int) -> list[psutil.Process]:
    try:
        children = psutil.Process(daemon_pid).children(recursive=True)
    except psutil.Error:
        return []
    found = []
    for child in children:
        try:
            if any(WORKER_MARK in part for part in child.cmdline()):
                found.append(child)
        except psutil.Error:
            continue
    return found


def gone_after(pids: list[int], limit: float) -> float | None:
    """Seconds until every pid is gone, or None after limit."""
    began = time.monotonic()
    while time.monotonic() - began < limit:
        if not any(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE for pid in pids):
            return time.monotonic() - began
        time.sleep(0.2)
    return None


async def daemon_with_worker(loaded: bool) -> tuple[subprocess.Popen, real_daemon.Session, list[int], asyncio.Task]:
    daemon = real_daemon.start()
    started.append(daemon.pid)
    session = await real_daemon.connect(240)
    await session.wait_ready(240)
    speaking = asyncio.create_task(session.call("speak_text", {"text": "Hello from the check."}, timeout=600))
    deadline = time.monotonic() + 120
    pids: list[int] = []
    while time.monotonic() < deadline and not pids:
        pids = [worker.pid for worker in workers(daemon.pid)]
        await asyncio.sleep(0.2)
    started.extend(pids)
    if loaded and pids:
        try:
            await asyncio.wait_for(asyncio.shield(speaking), timeout=600)  # synthesised once: loaded and idle
        except Exception:
            pass
    return daemon, session, pids, speaking


async def case(name: str, *, loaded: bool, clean: bool) -> None:
    daemon, session, pids, speaking = await daemon_with_worker(loaded)
    if not pids:
        check(name, False, "no voice worker started (is Kokoro installed with its weights?)")
        await real_daemon.stop(daemon, session)
        return
    began = time.monotonic()
    if clean:
        await real_daemon.stop(daemon, session, limit=STOP_BOUND + 10)
    else:
        real_daemon.kill_serving(daemon)  # the interpreter, not a Windows launcher; its worker is left to its watch
        await session.close()
    exited = time.monotonic() - began
    speaking.cancel()
    after = gone_after(pids, GIVE_UP)
    bound = WORKER_BOUND
    ok = after is not None and after <= bound and (not clean or exited <= STOP_BOUND + 5)
    state = "loaded and idle" if loaded else "still loading"
    how = "stopped (daemon_stop)" if clean else "killed hard"
    worker_text = f"gone {after:.1f} s later" if after is not None else f"still running {GIVE_UP:.0f} s later"
    check(
        name,
        ok,
        f"{how} with the worker {state}: daemon exited in {exited:.1f} s, worker {worker_text} (bound {bound:.0f} s)",
    )


async def run() -> int:
    real_daemon.write_config('[voice]\ntts_engine = "kokoro_tts"\ntts_device = "cpu"\n')
    await case("stop-while-loading", loaded=False, clean=True)
    await case("stop-while-idle", loaded=True, clean=True)
    await case("kill-while-idle", loaded=True, clean=False)
    await case("kill-while-loading", loaded=False, clean=False)
    return 0 if results and all(results) else 1


def end_everything() -> None:
    for pid in started:
        try:
            process = psutil.Process(pid)
            for child in process.children(recursive=True):
                child.kill()
            process.kill()
        except psutil.Error:
            continue


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run()))
    finally:
        end_everything()

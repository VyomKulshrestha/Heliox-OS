"""No real check may hang: real_daemon.time_limit ends a check that waits too long, with a CHECK line.

Until 13 November 2026 the real supervisor check hung on Windows for the job's 75 minutes on a wait with no limit.
Each real check now calls time_limit first. Here a stand-in check starts a child, suspends it as a frozen daemon
is, and then waits for ever: it must end within seconds, say so in a CHECK line naming the child, run its clean-up,
and leave nothing running.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent / "platform_real"

STAND_IN = r"""
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import psutil, real_daemon
marker = sys.argv[2]
real_daemon.time_limit("stand-in", 2, lambda: open(marker, "w").write("cleaned"))
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(f"child {child.pid}", flush=True)
time.sleep(0.5)
psutil.Process(child.pid).suspend()
child.wait()  # a wait with no limit of its own
"""


def test_a_check_that_waits_for_ever_ends_with_a_check_line_and_nothing_left(tmp_path):
    marker = tmp_path / "cleaned"
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", STAND_IN, str(HERE), str(marker)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    took = time.monotonic() - started

    assert done.returncode == 1, done.stdout + done.stderr
    assert took < 30
    child = int(next(line for line in done.stdout.splitlines() if line.startswith("child ")).split()[1])
    line = next(line for line in done.stdout.splitlines() if line.startswith("CHECK stand-in/time-limit: FAIL"))
    assert str(child) in line and "not finished after 2 s" in line
    assert marker.read_text() == "cleaned"
    deadline = time.monotonic() + 10
    while psutil.pid_exists(child) and time.monotonic() < deadline:
        try:
            if psutil.Process(child).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    alive = psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE
    if alive:
        psutil.Process(child).kill()
    assert not alive, "the suspended child outlived the time limit"

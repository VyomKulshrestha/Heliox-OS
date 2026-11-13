"""real_daemon.freeze, the one way the real checks freeze a daemon (frozen_daemon_check.py, the Live agent's day check):
the whole process tree is suspended for the time asked, and always resumed, also when the wait is cancelled.

A stand-in process (python sleeping, with a child of its own) takes the daemon's place; no daemon is started.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "platform_real"))

import real_daemon  # noqa: E402

TREE = (
    "import subprocess, sys, time; "
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "time.sleep(60)"
)


@pytest.fixture
def tree():
    parent = subprocess.Popen([sys.executable, "-c", TREE])
    root = psutil.Process(parent.pid)
    for _ in range(100):
        if root.children(recursive=True):
            break
        time.sleep(0.05)
    try:
        yield parent, [root, *root.children(recursive=True)]
    finally:
        real_daemon.end_tree(parent)


def _stopped(processes: list[psutil.Process]) -> list[bool]:
    return [process.status() == psutil.STATUS_STOPPED for process in processes]


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="psutil reports a suspended Windows process as running")
async def test_the_whole_tree_is_frozen_for_the_time_asked_then_resumed(tree):
    parent, processes = tree
    freezing = asyncio.create_task(real_daemon.freeze(parent, 1.0))
    await asyncio.sleep(0.5)
    assert all(_stopped(processes))
    frozen = await freezing
    assert sorted(frozen) == sorted(process.pid for process in processes)
    assert not any(_stopped(processes))


@pytest.mark.asyncio
async def test_a_cancelled_freeze_still_resumes_every_process(tree, monkeypatch):
    parent, processes = tree
    resumed: list[int] = []
    real_resume = psutil.Process.resume

    def resume(self):
        resumed.append(self.pid)
        return real_resume(self)

    monkeypatch.setattr(psutil.Process, "resume", resume)
    freezing = asyncio.create_task(real_daemon.freeze(parent, 60.0))
    await asyncio.sleep(0.3)
    freezing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await freezing
    assert sorted(resumed) == sorted(process.pid for process in processes)


@pytest.mark.asyncio
async def test_a_freeze_is_bounded(tree):
    parent, _processes = tree
    with pytest.raises(ValueError):
        await real_daemon.freeze(parent, real_daemon.FREEZE_LONGEST + 1)
    with pytest.raises(ValueError):
        await real_daemon.freeze(parent, 0)

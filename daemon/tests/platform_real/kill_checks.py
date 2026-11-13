"""Real kills on this machine (CI runners: Windows, macOS, Linux). Prints CHECK lines; exit code 1 if one fails.

1. worker-ends and worker-unload: a process that starts Gnomon's worker exactly as the daemon does (gnomon_worker._default_spawn: the
   real interpreter, pipes and parent watch) is killed hard (SIGKILL / TerminateProcess, no clean-up); the worker
   must be gone within WORKER_BOUND seconds. The parent is a small harness, not the whole daemon: on a runner without
   an NVIDIA card the daemon never starts this worker, but the spawn, the pipes and the watch are the daemon's own.
   worker-unload: the daemon's own stop path for that worker (the reviewer's close unloads the engine) ends a real
   worker process at once.
2. interrupted-task: a process holding the real durable task store and checkpoint store, in the middle of a
   three-step task (one step done, the second claimed and running), is killed hard. A new process then starts the
   way the daemon does (recover_incomplete) and asks for the task as the window does (resume_task): it must be
   offered as interrupted with 1 step done, 2 left and 1 uncertain, and nothing runs by itself.

    python kill_checks.py            # both
    python kill_checks.py worker     # the harness roles are internal: worker-parent, task-holder
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER_BOUND = 10.0  # seconds: the parent watch checks every 1 s; the rest is room for a slow runner
HERE = Path(__file__).resolve()


def hard_kill(pid: int) -> None:
    """No clean-up at all: SIGKILL on macOS and Linux, TerminateProcess on Windows (taskkill /F, without /T)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False)
    else:
        os.kill(pid, signal.SIGKILL)


def alive(pid: int) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _child(role: str, *extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(HERE), role, *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _first_line(process: subprocess.Popen, limit: float = 120.0) -> str:
    """The harness's first line, read in a thread so a harness that says nothing cannot hold the check: before 13
    November 2026 readline() waited with no limit, and the limit was looked at only after it returned."""
    import threading

    found: list[str] = []
    reader = threading.Thread(
        target=lambda: found.append(process.stdout.readline() if process.stdout else ""), daemon=True
    )
    reader.start()
    reader.join(limit)
    if reader.is_alive() or not found:
        raise TimeoutError(f"the harness said nothing in {limit:g} s")
    return found[0].strip()


# ── 1. the worker ends when the process that started it is killed ──


def worker_parent() -> None:
    from pilot.intelligence import gnomon_worker

    worker = gnomon_worker._default_spawn()
    print(json.dumps({"worker": worker.pid}), flush=True)
    time.sleep(3600)


def check_worker_ends() -> bool:
    parent = _child("worker-parent")
    try:
        worker = int(json.loads(_first_line(parent))["worker"])
    except Exception as error:
        print(
            f"CHECK kill/worker-ends: FAIL - the harness did not start a worker ({error}): {parent.stderr.read()[-400:]}"
        )
        parent.kill()
        return False
    time.sleep(2.0)  # the worker has started its parent watch
    if not alive(worker):
        print("CHECK kill/worker-ends: FAIL - the worker had already exited before the kill")
        return False
    import real_daemon

    hard_kill(real_daemon.serving_process(parent, "worker-parent").pid)  # the harness's interpreter, not a launcher
    killed_at = time.monotonic()
    while alive(worker) and time.monotonic() - killed_at < WORKER_BOUND * 3:
        time.sleep(0.1)
    waited = time.monotonic() - killed_at
    if alive(worker):
        print(
            f"CHECK kill/worker-ends: FAIL - the worker (pid {worker}) still runs {waited:.0f} s after its parent was killed"
        )
        hard_kill(worker)
        return False
    ok = waited <= WORKER_BOUND
    print(
        f"CHECK kill/worker-ends: {'PASS' if ok else 'FAIL'} - the worker was gone {waited:.1f} s after its parent "
        f"was killed hard (bound {WORKER_BOUND:.0f} s)"
    )
    return ok


def check_worker_unload() -> bool:
    """The daemon's own stop path for Gnomon (reviewer close -> engine unload) with a real worker process: started
    and answering its ping (no card needed for that), then unloaded; it must be gone at once, not left behind."""
    from types import SimpleNamespace

    from pilot.intelligence.gnomon_worker import GnomonWorkerEngine

    engine = GnomonWorkerEngine({"check": True}, arbiter=SimpleNamespace(disabled=False))
    try:
        engine._start(None)
    except Exception as error:
        print(f"CHECK kill/worker-unload: FAIL - the worker did not start ({type(error).__name__}: {error})")
        return False
    pids = [pid for pid in (engine.worker_pid, getattr(engine._process, "pid", None)) if pid]
    began = time.monotonic()
    engine.unload()
    while any(alive(pid) for pid in pids) and time.monotonic() - began < WORKER_BOUND * 3:
        time.sleep(0.1)
    waited = time.monotonic() - began
    left = [pid for pid in pids if alive(pid)]
    for pid in left:
        hard_kill(pid)
    ok = not left and waited <= WORKER_BOUND
    print(
        f"CHECK kill/worker-unload: {'PASS' if ok else 'FAIL'} - the worker (pids {pids}) was "
        f"{'gone' if not left else 'still running'} {waited:.1f} s after the stop path unloaded it "
        f"(bound {WORKER_BOUND:.0f} s)"
    )
    return ok


# ── 2. a task interrupted by a hard kill is offered, not repeated ──


def _plan(count: int):
    from pilot.actions import Action, ActionPlan, ActionType, NotifyParams

    return ActionPlan(
        actions=[
            Action(action_type=ActionType.NOTIFY, target=f"n{i}", parameters=NotifyParams(summary="S", body=f"{i}"))
            for i in range(count)
        ],
        explanation="three notifications",
        raw_input="send three notifications",
    )


async def _hold_task(folder: Path) -> None:
    from pilot.actions import ActionResult
    from pilot.workflows.checkpoints import WorkflowCheckpointStore
    from pilot.workflows.durable_tasks import DurableTaskStore, TaskStatus

    durable = DurableTaskStore(folder / "durable_tasks.db")
    await durable.initialize()
    checkpoints = WorkflowCheckpointStore(folder / "workflow_checkpoints.db")
    await checkpoints.initialize()
    created = await durable.create_task(task_id="t1", session_id="s1", user_input="send three notifications")
    await durable.transition("t1", TaskStatus.PLANNING)
    await durable.transition("t1", TaskStatus.EXECUTING, plan_id="p1")
    plan = _plan(3)
    await checkpoints.start_plan("p1", "send three notifications", plan)
    await checkpoints.record_result("p1", ActionResult(action=plan.actions[0], success=True, output="ok"))
    await durable.claim_action(task_id="t1", plan_id="p1", action_key="a2", lease_owner="killed-daemon")
    print(json.dumps({"resume_token": created.resume_token}), flush=True)
    await asyncio.sleep(3600)  # the second step "runs" until the kill


def task_holder(folder: str) -> None:
    asyncio.run(_hold_task(Path(folder)))


async def _offer_after_restart(folder: Path, resume_token: str) -> dict:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from pilot.config import PilotConfig
    from pilot.security.rpc_identity import RpcClientRole
    from pilot.server import PilotServer
    from pilot.workflows.checkpoints import WorkflowCheckpointStore
    from pilot.workflows.durable_tasks import DurableTaskStore

    durable = DurableTaskStore(folder / "durable_tasks.db")
    await durable.initialize()
    checkpoints = WorkflowCheckpointStore(folder / "workflow_checkpoints.db")
    await checkpoints.initialize()
    await durable.recover_incomplete()  # what the daemon does when it starts
    server = PilotServer(PilotConfig())
    server._durable_tasks = durable
    server._checkpoint_store = checkpoints
    server._memory = AsyncMock()
    ran: list[int] = []

    async def execute(remaining, **_kwargs):
        ran.append(len(remaining.actions))
        return []

    server._executor = SimpleNamespace(execute=execute)
    ws = MagicMock()
    ws.send = AsyncMock()
    server._client_roles[ws] = RpcClientRole.UI
    offer = await server._handle_resume_task({"task_id": "t1", "resume_token": resume_token}, ws)
    offer["ran"] = ran
    await durable.close()
    return offer


def check_interrupted_task() -> bool:
    with tempfile.TemporaryDirectory(prefix="heliox-kill-") as temp:
        holder = _child("task-holder", temp)
        try:
            token = json.loads(_first_line(holder))["resume_token"]
        except Exception as error:
            print(
                f"CHECK kill/interrupted-task: FAIL - the task holder did not start ({error}): {holder.stderr.read()[-400:]}"
            )
            holder.kill()
            return False
        hard_kill(holder.pid)
        holder.wait(30)
        offer = asyncio.run(_offer_after_restart(Path(temp), token))
    got = (offer.get("status"), offer.get("steps_done"), offer.get("steps_left"), offer.get("uncertain_steps"))
    ok = got == ("interrupted", 1, 2, 1) and offer["ran"] == []
    print(
        f"CHECK kill/interrupted-task: {'PASS' if ok else 'FAIL'} - after a hard kill mid-task the restart offers "
        f"status={got[0]}, {got[1]} done, {got[2]} left, {got[3]} uncertain; ran by itself: {offer['ran'] or 'nothing'}"
    )
    return ok


def main(argv: list[str]) -> int:
    if argv[:1] == ["worker-parent"]:
        worker_parent()
        return 0
    if argv[:1] == ["task-holder"]:
        task_holder(argv[1])
        return 0
    sys.path.insert(0, str(HERE.parent))
    import real_daemon

    real_daemon.time_limit("kill", 300)  # no check may hang: a CHECK line and exit 1 instead
    wanted = set(argv) or {"worker", "task"}
    results = []
    if "worker" in wanted:
        results.append(check_worker_ends())
        results.append(check_worker_unload())
    if "task" in wanted:
        results.append(check_interrupted_task())
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Collects REAL (state, action, outcome) telemetry for the Learned Risk
Gate's transition model — see pilot/security/risk_model.py.

Mirrors Ferrum-OS's scripts/collect_world_model_dataset.mjs in spirit
(gather real before/after tuples from actually running actions, not
fabricated numbers) but adapted for a critical safety difference: Ferrum
collects data by hammering a disposable QEMU VM it can freely destroy and
rebuild. Heliox runs on the user's real machine, so this script ONLY ever
runs the small set of ActionTypes that are genuinely safe to repeat
thousands of times for real:

  - File operations (write/copy/delete), entirely confined to one
    throwaway temp directory this script creates and removes itself.
  - Trivial process spawn/kill (a `python -c "..."` sleep, immediately
    killed) — standing in for SERVICE_START/SERVICE_STOP/OPEN_APPLICATION/
    SHELL_*/CODE_EXECUTE's effect on process count, since actually
    starting/stopping real system services or launching real
    applications thousands of times would NOT be safe to repeat.

Every other ActionType never touches the learned model at all (see
RiskTransitionModel.predict()'s LEARNABLE_ACTION_TYPES gate) — there is
nothing to collect data for outside this list, and no synthetic/fabricated
labels are generated for them. This is a deliberately narrower dataset
than Ferrum's, in exchange for every row being real, not a rule
restated as training data.

Usage:
    python scripts/collect_risk_training_data.py [--samples-per-type N] [--out PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pilot.actions import Action, ActionType, EmptyParams  # noqa: E402
from pilot.security.risk_model import encode  # noqa: E402
from pilot.security.risk_observation import capture_os_snapshot  # noqa: E402

FILE_ACTION_TYPES = [ActionType.FILE_WRITE, ActionType.FILE_COPY, ActionType.FILE_DELETE, ActionType.DOWNLOAD_FILE]
PROCESS_ACTION_TYPES = [
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.PTY_EXEC,
    ActionType.CODE_EXECUTE,
    ActionType.OPEN_APPLICATION,
    ActionType.SERVICE_START,
    ActionType.SERVICE_STOP,
    ActionType.PROCESS_KILL,
]


def _make_action(action_type: ActionType, target: str) -> Action:
    return Action(action_type=action_type, target=target, parameters=EmptyParams())


async def _collect_file_sample(action_type: ActionType, sandbox_dir: str) -> dict | None:
    """Runs one real file operation confined to sandbox_dir and records the
    real disk-usage delta it produced."""
    target_path = os.path.join(sandbox_dir, f"sample_{random.randint(0, 1_000_000)}.bin")
    size_bytes = random.randint(1024, 512 * 1024)  # 1KB-512KB, varied on purpose

    action = _make_action(action_type, target_path)
    before = capture_os_snapshot(disk_path=sandbox_dir)

    if action_type in (ActionType.FILE_WRITE, ActionType.DOWNLOAD_FILE):
        with open(target_path, "wb") as f:
            f.write(os.urandom(size_bytes))
    elif action_type == ActionType.FILE_COPY:
        src = os.path.join(sandbox_dir, "_copy_source.bin")
        if not os.path.exists(src):
            with open(src, "wb") as f:
                f.write(os.urandom(size_bytes))
        shutil.copyfile(src, target_path)
    elif action_type == ActionType.FILE_DELETE:
        with open(target_path, "wb") as f:
            f.write(os.urandom(size_bytes))
        # Re-snapshot after the write so the delete's OWN delta is measured
        # relative to a state that already includes the file being deleted.
        before = capture_os_snapshot(disk_path=sandbox_dir)
        os.remove(target_path)
    else:
        return None

    after = capture_os_snapshot(disk_path=sandbox_dir)
    embedding = encode(before, action)
    return {
        "embedding": embedding.tolist(),
        "disk_delta": after.disk_usage_fraction - before.disk_usage_fraction,
        "proc_delta": 0.0,  # file operations have no direct effect on process count
        "action_type": action_type.value,
        "source": "real_sandbox",
    }


_STARTS_ACTION_TYPES = (
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.PTY_EXEC,
    ActionType.CODE_EXECUTE,
    ActionType.OPEN_APPLICATION,
    ActionType.SERVICE_START,
)


async def _spawn_trivial_process() -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(0.4)",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.sleep(0.05)  # let it actually register in the process table
    return proc


async def _collect_process_sample(action_type: ActionType, sandbox_dir: str) -> dict | None:
    """Spawns/kills a trivial, harmless subprocess as a real stand-in for
    this action type's effect on process count — see module docstring for
    why real services/apps aren't launched directly.

    For "start" types, `before` is the baseline and `after` reflects the
    spawn. For "stop" types (SERVICE_STOP/PROCESS_KILL), the process is
    spawned FIRST so `before` reflects the already-running state a real
    stop/kill would act on, and `after` reflects its removal — measuring
    the marginal effect of ENDING a process, not a spawn-then-kill
    round-trip netting back to the same baseline.
    """
    action = _make_action(action_type, target="sandbox-process")

    if action_type in _STARTS_ACTION_TYPES:
        before = capture_os_snapshot(disk_path=sandbox_dir)
        proc = await _spawn_trivial_process()
        after = capture_os_snapshot(disk_path=sandbox_dir)
        await proc.wait()
    else:
        proc = await _spawn_trivial_process()
        before = capture_os_snapshot(disk_path=sandbox_dir)
        proc.terminate()
        await proc.wait()
        after = capture_os_snapshot(disk_path=sandbox_dir)

    embedding = encode(before, action)
    return {
        "embedding": embedding.tolist(),
        "disk_delta": 0.0,
        "proc_delta": (after.proc_count - before.proc_count) / 300.0,  # NOMINAL_PROC_CAPACITY
        "action_type": action_type.value,
        "source": "real_sandbox",
    }


async def collect(samples_per_type: int, out_path: str) -> None:
    sandbox_dir = tempfile.mkdtemp(prefix="heliox_risk_sandbox_")
    rows: list[dict] = []
    try:
        for action_type in FILE_ACTION_TYPES:
            for _ in range(samples_per_type):
                row = await _collect_file_sample(action_type, sandbox_dir)
                if row:
                    rows.append(row)

        for action_type in PROCESS_ACTION_TYPES:
            for _ in range(samples_per_type):
                row = await _collect_process_sample(action_type, sandbox_dir)
                if row:
                    rows.append(row)
                # Real process spawn/kill isn't free — avoid hammering the
                # OS scheduler harder than a real workload would.
                await asyncio.sleep(0.01)

        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(
            f"Collected {len(rows)} real samples across {len(FILE_ACTION_TYPES) + len(PROCESS_ACTION_TYPES)} action types -> {out_path}"
        )
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-type", type=int, default=350)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "risk_dataset.jsonl"))
    args = parser.parse_args()

    asyncio.run(collect(args.samples_per_type, args.out))


if __name__ == "__main__":
    main()

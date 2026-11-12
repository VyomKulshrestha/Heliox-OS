"""How much Heliox's unbounded stores grow per request: a scripted day of requests in a temporary data folder.

Runs the real request path (``PilotServer._handle_execute`` -> ``Executor``) with the real audit log, durable task
store, checkpoint store and experience ledger, each writing into a temporary folder. The planner returns a fixed
plan and system information is stubbed, so no model, network or GPU is used and nothing on this computer changes
outside the temporary folder.

    .venv\\Scripts\\python.exe benchmarks\\store_growth.py [requests]

It prints bytes per request for each store and what that comes to per day and per year at the stated number of
requests a day. Sizes are read after a SQLite checkpoint, so write-ahead logs are folded in.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from pilot.real_folders import use_scratch_folders

# Every Heliox folder under a scratch root of its own, before pilot.config loads: never the person's real ones
use_scratch_folders()  # a temporary root, removed when the script ends

from pilot.actions import Action, ActionPlan, ActionType, FileParams, SystemInfoParams, VerificationResult  # noqa: E402
from pilot.agents.executor import Executor  # noqa: E402
from pilot.config import PilotConfig  # noqa: E402
from pilot.intelligence.experience import ExperienceLedger  # noqa: E402
from pilot.security.audit import AuditLogger  # noqa: E402
from pilot.security.permissions import PermissionChecker  # noqa: E402
from pilot.security.validator import ActionValidator  # noqa: E402
from pilot.server import PilotServer  # noqa: E402
from pilot.system import sysinfo  # noqa: E402
from pilot.workflows.checkpoints import WorkflowCheckpointStore  # noqa: E402
from pilot.workflows.durable_tasks import DurableTaskStore  # noqa: E402

STORES = {
    "audit.jsonl": "Audit log",
    "durable_tasks.db": "Durable tasks",
    "workflow_checkpoints.db": "Checkpoints",
    "experience.db": "Experience ledger",
}


class _Ws:
    async def send(self, message: str) -> None:
        return None


class _Nothing:
    async def get_improvement_context(self, *_args, **_kwargs):
        return ""

    async def reflect(self, *_args, **_kwargs):
        return None

    async def record(self, *_args, **_kwargs):
        return None

    async def put_working(self, *_args, **_kwargs):
        return None

    def get_routing_summary(self, *_args, **_kwargs):
        return {"assigned_agents": []}

    def plan_requires_confirmation(self, _plan):
        return False


class _Memory:
    """Working memory and long-term memory are not what is measured here: every call does nothing."""

    def __getattr__(self, _name):
        async def nothing(*_args, **_kwargs):
            return None

        return nothing


class _Verifier:
    async def verify(self, plan, results):
        return VerificationResult(passed=all(result.success for result in results))


def _plans(sandbox: Path) -> list[ActionPlan]:
    """A day's mix: one-step reads and a two-step write-then-read."""
    note = str(sandbox / "note.txt")
    return [
        ActionPlan(
            actions=[Action(action_type=ActionType.SYSTEM_INFO, target="system", parameters=SystemInfoParams())],
            explanation="How is my computer doing?",
        ),
        ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.FILE_WRITE,
                    target=note,
                    parameters=FileParams(path=note, content="Remember the dentist on Friday."),
                ),
                Action(action_type=ActionType.FILE_READ, target=note, parameters=FileParams(path=note)),
            ],
            explanation="Write a note and read it back.",
        ),
    ]


def _bytes(folder: Path, name: str) -> int:
    path = folder / name
    if name.endswith(".db") and path.exists():
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return sum(p.stat().st_size for p in folder.glob(name + "*") if p.is_file())


async def measure(requests: int) -> dict[str, dict[str, float]]:
    async def section(heading):
        return f"=== {heading} ===\n  sample: {heading}"

    sysinfo.get_platform_info = lambda: {"sample": "Operating System"}
    for category in ("time", "cpu", "memory", "disk", "network", "battery"):
        setattr(sysinfo, f"_{category}_info", lambda category=category: section(category))

    with tempfile.TemporaryDirectory(prefix="heliox-growth-", ignore_cleanup_errors=True) as temp:
        data = Path(temp) / "data"
        sandbox = Path(temp) / "sandbox"
        data.mkdir()
        sandbox.mkdir()
        config = PilotConfig()
        server = PilotServer(config)
        server._reflector = server._multi_agent = _Nothing()
        server._memory = _Memory()
        server._permission_checker = _Nothing()
        server._orchestrator = server._destructive_critic = None
        server._verifier = _Verifier()

        async def broadcast(_method, _params):
            return None

        server._broadcast_notification = broadcast
        server._checkpoint_store = WorkflowCheckpointStore(data / "workflow_checkpoints.db")
        await server._checkpoint_store.initialize()
        server._durable_tasks = DurableTaskStore(data / "durable_tasks.db")
        await server._durable_tasks.initialize()
        ledger = ExperienceLedger(data / "experience.db")
        await ledger.initialize()
        server._experience_ledger = ledger
        executor = Executor(
            config, ActionValidator(config), PermissionChecker(config), AuditLogger(data / "audit.jsonl")
        )
        executor.set_experience_ledger(ledger)
        executor.set_durable_task_store(server._durable_tasks)
        server._executor = executor
        plans = _plans(sandbox)
        outcomes: dict[str, int] = {}

        class _Planner:
            index = 0

            async def plan(self, *_args, **_kwargs):
                plan = plans[self.index % len(plans)]
                self.index += 1
                return plan.model_copy(deep=True)

        server._planner = _Planner()
        before = {name: _bytes(data, name) for name in STORES}
        for number in range(requests):
            began = time.monotonic()
            response = await server._handle_execute({"input": f"request {number}"}, _Ws())
            if number < 3:
                print(f"request {number}: {time.monotonic() - began:.1f} s", file=sys.stderr)
            status = str(response.get("status"))
            outcomes[status] = outcomes.get(status, 0) + 1
        await ledger.close()
        await server._durable_tasks.close()
        after = {name: _bytes(data, name) for name in STORES}
        return {
            "outcomes": outcomes,
            **{
                STORES[name]: {"bytes": after[name], "per_request": (after[name] - before[name]) / requests}
                for name in STORES
            },
        }


def main() -> None:
    requests = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    found = asyncio.run(measure(requests))
    per_day = requests  # the scripted day is this many requests
    print(json.dumps({"requests": requests, "outcomes": found.pop("outcomes")}))
    for name, row in found.items():
        day = row["per_request"] * per_day
        print(
            f"{name}: {row['per_request']:.0f} bytes per request, {day / 1024:.0f} KB per day of {per_day} requests, "
            f"{day * 365 / 1024**2:.0f} MB per year"
        )
    sys.stdout.flush()
    # Stores this script does not own (opened by the server's request path) keep aiosqlite threads alive; the
    # numbers are printed, so end here instead of waiting on them.
    os._exit(0)


if __name__ == "__main__":
    main()

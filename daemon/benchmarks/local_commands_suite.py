"""Measure the built-in commands Heliox answers in code, without planning or a model.

Every example in the command guide (the list behind Ctrl+K and "what can I say?") is sent
to the daemon's control path on a fresh daemon whose stores live in a temporary folder.
The suite records whether each example was handled and how long the reply took. It never
plans or executes an action: stand-ins replace the clipboard, speech, settings file and
the parts that read this computer (processes, battery, the front window and folders), so
the numbers describe the control path itself, not the machine. The health check's
half-second processor sample is stood in for as well.

Run from ``daemon``:
    python benchmarks/local_commands_suite.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from benchmarks.react_latency import summarize

GB = 1024**3
SNAPSHOT = {
    "at": 0.0,
    "cpu": 12.0,
    "memory": {"total": 16 * GB, "used": 8 * GB, "percent": 50.0},
    "disks": [{"drive": "C:\\", "total": 500 * GB, "free": 200 * GB}],
    "battery": None,
    "apps": [{"app": "Chrome", "cpu": 5.0, "memory": 2 * GB, "processes": 12}],
}


class _MemoryVault:
    """Keeps the receipt signing key in memory, so the run never touches the OS credential store."""

    available = True

    def __init__(self) -> None:
        self._keys: dict[str, str] = {}

    async def get_key(self, name: str) -> str | None:
        return self._keys.get(name)

    async def store_key(self, name: str, value: str) -> None:
        self._keys[name] = value


def _server(root: Path) -> Any:
    from pilot.config import PilotConfig, TimelineConfig, UndoConfig
    from pilot.server import PilotServer
    from pilot.workflows.activity_timeline import ActivityTimeline
    from pilot.workflows.app_reminders import AppReminders
    from pilot.workflows.clipboard_history import ClipboardHistory
    from pilot.workflows.file_finder import FileFinder
    from pilot.workflows.focus import FocusSession, QuietHours
    from pilot.workflows.focus_guard import FocusGuard
    from pilot.workflows.focus_log import FocusLog
    from pilot.workflows.generators import Generators
    from pilot.workflows.goals import GoalStore
    from pilot.workflows.habits import HabitStore
    from pilot.workflows.notes import NoteStore
    from pilot.workflows.nudges import NudgeStore
    from pilot.workflows.quick_answers import QuickAnswers
    from pilot.workflows.reminders import ReminderStore
    from pilot.workflows.routines import ShortcutStore
    from pilot.workflows.snippets import SnippetStore
    from pilot.workflows.system_health import SystemHealth
    from pilot.workflows.task_receipts import TaskReceipts
    from pilot.workflows.todos import TodoStore
    from pilot.workflows.undo_journal import UndoJournal
    from pilot.workflows.user_profile import UserProfile
    from pilot.workflows.watchers import Probes, WatcherStore
    from pilot.workflows.workspaces import WorkspaceStore

    config = PilotConfig()
    config.save = MagicMock()  # settings examples must never touch the real config file
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._task_receipts = TaskReceipts(root / "receipts", vault=_MemoryVault(), key_dir=root / "keys")
    server._activity_timeline = ActivityTimeline(root / "timeline.db", config=TimelineConfig(enabled=True))
    server._undo_journal = UndoJournal(root / "undo", config=UndoConfig())
    server._reminders = ReminderStore(root / "reminders.json")
    server._app_reminders = AppReminders(root / "app_reminders.json", foreground=lambda: ("Code", "x"))
    server._notes = NoteStore(root / "notes.json")
    server._todos = TodoStore(root / "todos.json")
    server._todos.add("call the bank")
    server._nudges = NudgeStore(root / "nudges.json", timeline_on=lambda: True)
    server._focus = FocusSession(root / "focus.json", on_end=server._on_focus_end)
    server._focus_guard = FocusGuard(root / "distractions.json", focus_status=server._focus.status)
    server._focus_log = FocusLog(root / "focus_history.json")
    server._quiet_hours = QuietHours(root / "quiet.json", focus=server._focus, on_start=AsyncMock())
    server._workspaces = WorkspaceStore(root / "workspaces.json")
    server._workspaces.save("coding", ["Visual Studio Code"])
    server._shortcuts = ShortcutStore(root / "shortcuts.json", reserved=server._phrase_reserved)
    downloads = root / "Downloads"
    downloads.mkdir()
    (downloads / "report.pdf.crdownload").write_bytes(b"x")
    server._watchers = WatcherStore(
        probes=Probes(
            processes=lambda: [(10, "chrome.exe", 1.0)],
            alive=lambda pid, created: True,
            battery=lambda: (50.0, True),
            cpu=lambda: 80.0,
            downloads=downloads,
        )
    )
    clipboard = AsyncMock(return_value="Jane Doe, 12 Main Street")
    server._clipboard_history = ClipboardHistory(enabled=lambda: True, read=clipboard, write=AsyncMock())
    server._snippets = SnippetStore(root / "snippets.json", read=clipboard, write=AsyncMock())
    server._snippets.save("signature", "Best, Jane")
    server._system_health = SystemHealth(read=lambda: SNAPSHOT)
    server._quick_answers = QuickAnswers()
    server._file_finder = FileFinder(roots=lambda: [downloads], downloads=lambda: downloads)
    server._generators = Generators(write=AsyncMock())
    server._habits = HabitStore(root / "habits.json")
    server._habits.log("workout")
    server._habits.log("water")
    server._user_profile = UserProfile(root / "user_profile.json")
    server._goals = GoalStore(root / "goals.json")
    server._goals.create("launch the podcast")
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock()
    return server


async def _measure(iterations: int, warmup: int) -> dict[str, Any]:
    from pilot.workflows.command_guide import all_examples

    examples = all_examples()
    with tempfile.TemporaryDirectory(prefix="heliox-local-commands-") as folder:
        server = _server(Path(folder))
        # "Save this as my coding setup" would list this computer's open apps; stand in for it.
        apps = patch("pilot.workflows.workspaces.open_apps", AsyncMock(return_value=["Visual Studio Code"]))
        apps.start()
        try:
            handled: dict[str, bool] = {}
            per_example: dict[str, list[float]] = {}
            for example in examples:
                timings = []
                for run in range(warmup + iterations):
                    started = time.perf_counter()
                    reply = await server._control_reply(example)
                    expansion = await server._expand_request(example) if reply is None else None
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if run == 0:  # many examples change state, so judge each on its first, fresh run
                        handled[example] = reply is not None or expansion is not None
                    if run >= warmup:
                        timings.append(elapsed_ms)
                per_example[example] = timings
        finally:
            apps.stop()
            if server._focus is not None:
                await server._focus.close()
    all_timings = [timing for timings in per_example.values() for timing in timings]
    medians = sorted(
        ((sorted(timings)[len(timings) // 2], example) for example, timings in per_example.items()),
        reverse=True,
    )
    return {
        "examples": len(examples),
        "handled": sum(handled.values()),
        "unhandled": sorted(example for example, ok in handled.items() if not ok),
        "latency": summarize(all_timings),
        "slowest_examples": [{"example": example, "median_ms": round(median, 3)} for median, example in medians[:3]],
        "model_generate_calls": 0,
        "actions_executed": 0,
    }


def benchmark(iterations: int = 20, warmup: int = 2) -> dict[str, Any]:
    """Every command-guide example through the control path; per-call latency across all of them."""
    report = asyncio.run(_measure(iterations, warmup)) if not _running_loop() else None
    if report is None:
        raise RuntimeError("call benchmark_async from inside an event loop")
    return _finish(report, iterations, warmup)


async def benchmark_async(iterations: int = 20, warmup: int = 2) -> dict[str, Any]:
    return _finish(await _measure(iterations, warmup), iterations, warmup)


def _finish(report: dict[str, Any], iterations: int, warmup: int) -> dict[str, Any]:
    return {
        "benchmark": "local_commands",
        "iterations_per_example": iterations,
        "warmup_per_example": warmup,
        "python": platform.python_version(),
        "platform": platform.platform(),
        **report,
        "claim_boundary": (
            "In-process control path with stand-ins for the clipboard, speech, settings file and machine "
            "readings; no claim of speech recognition, text-to-speech, UI or end-to-end voice latency"
        ),
    }


def _running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = benchmark(iterations=args.iterations)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        latency = report["latency"]
        print(
            f"{report['handled']}/{report['examples']} handled; median {latency['median_ms']} ms, "
            f"p95 {latency['p95_ms']} ms across {latency['iterations']} replies"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

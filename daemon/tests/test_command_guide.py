"""Every example in the command guide must work as written, and none can become a shortcut."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, TimelineConfig, UndoConfig
from pilot.workflows.activity_timeline import ActivityTimeline
from pilot.workflows.app_reminders import AppReminders
from pilot.workflows.clipboard_history import ClipboardHistory
from pilot.workflows.command_guide import SPOKEN_HELP, all_examples, guide
from pilot.workflows.focus import FocusSession, QuietHours
from pilot.workflows.focus_guard import FocusGuard
from pilot.workflows.notes import NoteStore
from pilot.workflows.nudges import NudgeStore
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.routines import ShortcutStore, normalize_phrase
from pilot.workflows.system_health import SystemHealth
from pilot.workflows.task_receipts import TaskReceipts
from pilot.workflows.todos import TodoStore
from pilot.workflows.undo_journal import UndoJournal
from pilot.workflows.watchers import Probes, WatcherStore
from pilot.workflows.workspaces import WorkspaceStore
from tests.test_task_receipts import FakeVault

SNAPSHOT = {
    "at": 0.0,
    "cpu": 12.0,
    "memory": {"total": 16 * 1024**3, "used": 8 * 1024**3, "percent": 50.0},
    "disks": [{"drive": "C:\\", "total": 500 * 1024**3, "free": 200 * 1024**3}],
    "battery": None,
    "apps": [{"app": "Chrome", "cpu": 5.0, "memory": 2 * 1024**3, "processes": 12}],
}


@pytest.fixture
def server(tmp_path):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._task_receipts = TaskReceipts(tmp_path / "receipts", vault=FakeVault())
    server._activity_timeline = ActivityTimeline(tmp_path / "t.db", config=TimelineConfig(enabled=True))
    server._undo_journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    server._reminders = ReminderStore(tmp_path / "reminders.json")
    server._app_reminders = AppReminders(tmp_path / "app_reminders.json", foreground=lambda: ("Code", "x"))
    server._notes = NoteStore(tmp_path / "notes.json")
    server._todos = TodoStore(tmp_path / "todos.json")
    server._todos.add("call the bank")
    downloads = tmp_path / "Downloads"
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
    server._nudges = NudgeStore(tmp_path / "nudges.json", timeline_on=lambda: True)
    server._focus = FocusSession(tmp_path / "focus.json", on_end=server._on_focus_end)
    server._focus_guard = FocusGuard(tmp_path / "distractions.json", focus_status=server._focus.status)
    server._quiet_hours = QuietHours(tmp_path / "quiet.json", focus=server._focus, on_start=AsyncMock())
    server._workspaces = WorkspaceStore(tmp_path / "workspaces.json")
    server._workspaces.save("coding", ["Visual Studio Code"])
    server._shortcuts = ShortcutStore(tmp_path / "shortcuts.json", reserved=server._phrase_reserved)
    server._clipboard_history = ClipboardHistory(
        enabled=lambda: True, read=AsyncMock(return_value=""), write=AsyncMock()
    )
    server._system_health = SystemHealth(read=lambda: SNAPSHOT)
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock()
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("example", all_examples())
async def test_every_example_is_handled_without_planning(server, example):
    reply = await server._control_reply(example)
    expansion = await server._expand_request(example) if reply is None else None

    assert reply is not None or expansion is not None, example


@pytest.mark.parametrize("example", all_examples())
def test_no_example_can_become_a_shortcut(server, example):
    assert server._phrase_reserved(normalize_phrase(example)), example


@pytest.mark.asyncio
async def test_what_can_i_say_opens_the_palette(server):
    assert await server._control_reply("What can I say?") == SPOKEN_HELP
    server._broadcast_notification.assert_awaited_with("open_command_palette", {})
    assert await server._help_request("what can you do") is None  # left to the conversational answer


@pytest.mark.asyncio
async def test_command_guide_rpc(server):
    result = await server._handle_command_guide({}, MagicMock())

    assert result["status"] == "ok"
    assert [group["group"] for group in result["groups"]] == [group["group"] for group in guide()]
    assert all(group["examples"] for group in result["groups"])

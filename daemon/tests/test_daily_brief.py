"""Today view, daily brief and "undo the last N tasks": composed in code from local records."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionResult, ActionType, FileParams
from pilot.config import BriefConfig, PilotConfig, TimelineConfig, _merge_config
from pilot.workflows.activity_timeline import ActivityTimeline
from pilot.workflows.daily_brief import DailyBriefScheduler, compose_brief, day_window, today_overview
from pilot.workflows.task_receipts import TaskReceipts
from tests.test_task_receipts import FakeVault


def _record(receipts: TaskReceipts, task: str) -> None:
    action = Action(action_type=ActionType.FILE_WRITE, target="a.txt", parameters=FileParams(path="a.txt"))
    receipts.record_action(task, "p", action, ActionResult(action=action, success=True, output="ok"))


def _overview(**changes) -> dict:
    overview = {
        "day": "2026-10-23",
        "label": "today",
        "tasks": [],
        "usage": None,
        "undoable": [],
    }
    overview.update(changes)
    return overview


def test_brief_names_tasks_time_and_what_can_be_undone():
    overview = _overview(
        tasks=[
            {"status": "success", "request": "tidy downloads", "summary": ""},
            {"status": "error", "request": "email the report", "summary": ""},
        ],
        usage={"apps": [{"app": "Code", "seconds": 7200}, {"app": "Chrome", "seconds": 1500}]},
        undoable=[{"task_key": "a"}],
    )

    assert compose_brief(overview) == (
        'Today Heliox ran 2 tasks, 1 of them not cleanly; the latest was "email the report". '
        "You spent most of your time in Code (2 h), Chrome (25 min). 1 task can still be undone."
    )


def test_brief_on_a_quiet_day():
    assert compose_brief(_overview(label="yesterday")) == "Yesterday Heliox ran no tasks."


def test_day_window_today_ends_now_and_yesterday_is_a_whole_day():
    now = datetime(2026, 10, 23, 15, 30)

    start, end, label, iso_day = day_window("today", now)
    assert (datetime.fromtimestamp(start), datetime.fromtimestamp(end), label) == (
        datetime(2026, 10, 23),
        now,
        "today",
    )
    start, end, label, iso_day = day_window("yesterday", now)
    assert (iso_day, end - start) == ("2026-10-22", 86400)


@pytest.mark.asyncio
async def test_overview_combines_receipts_timeline_and_undo(tmp_path):
    receipts = TaskReceipts(tmp_path / "r", vault=FakeVault())
    _record(receipts, "t1")
    await receipts.finalize("t1", request="tidy downloads", status="success", summary="ok", source="voice")
    timeline = ActivityTimeline(tmp_path / "t.db", config=TimelineConfig(enabled=True))
    now = datetime.now()
    timeline.observe("Code", "server.py", now.timestamp() - 120)
    timeline.observe("Code", "server.py", now.timestamp() - 60)
    undo = SimpleNamespace(
        list_tasks=AsyncMock(
            return_value=[{"task_key": "t1", "created_at": now.timestamp() - 30, "undoable": 2, "undone": False}]
        )
    )

    overview = await today_overview(receipts=receipts, timeline=timeline, undo_journal=undo)

    assert [task["request"] for task in overview["tasks"]] == ["tidy downloads"]
    assert overview["usage"]["apps"][0]["app"] == "Code"
    assert [row["task_key"] for row in overview["undoable"]] == ["t1"]
    assert overview["brief"].startswith('Today Heliox ran 1 task; the latest was "tidy downloads".')


@pytest.mark.asyncio
async def test_timeline_off_leaves_usage_out(tmp_path):
    timeline = ActivityTimeline(tmp_path / "t.db", config=TimelineConfig(enabled=False))

    overview = await today_overview(receipts=None, timeline=timeline, undo_journal=None)

    assert overview["timeline_enabled"] is False and overview["usage"] is None


class Clock:
    def __init__(self, when: datetime) -> None:
        self.when = when

    def __call__(self) -> datetime:
        return self.when


@pytest.mark.asyncio
async def test_brief_goes_out_once_after_its_time(tmp_path):
    config = BriefConfig(enabled=True, time="18:00")
    clock = Clock(datetime(2026, 10, 23, 17, 59))
    delivered = []

    async def build():
        return {"day": clock().date().isoformat(), "brief": "hi"}

    async def deliver(overview):
        delivered.append(overview["day"])

    scheduler = DailyBriefScheduler(
        config=lambda: config, build=build, deliver=deliver, state_file=tmp_path / "brief.json", now=clock
    )

    assert await scheduler.tick() is False
    clock.when = datetime(2026, 10, 23, 20, 15)  # the computer woke up after 18:00
    assert await scheduler.tick() is True
    assert await scheduler.tick() is False
    restarted = DailyBriefScheduler(
        config=lambda: config, build=build, deliver=deliver, state_file=tmp_path / "brief.json", now=clock
    )
    assert await restarted.tick() is False
    clock.when = datetime(2026, 10, 24, 18, 0)
    assert await scheduler.tick() is True
    assert delivered == ["2026-10-23", "2026-10-24"]


@pytest.mark.asyncio
async def test_brief_off_or_with_a_bad_time_never_sends(tmp_path):
    config = BriefConfig(enabled=False, time="18:00")
    scheduler = DailyBriefScheduler(
        config=lambda: config,
        build=AsyncMock(),
        deliver=AsyncMock(),
        state_file=tmp_path / "brief.json",
        now=Clock(datetime(2026, 10, 23, 23, 0)),
    )
    assert await scheduler.tick() is False
    config.enabled, config.time = True, "25:00"
    assert await scheduler.tick() is False


def test_brief_settings_merge_and_validate():
    config = PilotConfig()
    _merge_config(config, {"brief": {"enabled": True, "time": "7:5", "speak": True}})
    assert (config.brief.enabled, config.brief.time, config.brief.speak) == (True, "18:00", True)
    _merge_config(config, {"brief": {"time": "07:05"}})
    assert config.brief.time == "07:05"


@pytest.mark.asyncio
async def test_update_config_rejects_a_bad_brief_time():
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    bad = await server._handle_update_config({"section": "brief", "values": {"time": "6pm"}}, MagicMock())
    good = await server._handle_update_config({"section": "brief", "values": {"time": "06:30"}}, MagicMock())

    assert bad["status"] == "error" and good["status"] == "ok" and config.brief.time == "06:30"


def _server_with_tasks(rows):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._undo_journal = SimpleNamespace(list_tasks=AsyncMock(return_value=rows))
    server._broadcast_notification = AsyncMock()
    return server


def _row(key, undoable=1, undone=False):
    return {"task_key": key, "undoable": undoable, "undone": undone}


@pytest.mark.asyncio
async def test_undo_the_last_two_tasks_opens_both_newest_first():
    server = _server_with_tasks([_row("new"), _row("old"), _row("older")])

    message = await server._undo_request("Undo the last two tasks")

    assert message == "Here are your last 2 tasks, newest first. Check each one, then press Undo."
    server._broadcast_notification.assert_awaited_once_with(
        "undo_offer", {"task_key": "new", "task_keys": ["new", "old"]}
    )


@pytest.mark.asyncio
async def test_tasks_that_cannot_be_undone_are_skipped_and_said():
    server = _server_with_tasks([_row("new"), _row("read-only", undoable=0), _row("old", undone=True)])

    message = await server._undo_request("undo my last 2 tasks")

    assert message.startswith("I can undo 1 of your last 2 tasks.")
    assert server._broadcast_notification.await_args.args[1]["task_keys"] == ["new"]


@pytest.mark.asyncio
async def test_undo_requests_are_bounded_and_other_text_passes_through():
    server = _server_with_tasks([_row("new")])

    assert await server._undo_request("undo the last 9 tasks") == "I can open undo for up to 5 tasks at a time."
    assert await server._undo_request("undo that") == "Here is what I can undo. Check it, then press Undo."
    assert await server._undo_request("undo the changes in my essay") is None


@pytest.mark.asyncio
async def test_daily_brief_now_broadcasts_and_speaks_when_asked(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server.config.brief.speak = True
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()

    result = await server._handle_daily_brief_now({}, MagicMock())

    assert result["text"] == "Today Heliox ran no tasks."
    assert server._broadcast_notification.await_args.args[0] == "daily_brief"
    server._speak_companion_text.assert_awaited_once()

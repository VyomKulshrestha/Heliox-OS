"""Focus sessions hold reminders and the brief, keep background speech quiet, and summarize at the end."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.system.companion_speech import SpeechChannel
from pilot.workflows.daily_brief import compose_report, day_window
from pilot.workflows.focus import FocusSession, compose_focus_summary, parse_focus_minutes


@pytest.mark.parametrize(
    ("text", "minutes"),
    [
        ("focus for 45 minutes", 45),
        ("Heliox, start a focus session for an hour", 60),
        ("do not disturb for 2 hours", 120),
        ("focus for half an hour", 30),
        ("focus for 900 minutes", 0),
        ("focus on the report", None),
    ],
)
def test_parse_focus_minutes(text, minutes):
    assert parse_focus_minutes(text) == minutes


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_a_session_holds_items_and_hands_them_over_at_the_end(tmp_path):
    clock = Clock()
    ended = []

    async def on_end(summary):
        ended.append(summary)

    focus = FocusSession(tmp_path / "focus.json", on_end=on_end, clock=clock)
    assert focus.hold("reminder", {"text": "x"}) is False

    focus.start(30)
    assert focus.hold("reminder", {"text": "stretch"}) is True
    clock.now += 600
    summary = await focus.stop()

    assert summary["duration"] == "10 minutes"
    assert [item["payload"]["text"] for item in ended[0]["held"]] == ["stretch"]
    assert focus.active is False and focus.hold("reminder", {"text": "y"}) is False
    assert await focus.stop() is None


@pytest.mark.asyncio
async def test_a_running_session_survives_a_restart(tmp_path):
    clock = Clock()
    FocusSession(tmp_path / "focus.json", on_end=AsyncMock(), clock=clock).start(45)

    restarted = FocusSession(tmp_path / "focus.json", on_end=AsyncMock(), clock=clock)

    assert restarted.active is True
    assert restarted.status()["remaining_seconds"] == 45 * 60


def test_focus_summary_mentions_time_and_what_was_held():
    summary = {
        "duration": "45 minutes",
        "held": [{"kind": "reminder", "payload": {"text": "stretch"}}, {"kind": "daily_brief", "payload": {}}],
    }

    text = compose_focus_summary(summary, {"apps": [{"app": "Code", "seconds": 2400}]})

    assert text == (
        "Focus session done: 45 minutes. You spent most of it in Code. "
        "While you focused: 1 reminder (stretch). Your daily brief is below."
    )
    assert compose_focus_summary({"duration": "5 minutes", "held": []}, None).endswith(
        "Nothing needed you while you focused."
    )


def _server(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock()
    server._focus = FocusSession(tmp_path / "focus.json", on_end=server._on_focus_end)
    return server


@pytest.mark.asyncio
async def test_focus_holds_reminders_and_quiets_background_speech_but_not_answers(tmp_path):
    server = _server(tmp_path)

    reply = await server._control_reply("focus for 25 minutes")
    assert reply.startswith("Focusing until ")
    await server._deliver_reminder({"id": "a", "text": "stretch", "due": 0, "late": False})
    assert all(call.args[0] != "reminder" for call in server._broadcast_notification.await_args_list)

    quiet = await server._speak_companion_text("Tip", channel=SpeechChannel.PROACTIVE_SUGGESTION)
    assert quiet.status == "suppressed"
    await server._speak_companion_text("Your result", channel=SpeechChannel.FINAL_ANSWER)
    server._speech_coordinator.speak.assert_awaited_once()

    summary = await server._control_reply("stop focusing")
    assert summary.startswith("Focus session done:") and "1 reminder (stretch)" in summary
    reminders = [call.args[1] for call in server._broadcast_notification.await_args_list if call.args[0] == "reminder"]
    assert reminders == [{"id": "a", "text": "stretch", "due": 0, "late": False, "held": True}]
    assert await server._control_reply("stop focusing") == "No focus session is running."


@pytest.mark.asyncio
async def test_focus_rpcs(tmp_path):
    server = _server(tmp_path)

    assert (await server._handle_focus_start({"minutes": 0}, MagicMock()))["status"] == "error"
    started = await server._handle_focus_start({"minutes": 15}, MagicMock())
    assert started["active"] is True
    assert (await server._handle_focus_status({}, MagicMock()))["remaining_seconds"] > 0
    stopped = await server._handle_focus_stop({}, MagicMock())
    assert stopped["summary"].startswith("Focus session done:")


def test_week_window_covers_the_last_seven_days():
    now = datetime(2026, 10, 23, 15, 0)

    start, end, label, iso_day = day_window("week", now)

    assert (iso_day, label) == ("2026-10-17", "this week")
    assert datetime.fromtimestamp(start) == datetime(2026, 10, 17)


def test_report_lists_tasks_time_and_undo():
    overview = {
        "label": "this week",
        "day": "2026-10-17",
        "brief": "This week Heliox ran 1 task.",
        "tasks": [
            {
                "finished_at": "2026-10-20T09:00:00+00:00",
                "request": "tidy | sort",
                "summary": "",
                "status": "success",
                "action_count": 3,
                "receipt_id": "abcdef1234567890",
            }
        ],
        "timeline_enabled": True,
        "usage": {"apps": [{"app": "Code", "seconds": 3600}]},
        "undoable": [{"task_key": "t1", "targets": ["report.pdf"], "undoable": 2}],
    }

    report = compose_report(overview)

    assert report.startswith("# Heliox report: this week (from 2026-10-17)")
    assert "| tidy \\| sort | success | 3 | `abcdef123456` |" in report
    assert "- Code: 1 h" in report and "- report.pdf (2 changes)" in report


@pytest.mark.asyncio
async def test_report_export_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / "Downloads").mkdir()
    server = _server(tmp_path)

    result = await server._handle_report_export({"day": "week"}, MagicMock())

    assert result["status"] == "ok"
    assert (
        (tmp_path / "Downloads")
        .joinpath(result["path"].split("Downloads")[-1].strip("\\/"))
        .read_text(encoding="utf-8")
        .startswith("# Heliox report: this week")
    )
    assert (await server._handle_report_export({"day": "month"}, MagicMock()))["status"] == "error"

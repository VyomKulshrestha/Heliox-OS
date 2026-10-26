"""Repeating reminders and reminders that wait for an app to open."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.app_reminders import AppReminders
from pilot.workflows.reminders import ReminderStore, describe_repeat, next_due, parse_repeating

NOW = datetime(2026, 10, 26, 14, 10)  # a Monday afternoon


@pytest.mark.parametrize(
    ("sentence", "first", "what", "label"),
    [
        ("remind me every hour to stand up", datetime(2026, 10, 26, 15, 10), "stand up", "every hour"),
        (
            "remind me to drink water every 30 minutes",
            datetime(2026, 10, 26, 14, 40),
            "drink water",
            "every 30 minutes",
        ),
        ("remind me every two hours to stretch", datetime(2026, 10, 26, 16, 10), "stretch", "every 2 hours"),
        (
            "remind me every weekday at 9 to check email",
            datetime(2026, 10, 27, 9, 0),
            "check email",
            "every weekday at 9:00 AM",
        ),
        (
            "remind me every Monday at 8:30 am to plan the week",
            datetime(2026, 11, 2, 8, 30),
            "plan the week",
            "every Monday at 8:30 AM",
        ),
        (
            "remind me every evening to water the plants",
            datetime(2026, 10, 26, 18, 0),
            "water the plants",
            "every day at 6:00 PM",
        ),
        (
            "remind me every weekend at 10 am to call mum",
            datetime(2026, 10, 31, 10, 0),
            "call mum",
            "every weekend day at 10:00 AM",
        ),
    ],
)
def test_parse_repeating(sentence, first, what, label):
    due, text, repeat = parse_repeating(sentence, NOW)
    assert datetime.fromtimestamp(due) == first
    assert text == what
    assert describe_repeat(repeat) == label


def test_repeating_limits_and_one_offs():
    assert parse_repeating("remind me every 5 minutes to blink", NOW).startswith(
        "A repeating reminder can come every 15"
    )
    assert parse_repeating("remind me in 20 minutes to stretch", NOW) is None
    assert parse_repeating("open every file", NOW) is None


def test_interval_reminders_skip_the_night():
    assert next_due({"minutes": 60}, datetime(2026, 10, 26, 21, 30)) == datetime(2026, 10, 27, 7, 0)
    assert next_due({"minutes": 60}, datetime(2026, 10, 27, 2, 0)) == datetime(2026, 10, 27, 7, 0)
    assert next_due({"minutes": 30}, datetime(2026, 10, 27, 7, 0)) == datetime(2026, 10, 27, 7, 30)
    # Fixed-time reminders are not moved: the person chose the time.
    assert next_due({"days": [0, 1, 2, 3, 4, 5, 6], "time": "23:00"}, NOW) == datetime(2026, 10, 26, 23, 0)


@pytest.mark.asyncio
async def test_repeating_reminder_is_rescheduled_not_removed(tmp_path):
    clock = MagicMock(return_value=NOW.timestamp())
    store = ReminderStore(tmp_path / "r.json", deliver=AsyncMock(), clock=clock)
    assert store.answer("remind me every hour to stand up", now=NOW) == (
        "Okay, I'll remind you every hour to stand up. The first one is at 3:10 PM."
    )
    assert store.answer("what are my reminders", now=NOW) == "You have 1 reminder. every hour: stand up."

    clock.return_value = datetime(2026, 10, 26, 15, 10, 5).timestamp()
    assert await store.tick() == 1
    assert store._deliver.await_args.args[0]["text"] == "stand up"
    assert datetime.fromtimestamp(store.list()[0]["due"]) == datetime(2026, 10, 26, 16, 10)

    # Away for three hours: the missed ones are skipped, the next keeps the same cadence.
    clock.return_value = datetime(2026, 10, 26, 19, 0).timestamp()
    assert await store.tick() == 0
    assert datetime.fromtimestamp(store.list()[0]["due"]) == datetime(2026, 10, 26, 19, 10)

    assert store.answer("stop reminding me to stand up", now=NOW) == "Cancelled 1 reminder."
    assert store.list() == []


@pytest.fixture
def front():
    return {"window": ("Code", "server.py")}


@pytest.fixture
def app_reminders(tmp_path, front):
    return AppReminders(tmp_path / "app_reminders.json", deliver=AsyncMock(), foreground=lambda: front["window"])


@pytest.mark.asyncio
async def test_app_reminder_fires_once_when_the_app_opens(app_reminders, front):
    assert app_reminders.answer("Remind me to submit my timesheet when I open Outlook") == (
        "Okay, next time you open outlook, I'll remind you to submit my timesheet."
    )
    assert await app_reminders.check() == 0
    front["window"] = ("OUTLOOK", "Inbox - jane@example.com - Outlook")
    assert await app_reminders.check() == 1
    app_reminders._deliver.assert_awaited_once()
    assert app_reminders._deliver.await_args.args[0]["text"] == "submit my timesheet"
    assert app_reminders.list() == []  # once only


@pytest.mark.asyncio
async def test_every_time_reminders_wait_for_each_switch(app_reminders, front):
    front["window"] = ("slack", "general - Slack")
    app_reminders.answer("every time I open Slack, remind me to set my status")
    assert await app_reminders.check() == 0  # it was already in front when set

    front["window"] = ("Code", "x")
    assert await app_reminders.check() == 0
    front["window"] = ("slack", "random - Slack")
    assert await app_reminders.check() == 1
    assert await app_reminders.check() == 0  # still in front: not again
    front["window"] = ("Code", "x")
    await app_reminders.check()
    front["window"] = ("slack", "general - Slack")
    assert await app_reminders.check() == 1
    assert len(app_reminders.list()) == 1


def test_app_reminder_phrases(app_reminders):
    assert app_reminders.answer("when I open Excel remind me to check the totals").startswith(
        "Okay, next time you open excel"
    )
    assert app_reminders.answer("remind me when I open the Teams app to share the deck").startswith(
        "Okay, next time you open teams"
    )
    assert app_reminders.answer("what app reminders do I have") == (
        "Your app reminders: when you open excel: check the totals; when you open teams: share the deck."
    )
    for text in ("remind me in 10 minutes to stretch", "remind me to call mum when I get home", "open outlook"):
        assert app_reminders.answer(text) is None, text


@pytest.mark.asyncio
async def test_rpcs_and_routing(tmp_path, front):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._reminders = ReminderStore(tmp_path / "r.json")
    server._app_reminders = AppReminders(
        tmp_path / "a.json", deliver=server._deliver_app_reminder, foreground=lambda: front["window"]
    )
    ws = MagicMock()

    # The app reminder is answered before the time reminders, which would ask "when?".
    reply = await server._control_reply("remind me to submit my timesheet when I open Outlook")
    assert reply.startswith("Okay, next time you open outlook")
    assert server._reminders.list() == []

    front["window"] = ("OUTLOOK", "Inbox - Outlook")
    await server._app_reminders.check()
    name, payload = server._broadcast_notification.await_args.args
    assert name == "reminder" and payload["text"] == "submit my timesheet" and payload["trigger"] == "app"
    assert server._speak_companion_text.await_args.args[0] == "Reminder: submit my timesheet"

    await server._control_reply("every time I open Slack remind me to set my status")
    listed = await server._handle_app_reminder_list({}, ws)
    assert [row["app"] for row in listed["reminders"]] == ["slack"]
    assert (await server._handle_app_reminder_delete({"id": listed["reminders"][0]["id"]}, ws))["reminders"] == []
    assert (await server._handle_app_reminder_delete({"id": "missing"}, ws))["status"] == "error"

"""Quick notes, usage nudges, and everyday settings by voice."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.focus import FocusSession
from pilot.workflows.notes import NoteStore
from pilot.workflows.nudges import NudgeStore


def test_notes_take_read_search_and_delete(tmp_path):
    notes = NoteStore(tmp_path / "n.json")

    assert notes.answer("Note that the plumber comes Tuesday") == "Noted."
    assert notes.answer("take a note: call the bank about the card") == "Noted."
    assert notes.answer("read my notes").startswith("Your latest notes: ")
    assert notes.answer("what did I note about the bank").endswith("call the bank about the card.")
    assert notes.answer("what did I note about the garden") == "You have no note about the garden."
    assert notes.answer("delete my last note") == 'Deleted your last note: "call the bank about the card".'
    assert [item["text"] for item in notes.list()] == ["the plumber comes Tuesday"]
    assert notes.answer("open the report") is None


@pytest.mark.parametrize(
    ("text", "match", "minutes"),
    [
        ("Tell me if I spend more than an hour on YouTube today", "youtube", 60),
        ("warn me after 30 minutes of Twitter", "twitter", 30),
        ("tell me when I spend over 2 hours in Slack", "slack", 120),
        ("nudge me after half an hour on reddit a day", "reddit", 30),
    ],
)
def test_nudge_phrases(tmp_path, text, match, minutes):
    store = NudgeStore(tmp_path / "n.json", timeline_on=lambda: True)

    assert store.answer(text).startswith("Okay. I'll tell you once a day")
    assert [(item["match"], item["minutes"]) for item in store.list()] == [(match, minutes)]


def test_nudges_need_the_timeline(tmp_path):
    store = NudgeStore(tmp_path / "n.json", timeline_on=lambda: False)

    assert "Turn it on in Settings first" in store.answer("warn me after 30 minutes of Twitter")
    assert store.list() == []


@pytest.mark.asyncio
async def test_a_nudge_fires_once_a_day_when_the_limit_is_passed(tmp_path):
    seconds = {"value": 20 * 60}
    clock = {"now": datetime(2026, 10, 24, 15, 0)}
    delivered = []

    async def deliver(nudge):
        delivered.append(nudge["text"])

    store = NudgeStore(
        tmp_path / "n.json",
        usage=lambda needle, start, end: seconds["value"],
        timeline_on=lambda: True,
        deliver=deliver,
        now=lambda: clock["now"],
    )
    store.add("YouTube", 30)

    assert await store.tick() == 0
    seconds["value"] = 31 * 60
    assert await store.tick() == 1
    assert await store.tick() == 0
    clock["now"] = datetime(2026, 10, 25, 15, 0)
    assert await store.tick() == 1
    assert delivered == ["You have spent 31 min on YouTube today."] * 2


def test_timeline_time_matching(tmp_path):
    from pilot.config import TimelineConfig
    from pilot.workflows.activity_timeline import ActivityTimeline

    timeline = ActivityTimeline(tmp_path / "t.db", config=TimelineConfig(enabled=True))
    for second in range(0, 601, 30):
        timeline.observe("chrome", "Cats - YouTube", 1_800_000_000 + second)
    for second in range(630, 931, 30):
        timeline.observe("Code", "server.py", 1_800_000_000 + second)

    assert timeline.time_matching("youtube", 1_800_000_000, 1_800_001_000) == 600
    assert timeline.time_matching("discord", 1_800_000_000, 1_800_001_000) == 0


def _server(tmp_path):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._notes = NoteStore(tmp_path / "notes.json")
    return server


@pytest.mark.asyncio
async def test_everyday_settings_change_by_voice(tmp_path):
    server = _server(tmp_path)

    assert await server._control_reply("turn on the activity timeline") == "The activity timeline is on."
    assert server.config.timeline.enabled is True
    assert await server._control_reply("send me a daily brief at 7 pm") == (
        "I'll send your daily brief every day at 19:00."
    )
    assert (server.config.brief.enabled, server.config.brief.time) == (True, "19:00")
    assert await server._control_reply("read the brief aloud") == "I'll read the daily brief aloud."
    assert server.config.brief.speak is True
    assert await server._control_reply("stop recording my activity") == "The activity timeline is off."
    methods = [call.args[0] for call in server._broadcast_notification.await_args_list]
    assert methods.count("config_updated") == 4


@pytest.mark.asyncio
async def test_security_settings_are_not_voice_settings(tmp_path):
    server = _server(tmp_path)

    for text in ("turn off approvals", "turn on root access", "turn off the gateway", "disable dry run"):
        assert await server._settings_request(text) is None


@pytest.mark.asyncio
async def test_focus_holds_nudges(tmp_path):
    server = _server(tmp_path)
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock()
    server._focus = FocusSession(tmp_path / "focus.json", on_end=server._on_focus_end)
    server._focus.start(30)

    await server._deliver_nudge({"id": "a", "match": "youtube", "minutes": 30, "text": "You have spent 31 min."})
    assert all(call.args[0] != "nudge" for call in server._broadcast_notification.await_args_list)

    summary = await server._focus.stop()
    assert "You have spent 31 min." in summary["text"]
    assert any(call.args[0] == "nudge" for call in server._broadcast_notification.await_args_list)


@pytest.mark.asyncio
async def test_note_and_nudge_rpcs(tmp_path):
    server = _server(tmp_path)
    server._nudges = NudgeStore(tmp_path / "nudges.json", timeline_on=lambda: True)

    added = await server._handle_note_add({"text": "buy milk"}, MagicMock())
    assert [item["text"] for item in added["notes"]] == ["buy milk"]
    note_id = added["notes"][0]["id"]
    assert (await server._handle_note_delete({"id": note_id}, MagicMock()))["notes"] == []

    nudged = await server._handle_nudge_add({"match": "YouTube", "minutes": 45}, MagicMock())
    assert nudged["nudges"][0]["minutes"] == 45
    assert (await server._handle_nudge_add({"match": "x", "minutes": 0}, MagicMock()))["status"] == "error"
    nudge_id = nudged["nudges"][0]["id"]
    assert (await server._handle_nudge_delete({"id": nudge_id}, MagicMock()))["nudges"] == []

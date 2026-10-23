"""Routines from finished tasks, voice shortcuts, and "do that again"."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionResult, ActionType, EmptyParams, FileParams
from pilot.config import PilotConfig
from pilot.workflows.routines import (
    ShortcutStore,
    parse_schedule,
    parse_shortcut_definition,
    routine_from_receipt,
)
from pilot.workflows.task_receipts import TaskReceipts
from tests.test_task_receipts import FakeVault


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("every day at 9", {"daily_at": "09:00", "weekdays": [], "label": "every day at 9:00 AM"}),
        ("every weekday at 8:30 am", {"daily_at": "08:30", "weekdays": [0, 1, 2, 3, 4]}),
        ("every monday at 6pm", {"daily_at": "18:00", "weekdays": [0], "label": "every Monday at 6:00 PM"}),
        ("every evening", {"daily_at": "18:00", "weekdays": []}),
        ("every 2 hours", {"every_minutes": 120, "label": "every 2 hours"}),
        ("every hour", {"every_minutes": 60, "label": "every hour"}),
    ],
)
def test_parse_schedule(text, expected):
    parsed = parse_schedule(text)

    assert parsed is not None
    assert {key: parsed[key] for key in expected} == expected


def test_unclear_schedules_are_not_guessed():
    assert parse_schedule("every so often") is None
    assert parse_schedule("every day at 27") is None


def _envelope(actions, request="tidy my downloads"):
    return {"receipt": {"receipt_id": "r1", "request": request, "actions": actions}}


def test_a_routine_may_use_only_the_actions_that_succeeded():
    routine = routine_from_receipt(
        _envelope(
            [
                {"action_type": "file_list", "executed": True, "success": True},
                {"action_type": "file_move", "executed": True, "success": True},
                {"action_type": "file_move", "executed": True, "success": True},
                {"action_type": "file_delete", "executed": True, "success": False},
                {"action_type": "shell_command", "executed": False, "success": False},
            ]
        )
    )

    assert routine["allowed_actions"] == ["file_list", "file_move"]
    assert routine["goal"] == "tidy my downloads"


def test_no_routine_from_a_task_with_nothing_completed():
    assert routine_from_receipt(_envelope([{"action_type": "file_list", "success": False}])) is None
    assert routine_from_receipt(_envelope([{"action_type": "file_list", "success": True}], request="")) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("When I say start work, open VS Code and Slack", ("start work", "open VS Code and Slack")),
        ("when i say start work open VS Code", ("start work", "open VS Code")),
        ('When I say "good night" lock the computer', ("good night", "lock the computer")),
        ("when I say focus then mute notifications", ("focus", "mute notifications")),
    ],
)
def test_parse_shortcut_definition(text, expected):
    assert parse_shortcut_definition(text) == expected


def test_shortcut_store_adds_matches_replaces_and_removes(tmp_path):
    store = ShortcutStore(tmp_path / "s.json", reserved={"stop"})

    store.add("Start work!", "open VS Code")
    store.add("start work", "open VS Code and Slack")

    assert len(store.list()) == 1
    assert store.match("Heliox, start work")["command"] == "open VS Code and Slack"
    assert store.match("start working") is None
    with pytest.raises(ValueError, match="already means something"):
        store.add("stop", "close everything")
    assert store.remove("start work") is True
    assert store.list() == []


def _receipts_with_task(tmp_path, request="tidy my downloads", action_type=ActionType.FILE_LIST):
    receipts = TaskReceipts(tmp_path / "r", vault=FakeVault())
    params = FileParams(path="~/Downloads") if action_type.value.startswith("file") else EmptyParams()
    action = Action(action_type=action_type, target="x", parameters=params)
    receipts.record_action("t1", "p1", action, ActionResult(action=action, success=True, output="ok"))
    return receipts, request


async def _server(tmp_path, with_task=True):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._shortcuts = ShortcutStore(tmp_path / "s.json", reserved=server._phrase_reserved)
    if with_task:
        receipts, request = _receipts_with_task(tmp_path)
        await receipts.finalize("t1", request=request, status="success", summary="ok", source="voice")
        server._task_receipts = receipts
    return server


@pytest.mark.asyncio
async def test_do_that_every_day_proposes_a_least_privilege_routine(tmp_path):
    server = await _server(tmp_path)

    reply = await server._control_reply("do that every weekday at 8:30 am")

    assert reply.startswith('I can run "tidy my downloads" every weekday at 8:30 AM, using only the 1 kind')
    method, offer = server._broadcast_notification.await_args.args
    assert method == "routine_offer"
    assert offer["allowed_actions"] == ["file_list"] and offer["daily_at"] == "08:30"
    assert offer["weekdays"] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_routine_needs_a_clear_time_and_a_finished_task(tmp_path):
    server = await _server(tmp_path, with_task=False)

    assert "Tell me when" in await server._control_reply("do that every so often")
    assert "no completed actions" in await server._control_reply("do that every day at 9")


@pytest.mark.asyncio
async def test_saying_a_shortcut_runs_its_command_and_do_that_again_repeats(tmp_path):
    server = await _server(tmp_path)

    reply = await server._control_reply("When I say start work, open VS Code and Slack")
    assert reply == 'Okay. When you say "start work", I\'ll run: open VS Code and Slack'

    assert await server._expand_request("start work") == (
        "open VS Code and Slack",
        'Shortcut "start work": open VS Code and Slack',
    )
    assert await server._expand_request("do that again") == ("tidy my downloads", 'Again: "tidy my downloads"')
    assert await server._expand_request("open the report") is None


@pytest.mark.asyncio
async def test_safety_phrases_cannot_become_shortcuts(tmp_path):
    server = await _server(tmp_path)

    for phrase in ("stop", "approve", "undo that", "what did you do today"):
        reply = await server._control_reply(f"when I say {phrase}, open notepad")
        assert "cannot be a shortcut" in reply, phrase
    assert server._shortcuts.list() == []


@pytest.mark.asyncio
async def test_nothing_to_repeat_is_said_plainly(tmp_path):
    server = await _server(tmp_path, with_task=False)

    assert await server._expand_request("do that again") is None
    assert await server._control_reply("do that again") == "There is no earlier task for me to repeat yet."


@pytest.mark.asyncio
async def test_shortcut_rpcs_and_routine_suggest(tmp_path):
    server = await _server(tmp_path)

    added = await server._handle_shortcut_add({"phrase": "good night", "command": "lock the computer"}, MagicMock())
    assert added["status"] == "ok" and added["shortcuts"][0]["phrase"] == "good night"
    assert (await server._handle_shortcut_add({"phrase": "no", "command": "x"}, MagicMock()))["status"] == "error"
    assert (await server._handle_shortcut_remove({"phrase": "good night"}, MagicMock()))["shortcuts"] == []

    receipt_id = (await server._task_receipts.list(1))[0]["receipt_id"]
    suggested = await server._handle_routine_suggest({"receipt_id": receipt_id}, MagicMock())
    assert suggested["status"] == "ok"
    assert suggested["routine"]["allowed_actions"] == ["file_list"]
    assert len(suggested["routine"]["daily_at"]) == 5

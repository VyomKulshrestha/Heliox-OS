"""The to-do list: voice phrases, the store, the RPCs and the daily brief."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.daily_brief import compose_brief, compose_report, today_overview
from pilot.workflows.todos import TodoStore


@pytest.fixture
def todos(tmp_path):
    return TodoStore(tmp_path / "todos.json")


def test_add_read_finish_and_clear(todos):
    assert todos.answer("Add Call the bank to my to-do list") == 'Added "Call the bank". You have 1 thing to do.'
    assert todos.answer("put renew passport on my list").startswith('Added "renew passport"')
    assert todos.answer("What's on my list?") == "You have 2 things to do: Call the bank; renew passport."

    assert todos.answer("mark call the bank as done") == 'Done: "Call the bank". 1 left.'
    assert todos.answer("check off renew passport") == 'Done: "renew passport". Your list is clear.'
    assert todos.answer("what do I have to do").startswith("Your list is empty")

    assert todos.answer("clear finished items") == "Cleared 2 finished items."
    assert todos.list() == []
    assert todos.answer("clear completed tasks") == "There are no finished items to clear."


def test_duplicates_and_limits(todos):
    todos.add("buy milk")
    assert todos.answer("add buy milk to my list") == '"buy milk" is already on your list'
    with pytest.raises(ValueError):
        todos.add("x" * 201)
    todos.set_done(todos.list()[0]["id"])
    todos.add("buy milk")  # a finished item does not block adding it again
    assert len(todos.list()) == 2


def test_mark_done_leaves_planner_tasks_alone(todos):
    todos.add("call the bank")
    # Not on the list and the list is not named: this is a task for the planner.
    assert todos.answer("mark this email as done") is None
    assert todos.answer("check off the first box") is None
    # Naming the list makes it a list request even without a match.
    assert todos.answer("mark the report as done on my list") == 'There is no "the report" on your list.'


def test_ambiguous_and_partial_matches(todos):
    todos.add("call the bank")
    todos.add("call mum")
    assert todos.answer("mark call as done on my list") == "Which one: call the bank; call mum?"
    assert todos.answer("cross off bank") == 'Done: "call the bank". 1 left.'
    assert todos.answer("remove call mum from my list") == 'Removed "call mum" from your list.'
    assert todos.answer("remove taxes from my list") == 'There is no "taxes" on your list.'


def test_unrelated_sentences_are_not_list_requests(todos):
    for text in (
        "add this song to my playlist",
        "add bob to my contacts list",
        "what do I need to do to install python",
        "open my list of projects",
        "note that the plumber comes Tuesday",
    ):
        assert todos.answer(text) is None, text


def test_brief_and_report_mention_open_items(todos):
    todos.add("call the bank")
    todos.add("renew passport")
    overview = {
        "label": "today",
        "day": "2026-10-26",
        "tasks": [],
        "usage": None,
        "undoable": [],
        "timeline_enabled": False,
        "todos": todos.open_items(),
    }
    overview["brief"] = compose_brief(overview)

    assert 'You have 2 things to do, starting with "call the bank".' in overview["brief"]
    assert "- [ ] renew passport" in compose_report(overview)


@pytest.mark.asyncio
async def test_overview_includes_open_todos(todos):
    todos.add("call the bank")
    done = todos.add("buy milk")
    todos.set_done(done["id"])

    overview = await today_overview(
        receipts=None, timeline=None, undo_journal=None, todos=todos, now=datetime(2026, 10, 26, 12)
    )

    assert [item["text"] for item in overview["todos"]] == ["call the bank"]


@pytest.mark.asyncio
async def test_todo_rpcs(tmp_path):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._todos = TodoStore(tmp_path / "todos.json")
    ws = MagicMock()

    added = await server._handle_todo_add({"text": "call the bank"}, ws)
    item_id = added["todos"][0]["id"]
    assert added["status"] == "ok"
    assert (await server._handle_todo_add({"text": ""}, ws))["status"] == "error"

    finished = await server._handle_todo_set_done({"id": item_id, "done": True}, ws)
    assert finished["todos"][0]["done"] is True
    assert (await server._handle_todo_set_done({"id": item_id, "done": "yes"}, ws))["status"] == "error"
    assert (await server._handle_todo_set_done({"id": "missing"}, ws))["status"] == "error"

    cleared = await server._handle_todo_clear_done({}, ws)
    assert cleared["removed"] == 1 and cleared["todos"] == []
    assert (await server._handle_todo_delete({"id": item_id}, ws))["status"] == "error"

    assert (
        await server._control_reply("add renew passport to my list")
        == 'Added "renew passport". You have 1 thing to do.'
    )
    assert (await server._handle_todo_list({}, ws))["todos"][0]["text"] == "renew passport"

"""Goal threads: long-lived goals with updates, next steps and status, surfaced in the morning."""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.day_briefing import compose_morning
from pilot.workflows.goals import GoalStore


@pytest.fixture
def goals(tmp_path):
    return GoalStore(tmp_path / "goals.json")


def test_create_update_next_step_and_read_back(goals):
    assert goals.answer("start a goal: launch the podcast").startswith('Started the goal "launch the podcast"')
    assert goals.answer("new goal to learn Spanish").startswith('Started the goal "learn Spanish"')
    assert (
        goals.answer("add to my podcast goal: recorded episode one")
        == 'Logged on "launch the podcast": recorded episode one.'
    )
    assert (
        goals.answer("set the next step on the podcast to book a guest")
        == 'Next step on "launch the podcast": book a guest.'
    )
    assert (
        goals.answer("what's my next step on the podcast") == 'Your next step on "launch the podcast" is: book a guest.'
    )
    assert goals.answer("where am I with learn Spanish").startswith('No next step set on "learn Spanish" yet')
    assert goals.answer("what are my goals") == (
        "You have 2 goals: launch the podcast (next: book a guest); learn Spanish."
    )


def test_status_done_and_delete(goals):
    goals.answer("start a goal to launch the podcast")
    goals.answer("start a goal to learn Spanish")
    assert goals.answer("pause the podcast goal") == 'Okay, "launch the podcast" is paused.'
    assert goals.answer("resume the podcast goal") == 'Okay, "launch the podcast" is active.'
    assert goals.answer("I finished the Spanish goal") == 'Nice work. I\'ve marked "learn Spanish" done.'
    assert [g["title"] for g in goals.list() if g["status"] != "done"] == ["launch the podcast"]
    assert goals.answer("drop the podcast goal") == 'Dropped the goal "launch the podcast".'
    # The finished goal stays (done, not deleted); the dropped one is gone.
    assert [(g["title"], g["status"]) for g in goals.list()] == [("learn Spanish", "done")]


def test_duplicate_and_limit(goals):
    goals.create("launch the podcast")
    assert goals.answer("start a goal to launch the podcast") == 'You already have a goal for "launch the podcast"'
    with pytest.raises(ValueError):
        goals.create("x")


def test_tasks_and_questions_pass_through(goals):
    goals.create("launch the podcast")
    for text in (
        "update chrome",
        "how is the weather",
        "open the podcast app",
        "start the dishwasher",
        "what's my cpu usage",
        "remind me to call mom",
        "log out",
        "where is my budget spreadsheet",
    ):
        assert goals.answer(text) is None, text


def test_next_steps_and_stale(goals):
    goals.create("launch the podcast")
    goals.create("learn Spanish")
    podcast = goals.find("podcast")
    goals.set_next(podcast["id"], "book a guest")
    assert goals.next_steps() == [{"title": "launch the podcast", "next_step": "book a guest"}]

    stale_goal = goals.find("learn Spanish")
    data = goals._load()
    for g in data:
        if g["id"] == stale_goal["id"]:
            g["updated_at"] = time.time() - 20 * 86400
    goals._save(data)
    assert [g["title"] for g in goals.stale()] == ["learn Spanish"]


def test_morning_names_goal_next_steps():
    now = datetime(2026, 10, 31, 8, 30)
    text = compose_morning(
        now,
        reminders=[],
        todos=[],
        habits=[],
        name="Vyom",
        goals=[{"title": "the podcast", "next_step": "book a guest"}],
    )
    assert "On the podcast, your next step is book a guest." in text


@pytest.mark.asyncio
async def test_rpcs_and_routing(tmp_path):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._goals = GoalStore(tmp_path / "g.json")
    ws = MagicMock()

    assert (await server._control_reply("start a goal: launch the podcast")).startswith("Started the goal")
    listed = await server._handle_goal_list({}, ws)
    goal_id = listed["goals"][0]["id"]
    assert listed["goals"][0]["title"] == "launch the podcast" and listed["goals"][0]["stale"] is False

    assert (await server._handle_goal_update({"id": goal_id, "text": "recorded episode one"}, ws))["status"] == "ok"
    assert (await server._handle_goal_set_next({"id": goal_id, "step": "book a guest"}, ws))["goals"][0][
        "next_step"
    ] == "book a guest"
    assert (await server._handle_goal_set_status({"id": goal_id, "status": "done"}, ws))["goals"][0]["status"] == "done"
    assert (await server._handle_goal_set_status({"id": goal_id, "status": "bogus"}, ws))["status"] == "error"
    assert (await server._handle_goal_create({"title": "x"}, ws))["status"] == "error"
    assert (await server._handle_goal_delete({"id": goal_id}, ws))["goals"] == []
    assert (await server._handle_goal_delete({"id": "missing"}, ws))["status"] == "error"
    assert server._phrase_reserved("what are my goals")
    assert not server._phrase_reserved("open chrome")

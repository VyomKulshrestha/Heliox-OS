"""Proactive suggestions: one timely nudge from local signals, suggestion-only."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.workflows.suggestions import MIN_GAP_SECONDS, Suggester, SuggestionLoop


def _suggester(**over):
    base = dict(
        goals=list,
        stale_goals=list,
        habits=list,
        open_todos=list,
        now=lambda: datetime(2026, 10, 31, 19, 0),
    )
    base.update(over)
    return Suggester(**base)


def test_priority_order():
    stale = _suggester(
        stale_goals=lambda: [{"id": "g1", "title": "the podcast", "next_step": "book a guest"}],
        habits=lambda: [{"name": "meditation", "streak": 4, "today": 0}],
    )
    assert stale.suggest()["kind"] == "stale_goal"

    habit = _suggester(habits=lambda: [{"name": "meditation", "streak": 4, "today": 0}])
    assert habit.suggest()["kind"] == "habit"

    todo = _suggester(open_todos=lambda: [{"text": "call the bank"}])
    assert todo.suggest()["kind"] == "todos"

    step = _suggester(goals=lambda: [{"id": "g2", "title": "learn Spanish", "status": "active", "next_step": "unit 4"}])
    assert step.suggest()["kind"] == "goal_step"

    assert _suggester().suggest() is None


def test_morning_holds_habit_and_todo_suggestions():
    morning = _suggester(
        habits=lambda: [{"name": "meditation", "streak": 4, "today": 0}],
        open_todos=lambda: [{"text": "call the bank"}],
        now=lambda: datetime(2026, 10, 31, 8, 0),
    )
    assert morning.suggest() is None  # a streak-reminder at 8am would be pushy


def test_spoken_and_answer_routing():
    s = _suggester(stale_goals=lambda: [{"id": "g1", "title": "the podcast", "next_step": "book a guest"}])
    spoken = s.answer("what should I do now")
    assert "the podcast" in spoken and 'say "remind me to book a guest"' in spoken
    assert _suggester().answer("any suggestions").startswith("Nothing pressing")
    for text in ("what's the weather", "open chrome", "what are my goals"):
        assert s.answer(text) is None, text


@pytest.mark.asyncio
async def test_loop_only_when_enabled_spaced_and_deduped():
    enabled = {"on": False}
    clock = {"t": 1000.0}
    delivered = []
    suggester = _suggester(stale_goals=lambda: [{"id": "g1", "title": "the podcast", "next_step": "book a guest"}])
    loop = SuggestionLoop(
        suggester,
        enabled=lambda: enabled["on"],
        deliver=AsyncMock(side_effect=lambda p: delivered.append(p)),
        clock=lambda: clock["t"],
        today=lambda: "2026-10-31",
    )
    assert await loop.tick() is False  # off by default
    enabled["on"] = True
    assert await loop.tick() is True  # first one goes out
    assert await loop.tick() is False  # same thing, same day: not again
    clock["t"] += MIN_GAP_SECONDS + 1
    assert await loop.tick() is False  # still the same suggestion key today
    assert len(delivered) == 1


def test_config_default_off():
    assert PilotConfig().suggestions.enabled is False
    assert _merge_config(PilotConfig(), {"suggestions": {"enabled": True}}).suggestions.enabled is True


@pytest.mark.asyncio
async def test_rpc_voice_toggle_and_delivery(tmp_path):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._focus = None
    server._suggester = _suggester(
        stale_goals=lambda: [{"id": "g1", "title": "the podcast", "next_step": "book a guest"}]
    )
    server._suggestions = object()
    ws = MagicMock()

    now = await server._handle_suggestion_now({}, ws)
    assert now["suggestion"]["kind"] == "stale_goal" and now["enabled"] is False

    assert (await server._control_reply("turn on suggestions")).startswith("Okay, I'll offer a suggestion")
    assert server.config.suggestions.enabled is True
    assert (await server._control_reply("turn off suggestions")).startswith("Okay, I'll keep suggestions")
    assert server.config.suggestions.enabled is False

    await server._deliver_suggestion({"key": "stale:g1", "text": "pick up the podcast", "say": None})
    name, payload = server._broadcast_notification.await_args.args
    assert name == "suggestion" and payload["text"] == "pick up the podcast"
    assert server._speak_companion_text.await_args.args[0] == "pick up the podcast"

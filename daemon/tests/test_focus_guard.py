"""Focus distractions: one heads-up per word per session, never blocking or recording."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.focus_guard import FocusGuard


@pytest.fixture
def state():
    return {"focus": {"active": False}, "front": ("msedge", "YouTube - Cat videos")}


@pytest.fixture
def guard(tmp_path, state):
    return FocusGuard(
        tmp_path / "focus_distractions.json",
        focus_status=lambda: state["focus"],
        deliver=AsyncMock(),
        foreground=lambda: state["front"],
    )


def test_add_list_and_remove_by_voice(guard):
    assert guard.answer("During focus, tell me if I open YouTube") == (
        "Okay. During focus I'll give you one heads-up if youtube is in front. Nothing gets blocked."
    )
    assert guard.answer("warn me if I open reddit while I'm focusing").startswith("Okay. During focus")
    assert guard.answer("What are my distractions?") == "During focus I'll give you a heads-up about youtube, reddit."
    assert guard.answer("stop warning me about reddit") == "I'll stop warning you about reddit."
    assert guard.answer("remove twitter from my distractions") == "twitter isn't on your list."
    assert guard.list() == ["youtube"]


def test_requests_without_focus_are_left_alone(guard):
    for text in (
        "tell me if I open YouTube",
        "tell me if I spend more than an hour on YouTube today",
        "tell me when Chrome closes",
        "open youtube",
    ):
        assert guard.answer(text) is None, text


@pytest.mark.asyncio
async def test_one_heads_up_per_word_per_session(guard, state):
    guard.add("YouTube")
    assert await guard.check() is None  # no focus session

    state["focus"] = {"active": True, "started_at": 100.0, "remaining_seconds": 1500}
    event = await guard.check()
    assert event == {
        "word": "YouTube",
        "app": "msedge",
        "text": "Heads-up: YouTube is open, and you're focusing for another 25 minutes.",
    }
    guard._deliver.assert_awaited_once_with(event)
    assert await guard.check() is None  # already mentioned this session

    state["focus"] = {"active": True, "started_at": 200.0, "remaining_seconds": 60}
    assert (await guard.check())["text"].endswith("another 1 minute.")


@pytest.mark.asyncio
async def test_other_windows_and_ended_sessions_are_quiet(guard, state):
    guard.add("reddit")
    state["focus"] = {"active": True, "started_at": 1.0, "remaining_seconds": 600}
    state["front"] = ("Code", "server.py - Heliox")
    assert await guard.check() is None
    state["focus"] = {"active": False}
    state["front"] = ("chrome", "reddit - front page")
    assert await guard.check() is None
    guard._deliver.assert_not_awaited()


def test_limits(guard):
    with pytest.raises(ValueError):
        guard.add("x")
    for number in range(20):
        guard.add(f"site{number}")
    assert guard.answer("during focus tell me if I open twitter") == "You can keep up to 20 distractions"


@pytest.mark.asyncio
async def test_rpcs_and_delivery(tmp_path, state):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._focus_guard = FocusGuard(
        tmp_path / "d.json",
        focus_status=lambda: state["focus"],
        deliver=server._deliver_distraction,
        foreground=lambda: state["front"],
    )
    ws = MagicMock()

    assert (await server._handle_focus_distraction_add({"word": "YouTube"}, ws))["words"] == ["YouTube"]
    assert (await server._handle_focus_distraction_add({"word": ""}, ws))["status"] == "error"
    assert (await server._handle_focus_distractions_list({}, ws))["words"] == ["YouTube"]

    state["focus"] = {"active": True, "started_at": 5.0, "remaining_seconds": 300}
    await server._focus_guard.check()
    name, payload = server._broadcast_notification.await_args.args
    assert name == "focus_distraction" and payload["word"] == "YouTube"
    assert server._speak_companion_text.await_args.args[0].startswith("Heads-up: YouTube is open")

    assert (await server._handle_focus_distraction_delete({"word": "youtube"}, ws))["words"] == []
    assert (await server._handle_focus_distraction_delete({"word": "youtube"}, ws))["status"] == "error"

"""After first-run setup the app greets once: the first getting-to-know question appears exactly once.

The app's side (tauri-app/ui/src/lib/firstRun.ts, ``afterSetup``) asks ``getting_to_know status`` and, when the
profile is empty and no questions are open, ``getting_to_know start``; it runs only while the setup marker is in the
app's storage and clears it afterwards, so a later app start does not greet. This drives the daemon's real handlers
the way that function does, including a reconnect that runs it twice, and checks what the person would see.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.personal.getting_to_know import QUESTIONS
from pilot.workflows.user_profile import UserProfile


def make_server(tmp_path):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "user_profile.json")
    return server


async def after_setup(server) -> None:
    """firstRun.ts afterSetup, the getting-to-know part, line for line."""
    status = await server._handle_getting_to_know({"action": "status"}, None)
    if status.get("profile_empty") and not status.get("active"):
        await server._handle_getting_to_know({"action": "start"}, None)


def questions_shown(server) -> list[str]:
    return [
        call.args[1]["message"]
        for call in server._broadcast_notification.await_args_list
        if call.args[0] == "voice_result" and call.args[1].get("source") == "getting_to_know"
    ]


@pytest.mark.asyncio
async def test_the_first_question_is_shown_once_even_when_the_app_reconnects(tmp_path):
    server = make_server(tmp_path)
    await after_setup(server)
    await after_setup(server)  # the UI reconnected and ran it again before the person answered
    shown = questions_shown(server)
    assert len(shown) == 1 and shown[0].endswith(QUESTIONS[0])
    server._speak_companion_text.assert_not_awaited()  # voice is off after setup: the chat only
    # A start that bypasses the status check (two windows at once) also changes nothing.
    again = await server._handle_getting_to_know({"action": "start"}, None)
    assert again["already_active"] and len(questions_shown(server)) == 1


@pytest.mark.asyncio
async def test_no_question_when_the_person_already_has_a_profile_or_turned_it_off(tmp_path):
    server = make_server(tmp_path / "a")
    await server._user_profile.answer_async("call me Sam")
    await after_setup(server)
    assert questions_shown(server) == []
    direct = await server._handle_getting_to_know({"action": "start"}, None)
    assert direct == {"status": "ok", "question": "", "skipped": "profile"}

    server = make_server(tmp_path / "b")
    await server._handle_offers({"action": "intro", "on": False}, None)
    status = await server._handle_getting_to_know({"action": "status"}, None)
    assert status["intro"] is False
    await after_setup(server)  # the app does not read "intro"; the daemon still asks nothing
    assert questions_shown(server) == []


@pytest.mark.asyncio
async def test_the_greeting_does_not_come_back_once_answered(tmp_path):
    server = make_server(tmp_path)
    await after_setup(server)
    for answer in ("call me Sam", "skip", "skip"):
        await server._control_reply(answer)
    await after_setup(server)  # a later run of the same code, if the marker were left behind
    assert len(questions_shown(server)) == 1

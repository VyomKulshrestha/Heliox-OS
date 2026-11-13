"""Short things a person says while a task they handed over is running. A short utterance is never handed to the
running task as a change and never a go-ahead; what a handler on this computer recognises exactly still works:
answers and settings in code at once, a one-step command (a media key, the volume, an app) after the task. Anything
else short is not taken, with one sentence when the person said the wake phrase and silence for a follow-up.

Before this rule every one of these went to the running task as a correction (the planner replanned the task with
"Volume down." as the change), with and without the wake phrase. No model, no audio device; real handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.quick_answers import QuickAnswers
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.user_profile import UserProfile


def running_server(tmp_path, *, interactive: bool = True, approval: bool = False) -> PilotServer:
    config = PilotConfig()
    config.save = lambda *a, **k: None
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._handle_interject = AsyncMock(return_value={"status": "ok", "message": ""})
    server._queue_interactive_request = AsyncMock(return_value={"status": "queued", "task_id": "t2"})
    server._interaction_runtime.transition = AsyncMock()
    server._quick_answers = QuickAnswers()
    server._user_profile = UserProfile(tmp_path / "user_profile.json")
    server._reminders = ReminderStore(tmp_path / "reminders.json", deliver=AsyncMock(), deliver_missed=AsyncMock())
    server._interactive_request_active = interactive
    if approval:
        import asyncio

        from pilot.server import PendingConfirmation

        server._pending_confirms["plan-1"] = PendingConfirmation(plan_id="plan-1", event=asyncio.Event())
    return server


def outcome(server: PilotServer) -> str:
    if server._handle_interject.await_args is not None:
        mode = server._handle_interject.await_args.args[0].get("mode")
        return "correction" if mode is None else mode
    if server._queue_interactive_request.await_count:
        return "queued"
    statuses = [
        c.args[1].get("status")
        for c in server._broadcast_notification.await_args_list
        if c.args and c.args[0] == "voice_result"
    ]
    if "not_acted_on" in statuses:
        return "not taken, told" if server._speak_voice_response.await_count else "not taken, silent"
    if "answered" in statuses or "success" in statuses:
        return "answered here"
    return "nothing"


@pytest.mark.asyncio
@pytest.mark.parametrize("addressed", ["wake", "follow_up"])
@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Volume down.", "queued"),  # a media key: after the task, never into it
        ("Pause the music.", "queued"),
        ("Next track.", "queued"),
        ("Mute.", "queued"),
        ("Louder.", "queued"),
        ("Open Notepad.", "queued"),
        ("What time is it?", "answered here"),  # answered in code at once
        ("Stop interrupting me.", "answered here"),  # the profile's quiet request
        ("Set a timer for ten minutes.", "answered here"),  # a timer, set at once
        ("No, the other one.", "correction"),  # starts like a correction: still one
        ("Use the other folder instead.", "correction"),  # four words or more: a change, as before
        ("Stop.", "stop"),
    ],
)
async def test_short_things_said_during_a_task(tmp_path, said, expected, addressed):
    server = running_server(tmp_path)
    voice_turn_addressed.set(addressed)
    await server._voice_command_dispatch(said)
    assert outcome(server) == expected
    server._handle_execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Send.", "And after that?", "What's connected?", "Save."])
async def test_a_short_unknown_word_is_never_a_change_to_the_task(tmp_path, said):
    server = running_server(tmp_path)
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch(said)
    assert outcome(server) == "not taken, told"
    assert "didn't take that" in server._speak_voice_response.await_args.args[0]
    follow = running_server(tmp_path)
    voice_turn_addressed.set("follow_up")
    await follow._voice_command_dispatch(said)
    assert outcome(follow) == "not taken, silent"  # "Send." alone stays nothing


@pytest.mark.asyncio
async def test_with_only_an_approval_waiting_local_commands_run_as_before(tmp_path):
    server = running_server(tmp_path, interactive=False, approval=True)
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("What time is it?")
    assert outcome(server) == "answered here"
    server = running_server(tmp_path, interactive=False, approval=True)
    await server._voice_command_dispatch("Send.")
    assert outcome(server) == "not taken, told"
    assert "approval is still waiting" in server._speak_voice_response.await_args.args[0]

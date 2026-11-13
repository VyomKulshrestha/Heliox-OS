"""One-step commands said while a task runs: a system call that cannot land in the task's window runs at once; a key
press, or anything that brings a window to the front, waits for the task and is named when Heliox says so. The
volume commands are the system's own volume call where it is there (Core Audio on Windows, osascript on macOS,
pactl or amixer on Linux), and wait only where it is missing. Fakes only."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.actions import ActionType
from pilot.config import PilotConfig
from pilot.conversation import quick
from pilot.conversation.router import normalize, route
from pilot.server import PilotServer
from pilot.system import volume
from pilot.system.voice import voice_turn_addressed


def plan(text: str, platform: str = "windows") -> quick.QuickPlan:
    routed = route(text)
    made = quick.from_words(normalize(text), routed.verb, routed.target, platform=platform)
    assert made is not None, text
    return made


VOLUME = ["Volume down.", "Louder.", "Quieter.", "Mute.", "Unmute.", "Set the volume to 30."]
READ_ONLY = ["Check the battery."]
ALWAYS_WAIT = ["Next track.", "Pause the music.", "Open Notepad."]


@pytest.fixture(autouse=True)
def notepad(monkeypatch):
    monkeypatch.setattr(quick, "_app_name", lambda target: "Notepad" if "notepad" in target else None)


@pytest.mark.parametrize("platform", ["windows", "macos", "linux"])
def test_which_commands_run_beside_a_task(platform):
    for text in READ_ONLY:
        assert quick.runs_beside_a_task(plan(text, platform), volume_call=False), text
    for text in VOLUME:
        assert quick.runs_beside_a_task(plan(text, platform), volume_call=True), (text, platform)
        assert not quick.runs_beside_a_task(plan(text, platform), volume_call=False), (text, platform)
    for text in ALWAYS_WAIT:  # media keys are key presses; opening an app brings a window to the front
        assert not quick.runs_beside_a_task(plan(text, platform), volume_call=True), (text, platform)


def test_volume_up_and_down_are_volume_calls_not_key_presses():
    for text, step in (("Volume down.", -10), ("Louder.", 10), ("Quieter.", -10), ("Turn the volume up.", 10)):
        action = plan(text).plan.actions[0]
        assert action.action_type is ActionType.VOLUME_SET and action.parameters.step == step


@pytest.mark.parametrize(
    ("said", "sentence"),
    [
        ("Open Notepad.", "I'll open Notepad when this is done."),
        ("Volume down.", "I'll turn the volume down when this is done."),
        ("Louder.", "I'll turn the volume up when this is done."),
        ("Next track.", "I'll skip to the next track when this is done."),
        ("Pause the music.", "I'll pause the music when this is done."),
        ("Mute.", "I'll mute the sound when this is done."),
        ("Set the volume to 30.", "I'll set the volume to 30 when this is done."),
    ],
)
def test_a_waiting_command_is_named(said, sentence):
    assert quick.queued_sentence(plan(said), said) == sentence


def running(monkeypatch, *, volume_call: bool) -> PilotServer:
    monkeypatch.setattr(volume, "volume_call_available", lambda: volume_call)
    config = PilotConfig()
    config.save = lambda *a, **k: None
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._handle_interject = AsyncMock(return_value={"status": "ok", "message": ""})
    server._queue_interactive_request = AsyncMock(return_value={"status": "queued", "task_id": "t2"})
    server._interaction_runtime.transition = AsyncMock()
    server._interactive_request_active = True
    calls = {
        ActionType.VOLUME_MUTE: AsyncMock(return_value="Already unmuted."),
        ActionType.VOLUME_SET: AsyncMock(return_value="Volume is at 20."),
    }
    server._executor = type("Executor", (), {"_dispatch_table": calls})()
    server._calls = calls
    voice_turn_addressed.set("wake")
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "kind", "sentence"),
    [
        ("Volume down.", ActionType.VOLUME_SET, "Volume is at 20."),
        ("Louder.", ActionType.VOLUME_SET, "Volume is at 20."),
        ("Unmute.", ActionType.VOLUME_MUTE, "Already unmuted."),
    ],
)
async def test_volume_runs_at_once_during_a_task_and_says_what_is_true(said, kind, sentence, monkeypatch):
    server = running(monkeypatch, volume_call=True)
    await server._voice_command_dispatch(said)
    server._calls[kind].assert_awaited_once()
    server._queue_interactive_request.assert_not_awaited()
    server._handle_interject.assert_not_awaited()
    assert server._speak_voice_response.await_args.args[0] == sentence


@pytest.mark.asyncio
async def test_without_the_volume_call_volume_waits_and_is_named(monkeypatch):
    server = running(monkeypatch, volume_call=False)
    await server._voice_command_dispatch("Volume down.")
    server._calls[ActionType.VOLUME_SET].assert_not_awaited()
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == "I'll turn the volume down when this is done."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "sentence"),
    [
        ("Next track.", "I'll skip to the next track when this is done."),
        ("Open Notepad.", "I'll open Notepad when this is done."),
    ],
)
async def test_key_presses_and_windows_wait(said, sentence, monkeypatch):
    server = running(monkeypatch, volume_call=True)
    await server._voice_command_dispatch(said)
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == sentence
    server._handle_interject.assert_not_awaited()

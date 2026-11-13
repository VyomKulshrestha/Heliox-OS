"""One-step commands said while a task runs: a system call that cannot land in the task's window runs at once; a key
press, or anything that brings a window to the front, waits for the task and is named when Heliox says so. Per
system, from how each action is carried out (pilot/conversation/quick.py, runs_beside_a_task). Fakes only."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.actions import ActionType
from pilot.config import PilotConfig
from pilot.conversation import quick
from pilot.conversation.router import normalize, route
from pilot.server import PilotServer
from pilot.system.voice import voice_turn_addressed


def plan(text: str, platform: str) -> quick.QuickPlan:
    routed = route(text)
    made = quick.from_words(normalize(text), routed.verb, routed.target, platform=platform)
    assert made is not None, text
    return made


AT_ONCE_EVERYWHERE = ["Check the battery."]
AT_ONCE_ON_MAC_AND_LINUX = ["Mute.", "Unmute.", "Set the volume to 30."]
ALWAYS_WAIT = ["Volume down.", "Louder.", "Next track.", "Pause the music.", "Open Notepad."]


@pytest.mark.parametrize("platform", ["windows", "macos", "linux"])
def test_which_commands_run_beside_a_task_on_each_system(platform, monkeypatch):
    monkeypatch.setattr(quick, "_app_name", lambda target: "Notepad" if "notepad" in target else None)
    for text in AT_ONCE_EVERYWHERE:
        assert quick.runs_beside_a_task(plan(text, platform), platform), (text, platform)
    for text in AT_ONCE_ON_MAC_AND_LINUX:
        # Windows mutes and sets the volume with WScript.Shell SendKeys: key presses, so they wait there
        assert quick.runs_beside_a_task(plan(text, platform), platform) is (platform != "windows"), (text, platform)
    for text in ALWAYS_WAIT:
        # media keys are pyautogui key presses on every system; opening an app brings a window to the front
        assert not quick.runs_beside_a_task(plan(text, platform), platform), (text, platform)


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
def test_a_waiting_command_is_named(said, sentence, monkeypatch):
    monkeypatch.setattr(quick, "_app_name", lambda target: "Notepad" if "notepad" in target else None)
    assert quick.queued_sentence(plan(said, "windows"), said) == sentence


def running(platform: str, monkeypatch) -> PilotServer:
    monkeypatch.setattr(quick, "_platform", lambda: platform)
    monkeypatch.setattr(quick, "_app_name", lambda target: "Notepad" if "notepad" in target else None)
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
    mute = AsyncMock(return_value="Volume muted")
    server._executor = type("Executor", (), {"_dispatch_table": {ActionType.VOLUME_MUTE: mute}})()
    server._mute = mute
    voice_turn_addressed.set("wake")
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["macos", "linux"])
async def test_mute_runs_at_once_during_a_task_where_it_is_a_system_call(platform, monkeypatch):
    server = running(platform, monkeypatch)
    await server._voice_command_dispatch("Mute.")
    server._mute.assert_awaited_once()
    server._queue_interactive_request.assert_not_awaited()
    server._handle_interject.assert_not_awaited()


@pytest.mark.asyncio
async def test_mute_waits_for_the_task_on_windows_and_is_named(monkeypatch):
    server = running("windows", monkeypatch)
    await server._voice_command_dispatch("Mute.")
    server._mute.assert_not_awaited()
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == "I'll mute the sound when this is done."


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["windows", "macos", "linux"])
@pytest.mark.parametrize(
    ("said", "sentence"),
    [
        ("Volume down.", "I'll turn the volume down when this is done."),
        ("Open Notepad.", "I'll open Notepad when this is done."),
    ],
)
async def test_key_presses_and_windows_wait_on_every_system(platform, said, sentence, monkeypatch):
    server = running(platform, monkeypatch)
    await server._voice_command_dispatch(said)
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == sentence
    server._handle_interject.assert_not_awaited()

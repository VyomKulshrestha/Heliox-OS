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
MEDIA = ["Next track.", "Pause the music.", "Previous track."]
ALWAYS_WAIT = ["Open Notepad."]


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
    for text in ALWAYS_WAIT:  # opening an app brings a window to the front
        assert not quick.runs_beside_a_task(plan(text, platform), volume_call=True, media_call=True), text
    for text in MEDIA:  # the player itself on Linux when the bus answers; a key press otherwise, which waits
        reachable = quick.runs_beside_a_task(plan(text, platform), volume_call=True, media_call=True)
        assert reachable is (platform == "linux"), (text, platform)
        assert not quick.runs_beside_a_task(plan(text, platform), volume_call=True, media_call=False), (text, platform)


@pytest.mark.parametrize("host", ["linux", "win32", "darwin"])
def test_the_plan_s_platform_decides_not_the_host(host, monkeypatch):
    """The computer the tests run on never decides: a plan made for Windows or macOS waits even on a Linux host
    whose session bus answers, and the bus is not even asked; a plan for Linux asks it."""
    from pilot.system import mpris

    asked = []
    monkeypatch.setattr(mpris.sys, "platform", host)
    monkeypatch.setattr(mpris, "media_call_available", lambda bus_factory=None: asked.append(1) or True)
    for platform in ("windows", "macos", "win32", "darwin"):
        assert not quick.runs_beside_a_task(plan("Next track.", platform)), (platform, host)
    assert asked == []
    assert quick.runs_beside_a_task(plan("Next track.", "linux"))
    assert asked == [1]
    assert mpris.goes_to_player("nexttrack", "linux", reachable=False) is False
    assert mpris.media_command("nexttrack", platform="win32") is None  # nothing asked of any bus, no key here


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


def media_state(monkeypatch, platform: str, *, bus: bool, player_says: str | None = "Next track in Spotify.") -> list:
    """The system the request is made on and what the media player does, set explicitly so the host never decides:
    ``bus`` whether the session bus answers, ``player_says`` what pilot.system.mpris reports after the call (None:
    the player could not be asked after all). Returns the requests the player got."""
    from pilot.system import mpris

    got: list[str] = []

    def media_command(request, *, platform=None, **_):
        if not mpris.goes_to_player(request, platform, reachable=bus):
            return None
        got.append(request)
        return player_says

    monkeypatch.setattr(quick, "_platform", lambda: platform)
    monkeypatch.setattr(mpris, "media_call_available", lambda bus_factory=None: bus)
    monkeypatch.setattr(mpris, "media_command", media_command)
    return got


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "bus"), [("win32", True), ("darwin", True), ("win32", False), ("linux", False)]
)  # Windows and macOS: a key press whatever a bus would say; Linux without a bus: a key press too
@pytest.mark.parametrize(
    ("said", "sentence"),
    [
        ("Next track.", "I'll skip to the next track when this is done."),
        ("Pause the music.", "I'll pause the music when this is done."),
        ("Open Notepad.", "I'll open Notepad when this is done."),
    ],
)
async def test_key_presses_and_windows_wait(said, sentence, platform, bus, monkeypatch):
    got = media_state(monkeypatch, platform, bus=bus)
    server = running(monkeypatch, volume_call=True)
    await server._voice_command_dispatch(said)
    assert got == []  # nothing changed yet, so "when this is done" is true
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == sentence
    server._handle_interject.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "asked", "player_says"),
    [("Next track.", "nexttrack", "Next track in Spotify."), ("Pause the music.", "pause", "Paused Spotify.")],
)
async def test_on_linux_with_a_player_the_command_is_answered_beside_the_task(said, asked, player_says, monkeypatch):
    got = media_state(monkeypatch, "linux", bus=True, player_says=player_says)
    server = running(monkeypatch, volume_call=True)
    await server._voice_command_dispatch(said)
    assert got == [asked]
    server._queue_interactive_request.assert_not_awaited()
    server._handle_interject.assert_not_awaited()
    assert [c.args[0] for c in server._speak_voice_response.await_args_list] == [player_says]  # never "when done"


@pytest.mark.asyncio
async def test_on_linux_a_player_that_cannot_be_asked_after_all_waits_and_is_named(monkeypatch):
    """The bus answered the check, then the call found nothing to reach: nothing happened, so it waits and the
    sentence says it will happen later."""
    media_state(monkeypatch, "linux", bus=True, player_says=None)
    server = running(monkeypatch, volume_call=True)
    await server._voice_command_dispatch("Next track.")
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == "I'll skip to the next track when this is done."

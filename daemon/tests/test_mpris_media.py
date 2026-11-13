"""Next, previous, play and pause through the media player itself on Linux (MPRIS), with a fake session bus: which
player is asked, what Heliox says from what the player reports afterwards, the key press only when MPRIS cannot be
reached, and the media commands running at once during a task where MPRIS is there. No bus, no device."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.actions import ActionType
from pilot.config import PilotConfig
from pilot.conversation import quick
from pilot.conversation.router import normalize, route
from pilot.server import PilotServer
from pilot.system import mpris
from pilot.system.voice import voice_turn_addressed

TRACKS = [("Blue", ["Joni Mitchell"]), ("A Case of You", ["Joni Mitchell"]), ("River", ["Joni Mitchell"])]


class FakePlayer:
    def __init__(self, identity: str, status: str, track: int = 0) -> None:
        self.identity, self.status, self.track = identity, status, track


class FakeBus:
    def __init__(self, **players: FakePlayer) -> None:
        self.players = {mpris.PREFIX + name: player for name, player in players.items()}
        self.calls: list[tuple[str, str]] = []

    def names(self):
        return ["org.freedesktop.DBus", ":1.42", *self.players]

    def get(self, name, interface, prop):
        player = self.players[name]
        if prop == "Identity":
            return player.identity
        if prop == "PlaybackStatus":
            return player.status
        if prop == "Metadata":
            title, artists = TRACKS[player.track]
            return {"xesam:title": ("s", title), "xesam:artist": ("as", artists)}
        raise KeyError(prop)

    def call(self, name, method):
        self.calls.append((name, method))
        player = self.players[name]
        if method == "Pause":
            player.status = "Paused"
        elif method == "Play":
            player.status = "Playing"
        elif method == "PlayPause":
            player.status = "Paused" if player.status == "Playing" else "Playing"
        elif method == "Next":
            player.track = (player.track + 1) % len(TRACKS)
        elif method == "Previous":
            player.track = (player.track - 1) % len(TRACKS)

    def close(self):
        pass


def ask(bus: FakeBus, request: str) -> str | None:
    return mpris.media_command(request, bus_factory=lambda: bus, sleep=lambda _s: None)


def test_the_player_that_is_playing_is_asked_and_the_sentence_is_what_it_reports():
    bus = FakeBus(spotify=FakePlayer("Spotify", "Playing"), firefox=FakePlayer("Firefox", "Paused"))
    assert ask(bus, "pause") == "Paused Spotify."
    assert bus.calls == [(mpris.PREFIX + "spotify", "Pause")]
    assert ask(bus, "pause") == "Firefox and Spotify are all paused, so I didn't change any of them. Say which one."


def test_play_next_and_previous():
    bus = FakeBus(spotify=FakePlayer("Spotify", "Paused"))
    assert ask(bus, "play") == 'Playing "Blue" by Joni Mitchell in Spotify.'
    assert ask(bus, "nexttrack") == 'Now playing "A Case of You" by Joni Mitchell.'
    assert ask(bus, "prevtrack") == 'Now playing "Blue" by Joni Mitchell.'
    assert ask(bus, "play") == "Spotify is already playing."  # nothing toggled


def test_pause_when_nothing_plays_changes_nothing():
    bus = FakeBus(vlc=FakePlayer("VLC media player", "Paused"))
    assert ask(bus, "pause") == "VLC media player is already paused."
    assert bus.calls == []


def test_no_player_and_two_players_playing_are_said_and_nothing_is_pressed():
    assert ask(FakeBus(), "pause") == "No media player is running."
    both = FakeBus(spotify=FakePlayer("Spotify", "Playing"), firefox=FakePlayer("Firefox", "Playing"))
    assert ask(both, "nexttrack") == (
        "Firefox and Spotify are all playing, so I didn't change any of them. Say which one."
    )
    assert both.calls == []


def test_with_none_playing_the_only_paused_one_is_asked_over_a_stopped_one():
    bus = FakeBus(spotify=FakePlayer("Spotify", "Stopped"), mpv=FakePlayer("mpv", "Paused"))
    assert ask(bus, "play") == 'Playing "Blue" by Joni Mitchell in mpv.'


def test_without_a_session_bus_the_caller_falls_back():
    def no_bus():
        raise OSError("no session bus")

    assert mpris.media_command("pause", bus_factory=no_bus) is None
    assert not mpris.media_call_available(no_bus)
    assert mpris.media_call_available(lambda: FakeBus())


def test_pause_and_play_are_told_apart_from_the_words():
    for said, wanted in (("Pause the music.", "pause"), ("Resume the music.", "play"), ("Play the music.", "play")):
        routed = route(said)
        plan = quick.from_words(normalize(said), routed.verb, routed.target, platform="linux")
        assert plan.plan.actions[0].target == wanted and plan.plan.actions[0].parameters.key == "playpause"


@pytest.mark.parametrize("platform", ["windows", "macos", "linux"])
def test_media_commands_run_beside_a_task_only_where_the_player_can_be_asked(platform):
    routed = route("Next track.")
    plan = quick.from_words(normalize("Next track."), routed.verb, routed.target, platform=platform)
    assert quick.runs_beside_a_task(plan, media_call=True) is (platform == "linux")  # elsewhere a key press: waits
    assert not quick.runs_beside_a_task(plan, media_call=False)


@pytest.mark.asyncio
async def test_the_executor_asks_the_player_and_presses_the_key_only_without_mpris(monkeypatch):
    from pilot.actions import Action, KeyboardParams
    from pilot.agents import executor as executor_module
    from pilot.system import input_control

    pressed = AsyncMock(return_value="Pressed 'nexttrack' x1")
    monkeypatch.setattr(input_control, "keyboard_press", pressed)
    action = Action(
        action_type=ActionType.KEYBOARD_PRESS, target="nexttrack", parameters=KeyboardParams(key="nexttrack")
    )
    run = executor_module.Executor._exec_keyboard_press
    monkeypatch.setattr(mpris, "media_command", lambda request: 'Now playing "River".')
    assert await run(None, action) == 'Now playing "River".'
    pressed.assert_not_awaited()
    monkeypatch.setattr(mpris, "media_command", lambda request: None)
    assert await run(None, action) == "Pressed 'nexttrack' x1"


def running(monkeypatch, *, media: bool) -> PilotServer:
    monkeypatch.setattr(quick, "_platform", lambda: "linux")  # set here: the host never decides
    monkeypatch.setattr(mpris, "media_call_available", lambda bus_factory=None: media)
    monkeypatch.setattr(mpris, "media_command", lambda request, **_: "Paused Spotify." if media else None)
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
    voice_turn_addressed.set("wake")
    return server


@pytest.mark.asyncio
async def test_pause_runs_at_once_during_a_task_on_linux(monkeypatch):
    server = running(monkeypatch, media=True)
    await server._voice_command_dispatch("Pause the music.")
    server._queue_interactive_request.assert_not_awaited()
    server._handle_interject.assert_not_awaited()
    assert server._speak_voice_response.await_args.args[0] == "Paused Spotify."


@pytest.mark.asyncio
async def test_without_mpris_it_waits_for_the_task_and_is_named(monkeypatch):
    server = running(monkeypatch, media=False)
    await server._voice_command_dispatch("Pause the music.")
    server._queue_interactive_request.assert_awaited_once()
    assert server._speak_voice_response.await_args.args[0] == "I'll pause the music when this is done."

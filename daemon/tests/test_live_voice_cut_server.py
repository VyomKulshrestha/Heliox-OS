"""Live's voice cut at the first keystroke, through the real server: PilotServer._live_speak, the real
CompanionSpeechCoordinator and a stand-in player that records when audio starts and stops (no model, no sound device).

The engine ticks every TICK_SECONDS as it does in the daemon. What the code guarantees, and what is asserted: the
player is told to stop at the engine's first look after the keystroke (counted looks, not a time), with only a
generous wall-clock ceiling for a slow runner. On a quiet laptop the stop came 0.20 to 0.22 s after the keystroke."""

from __future__ import annotations

import asyncio
import time
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation
from pilot.system.companion_speech import CompanionSpeechCoordinator

SYNTHESIS_SECONDS = 0.3  # the stand-in voice makes its audio for this long before playing it
AUDIO_SECONDS = 3.0
CEILING_SECONDS = 1.5  # keystroke to stop, on the slowest runner: far above the ~0.2 s it takes
WAIT_SECONDS = 20.0  # for anything to happen at all (a ceiling, not an expectation)


class Player:
    """A stand-in for the voice: synthesis, then audio; records when audio started and when it was told to stop."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, float]] = []
        self.audio_started: list[tuple[str, float]] = []
        self.stopped: list[tuple[str, float]] = []
        self.finished: list[str] = []

    async def __call__(self, text: str, recorder) -> bool:
        self.asked.append((text, time.monotonic()))
        try:
            await asyncio.sleep(SYNTHESIS_SECONDS)
            self.audio_started.append((text, time.monotonic()))
            await asyncio.sleep(AUDIO_SECONDS)
        except asyncio.CancelledError:
            self.stopped.append((text, time.monotonic()))
            raise
        self.finished.append(text)
        return False


class Field:
    """The text being typed, and every time the engine looked at it."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.looks: list[float] = []


def server_with(player: Player):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._speech_coordinator = CompanionSpeechCoordinator(speaker=player)
    server._voice_listener = None
    return server


def live_on(server, field: Field, events: list[tuple[str, dict]]) -> LiveCompanion:
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=False, muted_kinds=[])

    async def observe():
        field.looks.append(time.monotonic())
        return Observation(text=field.text, app="notepad.exe", title="budget.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        events.append((event, payload))

    live = LiveCompanion(
        settings=lambda: settings, observe=observe, emit=emit, speak=server._live_speak, clock=time.monotonic,
        today=lambda: date(2026, 11, 9),
    )  # fmt: skip
    server._live = live
    return live


async def tick_until(live: LiveCompanion, done, limit: float = WAIT_SECONDS) -> None:
    end = time.monotonic() + limit
    while time.monotonic() < end and not done():
        await live.tick()
        await asyncio.sleep(TICK_SECONDS)


def idle(server) -> bool:
    status = server._speech_coordinator.status()
    return not status["active"] and not status["queued"]


@pytest.mark.asyncio
async def test_a_keystroke_stops_the_voice_at_the_next_look_and_the_card_stays():
    player = Player()
    server = server_with(player)
    field = Field("Hotel 12 x 15 = 170 units.")
    events: list[tuple[str, dict]] = []
    live = live_on(server, field, events)
    await tick_until(live, lambda: bool(player.audio_started))
    assert player.audio_started, "the remark was never played"
    said, started = player.audio_started[0]
    assert said == "12 × 15 is 180."  # the short form
    await asyncio.sleep(max(0.0, started + 0.4 - time.monotonic()))  # the person types again 0.4 s into the audio
    await live.tick()
    field.text += " Next"  # just after a look: the worst moment, seen only at the next one
    typed_at = time.monotonic()
    await tick_until(live, lambda: bool(player.stopped))
    assert player.stopped, "the voice was never told to stop"
    stopped_at = player.stopped[0][1]
    looks_between = [t for t in field.looks if typed_at < t <= stopped_at]
    assert len(looks_between) == 1, looks_between  # told to stop at the first look after the keystroke
    assert stopped_at - typed_at <= CEILING_SECONDS, stopped_at - typed_at
    await tick_until(live, lambda: idle(server))
    looks = len(field.looks)
    await tick_until(live, lambda: len(field.looks) >= looks + 5)  # five more looks: nothing comes back
    assert idle(server) and player.finished == [] and len(player.audio_started) == 1  # nothing more of it
    assert [p["kind"] for e, p in events if e == "live_interjection"] == ["arithmetic"]
    assert not any(e == "live_interjection_withdrawn" for e, _ in events)  # the card is still up


@pytest.mark.asyncio
async def test_a_remark_cut_while_its_audio_is_being_made_is_never_played():
    player = Player()
    server = server_with(player)
    field = Field("Hotel 12 x 15 = 170 units.")
    events: list[tuple[str, dict]] = []
    live = live_on(server, field, events)
    await tick_until(live, lambda: bool(player.asked))
    assert player.asked and not player.audio_started
    field.text += " Next"  # typed while the voice is still being made
    await tick_until(live, lambda: bool(player.stopped))
    await tick_until(live, lambda: idle(server))  # the coordinator has let the request go
    looks = len(field.looks)
    await tick_until(live, lambda: len(field.looks) >= looks + 5)
    assert player.stopped and idle(server)
    assert player.audio_started == [] and player.finished == []  # nothing was ever played
    assert not any(e == "live_interjection_withdrawn" for e, _ in events)

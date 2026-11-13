"""When Live's voice yields and what happens to a slip once its text has gone (pilot.live.engine, pilot.live.policy):

1. the voice says the short form and stops at the first keystroke (the card stays up, nothing is said again);
2. a slip that is the last thing typed gets its card at the ordinary pause and its voice only after the longer one,
   and only if the card is still unanswered and the person has not gone on;
3. a slip sent before the catch is raised once as a card saying what was sent, never spoken, never for a doubled
   word; a slip left behind in another window is a card only."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation

MONDAY = date(2026, 11, 9)


class World:
    def __init__(self, text: str = "", window: tuple[str, str] = ("slack.exe", "#team - Slack")) -> None:
        self.t = 100.0
        self.text = text
        self.window = window
        self.events: list[tuple[str, dict]] = []
        self.said: list[str] = []
        self.cut: list[str] = []
        self.playing = asyncio.Event()

    def clock(self) -> float:
        return self.t


def build(world: World, **overrides) -> LiveCompanion:
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        app, title = world.window
        return Observation(text=world.text, app=app, title=title, source="focused_field")

    async def emit(event, payload):
        world.events.append((event, payload))

    async def speak(message, urgent):
        world.said.append(message)
        try:
            await world.playing.wait()  # plays until the test lets it end, or Live cuts it
        except asyncio.CancelledError:
            world.cut.append(message)
            raise
        return "spoken"

    return LiveCompanion(
        settings=lambda: settings, observe=observe, emit=emit, speak=speak, clock=world.clock, today=lambda: MONDAY,
        **overrides,
    )  # fmt: skip


async def run_for(live: LiveCompanion, world: World, seconds: float) -> None:
    for _ in range(round(seconds / TICK_SECONDS)):
        world.t += TICK_SECONDS
        await live.tick()
        for _ in range(3):
            await asyncio.sleep(0)


def cards(world: World) -> list[dict]:
    return [p for e, p in world.events if e == "live_interjection"]


@pytest.mark.asyncio
async def test_the_voice_says_the_short_form_and_stops_at_the_first_keystroke():
    world = World("Invoice total: 12 x 15 = 170 units.")
    live = build(world)
    await run_for(live, world, 3.0)
    assert world.said == ["12 × 15 is 180."]  # the title alone; "You wrote 170." stays on the card
    assert cards(world)[0]["message"] == "You wrote 170."
    world.text += " Next"  # the person types again while it is being said
    await run_for(live, world, 0.4)
    assert world.cut == ["12 × 15 is 180."]
    assert not any(e == "live_interjection_withdrawn" for e, _ in world.events)  # the card stays up
    await run_for(live, world, 10.0)
    assert world.said == ["12 × 15 is 180."]  # never said again by itself


@pytest.mark.asyncio
async def test_a_trailing_slip_shows_first_and_speaks_later_only_if_left_alone():
    world = World("Then we meet on Thursday 25 November")
    live = build(world)
    await run_for(live, world, 2.0)
    [card] = cards(world)
    assert not card["spoken"] and world.said == []  # the card at the ordinary pause, silently
    await run_for(live, world, 2.0)
    assert world.said == ["25 November is a Wednesday."]  # after the longer pause, still unanswered
    assert ("live_interjection_spoken", {"id": card["id"]}) in world.events


@pytest.mark.asyncio
async def test_a_trailing_slip_the_person_goes_on_from_is_never_said():
    world = World("Then we meet on Thursday 25 November")
    live = build(world)
    await run_for(live, world, 2.0)
    assert len(cards(world)) == 1
    world.text += " at noon"
    await run_for(live, world, 8.0)
    assert world.said == []


@pytest.mark.asyncio
async def test_a_slip_sent_before_the_catch_is_a_card_saying_what_was_sent():
    world = World("Then we meet on Thursday 25 November")
    live = build(world)
    await run_for(live, world, 0.6)
    world.text = ""  # sent
    await run_for(live, world, 4.0)
    [card] = cards(world)
    assert card["title"] == "You sent Thursday 25 November; 25 November is a Wednesday"
    assert not card["spoken"] and world.said == [] and card["fix"] == ""
    await run_for(live, world, 10.0)
    assert len(cards(world)) == 1  # once


@pytest.mark.asyncio
async def test_a_card_up_when_the_message_is_sent_stays_up():
    world = World("Then we meet on Thursday 25 November")
    live = build(world)
    await run_for(live, world, 2.0)
    [card] = cards(world)
    world.text = ""
    await run_for(live, world, 4.0)
    assert not any(e == "live_interjection_withdrawn" and p["id"] == card["id"] for e, p in world.events)


@pytest.mark.asyncio
async def test_a_doubled_word_sent_is_not_raised_afterwards():
    world = World("Please send the the report")
    live = build(world)
    await run_for(live, world, 0.4)
    world.text = ""
    await run_for(live, world, 6.0)
    assert cards(world) == []


@pytest.mark.asyncio
async def test_a_slip_left_in_another_window_is_a_card_only():
    world = World("The review is on Monday 1 December.", window=("OUTLOOK.EXE", "Review - Message (HTML)"))
    live = build(world)
    await run_for(live, world, 0.4)
    world.window, world.text = ("chrome.exe", "Planning - Google Docs"), "Notes for the review"
    await run_for(live, world, 8.0)
    [card] = cards(world)
    assert card["kind"] == "weekday_date" and not card["spoken"] and world.said == []

"""What the person says reaches Live's checks: spoken slips, spoken promises and the grounded check, using the
transcripts the voice probe recorded from the streaming recogniser."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from pilot.live.engine import TICK_SECONDS, LiveCompanion

TUESDAY = date(2026, 11, 10)


def build(tmp_path, grounded=None):
    clock = [100.0]
    events: list[tuple[str, dict]] = []
    spoken: list[str] = []
    settings = SimpleNamespace(
        enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[], grounded_checks=grounded is not None
    )

    async def observe():
        return None

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        spoken.append((message, urgent))

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "rel.json",
        timing_path=tmp_path / "timing.json",
        clock=lambda: clock[0],
        today=lambda: TUESDAY,
        wallclock=lambda: datetime(2026, 11, 10, 11, 0),
        grounded=grounded,
    )
    return live, clock, events, spoken


async def run_for(live, clock, seconds):
    for _ in range(int(seconds / TICK_SECONDS)):
        clock[0] += TICK_SECONDS
        await live.tick()
    for _ in range(3):
        await asyncio.sleep(0)


def cards(events):
    return [p for e, p in events if e == "live_interjection"]


@pytest.mark.asyncio
async def test_a_spoken_slip_is_caught_and_said_back(tmp_path):
    live, clock, events, spoken = build(tmp_path)
    live.offer_speech("Remind me that the launch call is monday the twelfth of november at ten")  # as recorded
    await run_for(live, clock, 1)
    [card] = cards(events)
    assert card["title"] == "The twelfth of November is a Thursday" and card["source"] == "speech"
    # Said back on the correction channel, so the planner's "Opening it now." waits instead of cutting it off.
    assert spoken == [("The twelfth of November is a Thursday. You said Monday.", True)]


@pytest.mark.asyncio
async def test_a_spoken_promise_gets_the_reminder_offer(tmp_path):
    live, clock, events, _ = build(tmp_path)
    # What the recogniser made of it: "frida" after "by" is read as Friday (pilot.live.spoken), so the offer comes.
    live.offer_speech("I'll send pria the deck by frida")
    await run_for(live, clock, 1)
    [card] = cards(events)
    assert card["kind"] == "commitment" and card["fix_kind"] == "ask" and "Friday" in card["fix"]
    assert card["evidence"] == "I'll send pria the deck by frida"  # the card quotes what was heard
    live.offer_speech("lunch with Frida after the stand up")  # a person called Frida is not a Friday
    await run_for(live, clock, 1)
    assert len(cards(events)) == 1


@pytest.mark.asyncio
async def test_the_grounded_check_reads_the_written_form_of_what_was_said(tmp_path):
    seen: list[str] = []

    class Grounded:
        async def check(self, text):
            seen.append(text)
            return []

        def stats(self):
            return {}

    live, clock, _, _ = build(tmp_path, grounded=Grounded())
    live.offer_speech("move my dentist appointment to the twelfth of november")
    await run_for(live, clock, 0.4)
    assert seen and "12" in seen[0] and "twelfth" not in seen[0]

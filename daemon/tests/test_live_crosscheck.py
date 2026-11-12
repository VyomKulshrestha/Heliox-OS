"""A date said to Heliox against the date just typed for the same thing (pilot.live.crosscheck)."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live.crosscheck import WINDOW_SECONDS, said_vs_typed
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation

MONDAY = date(2026, 11, 9)
TYPED = [(100.0, "Notepad", "Dentist appointment on Thursday 12 November at 4.", "")]


def found(said, typed=TYPED, now=150.0):
    return [(c.title, c.message, c.detail) for c in said_vs_typed(said, typed, now, MONDAY)]


def test_a_different_day_for_the_same_thing_is_raised_with_where_it_was_written():
    assert found("remind me about the dentist on Friday") == [
        (
            "You wrote Thursday 12 November for the dentist",
            "You said Friday, which is Friday 13 November.",
            "A few minutes ago in Notepad: Thursday 12 November.",
        )
    ]


@pytest.mark.parametrize(
    "said",
    [
        "remind me about the dentist on Thursday",  # the same day
        "remind me about the gym on Friday",  # another thing
        "move the dentist to Friday",  # changed on purpose
        "reschedule the dentist for 13 November",
        "what's on Friday",  # no subject, not scheduling
        "the dentist was nice",  # no date
    ],
)
def test_near_misses_do_not_fire(said):
    assert found(said) == []


def test_old_or_pre_existing_text_does_not_count():
    assert found("remind me about the dentist on Friday", now=100.0 + WINDOW_SECONDS + 1) == []
    already = [(100.0, "Outlook", TYPED[0][2], "Dentist appointment on Thursday 12 November")]
    assert found("remind me about the dentist on Friday", typed=already) == []
    both = [(100.0, "Notepad", "Training on Monday 16 November, and training on Friday 13 November.", "")]
    assert found("remind me about training on Friday", typed=both) == []


@pytest.mark.asyncio
async def test_live_says_it_when_the_person_finishes_speaking(tmp_path):
    clock = SimpleNamespace(t=100.0)
    text = [""]
    spoken = []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        return Observation(text=text[0], app="notepad.exe", title="todo.txt - Notepad", source="focused_field")

    async def emit(event, payload):
        return None

    async def speak(message, urgent):
        spoken.append(message)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "r.json",
        clock=lambda: clock.t,
        today=lambda: MONDAY,
    )
    for typed in ("Dentist", "Dentist appointment on Thursday 12 November at 4."):
        text[0] = typed
        for _ in range(int(2 / TICK_SECONDS)):
            clock.t += TICK_SECONDS
            await live.tick()
    live.offer_speech("remind me about the dentist on Friday")
    for _ in range(5):
        await asyncio.sleep(0)
    assert spoken == ["You wrote Thursday 12 November for the dentist. You said Friday, which is Friday 13 November."]
    settings.muted_kinds = ["said_vs_typed"]
    live.offer_speech("remind me about the dentist on Saturday")
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(spoken) == 1

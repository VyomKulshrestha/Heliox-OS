"""What Live keeps from speech that was not said to Heliox: a silent card at most, its words for 90 seconds in
memory, nothing in the conversation and nothing for any model."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.conversation.context import ConversationMemory
from pilot.live.conversation_link import LiveConversationLink
from pilot.live.engine import TICK_SECONDS, LiveCompanion, Observation


def build(tmp_path):
    clock = SimpleNamespace(t=100.0)
    events, spoken = [], []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        return Observation(text="", app="notepad.exe", title="x - Notepad", source="focused_field", editable=False)

    async def emit(event, payload):
        events.append((event, dict(payload)))

    async def speak(message, urgent):
        spoken.append(message)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "r.json",
        clock=lambda: clock.t,
        today=lambda: date(2026, 11, 9),
    )
    return live, clock, events, spoken


async def run_for(live, clock, seconds):
    for _ in range(int(round(seconds / TICK_SECONDS))):
        clock.t += TICK_SECONDS
        await live.tick()
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_overheard_speech_gives_no_card_and_nothing_reaches_a_model(tmp_path):
    live, clock, events, spoken = build(tmp_path)
    live.offer_speech("lunch with Sam is on Tuesday 14 November", addressed=False)
    live.offer_speech("I'll send Tom the deck by tomorrow.", addressed=False)
    await run_for(live, clock, 3)
    assert [e for e, _ in events if e == "live_interjection"] == [] and spoken == []
    memory = ConversationMemory(clock=lambda: clock.t)
    link = LiveConversationLink(
        live=lambda: live,
        layer=lambda: SimpleNamespace(memory=memory),
        setting=lambda: "on",
        model_local=lambda: True,
        clock=lambda: clock.t,
    )
    assert link.context_line() == "" and memory.recent() == []


@pytest.mark.asyncio
async def test_a_sentence_first_offered_as_overheard_and_then_dispatched_is_answered_aloud(tmp_path):
    live, clock, events, spoken = build(tmp_path)
    # The listener offers the words while it still decides whether the wake phrase was said, then dispatches them.
    live.offer_speech("remind me on Tuesday 14 November", addressed=False)
    live.offer_speech("remind me on Tuesday 14 November")
    await run_for(live, clock, 1)
    assert spoken == ["The fourteenth of November is a Saturday. You said Tuesday."]
    [card] = [p for e, p in events if e == "live_interjection"]
    assert card["spoken"] is True and not live._find(card["id"]).get("overheard")


def test_a_card_from_addressed_speech_still_reaches_the_model_as_before():
    live = MagicMock()
    live.last_raised.return_value = {
        "id": "c1",
        "kind": "weekday_date",
        "title": "14 November is a Saturday",
        "message": "You said Tuesday.",
        "evidence": "Tuesday 14 November",
        "source": "speech",
        "spoken": False,
        "at": 0.0,
        "overheard": False,
    }
    memory = ConversationMemory(clock=lambda: 0.0)
    link = LiveConversationLink(
        live=lambda: live,
        layer=lambda: SimpleNamespace(memory=memory),
        setting=lambda: "auto",
        model_local=lambda: False,
        clock=lambda: 1.0,
    )
    assert "14 November is a Saturday" in link.context_line()

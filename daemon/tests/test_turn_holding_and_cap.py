"""A holding phrase the answer would cut a moment later is not started, and a reply is not cut at "9 a.m.".

Companion acceptance run (24 November): 11 of 11 holding phrases heard were cut by the answer 0 to 0.6 s after they
began, because the hold came at about the model's usual wait for its first text. And a statement's one-sentence rule
ended a reply on "and a nine a m": the splitter took the full stop of "a.m." for the end of the sentence, so "deadline
leaves no slack" became a second sentence and was left out. Fakes only; nothing waits on real time."""

from __future__ import annotations

import asyncio

import pytest

from pilot.conversation.context import ContextSources
from pilot.conversation.layer import HOLD_LATE_SECONDS, HOLD_NO_TEXT_SECONDS, ConversationLayer, hold_due
from pilot.conversation.style import SentenceSplitter


def test_the_hold_waits_past_the_usual_wait_for_a_first_sentence():
    assert hold_due(HOLD_NO_TEXT_SECONDS, 0.0) == HOLD_NO_TEXT_SECONDS  # no history: as before
    assert hold_due(HOLD_NO_TEXT_SECONDS, 1.4) == pytest.approx(1.4 + HOLD_LATE_SECONDS)  # Haiku's usual 1.4 s
    assert hold_due(3.0, 0.5) == 3.0  # never sooner than its own limit


@pytest.mark.asyncio
async def test_the_layer_learns_the_usual_wait_from_its_spoken_turns():
    class Model:
        name = provider = "fake"
        streams, local = True, False

        async def stream(self, _system, _messages):
            yield "Canberra. "

    now = [100.0]
    said: list[str] = []

    async def sink(text, _channel, _key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: Model(), sources=ContextSources(), clock=lambda: now[0])
    assert layer._usual_first_sentence() == 0.0
    await layer.take("What's the capital of Australia?", source="voice")
    await asyncio.sleep(0)
    assert said == ["Canberra."] and len(layer._first_sentence_waits) == 1
    layer._first_sentence_waits.extend([1.2, 1.5, 1.6, 2.0])
    assert layer._usual_first_sentence() == 1.5  # the median of 0.0, 1.2, 1.5, 1.6, 2.0


def pieces(text: str) -> list[str]:
    splitter = SentenceSplitter(early=False)
    return [*splitter.feed(text), *splitter.flush()]


def test_a_time_of_day_does_not_end_the_sentence():
    reply = (
        "Four hours of work after a late series means very little sleep, and a 9 a.m. deadline leaves no slack. "
        "Could you start tonight?"
    )
    got = pieces(reply)
    assert got[-1] == "Could you start tonight?"
    assert any("9 a.m. deadline leaves no slack." in piece for piece in got)


def test_a_time_of_day_at_the_end_of_a_sentence_still_ends_it():
    assert pieces("Call at 9 a.m. Then send the notes.") == ["Call at 9 a.m.", "Then send the notes."]

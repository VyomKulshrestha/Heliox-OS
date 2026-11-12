"""The layer reads how a turn was addressed, and tells the listener a turn was not understood, through the real names
in pilot.system.voice (no getattr guards left)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from pilot.conversation import adapters
from pilot.conversation.layer import SORRY, ConversationLayer
from pilot.system.voice import ContinuousVoiceListener, voice_turn_addressed


class Sink:
    def __init__(self):
        self.said = []

    async def __call__(self, text, channel, key):
        self.said.append(text)
        return "spoken"


def test_the_adapter_reads_the_listeners_own_context_variable():
    async def in_a_dispatch(addressed: str) -> str:
        voice_turn_addressed.set(addressed)  # what ContinuousVoiceListener._dispatch_command does per task
        return adapters._turn_addressed()

    assert asyncio.run(in_a_dispatch("doubtful_follow_up")) == "doubtful_follow_up"
    assert asyncio.run(in_a_dispatch("wake")) == "wake"
    assert adapters._turn_addressed() == ""  # outside a dispatch: unknown


def test_the_adapter_calls_the_listeners_own_method():
    listener = MagicMock(spec=ContinuousVoiceListener)
    server = MagicMock(_voice_listener=listener)
    adapters._not_understood(server)
    listener.note_not_understood.assert_called_once_with()
    adapters._not_understood(MagicMock(_voice_listener=None))  # voice off: nothing to tell


async def test_a_doubtful_follow_up_through_the_real_names_says_sorry_once_and_tells_the_listener():
    listener = MagicMock(spec=ContinuousVoiceListener)
    server = MagicMock(_voice_listener=listener)
    model_calls = []

    class Model:
        name, provider, streams, local = "fake", "fake", True, False

        async def stream(self, system, messages):
            model_calls.append(messages)
            yield "Okay."

    sink = Sink()
    layer = ConversationLayer(
        sink=sink,
        model=lambda: Model(),
        hold_after=60.0,
        addressed=adapters._turn_addressed,
        not_understood=lambda: adapters._not_understood(server),
    )
    voice_turn_addressed.set("doubtful_follow_up")
    await layer.take("and then the blue one went over there")
    await layer.take("no I mean the other thing")
    assert sink.said == [SORRY]
    assert listener.note_not_understood.call_count == 2
    assert model_calls == []
    voice_turn_addressed.set("wake")
    await layer.take("what's the capital of France")
    assert len(model_calls) == 1

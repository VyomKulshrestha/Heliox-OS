"""One late answer costs one turn: the next turn asks the model again, and only late answers in a row rest it.

A slow turn as the person hears it (cloud model, the code's own settings): at 0 s their turn reaches the layer; at
2 s "One moment."; at 4.5 s with no first word, "The language model is slow right now, so I stopped waiting. Ask me
again." The next turn asks the model again at once."""

from __future__ import annotations

import asyncio
import json
import threading

import httpx
import pytest

from pilot.conversation.fast_model import SLOW_PAUSES, HttpFastModel
from pilot.conversation.layer import AGAIN_ALTERNATES, FALLBACK, ConversationLayer


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Body(httpx.AsyncByteStream):
    def __init__(self, delay: float, text: str) -> None:
        self.delay, self.text = delay, text

    async def __aiter__(self):
        await asyncio.sleep(self.delay)
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": self.text}}
        yield f"data: {json.dumps(event)}\n\n".encode()
        yield b'data: {"type":"message_delta","usage":{"output_tokens":3}}\n\n'


def model_with(delays: list[float], clock: Clock, built_in: list[str] | None = None) -> tuple[HttpFastModel, list]:
    asked: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        delay = delays.pop(0) if delays else 0.0
        asked.append(delay)
        return httpx.Response(200, stream=Body(delay, "It's sunny."), headers={"content-type": "text/event-stream"})

    def factory():
        if built_in is not None:
            built_in.append(threading.current_thread().name)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def key() -> str:
        return "k"

    model = HttpFastModel(provider="claude", name="claude-haiku-5-5", client_factory=factory, api_key=key, clock=clock)
    return model, asked


def layer_with(model, said: list[str]) -> ConversationLayer:
    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    # The deadline and the spoken sign scaled down so the test does not wait seconds; the rule is the same.
    return ConversationLayer(sink=sink, model=lambda: model, hold_after=0.25, first_text_deadline=0.6)


def test_late_answers_in_a_row_rest_the_model_a_little_longer_each_time_and_an_answer_resets_it():
    clock = Clock()
    model, _ = model_with([], clock)
    assert [model.note_slow() for _ in range(5)] == [0.0, 5.0, 15.0, 30.0, 30.0]
    assert SLOW_PAUSES[0] == 0.0 and max(SLOW_PAUSES) <= 30.0
    model.note_answered()
    assert model.note_slow() == 0.0  # a late answer after a good one costs only its own turn


async def test_one_late_answer_then_the_next_turn_asks_the_model_again():
    clock = Clock()
    said: list[str] = []
    model, asked = model_with([1.5, 0.0], clock)
    layer = layer_with(model, said)
    await layer.take("Why is the sky blue?")
    assert said[-1] == FALLBACK["slow"]
    assert said[0] in ("One moment.", "Let me think.", "Give me a second.")  # the spoken sign came first
    said.clear()
    await layer.take("And tomorrow?")
    assert len(asked) == 2  # asked again, no rest after one late answer
    assert said == ["It's sunny."]


async def test_three_late_answers_in_a_row_then_a_short_rest_then_recovery():
    clock = Clock()
    said: list[str] = []
    model, asked = model_with([1.5, 1.5, 0.0], clock)
    layer = layer_with(model, said)
    await layer.take("Why is the sky blue?")  # late: costs this turn
    await layer.take("And tomorrow?")  # late again: rests 5 s
    assert len(asked) == 2
    said.clear()
    await layer.take("And the weekend?")  # within the rest: not asked, said plainly
    assert len(asked) == 2 and said == [AGAIN_ALTERNATES["slow"]]  # the second line in a row differs
    clock.t += SLOW_PAUSES[1] + 0.1
    said.clear()
    await layer.take("And the weekend?")  # the rest is over: asked, answered, and the count starts again
    assert len(asked) == 3 and said == ["It's sunny."]
    assert model.note_slow() == 0.0


async def test_the_http_client_is_built_off_the_event_loop():
    clock = Clock()
    built_in: list[str] = []
    model, _ = model_with([0.0], clock, built_in)
    said: list[str] = []
    await layer_with(model, said).take("Why is the sky blue?")
    assert built_in and built_in[0] != threading.main_thread().name


@pytest.mark.parametrize("reason", ["offline", "no_key", "credit"])
def test_a_real_outage_keeps_its_own_plainer_handling(reason):
    assert "slow" not in FALLBACK[reason]

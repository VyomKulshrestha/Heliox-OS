"""The turn trace (pilot.conversation.trace): the stamps a spoken model answer leaves, in order, with times only."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pilot.conversation import trace
from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.layer import ConversationLayer

MODEL_ROAD = [
    "route decided",
    "context ready",
    "request sent",
    "model first text",
    "first sentence complete",
    "sentence to voice",
]
QUESTION = "How long does pasta take to cook?"


def _sse(text: str) -> bytes:
    event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
    return f"data: {json.dumps(event)}\n\n".encode()


def _model() -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse("About ten minutes. ") + _sse("Taste it at eight.")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async def key() -> str:
        return "k"

    return HttpFastModel(
        provider="claude",
        name="claude-haiku-5-5",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )


@pytest.fixture
def heard(monkeypatch):
    found: list[tuple[str, dict]] = []
    monkeypatch.setattr(trace, "listener", lambda name, at, **data: found.append((name, data)))
    return found


async def _ear(text: str, channel: str, key: str) -> str:
    return "spoken"


async def test_a_model_answer_leaves_every_stamp_in_order(heard):
    stamps = trace.begin()
    trace.mark("words received")
    model = _model()
    result = await ConversationLayer(sink=_ear, model=lambda: model, hold_after=60.0).take(QUESTION)
    assert result.done
    names = [name for name, _at in stamps]
    assert names == ["words received", *MODEL_ROAD]
    assert [at for _name, at in stamps] == sorted(at for _name, at in stamps)
    assert result.trace is stamps
    assert [name for name, _data in heard] == names
    data = dict(heard)
    assert data["route decided"]["route"] and data["request sent"]["warm"] is False  # no warm-up ran: cold
    assert data["sentence to voice"]["hold"] is False


async def test_the_holding_phrase_is_stamped_as_such(heard):
    trace.begin()

    def handler(request: httpx.Request) -> httpx.Response:
        async def slow():
            await asyncio.sleep(0.2)
            yield _sse("About ten minutes. ")

        return httpx.Response(200, content=slow(), headers={"content-type": "text/event-stream"})

    async def key() -> str:
        return "k"

    model = HttpFastModel(
        provider="claude",
        name="claude-haiku-5-5",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )
    await ConversationLayer(sink=_ear, model=lambda: model, hold_after=0.05).take(QUESTION)
    holds = [data.get("hold") for name, data in heard if name == "sentence to voice"]
    assert holds[0] is True and False in holds


def test_the_summary_carries_times_only():
    stamps = [("words received", 10.0), ("route decided", 10.012), ("sentence to voice", 10.5)]
    assert trace.summary(stamps) == "words received +0, route decided +12, sentence to voice +500"
    assert trace.summary([]) == ""


def test_a_listener_that_fails_never_stops_the_turn(monkeypatch):
    def broken(name, at, **data):
        raise RuntimeError("bench log closed")

    monkeypatch.setattr(trace, "listener", broken)
    stamps = trace.begin()
    trace.mark("words received")
    assert [name for name, _at in stamps] == ["words received"]


async def test_a_spoken_model_question_is_traced_from_the_dispatch(heard):
    """Through the server's own dispatch (the session runner): the stamps before the layer come first."""
    from benchmarks.conversation_eval.sessions import run_session

    session = {"name": "trace", "turns": [{"say": "Hey Heliox, how long does pasta take to cook?"}]}
    _record, failures = await run_session(session, "claude")
    assert not failures, failures
    # The warm-up started at the wake phrase finishes in its own time, so its stamp is checked apart from the turn's.
    names = [name for name, _data in heard if name != "warm-up done"]
    assert names[:3] == ["words received", "settled", "code handlers done"]
    assert names[3:] == MODEL_ROAD
    handlers = dict(heard)["code handlers done"]
    assert handlers["handlers_ms"] >= 0 and handlers["slowest"].startswith("_") and "answered_by" not in handlers
    for _name, data in heard:
        assert not any(isinstance(value, str) and "pasta" in value for value in data.values())  # never the words

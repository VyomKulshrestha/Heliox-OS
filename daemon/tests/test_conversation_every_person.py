"""What each kind of person hears from the fast layer, over each provider's own wire format.

Claude key, an OpenAI-compatible endpoint, OpenRouter, Gemini, local Ollama only, and no model at all. For each: the
first sentence is spoken before the stream ends; a stream that dies midway ends in a plain sentence, not silence; a
401 or a missing key says what to do in Settings (a local server that is not running says to start it); a model
that is slow to answer gets the short spoken sign before 2.5 s. The streams are stubbed with httpx's mock transport,
so the real HttpFastModel parses each format; no network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.layer import (
    FALLBACK,
    HOLD_AFTER_SECONDS,
    HOLD_PHRASES,
    LOCAL_LOADING,
    LOST_MID_REPLY,
    ConversationLayer,
)

PROVIDERS = ["claude", "openai", "openrouter", "gemini", "ollama"]
NAMES = {
    "claude": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1:8b",
}


def wire(provider: str, text: str) -> bytes:
    """One piece of reply text as the provider sends it."""
    if provider == "ollama":
        return (json.dumps({"message": {"content": text}, "done": False}) + "\n").encode()
    if provider == "gemini":
        event = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    elif provider == "claude":
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
    else:
        event = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(event)}\n\n".encode()


def wire_end(provider: str) -> bytes:
    if provider == "ollama":
        return (
            json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 9, "eval_count": 3}) + "\n"
        ).encode()
    if provider == "gemini":
        return b'data: {"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":3}}\n\n'
    if provider == "claude":
        return b'data: {"type":"message_delta","usage":{"output_tokens":3}}\n\n'
    return b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3}}\n\ndata: [DONE]\n\n'


class Body(httpx.AsyncByteStream):
    """A response body whose pieces come when the test lets them: waits, a dropped connection, or the end."""

    def __init__(self, steps: list) -> None:
        self.steps = steps

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for step in self.steps:
            if isinstance(step, bytes):
                yield step
            elif isinstance(step, asyncio.Event):
                await asyncio.wait_for(step.wait(), 5)
            elif isinstance(step, float):
                await asyncio.sleep(step)
            elif isinstance(step, BaseException):
                raise step


def model_for(provider: str, steps: list | None = None, *, status: int = 200, key: str | None = "test-key"):
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "invalid x-api-key"}})
        return httpx.Response(200, stream=Body(steps or []), headers={"content-type": "text/event-stream"})

    async def api_key() -> str | None:
        return key

    return HttpFastModel(
        provider=provider,
        name=NAMES[provider],
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=None if provider == "ollama" else api_key,
        base_url="http://127.0.0.1:11434" if provider == "ollama" else "",
    )


class Ear:
    """The sink: what the person hears, with an event when the first answer sentence is heard."""

    def __init__(self) -> None:
        self.heard: list[str] = []
        self.first = asyncio.Event()

    async def __call__(self, text: str, channel: str, key: str) -> str:
        self.heard.append(text)
        if channel == "final_answer":
            self.first.set()
        return "spoken"


def layer_for(model, ear: Ear, **kwargs) -> ConversationLayer:
    return ConversationLayer(sink=ear, model=lambda: model, **kwargs)


QUESTION = "How long does pasta take to cook?"


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_the_first_sentence_is_heard_before_the_stream_ends(provider):
    ear = Ear()
    # The stream stays open after the first sentence until the test sees that sentence was heard.
    steps = [wire(provider, "About ten minutes. "), ear.first, wire(provider, "Taste it at eight."), wire_end(provider)]
    result = await layer_for(model_for(provider, steps), ear, hold_after=60.0).take(QUESTION)
    assert result.done
    assert ear.heard == ["About ten minutes.", "Taste it at eight."]


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_stream_that_dies_midway_ends_in_a_plain_sentence(provider):
    ear = Ear()
    steps = [wire(provider, "About ten minutes. "), wire(provider, "Then you"), httpx.ReadError("connection reset")]
    result = await layer_for(model_for(provider, steps), ear, hold_after=60.0).take(QUESTION)
    assert result.done
    assert ear.heard[0] == "About ten minutes."
    assert ear.heard[-1] == LOST_MID_REPLY
    assert not any(line.startswith("Then you") for line in ear.heard)  # no half sentence trailing off


@pytest.mark.parametrize("provider", ["claude", "openai", "openrouter", "gemini"])
@pytest.mark.parametrize("how", ["rejected", "missing"])
async def test_a_rejected_or_missing_key_says_where_to_fix_it(provider, how):
    ear = Ear()
    model = model_for(provider, status=401) if how == "rejected" else model_for(provider, key=None)
    result = await layer_for(model, ear, hold_after=60.0).take(QUESTION)
    assert result.done
    assert ear.heard == [FALLBACK["no_key"]]
    assert "Settings, under Model" in ear.heard[0]


async def test_a_local_server_that_is_not_running_says_to_start_it():
    ear = Ear()
    error = httpx.ConnectError("refused")

    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    model = HttpFastModel(
        provider="ollama",
        name=NAMES["ollama"],
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="http://127.0.0.1:11434",
    )
    result = await layer_for(model, ear, hold_after=60.0).take(QUESTION)
    assert result.done and ear.heard == [FALLBACK["local_down"]]
    assert "Start Ollama" in ear.heard[0]


async def test_no_model_at_all_says_one_needs_setting_up():
    ear = Ear()
    result = await ConversationLayer(sink=ear, model=lambda: None).take(QUESTION)
    assert result.done and ear.heard == [FALLBACK["none"]]


def test_the_spoken_sign_comes_before_two_and_a_half_seconds():
    assert HOLD_AFTER_SECONDS < 2.5


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_slow_model_gets_the_short_spoken_sign_first(provider):
    """The hold delay scaled down (the real one is HOLD_AFTER_SECONDS) so the test does not wait seconds."""
    ear = Ear()
    steps = [0.3, wire(provider, "About ten minutes."), wire_end(provider)]
    await layer_for(model_for(provider, steps), ear, hold_after=0.05).take(QUESTION)
    sign = LOCAL_LOADING if provider == "ollama" else HOLD_PHRASES[0]
    assert ear.heard == [sign, "About ten minutes."]

"""A Wi-Fi sign-in page, or a reply with no text, on the conversation road: the person hears what happened, never
silence (found by tests/platform_real/network_faults_check.py with real sockets)."""

from __future__ import annotations

import httpx
import pytest

from pilot.conversation.fast_model import FastModelUnavailable, HttpFastModel
from pilot.conversation.layer import FALLBACK, ConversationLayer

PORTAL = b"<html><body><h1>Sign in to the hotel Wi-Fi</h1></body></html>"


def model_answering(status: int, content_type: str | None, body: bytes, provider: str = "claude") -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": content_type} if content_type is not None else {}
        return httpx.Response(status, content=body, headers=headers)

    async def key() -> str:
        return "k"

    return HttpFastModel(
        provider=provider,
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=None if provider == "ollama" else key,
        base_url="http://127.0.0.1:11434" if provider == "ollama" else "",
    )


async def heard(model) -> list[str]:
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    await ConversationLayer(sink=sink, model=lambda: model, hold_after=60.0).take("What do you think?")
    return said


async def test_a_sign_in_page_is_said_as_one_and_held_like_a_network_that_is_down():
    model = model_answering(200, "text/html; charset=utf-8", PORTAL)
    with pytest.raises(FastModelUnavailable) as caught:
        async for _ in model.stream("system", [{"role": "user", "content": "hi"}]):
            pass
    assert caught.value.reason == "not_api" and "text/html" in str(caught.value)
    said = await heard(model_answering(200, "text/html", PORTAL))
    assert said == [FALLBACK["not_api"]]
    assert "sign in" in said[0]


@pytest.mark.parametrize("provider", ["claude", "ollama"])
async def test_a_stream_that_carries_no_text_is_said_not_left_silent(provider):
    empty = b'data: {"type":"message_delta","usage":{"output_tokens":0}}\n\n' if provider == "claude" else b"{}\n"
    kind = "text/event-stream" if provider == "claude" else "application/x-ndjson"
    said = await heard(model_answering(200, kind, empty, provider=provider))
    assert said == [FALLBACK["empty"]]


@pytest.mark.parametrize("kind", ["text/event-stream", "application/json", "application/x-ndjson", ""])
async def test_the_providers_own_content_types_are_read_as_before(kind):
    text = b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Fine, thanks."}}\n\n'
    said = await heard(model_answering(200, kind, text))
    assert said == ["Fine, thanks."]


STREAM = b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Fine, thanks."}}\n\n'


@pytest.mark.parametrize("kind", ["text/plain", "application/octet-stream", "text/x-my-proxy"])
async def test_a_local_or_proxy_server_streaming_events_under_its_own_type_still_answers(kind):
    assert await heard(model_answering(200, kind, STREAM)) == ["Fine, thanks."]


async def test_a_text_page_that_is_not_an_answer_is_said_as_not_the_api_naming_its_type():
    model = model_answering(200, "text/plain", b"Please accept the terms of use to continue.")
    with pytest.raises(FastModelUnavailable) as caught:
        async for _ in model.stream("system", [{"role": "user", "content": "hi"}]):
            pass
    assert caught.value.reason == "not_api" and "text/plain" in str(caught.value)
    said = await heard(model_answering(200, "text/plain", b"Please accept the terms of use to continue."))
    assert said == [FALLBACK["not_api"]]


async def test_with_no_content_type_it_is_read_and_nothing_in_it_is_empty():
    assert await heard(model_answering(200, None, STREAM)) == ["Fine, thanks."]
    assert await heard(model_answering(200, None, b"")) == [FALLBACK["empty"]]


@pytest.mark.parametrize("kind", ["application/xhtml+xml", "text/html"])
async def test_a_web_page_is_refused_at_once_whatever_it_holds(kind):
    assert await heard(model_answering(200, kind, STREAM)) == [FALLBACK["not_api"]]

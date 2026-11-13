"""What a fast-model call that did not answer costs in the budget (14 November).

A provider that answers with an error (a 4xx or a 5xx) generated nothing and bills nothing, so nothing is booked.
Before, such a call went through the abandoned-call estimate and was booked at about 100 output tokens a second for the
time it was open: a Gemini call refused after 0.76 s was booked as 76 output tokens and $0.00016. A call the provider
had accepted and that was cut (a stream cut after the 200, or a timeout before the first byte, when the provider may
still be generating) keeps the estimate, as a provider may bill it."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from pilot.conversation.fast_model import FastModelUnavailable, HttpFastModel


class Budget:
    def __init__(self) -> None:
        self.booked: list[dict] = []

    def check_budget(self, provider: str) -> None:
        return None

    async def record_usage(self, provider, model, input_tokens, output_tokens, **kwargs) -> None:
        self.booked.append({"input": input_tokens, "output": output_tokens, **kwargs})


def model_with(handler, budget: Budget, *, clock=None) -> HttpFastModel:
    async def key() -> str:
        return "k"

    kwargs = {"clock": clock} if clock is not None else {}
    return HttpFastModel(
        provider="gemini",
        name="gemini-3.5-flash-lite",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
        budget=budget,
        request_timeout=0.2,
        **kwargs,
    )


async def ask(model: HttpFastModel) -> list[str]:
    return [piece async for piece in model.stream("s", [{"role": "user", "content": "hi"}])]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (
            404,
            {
                "error": {
                    "code": 404,
                    "message": "This model is no longer available to new users.",
                    "status": "NOT_FOUND",
                }
            },
        ),
        (
            400,
            {"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}},
        ),
        (500, {"error": {"code": 500, "message": "Internal error.", "status": "INTERNAL"}}),
        (503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}),
    ],
)
async def test_an_error_answer_books_nothing(status, body):
    budget = Budget()
    now = [0.0]

    def handler(request: httpx.Request) -> httpx.Response:
        now[0] += 0.76  # open for as long as the refused call in the provider comparison
        return httpx.Response(status, json=body)

    model = model_with(handler, budget, clock=lambda: now[0])
    with pytest.raises(FastModelUnavailable):
        await ask(model)
    assert budget.booked == []
    assert model.calls[-1].usage_source == "refused" and model.calls[-1].output_tokens == 0


async def test_a_timeout_before_the_first_byte_keeps_the_estimate():
    """The provider may have the request and be generating: it may bill, so the estimate stays (the safe side)."""
    budget = Budget()

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    with pytest.raises(FastModelUnavailable):
        await ask(model_with(handler, budget))
    assert len(budget.booked) == 1 and budget.booked[0]["usage_source"] == "abandoned_estimate"


async def test_a_stream_cut_after_the_answer_began_keeps_the_estimate():
    budget = Budget()

    class Cut(httpx.AsyncByteStream):
        async def __aiter__(self):
            event = {"candidates": [{"content": {"parts": [{"text": "Mostly the processor. "}]}}]}
            yield f"data: {json.dumps(event)}\r\n\r\n".encode()
            raise httpx.ReadError("connection reset")

    now = [0.0]

    def handler(request: httpx.Request) -> httpx.Response:
        now[0] += 0.5  # open for half a second before the cut
        return httpx.Response(200, stream=Cut(), headers={"content-type": "text/event-stream"})

    model = model_with(handler, budget, clock=lambda: now[0])
    with pytest.raises(FastModelUnavailable):
        await ask(model)
    assert len(budget.booked) == 1
    assert budget.booked[0]["usage_source"] == "abandoned_estimate" and budget.booked[0]["output"] > 0


async def test_a_web_page_instead_of_the_provider_books_nothing():
    budget = Budget()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>Sign in to the Wi-Fi</html>", headers={"content-type": "text/html"})

    with pytest.raises(FastModelUnavailable):
        await ask(model_with(handler, budget))
    assert budget.booked == []


async def test_a_finished_answer_books_what_the_provider_reported():
    budget = Budget()

    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "candidates": [{"content": {"parts": [{"text": "Mostly the processor."}]}}],
            "usageMetadata": {"promptTokenCount": 1136, "candidatesTokenCount": 7},
        }
        return httpx.Response(200, content=f"data: {json.dumps(event)}\r\n\r\n".encode(), headers={"content-type": "x"})

    assert await ask(model_with(handler, budget)) == ["Mostly the processor."]
    await asyncio.sleep(0)
    assert budget.booked == [
        {
            "input": 1136,
            "output": 7,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "usage_source": "reported",
        }
    ]

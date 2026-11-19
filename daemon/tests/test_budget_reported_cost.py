"""What a call is booked at: the provider's own cost when it reports one (OpenRouter's usage.cost), the price tables
otherwise, and never less than the abandoned-call estimate for a call cut short. Fake responses only: no call leaves
the machine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest

from pilot.conversation.fast_model import HttpFastModel
from pilot.models import prices
from pilot.models.budget_tracker import BudgetTracker, claude_cost_usd
from pilot.models.cloud import (
    CloudCallUsage,
    UsageCapture,
    _note_billed_call,
    _openai_usage,
    _usage_capture,
    estimate_abandoned_call,
    reported_cost,
)
from pilot.models.router import ModelRouter

MODEL = "google/gemini-3.5-flash-lite"  # a model with a current list price (pilot.models.prices)


@dataclass
class FakeModelConfig:
    budget_enabled: bool = True
    budget_monthly_limit_usd: float = 100.0
    max_tokens_per_task: int = 100_000
    max_usd_per_task: float = 10.0


async def tracker_in(tmp_path) -> BudgetTracker:
    tracker = BudgetTracker(FakeModelConfig(), str(tmp_path / "budget.db"))
    await tracker.initialize()
    return tracker


async def booked(tracker: BudgetTracker) -> float:
    return float((await tracker.get_stats())["cost_usd"])


def listed(input_tokens: int, output_tokens: int, model: str = MODEL) -> float:
    return prices.cost_usd("openrouter", model, input_tokens, output_tokens)


# ── the tracker ──


async def test_a_reported_cost_is_booked_as_it_is(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        list_price = listed(1587, 5)
        await tracker.record_usage("openrouter", MODEL, 1587, 5, usage_source="reported", cost_usd=0.00157858)
        assert await booked(tracker) == pytest.approx(0.00157858, abs=1e-6)
        assert list_price < 0.00157858  # a pinned host can charge many times the list price
    finally:
        await tracker.close()


async def test_without_a_reported_cost_the_price_tables_decide_as_before(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        await tracker.record_usage("openrouter", MODEL, 1000, 100, usage_source="reported")
        assert await booked(tracker) == pytest.approx(listed(1000, 100), abs=1e-9)
    finally:
        await tracker.close()


@pytest.mark.parametrize("junk", [-1.0, float("nan"), float("inf"), True, "0.5"])
async def test_a_cost_that_is_not_a_usable_number_is_ignored(tmp_path, junk):
    tracker = await tracker_in(tmp_path)
    try:
        await tracker.record_usage("openrouter", MODEL, 1000, 100, usage_source="reported", cost_usd=junk)
        assert await booked(tracker) == pytest.approx(listed(1000, 100), abs=1e-9)
    finally:
        await tracker.close()


async def test_an_estimate_after_a_reported_cost_above_the_list_price_is_raised_by_that_ratio(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        ratio = 15.0
        await tracker.record_usage(
            "openrouter", MODEL, 1000, 100, usage_source="reported", cost_usd=listed(1000, 100) * ratio
        )
        first = await booked(tracker)
        await tracker.record_usage("openrouter", MODEL, 2000, 400, usage_source="abandoned_estimate")
        assert await booked(tracker) - first == pytest.approx(listed(2000, 400) * ratio, rel=1e-6)
        # Another model on the same provider keeps its own price.
        other = "anthropic/claude-haiku-5.5"
        before = await booked(tracker)
        await tracker.record_usage("openrouter", other, 2000, 400, usage_source="abandoned_estimate")
        assert await booked(tracker) - before == pytest.approx(listed(2000, 400, other), rel=1e-6)
    finally:
        await tracker.close()


async def test_a_reported_cost_below_the_list_price_never_lowers_an_estimate(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        await tracker.record_usage("openrouter", MODEL, 1000, 100, usage_source="reported", cost_usd=0.0)
        await tracker.record_usage("openrouter", MODEL, 2000, 400, usage_source="abandoned_estimate")
        assert await booked(tracker) == pytest.approx(listed(2000, 400), rel=1e-6)
    finally:
        await tracker.close()


async def test_a_call_cut_short_is_never_booked_below_its_estimate(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        # Claude reported the prompt at the start, and only a few output tokens had arrived when it was cut.
        await tracker.record_usage(
            "claude", "claude-haiku-5-5", 900, 12, usage_source="reported", at_least_tokens=(1000, 400)
        )
        assert await booked(tracker) == pytest.approx(claude_cost_usd("claude-haiku-5-5", 1000, 400), rel=1e-6)
        # A reported cost above the estimate stands.
        before = await booked(tracker)
        await tracker.record_usage(
            "openrouter", MODEL, 900, 12, usage_source="reported", cost_usd=1.0, at_least_tokens=(1000, 400)
        )
        assert await booked(tracker) - before == pytest.approx(1.0, rel=1e-6)
        # A cut call's partial tokens say nothing about the host's price: later estimates are not raised by it.
        before = await booked(tracker)
        await tracker.record_usage("openrouter", MODEL, 1000, 100, usage_source="abandoned_estimate")
        assert await booked(tracker) - before == pytest.approx(listed(1000, 100), abs=2e-6)
    finally:
        await tracker.close()


# ── OpenRouter's usage object ──

NOT_BYOK = {
    "prompt_tokens": 1587,
    "completion_tokens": 5,
    "cost": 0.00157858,
    "is_byok": False,
    "cost_details": {"upstream_inference_cost": 0.00157858},
}


def test_reported_cost_reads_openrouter_usage():
    assert reported_cost(NOT_BYOK) == pytest.approx(0.00157858)  # upstream repeats cost: not added
    byok = {**NOT_BYOK, "cost": 0.00007, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.0014}}
    assert reported_cost(byok) == pytest.approx(0.00147)  # the person's own key is billed apart
    for junk in ({}, {"cost": None}, {"cost": -1}, {"cost": "0.1"}, {"cost": True}, None, "usage"):
        assert reported_cost(junk) is None


def test_the_planner_keeps_the_cost_for_openrouter_only():
    capture = UsageCapture()
    token = _usage_capture.set(capture)
    try:
        _note_billed_call("openrouter", MODEL, {"usage": NOT_BYOK}, _openai_usage)
        _note_billed_call("openai", "gpt-6-luna", {"usage": NOT_BYOK}, _openai_usage)
    finally:
        _usage_capture.reset(token)
    assert capture.calls[0].cost_usd == pytest.approx(0.00157858)
    assert capture.calls[1].cost_usd is None and capture.calls[1].reported


class Recorded:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def check_budget(self, provider: str) -> None:
        return None

    async def record_usage(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


async def test_the_planner_books_the_reported_cost():
    budget = Recorded()
    capture = UsageCapture(
        calls=[
            CloudCallUsage("openrouter", MODEL, 1587, 5, reported=True, cost_usd=0.00157858),
            CloudCallUsage("openai", "gpt-6-luna", 100, 10, reported=True),
        ]
    )
    await ModelRouter._record_billed_cloud_calls(SimpleNamespace(_budget_tracker=budget), capture, 10, "x")
    assert budget.calls[0][1]["cost_usd"] == pytest.approx(0.00157858)
    assert "cost_usd" not in budget.calls[1][1]  # nothing reported: the tables decide, as before


# ── the talk model's stream ──


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def openai_like(text: str, usage: dict) -> bytes:
    pieces = [sse({"choices": [{"delta": {"role": "assistant", "content": word + " "}}]}) for word in text.split()]
    return b"".join([*pieces, sse({"choices": [], "usage": usage}), b"data: [DONE]\n\n"])


def talk_model(provider: str, name: str, body: bytes, clock: Clock, budget, advance: float = 0.0) -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        clock.t += advance
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async def key() -> str:
        return "test-key-not-real"

    return HttpFastModel(
        provider=provider,
        name=name,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
        budget=budget,
        clock=clock,
    )


async def test_a_finished_openrouter_reply_books_openrouter_s_cost(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        model = talk_model("openrouter", MODEL, openai_like("It is sunny.", NOT_BYOK), Clock(), tracker)
        text = "".join([piece async for piece in model.stream("Be brief.", [{"role": "user", "content": "Weather?"}])])
        assert text.strip() == "It is sunny."
        assert model.calls[0].usage_source == "reported"
        assert await booked(tracker) == pytest.approx(0.00157858, abs=1e-6)
    finally:
        await tracker.close()


async def test_another_openai_compatible_host_s_cost_field_is_not_read():
    budget = Recorded()
    model = talk_model("openai", "gpt-6-luna", openai_like("It is sunny.", NOT_BYOK), Clock(), budget)
    [piece async for piece in model.stream("Be brief.", [{"role": "user", "content": "Weather?"}])]
    assert "cost_usd" not in budget.calls[0][1]


async def test_an_openrouter_reply_cut_short_is_booked_at_the_estimate_raised_by_the_reported_ratio(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        clock = Clock()
        # A finished call first: the host charged 15 times the list price for it.
        usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cost": listed(1000, 100) * 15, "is_byok": False}
        finished = talk_model("openrouter", MODEL, openai_like("Yes.", usage), clock, tracker)
        [piece async for piece in finished.stream("Be brief.", [{"role": "user", "content": "Ok?"}])]
        before = await booked(tracker)
        # Then a reply the layer cuts at its question: the usage chunk never arrives.
        long_reply = "Do you want the long version? " + "More words follow here. " * 20
        cut = talk_model("openrouter", MODEL, openai_like(long_reply, usage), clock, tracker, advance=2.0)
        stream = cut.stream("Be brief.", [{"role": "user", "content": "Tell me."}])
        await anext(stream)
        await stream.aclose()
        record = cut.calls[0]
        assert record.usage_source == "abandoned_estimate" and record.cut_off
        estimate = (record.input_tokens, record.output_tokens)
        assert estimate[1] > 0  # two seconds open: output estimated, up to the request's limit
        assert await booked(tracker) - before == pytest.approx(listed(*estimate) * 15, abs=2e-6)  # stats round to 1e-6
    finally:
        await tracker.close()


async def test_a_claude_reply_cut_short_after_its_prompt_was_reported_is_not_booked_below_the_estimate(tmp_path):
    tracker = await tracker_in(tmp_path)
    try:
        clock = Clock()
        body = b"".join(
            [
                sse({"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}}),
                sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Well, "}}),
                sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "it is."}}),
                sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}}),
            ]
        )
        model = talk_model("claude", "claude-haiku-5-5", body, clock, tracker, advance=1.0)
        stream = model.stream("Be brief.", [{"role": "user", "content": "Is it?"}])
        await anext(stream)
        await stream.aclose()
        record = model.calls[0]
        assert record.usage_source == "reported" and record.input_tokens == 20  # what had arrived
        estimate = _estimate_of(model, "Is it?")
        assert estimate[1] > record.output_tokens  # one second open: more output than had arrived
        booked_cost = await booked(tracker)
        assert booked_cost > claude_cost_usd("claude-haiku-5-5", 20, record.output_tokens)
        assert booked_cost == pytest.approx(claude_cost_usd("claude-haiku-5-5", *estimate), rel=1e-6)
    finally:
        await tracker.close()


def _estimate_of(model: HttpFastModel, words: str) -> tuple[int, int]:
    _url, _headers, body = model.request("Be brief.", [{"role": "user", "content": words}], "test-key-not-real")
    estimate = estimate_abandoned_call(model.provider, model.name, body, 1.0)
    return estimate.input_tokens, estimate.output_tokens

"""Claude spend is recorded from the usage Anthropic reports, priced per model.

Every call goes through a mocked HTTP transport; nothing here reaches the network.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pilot.config import ModelConfig, PilotConfig
from pilot.models import cloud as cloud_module
from pilot.models.budget_tracker import (
    CLAUDE_MODEL_PRICES,
    BudgetExceededError,
    BudgetTracker,
    TaskBudgetExceededError,
    claude_cost_usd,
    claude_price_for,
    current_task_id,
)
from pilot.models.cloud import CloudClient
from pilot.models.router import ModelRouter

PROMPT = "plan the work " * 50


class _Vault:
    async def get_key(self, provider: str) -> str | None:
        return "test-key" if provider == "claude" else None


def _message(text: str = "plan", usage: dict | None = None, stop_reason: str = "end_turn") -> dict:
    body: dict = {
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
    }
    if usage is not None:
        body["usage"] = usage
    return body


# Planning call from the report: 10,000 input tokens, 2,500 output tokens, most
# of the output being thinking that never shows up as text.
THINKING_HEAVY = {"input_tokens": 10_000, "output_tokens": 2_500}


class _Harness:
    def __init__(self, router: ModelRouter, tracker: BudgetTracker, calls: list[httpx.Request]) -> None:
        self.router = router
        self.tracker = tracker
        self.calls = calls


async def _harness(tmp_path, handler, *, model="claude-opus-5-5", monthly=100.0, per_task=100.0) -> _Harness:
    cfg = PilotConfig()
    cfg.model = ModelConfig(
        provider="cloud",
        cloud_provider="claude",
        cloud_model=model,
        budget_enabled=True,
        budget_monthly_limit_usd=monthly,
        max_usd_per_task=per_task,
        max_tokens_per_task=10_000_000,
        max_tokens_per_action=1_000_000,
    )
    tracker = BudgetTracker(cfg.model, str(tmp_path / "b.db"))
    await tracker.initialize()
    calls: list[httpx.Request] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    with (
        patch("pilot.models.router.OllamaClient"),
        patch("pilot.models.router.CloudClient"),
        patch("pilot.models.router.LLMCache"),
        patch("pilot.models.router.RedisCacheAdapter"),
    ):
        router = ModelRouter(cfg, MagicMock())
    client = CloudClient(cfg, _Vault())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(counting))
    router._cloud = client

    async def via_cloud(prompt, *, system="", json_mode=False, temperature=0.1, stream_callback=None):
        return await client.generate(prompt, system=system, json_mode=json_mode, temperature=temperature)

    router._generate_with_cache = via_cloud
    router._rate_limiter = MagicMock()
    router._rate_limiter.acquire = AsyncMock()
    router.set_budget_tracker(tracker)
    return _Harness(router, tracker, calls)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(cloud_module.asyncio, "sleep", instant)


def _ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


async def _rows(tracker: BudgetTracker) -> list[tuple]:
    async with tracker._pool.read() as db:
        cursor = await db.execute(
            "SELECT provider, model, input_tokens, output_tokens, cache_read_tokens, "
            "cache_creation_tokens, cost_usd, usage_source FROM token_usage ORDER BY id"
        )
        rows = await cursor.fetchall()
        await cursor.close()
    return [tuple(r) for r in rows]


# ---------------------------------------------------------------- pricing


def test_price_table_matches_skill_prices():
    assert (CLAUDE_MODEL_PRICES["claude-opus-5-5"].input, CLAUDE_MODEL_PRICES["claude-opus-5-5"].output) == (4.0, 20.0)
    assert (CLAUDE_MODEL_PRICES["claude-sonnet-5-5"].input, CLAUDE_MODEL_PRICES["claude-sonnet-5-5"].output) == (
        2.0,
        10.0,
    )
    assert (CLAUDE_MODEL_PRICES["claude-haiku-5-5"].input, CLAUDE_MODEL_PRICES["claude-haiku-5-5"].output) == (
        0.10,
        0.50,
    )


def test_cost_is_priced_per_model():
    assert claude_cost_usd("claude-opus-5-5", 10_000, 2_500) == pytest.approx(0.09)
    assert claude_cost_usd("claude-sonnet-5-5", 10_000, 2_500) == pytest.approx(0.045)
    assert claude_cost_usd("claude-haiku-5-5", 10_000, 2_500) == pytest.approx(0.00225)


def test_haiku_55_long_prompt_uses_higher_rate():
    assert claude_cost_usd("claude-haiku-5-5", 150_000, 1_000) == pytest.approx(150_000 * 0.5e-6 + 1_000 * 2.5e-6)


def test_unknown_claude_model_uses_most_expensive_known_rate():
    price, known = claude_price_for("claude-future-9")
    assert not known
    assert (price.input, price.output) == (10.0, 50.0)
    assert claude_cost_usd("claude-future-9", 10_000, 2_500) == pytest.approx(0.225)
    # A newer minor version of a known family is not assumed to share its price.
    assert not claude_price_for("claude-opus-5-6")[1]
    assert not claude_price_for("")[1]


def test_dated_snapshot_of_known_model_is_known():
    price, known = claude_price_for("claude-opus-5-5-20260401")
    assert known
    assert price.output == 20.0


def test_cache_tokens_are_priced_separately():
    # 1000*4 + 100*20 + 50000*0.20 + 2000*4*1.25 = 26,000 micro-dollars
    cost = claude_cost_usd("claude-opus-5-5", 1_000, 100, cache_read_tokens=50_000, cache_creation_tokens=2_000)
    assert cost == pytest.approx(0.026)


# ------------------------------------------------------- recording through the router


@pytest.mark.asyncio
async def test_thinking_heavy_usage_is_recorded_exactly(tmp_path):
    h = await _harness(tmp_path, lambda r: _ok(_message("short visible text", THINKING_HEAVY)))
    await h.router.generate(PROMPT)

    rows = await _rows(h.tracker)
    assert rows == [("claude", "claude-opus-5-5", 10_000, 2_500, 0, 0, pytest.approx(0.09), "reported")]
    stats = await h.tracker.get_stats()
    assert stats["input_tokens"] == 10_000
    assert stats["output_tokens"] == 2_500
    await h.tracker.close()


@pytest.mark.asyncio
async def test_cache_tokens_from_usage_are_recorded(tmp_path):
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "cache_read_input_tokens": 50_000,
        "cache_creation_input_tokens": 2_000,
    }
    h = await _harness(tmp_path, lambda r: _ok(_message("x", usage)))
    await h.router.generate(PROMPT)
    row = (await _rows(h.tracker))[0]
    assert row[4:6] == (50_000, 2_000)
    assert row[6] == pytest.approx(0.026)
    await h.tracker.close()


@pytest.mark.asyncio
async def test_model_decides_the_price(tmp_path):
    h = await _harness(tmp_path, lambda r: _ok(_message("x", THINKING_HEAVY)), model="claude-sonnet-5-5")
    await h.router.generate(PROMPT)
    assert (await _rows(h.tracker))[0][6] == pytest.approx(0.045)
    await h.tracker.close()


@pytest.mark.asyncio
async def test_unknown_model_is_recorded_at_the_safe_rate(tmp_path):
    h = await _harness(tmp_path, lambda r: _ok(_message("x", THINKING_HEAVY)), model="claude-future-9")
    await h.router.generate(PROMPT)
    assert (await _rows(h.tracker))[0][6] == pytest.approx(0.225)
    await h.tracker.close()


@pytest.mark.asyncio
async def test_missing_usage_falls_back_to_the_estimate(tmp_path):
    text = "a" * 400
    h = await _harness(tmp_path, lambda r: _ok(_message(text, usage=None)))
    await h.router.generate(PROMPT)
    provider, model, tokens_in, tokens_out, _, _, cost, source = (await _rows(h.tracker))[0]
    assert (provider, model, source) == ("claude", "claude-opus-5-5", "estimate")
    assert tokens_in == h.router._estimate_input_tokens(PROMPT)
    assert tokens_out == 100
    assert cost == pytest.approx(claude_cost_usd("claude-opus-5-5", tokens_in, 100))
    await h.tracker.close()


@pytest.mark.asyncio
async def test_billed_but_rejected_response_is_still_recorded(tmp_path):
    refusal = _message("", THINKING_HEAVY, stop_reason="refusal")
    h = await _harness(tmp_path, lambda r: _ok(refusal))
    with pytest.raises(RuntimeError, match="declined"):
        await h.router.generate(PROMPT)
    rows = await _rows(h.tracker)
    assert len(rows) == 1
    assert rows[0][2:4] == (10_000, 2_500)
    assert rows[0][6] == pytest.approx(0.09)
    await h.tracker.close()


@pytest.mark.asyncio
async def test_rate_limit_records_nothing(tmp_path):
    h = await _harness(tmp_path, lambda r: httpx.Response(429, json={"error": {"type": "rate_limit_error"}}))
    with pytest.raises(RuntimeError):
        await h.router.generate(PROMPT)
    assert await _rows(h.tracker) == []
    await h.tracker.close()


@pytest.mark.asyncio
async def test_network_error_records_nothing(tmp_path):
    def boom(request):
        raise httpx.ConnectError("down")

    h = await _harness(tmp_path, boom)
    with pytest.raises(RuntimeError):
        await h.router.generate(PROMPT)
    assert await _rows(h.tracker) == []
    await h.tracker.close()


@pytest.mark.asyncio
async def test_retry_after_429_records_only_the_billed_response(tmp_path):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, json={})
        return _ok(_message("x", THINKING_HEAVY))

    h = await _harness(tmp_path, handler)
    await h.router.generate(PROMPT)
    assert len(h.calls) == 2
    assert len(await _rows(h.tracker)) == 1
    await h.tracker.close()


@pytest.mark.asyncio
async def test_cache_hit_records_nothing(tmp_path):
    h = await _harness(tmp_path, lambda r: _ok(_message("x", THINKING_HEAVY)))
    h.router._generate_with_cache = AsyncMock(return_value="cached text")
    assert await h.router.generate(PROMPT) == "cached text"
    assert await _rows(h.tracker) == []
    await h.tracker.close()


@pytest.mark.asyncio
async def test_request_body_is_unchanged(tmp_path):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return _ok(_message("x", THINKING_HEAVY))

    h = await _harness(tmp_path, handler)
    await h.router.generate(PROMPT)
    assert seen["body"]["model"] == "claude-opus-5-5"
    await h.tracker.close()


# ------------------------------------------------------------------ caps


@pytest.mark.asyncio
async def test_per_task_cap_trips_on_real_numbers_and_fails_closed(tmp_path):
    # Each call costs $0.09 against a $0.10 task cap. By the old estimate
    # (visible text only) a call cost a fraction of a cent and never tripped.
    h = await _harness(tmp_path, lambda r: _ok(_message("ok", THINKING_HEAVY)), per_task=0.10)
    h.tracker.start_task("task-1")
    token = current_task_id.set("task-1")
    try:
        await h.router.generate(PROMPT)  # $0.09, under the cap
        await h.router.generate(PROMPT)  # allowed (check happens before the call): $0.18
        assert len(h.calls) == 2
        with pytest.raises(TaskBudgetExceededError, match=r"USD budget.*No more model calls") as excinfo:
            await h.router.generate(PROMPT)
        assert "$0.1800" in str(excinfo.value)
        assert len(h.calls) == 2  # no HTTP request was made
        with pytest.raises(TaskBudgetExceededError):
            await h.router.generate(PROMPT)
        assert len(h.calls) == 2
    finally:
        current_task_id.reset(token)
    await h.tracker.close()


@pytest.mark.asyncio
async def test_monthly_cap_trips_on_real_numbers(tmp_path):
    h = await _harness(tmp_path, lambda r: _ok(_message("ok", THINKING_HEAVY)), monthly=0.10)
    await h.router.generate(PROMPT)
    await h.router.generate(PROMPT)
    with pytest.raises(BudgetExceededError, match="Monthly API budget"):
        await h.router.generate(PROMPT)
    assert len(h.calls) == 2
    await h.tracker.close()


# ---------------------------------------------------------------- abandoned requests


class _Started(list):
    """Request bodies the mock provider received; ``sent`` is set once the first arrives."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = asyncio.Event()


def _hanging(started: _Started):
    async def handler(request: httpx.Request) -> httpx.Response:
        started.append(json.loads(request.content))
        started.sent.set()
        await asyncio.Event().wait()  # the provider is still working when Heliox stops waiting
        raise AssertionError("unreachable")

    return handler


@pytest.mark.asyncio
async def test_a_request_cancelled_while_open_is_recorded_as_an_estimate(tmp_path):
    started = _Started()
    h = await _harness(tmp_path, _hanging(started))
    try:
        task = asyncio.create_task(h.router.generate(PROMPT, system="Review the plan."))
        await asyncio.wait_for(started.sent.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        [(provider, model, input_tokens, output_tokens, _, _, cost, source)] = await _rows(h.tracker)
        body = json.dumps(started[0], ensure_ascii=False)
        assert (provider, model, source) == ("claude", "claude-opus-5-5", "abandoned_estimate")
        assert input_tokens == -(-len(body) // 4)  # the whole request body, system prompt included
        assert output_tokens <= 1  # open for well under a hundredth of a second
        assert cost > 0
    finally:
        await h.tracker.close()  # its SQLite worker threads must not outlive this test's loop


@pytest.mark.asyncio
async def test_a_request_that_times_out_is_recorded_with_output_bounded_by_its_open_time(tmp_path):
    started = _Started()
    h = await _harness(tmp_path, _hanging(started))
    try:
        loop = asyncio.get_running_loop()
        begin = loop.time()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(h.router.generate(PROMPT), timeout=0.3)
        waited = loop.time() - begin

        [row] = await _rows(h.tracker)
        assert row[7] == "abandoned_estimate"
        assert 20 <= row[3] <= -(-waited * cloud_module.ABANDONED_OUTPUT_TOKENS_PER_SECOND // 1) + 1
    finally:
        await h.tracker.close()


@pytest.mark.asyncio
async def test_a_request_that_never_connected_is_not_billed(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    h = await _harness(tmp_path, refuse)
    try:
        with pytest.raises(RuntimeError):
            await h.router.generate(PROMPT)
        assert await _rows(h.tracker) == []
    finally:
        await h.tracker.close()


def test_the_output_estimate_never_exceeds_the_requests_own_limit():
    payload = {"model": "claude-opus-5-5", "max_tokens": 500, "messages": [{"role": "user", "content": "x" * 400}]}
    call = cloud_module.estimate_abandoned_call("claude", "claude-opus-5-5", payload, 60.0)
    assert call.output_tokens == 500  # 60 s at 100 tokens a second would be 6,000
    assert call.input_tokens == -(-len(json.dumps(payload)) // 4)
    gemini = {"contents": [], "generationConfig": {"maxOutputTokens": 64}}
    assert cloud_module.estimate_abandoned_call("gemini", "g", gemini, 5.0).output_tokens == 64

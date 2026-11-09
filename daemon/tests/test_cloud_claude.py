"""Tests for the Claude (Anthropic Messages API) path of CloudClient."""

from __future__ import annotations

import json

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.models.cloud import (
    CLAUDE_MAX_TOKENS,
    DEFAULT_MODELS,
    PROVIDER_ENDPOINTS,
    CloudClient,
    claude_supports_effort,
)


class _FakeVault:
    async def get_key(self, provider: str) -> str | None:
        return "test-key" if provider == "claude" else None


def _client(handler, model: str = "") -> CloudClient:
    config = PilotConfig()
    config.model.cloud_provider = "claude"
    config.model.cloud_model = model
    client = CloudClient(config, _FakeVault())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _ok(content: list[dict], stop_reason: str = "end_turn", **extra) -> httpx.Response:
    return httpx.Response(200, json={"content": content, "stop_reason": stop_reason, **extra})


def test_default_model_is_current_opus():
    assert DEFAULT_MODELS["claude"] == "claude-opus-5-5"


@pytest.mark.asyncio
async def test_request_body_has_no_sampling_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.read())
        return _ok([{"type": "text", "text": "ok"}])

    result = await _client(handler).generate("hi", system="be brief", temperature=0.1)

    assert result == "ok"
    body = seen["body"]
    assert seen["url"] == PROVIDER_ENDPOINTS["claude"]
    assert seen["headers"]["x-api-key"] == "test-key"
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == CLAUDE_MAX_TOKENS == 16000
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in body


@pytest.mark.asyncio
async def test_explicit_model_is_passed_through():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return _ok([{"type": "text", "text": "ok"}])

    await _client(handler, model="claude-sonnet-5-5").generate("hi")
    assert seen["body"]["model"] == "claude-sonnet-5-5"
    assert "temperature" not in seen["body"]


@pytest.mark.asyncio
async def test_system_message_in_message_list_moves_to_system_field():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return _ok([{"type": "text", "text": "ok"}])

    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    await _client(handler).generate(prompt)
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["messages"] == [{"role": "user", "content": "q"}]


@pytest.mark.asyncio
async def test_text_is_taken_from_text_blocks_after_thinking_block():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(
            [
                {"type": "thinking", "thinking": "", "signature": "x"},
                {"type": "text", "text": "part one "},
                {"type": "text", "text": "part two"},
            ]
        )

    assert await _client(handler).generate("hi") == "part one part two"


@pytest.mark.asyncio
async def test_refusal_raises_clear_error_with_category():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"})

    with pytest.raises(RuntimeError, match="declined this request.*cyber"):
        await _client(handler).generate("hi")


@pytest.mark.asyncio
async def test_max_tokens_with_text_returns_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([{"type": "text", "text": "partial"}], stop_reason="max_tokens")

    assert await _client(handler).generate("hi") == "partial"


@pytest.mark.asyncio
async def test_max_tokens_without_text_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([{"type": "thinking", "thinking": ""}], stop_reason="max_tokens")

    with pytest.raises(RuntimeError, match="max_tokens"):
        await _client(handler).generate("hi")


@pytest.mark.asyncio
async def test_429_is_retried(monkeypatch):
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("pilot.models.cloud.asyncio.sleep", no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})
        return _ok([{"type": "text", "text": "ok"}])

    assert await _client(handler).generate("hi") == "ok"
    assert calls["n"] == 3


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-5", True),
        ("claude-future-9-9", True),
        ("claude-haiku-4-5", False),
        ("claude-sonnet-4-5", False),
        ("claude-sonnet-4-20250514", False),
        ("claude-3-5-sonnet-latest", False),
    ],
)
def test_claude_supports_effort(model, expected):
    assert claude_supports_effort(model) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "has_effort"),
    [("claude-opus-5-5", True), ("claude-haiku-4-5", False), ("claude-future-9-9", True)],
)
async def test_effort_field_only_sent_to_supporting_models(model, has_effort):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return _ok([{"type": "text", "text": "ok"}])

    await _client(handler, model=model).generate("hi")
    assert ("output_config" in seen["body"]) is has_effort

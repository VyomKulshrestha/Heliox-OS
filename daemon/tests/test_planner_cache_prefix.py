"""The planner's request is laid out so a provider can cache its fixed part.

The planner's rules and action catalog (about 10,200 tokens as Anthropic counts them) are the same bytes for every
plan: they go first, alone, marked as the end of the stable prefix, and everything that changes per request comes after
them. Each adapter turns the marker into its provider's own caching field, or drops it, so a provider without explicit
caching gets nothing it would reject. Fakes only: no network, no key."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pilot.agents.planner import Planner
from pilot.models import cloud
from pilot.models.cloud import CACHE_PREFIX

W = "C:\\w"


class Recorder:
    def __init__(self) -> None:
        self.sent: list[list[dict[str, Any]]] = []

    async def generate(self, messages, **_kwargs) -> str:
        self.sent.append([dict(m) for m in messages])
        path = f"{W}\\notes\\a.txt"
        return json.dumps(
            {
                "explanation": "Write it.",
                "actions": [
                    {"action_type": "file_write", "target": path, "parameters": {"path": path, "content": "x"}}
                ],
            }
        )


class Memory:
    def __init__(self, turns: list[tuple[str, str]] | None = None, context: str = "") -> None:
        self.turns = turns or []
        self.context = context

    async def get_context(self, *_args, **_kwargs) -> str:
        return self.context

    async def get_history(self, *, limit: int = 50, **_kwargs) -> list[dict[str, Any]]:
        rows = [
            {"id": i, "timestamp": "", "user_input": r, "success": True, "explanation": e}
            for i, (r, e) in enumerate(self.turns)
        ]
        return list(reversed(rows))[:limit]


class Specialists:
    def get_routing_summary(self, user_input: str) -> dict[str, str]:
        return {"enhanced_prompt": f"File specialist for: {user_input[:20]}"}


@pytest.mark.asyncio
async def test_the_fixed_part_is_the_same_bytes_for_two_requests_and_two_sessions_and_comes_first() -> None:
    first, second = Recorder(), Recorder()
    planner_a = Planner(first, Memory())  # type: ignore[arg-type]
    planner_b = Planner(second, Memory([("Open notepad", "Opened Notepad.")], context="Prefers short notes."))
    planner_b.set_multi_agent_router(Specialists())

    await planner_a.plan(f"Write quokka to {W}\\notes\\a.txt", session_id="one", force_model=True)
    await planner_b.plan(
        f"Write wombat to {W}\\notes\\a.txt",
        session_id="two",
        screen_context="Notepad is open on the left",
        force_model=True,
    )

    a, b = first.sent[0], second.sent[0]
    assert a[0] == b[0]
    assert a[0]["role"] == "system" and a[0][CACHE_PREFIX] is True
    # Nothing that changes per request is in the fixed part.
    for changing in ("quokka", "wombat", "Open notepad", "Prefers short notes", "Notepad is open", "File specialist"):
        assert changing not in a[0]["content"]
    # The changing parts come after it, and the current request is last.
    assert [m["role"] for m in a] == ["system", "user"]
    assert [m["role"] for m in b] == ["system", "system", "user"]
    assert CACHE_PREFIX not in b[1] and "File specialist" in b[1]["content"]
    assert "Open notepad" in b[2]["content"] and "Notepad is open" in b[2]["content"]
    assert b[2]["content"].rstrip().endswith("Answer with the JSON object only.")
    assert "wombat" in b[2]["content"].rsplit("Current user request", 1)[1]


class _Response:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.body


ANSWERS = {
    "claude": {"content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn"},
    "chat": {"choices": [{"message": {"content": "{}"}}]},
    "gemini": {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]},
}

MARKED = [
    {"role": "system", "content": "RULES AND CATALOG", CACHE_PREFIX: True},
    {"role": "system", "content": "SPECIALIST"},
    {"role": "user", "content": "Write it", "type": "goal"},
]


@pytest.fixture
def payloads(monkeypatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def fake_post(self, provider, model, url, payload, headers):
        sent.append(json.loads(json.dumps(payload)))
        kind = "claude" if provider == "claude" else "gemini" if provider == "gemini" else "chat"
        return _Response(ANSWERS[kind])

    monkeypatch.setattr(cloud.CloudClient, "_post", fake_post)
    monkeypatch.setattr(cloud, "_api_json", lambda resp, provider: resp.json())
    return sent


def _client() -> cloud.CloudClient:
    return cloud.CloudClient.__new__(cloud.CloudClient)


def _no_marker_anywhere(payload: Any) -> bool:
    return CACHE_PREFIX not in json.dumps(payload)


@pytest.mark.asyncio
async def test_anthropic_gets_a_cache_breakpoint_on_the_fixed_part_only(payloads) -> None:
    await _client()._call_anthropic("k", "claude-haiku-5-5", MARKED, "", 0.1)

    [payload] = payloads
    assert payload["system"] == [
        {"type": "text", "text": "RULES AND CATALOG", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "SPECIALIST"},
    ]
    assert payload["messages"] == [{"role": "user", "content": "Write it"}]
    assert _no_marker_anywhere(payload)


@pytest.mark.asyncio
async def test_anthropic_without_a_marker_is_sent_as_before(payloads) -> None:
    plain = [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "x"}]
    await _client()._call_anthropic("k", "claude-haiku-5-5", plain, "", 0.1)
    assert payloads[0]["system"] == "A\n\nB"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["anthropic/claude-haiku-5.5", "google/gemini-3.5-flash-lite"])
async def test_openrouter_breakpoint_models_get_cache_control_in_a_content_part(payloads, model) -> None:
    await _client()._call_openai_compat("openrouter", "k", model, MARKED, "", True, 0.1)

    messages = payloads[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "RULES AND CATALOG", "cache_control": {"type": "ephemeral"}}],
    }
    assert messages[1] == {"role": "system", "content": "SPECIALIST"}
    assert _no_marker_anywhere(payloads[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model"),
    [("openai", "gpt-6-luna"), ("meta", "muse-spark-1.3"), ("openrouter", "openai/gpt-6-luna")],
)
async def test_providers_that_cache_by_themselves_get_plain_messages(payloads, provider, model) -> None:
    await _client()._call_openai_compat(provider, "k", model, MARKED, "", True, 0.1)

    messages = payloads[0]["messages"]
    assert [m["content"] for m in messages] == ["RULES AND CATALOG", "SPECIALIST", "Write it"]
    assert all(isinstance(m["content"], str) for m in messages)
    assert "cache_control" not in json.dumps(payloads[0])
    assert _no_marker_anywhere(payloads[0])


@pytest.mark.asyncio
async def test_gemini_gets_the_fixed_part_first_in_its_system_text(payloads) -> None:
    await _client()._call_gemini_native("k", "gemini-3.5-flash-lite", MARKED, "", True, 0.1, max_retries=0)

    payload = payloads[0]
    assert payload["systemInstruction"]["parts"][0]["text"] == "RULES AND CATALOG\n\nSPECIALIST"
    assert _no_marker_anywhere(payload)


@pytest.mark.asyncio
async def test_ollama_chat_gets_no_marker(monkeypatch) -> None:
    from pilot.config import PilotConfig
    from pilot.models.ollama import OllamaClient

    client = OllamaClient(config=PilotConfig())
    sent: list[dict[str, Any]] = []

    async def fake_max_context(model: str):
        return None

    async def fake_post(url, json=None, **_kwargs):  # noqa: A002
        sent.append(json)
        import httpx

        return httpx.Response(200, json={"message": {"content": "{}"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(client, "_max_context", fake_max_context)
    monkeypatch.setattr(client._client, "post", fake_post)
    try:
        await client.chat("llama", MARKED, json_mode=True)
    finally:
        await client.close()
    assert _no_marker_anywhere(sent[0])
    assert [m["content"] for m in sent[0]["messages"]] == ["RULES AND CATALOG", "SPECIALIST", "Write it"]

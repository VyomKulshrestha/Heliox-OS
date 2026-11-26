"""Every provider adapter passes every system message on, in order, and keeps the other turns in order.

The Anthropic and Gemini adapters once kept only the last system message of a chat, so a history summary sent after
the planner's prompt replaced the prompt. This test sends one chat with two system messages around the turns through
each adapter (fakes for the network, no model) and fails if any adapter drops or reorders one."""

from __future__ import annotations

from typing import Any

import pytest

from pilot.models import cloud
from pilot.models.llamacpp import render_prompt
from pilot.models.ollama import OllamaClient
from pilot.models.subscription_cli import _serialize_messages

CHAT = [
    {"role": "system", "content": "FIRST SYSTEM"},
    {"role": "user", "content": "first question"},
    {"role": "system", "content": "SECOND SYSTEM"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
]
TURNS = ["first question", "first answer", "second question"]


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": "ok"}],
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "choices": [{"message": {"content": "ok"}}],
            "message": {"content": "ok"},
        }


def _in_order(text: str, parts: list[str]) -> bool:
    positions = [text.find(part) for part in parts]
    return all(p >= 0 for p in positions) and positions == sorted(positions)


@pytest.fixture
def cloud_payloads(monkeypatch) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    async def fake_post(self, provider, model, url, payload, headers):
        payloads.append(payload)
        return _Response()

    monkeypatch.setattr(cloud.CloudClient, "_post", fake_post)
    monkeypatch.setattr(cloud, "_api_json", lambda resp, provider: resp.json())
    return payloads


@pytest.mark.asyncio
async def test_anthropic_keeps_every_system_message_and_the_turn_order(cloud_payloads) -> None:
    client = cloud.CloudClient.__new__(cloud.CloudClient)
    await client._call_anthropic("k", "claude-haiku-5-5", CHAT, "", 0.1)
    [payload] = cloud_payloads
    assert payload["system"] == "FIRST SYSTEM\n\nSECOND SYSTEM"
    assert [m["content"] for m in payload["messages"]] == TURNS


@pytest.mark.asyncio
async def test_gemini_keeps_every_system_message_and_the_turn_order(cloud_payloads) -> None:
    client = cloud.CloudClient.__new__(cloud.CloudClient)
    await client._call_gemini_native("k", "gemini-x", CHAT, "", True, 0.1, max_retries=0)
    [payload] = cloud_payloads
    assert payload["systemInstruction"]["parts"][0]["text"] == "FIRST SYSTEM\n\nSECOND SYSTEM"
    assert [c["parts"][0]["text"] for c in payload["contents"]] == TURNS


@pytest.mark.asyncio
async def test_openai_compatible_providers_get_the_chat_unchanged(cloud_payloads) -> None:
    client = cloud.CloudClient.__new__(cloud.CloudClient)
    await client._call_openai_compat("openrouter", "k", "some/model", CHAT, "", False, 0.1)
    [payload] = cloud_payloads
    assert payload["messages"] == CHAT


@pytest.mark.asyncio
async def test_ollama_gets_the_chat_unchanged() -> None:
    sent: list[dict[str, Any]] = []

    class _Client:
        async def post(self, url, json):
            sent.append(json)
            return _Response()

    client = OllamaClient.__new__(OllamaClient)
    client._client = _Client()
    client._base_url = "http://127.0.0.1:11434"
    client._keep_alive_seconds = lambda: 60

    async def options(model, text, temperature):
        return {"temperature": temperature}

    client._options = options
    await client.chat("local-model", CHAT)
    assert sent[0]["messages"] == CHAT


def test_subscription_command_lines_keep_every_system_message_in_order() -> None:
    text = _serialize_messages(CHAT, "", include_guard=False)
    assert _in_order(text, ["FIRST SYSTEM", "first question", "SECOND SYSTEM", "first answer", "second question"])


def test_llama_cpp_reads_every_message_in_order_and_not_a_python_list() -> None:
    text = render_prompt(CHAT)
    assert _in_order(text, ["FIRST SYSTEM", "first question", "SECOND SYSTEM", "first answer", "second question"])
    assert "{'role'" not in text and text.endswith("<|assistant|>\n")
    assert render_prompt("hi", "rules") == "<|system|>\nrules\n<|user|>\nhi\n<|assistant|>\n"

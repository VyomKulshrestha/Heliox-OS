"""The fast model's stable system prompt carries one cache breakpoint (the same words), and the session runner warms
the fast model's connection before each dispatch, as the daemon does at a wake or a follow-up."""

from __future__ import annotations

import asyncio

from benchmarks.conversation_eval.sessions import run_session
from pilot.conversation.fast_model import claude_payload
from pilot.conversation.style import SYSTEM_PROMPT


def test_the_system_prompt_is_sent_with_one_cache_breakpoint_and_unchanged_words():
    body = claude_payload("claude-haiku-5-5", 400, SYSTEM_PROMPT, [{"role": "user", "content": "hi"}], "low", True)
    assert body["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # the turn's own context stays after the breakpoint


def test_the_session_runner_warms_the_connection_before_each_dispatch(monkeypatch):
    warmed: list[int] = []
    from pilot.conversation import fast_model

    original = fast_model.HttpFastModel.warm

    async def counting(self):
        warmed.append(1)
        return await original(self)

    monkeypatch.setattr(fast_model.HttpFastModel, "warm", counting)
    session = {
        "name": "warm",
        "turns": [
            {"say": "Hey Heliox, what's a good name for a cat?", "model": "Whiskers."},
            {"say": "Hey Heliox, and a dog?", "gap": 3, "model": "Biscuit."},
        ],
    }
    record, failures = asyncio.run(run_session(session, "claude"))
    assert failures == [], failures
    assert len(warmed) == 2  # one per dispatch
    assert sum(record.model_calls.values()) == 2  # the warm is not counted as a model call


# ── warm-ups: every cloud provider, never a local model, never for speech not said to Heliox ──


def _model(provider: str, sent: list, *, fail: bool = False, clock=None):
    import httpx

    from pilot.conversation.fast_model import HttpFastModel

    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("no network")
        sent.append((request.method, str(request.url), dict(request.headers)))
        return httpx.Response(200, json={})

    async def key() -> str:
        return "test-key"

    kwargs = {"clock": clock} if clock is not None else {}
    return HttpFastModel(
        provider=provider,
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=None if provider == "ollama" else key,
        base_url="http://127.0.0.1:11434" if provider == "ollama" else "",
        **kwargs,
    )


import pytest  # noqa: E402


@pytest.mark.parametrize(
    ("provider", "method", "url"),
    [
        ("claude", "POST", "https://api.anthropic.com/v1/messages/count_tokens"),
        ("openai", "GET", "https://api.openai.com/v1/models"),
        ("openrouter", "GET", "https://openrouter.ai/api/v1/models"),
        ("gemini", "GET", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"),
    ],
)
async def test_every_cloud_provider_is_warmed_with_a_free_request(provider, method, url):
    sent: list = []
    await _model(provider, sent).warm()
    assert [(m, u) for m, u, _h in sent] == [(method, url)]
    headers = sent[0][2]
    assert "test-key" in (
        headers.get("x-api-key", "") + headers.get("authorization", "") + headers.get("x-goog-api-key", "")
    )
    assert "test-key" not in sent[0][1]  # never in the URL


async def test_a_local_model_is_never_warmed():
    sent: list = []
    await _model("ollama", sent).warm()
    assert sent == []


async def test_a_warm_up_that_fails_is_silent_and_not_a_provider_fault():
    model = _model("claude", [], fail=True)
    await model.warm()  # raises nothing
    assert model._offline_until == 0.0 and model._hold_reason == ""


async def test_warm_ups_are_at_most_one_every_few_seconds():
    sent: list = []
    now = [100.0]
    model = _model("claude", sent, clock=lambda: now[0])
    await model.warm()
    now[0] += 1.0
    await model.warm()
    now[0] += 3.0
    await model.warm()
    assert len(sent) == 2


def _server(sent: list):
    from pilot.config import PilotConfig
    from pilot.conversation import layer_for
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._model_router = object()
    model = _model("claude", sent)
    layer_for(server)._model = lambda: model
    return server


@pytest.mark.parametrize(
    ("addressed", "warmed"),
    [
        ("wake", True),
        ("button", True),
        ("follow_up", True),
        ("continuation", True),
        ("live_reply", True),
        ("", False),
        ("doubtful_follow_up", False),
        ("overheard", False),
        ("not addressed", False),
    ],
)
async def test_only_speech_said_to_heliox_opens_a_connection(addressed, warmed):
    """A warm-up carries no words, but a connection made because someone spoke near the microphone would still tell
    the provider something: only the wake phrase, the button or an open follow-up window may cause one."""
    sent: list = []
    server = _server(sent)
    assert server.warm_for_speech(addressed) is warmed
    await asyncio.sleep(0.05)
    assert bool(sent) is warmed


async def test_a_live_caption_of_any_speech_never_warms_the_model(monkeypatch):
    sent: list = []
    server = _server(sent)
    monkeypatch.setattr("pilot.system.local_tts_worker.prewarm_local_voice", lambda voice: None)

    async def nowhere(*_args, **_kwargs):
        return None

    server._broadcast_notification = nowhere
    await server._voice_partial("so anyway I told him")  # captions come for every utterance, addressed or not
    await asyncio.sleep(0.05)
    assert sent == []


def test_the_fast_model_and_the_planner_keep_idle_connections_120_seconds():
    from pilot.config import PilotConfig
    from pilot.conversation.fast_model import KEEPALIVE_SECONDS, build_fast_model
    from pilot.models.cloud import IDLE_CONNECTION_SECONDS, CloudClient

    config = PilotConfig()
    config.model.provider, config.model.cloud_provider = "cloud", "claude"

    class Vault:
        async def get_key(self, provider):
            return None

    model = build_fast_model(config, vault=Vault(), router=None, budget=None)
    client = model.client_factory()
    assert client._transport._pool._keepalive_expiry == KEEPALIVE_SECONDS == 120.0
    planner = CloudClient(config, Vault())
    assert planner._client._transport._pool._keepalive_expiry == IDLE_CONNECTION_SECONDS == 120.0


@pytest.mark.parametrize(("addressed", "warmed"), [("wake", True), ("follow_up", True), ("doubtful_follow_up", False)])
async def test_the_listeners_status_warms_the_model_only_for_speech_the_rule_took(monkeypatch, addressed, warmed):
    sent: list = []
    server = _server(sent)
    monkeypatch.setattr("pilot.system.local_tts_worker.prewarm_local_voice", lambda voice: None)

    async def nowhere(*_args, **_kwargs):
        return None

    server._broadcast_notification = nowhere
    status = "wake_detected" if addressed == "wake" else "follow_up_detected"
    await server._voice_status_broadcast(status, {"transcript": "x", "addressed": addressed})
    await asyncio.sleep(0.05)
    assert bool(sent) is warmed

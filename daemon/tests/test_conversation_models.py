"""The fast model per kind of person (Claude key, another cloud, local only, none), with fake clients: request
shapes, streaming, cut-off, failure reasons and budget accounting. No network, no SDK needed."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.conversation import fast_model as fm
from pilot.conversation.fast_model import (
    ClaudeFastModel,
    FastModelUnavailable,
    HttpFastModel,
    RouterFastModel,
    build_fast_model,
    describe_setup,
    fast_model_name,
)


class FakeBudget:
    def __init__(self, exceeded=False):
        self.exceeded = exceeded
        self.records = []
        self.checked = []

    def check_budget(self, provider):
        self.checked.append(provider)
        if self.exceeded:
            raise RuntimeError("Monthly API budget of $10.00 exceeded")

    async def record_usage(self, provider, model, input_tokens, output_tokens, **kwargs):
        self.records.append((provider, model, input_tokens, output_tokens, kwargs.get("usage_source")))


async def collect(stream, stop_after=None):
    out = []
    gen = stream
    try:
        async for piece in gen:
            out.append(piece)
            if stop_after is not None and len(out) >= stop_after:
                break
    finally:
        await gen.aclose()
    return out


async def key():
    return "test-key"


# ── Claude through the SDK (faked) ──


class FakeClaudeStream:
    def __init__(self, pieces, usage, stop_reason="end_turn", error=None):
        self.pieces = pieces
        self.usage = usage
        self.stop_reason = stop_reason
        self.error = error
        self.current_message_snapshot = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=1)
        )

    @property
    def text_stream(self):
        async def gen():
            for piece in self.pieces:
                if self.error is not None:
                    raise self.error
                yield piece

        return gen()

    async def get_final_message(self):
        return SimpleNamespace(usage=self.usage, stop_reason=self.stop_reason)


class FakeClaudeClient:
    def __init__(self, stream, *, enter_error=None):
        self._stream = stream
        self.enter_error = enter_error
        self.kwargs = None
        self.closed = False
        client = self

        class Manager:
            async def __aenter__(self):
                if client.enter_error is not None:
                    raise client.enter_error
                return client._stream

            async def __aexit__(self, *exc):
                client.closed = True

        self.messages = SimpleNamespace(stream=lambda **kwargs: self._record(kwargs, Manager()))

    def _record(self, kwargs, manager):
        self.kwargs = kwargs
        return manager


def usage(i=120, o=30):
    return SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def claude(client, budget=None, **kwargs):
    return ClaudeFastModel(api_key=key, budget=budget, client_factory=lambda _key: client, **kwargs)


def test_the_claude_request_matches_what_haiku_5_5_accepts():
    client = FakeClaudeClient(FakeClaudeStream(["Hi."], usage()))
    model = claude(client)
    asyncio.run(collect(model.stream("system", [{"role": "user", "content": "hi"}])))
    sent = client.kwargs
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["thinking"] == {"type": "disabled"}  # accepted at effort low, medium and high
    assert sent["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & set(sent)  # API version 2026-10-21 rejects every one
    assert sent["max_tokens"] == 400 and sent["system"] == "system"


def test_a_finished_claude_stream_records_the_reported_usage():
    budget = FakeBudget()
    client = FakeClaudeClient(FakeClaudeStream(["Hello ", "there."], usage(150, 12)))
    out = asyncio.run(collect(claude(client, budget).stream("s", [{"role": "user", "content": "hi"}])))
    assert out == ["Hello ", "there."]
    assert budget.checked == ["claude"]
    assert budget.records == [("claude", "claude-haiku-5-5", 150, 12, "reported")]
    assert client.closed


def test_a_claude_stream_cut_off_records_what_it_had_reported():
    budget = FakeBudget()
    client = FakeClaudeClient(FakeClaudeStream(["one ", "two ", "three ", "four"], usage(150, 40)))
    model = claude(client, budget)
    out = asyncio.run(collect(model.stream("s", [{"role": "user", "content": "hi"}]), stop_after=2))
    assert out == ["one ", "two "]
    provider, name, input_tokens, output_tokens, source = budget.records[0]
    assert (provider, input_tokens, source) == ("claude", 150, "reported")
    assert output_tokens >= 2  # at least the text received, never zero
    assert model.calls[0].cut_off


def test_a_claude_stream_cut_before_any_usage_records_the_abandoned_estimate():
    budget = FakeBudget()
    stream = FakeClaudeStream(["a"], usage())
    stream.current_message_snapshot = None
    client = FakeClaudeClient(stream, enter_error=httpx.ConnectError("down"))
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(claude(client, budget).stream("s", [{"role": "user", "content": "hi"}])))
    assert caught.value.reason == "offline"
    assert budget.records and budget.records[0][-1] == "abandoned_estimate"


def test_an_exhausted_budget_or_missing_key_never_calls_the_model():
    client = FakeClaudeClient(FakeClaudeStream(["x"], usage()))
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(claude(client, FakeBudget(exceeded=True)).stream("s", [])))
    assert caught.value.reason == "budget" and client.kwargs is None

    async def no_key():
        return None

    model = ClaudeFastModel(api_key=no_key, budget=None, client_factory=lambda _k: client)
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [])))
    assert caught.value.reason == "no_key"


def test_after_a_connection_failure_the_next_turn_falls_back_at_once():
    client = FakeClaudeClient(FakeClaudeStream(["x"], usage()), enter_error=httpx.ConnectError("down"))
    model = claude(client)
    for _ in range(2):
        with pytest.raises(FastModelUnavailable) as caught:
            asyncio.run(collect(model.stream("s", [])))
        assert caught.value.reason == "offline"
    assert len(model.calls) == 1  # the second turn did not wait on the network


# ── HTTP providers (fake httpx client) ──


class FakeResponse:
    def __init__(self, lines, status=200):
        self.lines = lines
        self.status_code = status

    async def aread(self):
        return b""

    def raise_for_status(self):
        request = httpx.Request("POST", "http://x")
        raise httpx.HTTPStatusError("bad", request=request, response=httpx.Response(self.status_code, request=request))

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeHttp:
    def __init__(self, lines, status=200, error=None):
        self.lines = lines
        self.status = status
        self.error = error
        self.sent = None
        http = self

        class Ctx:
            async def __aenter__(self):
                if http.error is not None:
                    raise http.error
                return FakeResponse(http.lines, http.status)

            async def __aexit__(self, *exc):
                return False

        self._ctx = Ctx

    def stream(self, method, url, json=None, headers=None, timeout=None):
        self.sent = (method, url, json, headers)
        return self._ctx()


def http_model(provider, lines, name="m", budget=None, status=200, error=None, **kwargs):
    client = FakeHttp(lines, status=status, error=error)
    model = HttpFastModel(
        provider=provider,
        name=name,
        client_factory=lambda: client,
        api_key=None if provider == "ollama" else key,
        budget=budget,
        base_url="http://127.0.0.1:11434",
        **kwargs,
    )
    return model, client


OPENAI_LINES = [
    'data: {"choices":[{"delta":{"content":"Hi "}}]}',
    'data: {"choices":[{"delta":{"content":"there."}}]}',
    'data: {"choices":[],"usage":{"prompt_tokens":90,"completion_tokens":4}}',
    "data: [DONE]",
]


@pytest.mark.parametrize("provider", ["openai", "openrouter", "meta"])
def test_openai_compatible_providers_stream_and_report_usage(provider):
    budget = FakeBudget()
    model, client = http_model(provider, OPENAI_LINES, name="gpt-x", budget=budget)
    assert asyncio.run(collect(model.stream("sys", [{"role": "user", "content": "hi"}]))) == ["Hi ", "there."]
    _method, _url, body, headers = client.sent
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert "temperature" not in body and headers["Authorization"] == "Bearer test-key"
    assert budget.records == [(provider, "gpt-x", 90, 4, "reported")]
    assert not model.local


def test_gemini_streams_over_sse_without_thinking_on_flash():
    lines = [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello."}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":" Bye."}]}}],"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":3}}',
    ]
    budget = FakeBudget()
    model, client = http_model("gemini", lines, name="gemini-2.5-flash", budget=budget)
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert asyncio.run(collect(model.stream("sys", messages))) == ["Hello.", " Bye."]
    _m, url, body, _h = client.sent
    assert ":streamGenerateContent?alt=sse" in url
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert budget.records == [("gemini", "gemini-2.5-flash", 50, 3, "reported")]


def test_a_local_model_streams_from_the_local_server_and_nothing_leaves_the_machine():
    lines = [
        json.dumps({"message": {"content": "Local "}, "done": False}),
        json.dumps({"message": {"content": "reply."}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 70, "eval_count": 5}),
    ]
    budget = FakeBudget()
    model, client = http_model("ollama", lines, name="llama3.1:8b", budget=budget)
    assert model.local
    assert asyncio.run(collect(model.stream("sys", [{"role": "user", "content": "hi"}]))) == ["Local ", "reply."]
    _m, url, body, headers = client.sent
    assert url == "http://127.0.0.1:11434/api/chat" and "Authorization" not in headers
    assert body["options"] == {"num_predict": 400}
    assert budget.checked == []  # a local model is never blocked by the money budget
    assert budget.records == [("ollama", "llama3.1:8b", 70, 5, "reported")]


def test_a_local_server_that_is_not_running_says_so():
    model, _ = http_model("ollama", [], error=httpx.ConnectError("refused"))
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [])))
    assert caught.value.reason == "local_down"
    model, _ = http_model("ollama", [], status=404)
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [])))
    assert caught.value.reason == "local_missing"


@pytest.mark.parametrize(("status", "reason"), [(401, "no_key"), (429, "busy"), (402, "budget"), (500, "busy")])
def test_cloud_errors_become_reasons(status, reason):
    model, _ = http_model("openai", [], status=status)
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [])))
    assert caught.value.reason == reason


def test_a_cut_off_cloud_stream_without_usage_records_the_abandoned_estimate():
    budget = FakeBudget()
    model, _ = http_model("openai", OPENAI_LINES, budget=budget)
    asyncio.run(collect(model.stream("s", [{"role": "user", "content": "hi"}]), stop_after=1))
    assert budget.records[0][-1] == "abandoned_estimate"


def test_the_router_road_is_one_piece_and_reports_reasons():
    class Router:
        async def generate(self, prompt, *, system="", temperature=0.1):
            return "One piece."

    assert asyncio.run(collect(RouterFastModel(Router()).stream("s", []))) == ["One piece."]

    class Broken:
        async def generate(self, prompt, *, system="", temperature=0.1):
            raise RuntimeError("Monthly API budget exceeded")

    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(RouterFastModel(Broken()).stream("s", [])))
    assert caught.value.reason == "budget"


# ── which model each kind of person gets ──


def config(provider, cloud="", cloud_model="", **conversation):
    cfg = PilotConfig()
    cfg.model.provider = provider
    cfg.model.cloud_provider = cloud
    cfg.model.cloud_model = cloud_model
    for name, value in conversation.items():
        setattr(cfg.conversation, name, value)
    return cfg


VAULT = SimpleNamespace(get_key=lambda provider: asyncio.sleep(0, result="k"))


def test_a_claude_key_gets_haiku_streamed_through_the_sdk(monkeypatch):
    monkeypatch.setattr(fm, "sdk_available", lambda: True)
    model = build_fast_model(config("cloud", "claude", "claude-opus-5-5"), vault=VAULT, router=object(), budget=None)
    assert isinstance(model, ClaudeFastModel) and model.name == "claude-haiku-5-5" and not model.local
    chosen = build_fast_model(
        config("cloud", "claude", claude_model="claude-sonnet-5-5"), vault=VAULT, router=object(), budget=None
    )
    assert chosen.name == "claude-sonnet-5-5"


def test_without_the_sdk_claude_falls_back_to_the_ordinary_call(monkeypatch, caplog):
    monkeypatch.setattr(fm, "sdk_available", lambda: False)
    monkeypatch.setattr(fm, "_warned", set())
    model = build_fast_model(config("cloud", "claude"), vault=VAULT, router=object(), budget=None)
    assert isinstance(model, RouterFastModel)
    assert "anthropic package" in caplog.text


@pytest.mark.parametrize(
    ("cloud", "cloud_model", "expected"),
    [("gemini", "", "gemini-2.5-flash"), ("openai", "gpt-4o", "gpt-4o"), ("openrouter", "", "openrouter/auto")],
)
def test_other_cloud_providers_stream_their_configured_model(cloud, cloud_model, expected):
    model = build_fast_model(config("cloud", cloud, cloud_model), vault=VAULT, router=object(), budget=None)
    assert isinstance(model, HttpFastModel) and model.provider == cloud and model.name == expected


def test_each_provider_has_its_own_fast_model_setting():
    cfg = config("cloud", "openai", "gpt-4o", openai_model="gpt-4o-mini")
    assert fast_model_name(cfg, "openai") == "gpt-4o-mini"
    assert fast_model_name(cfg, "claude") == "claude-haiku-5-5"


def test_a_local_only_person_gets_their_local_model_streamed():
    cfg = config("ollama")
    cfg.model.ollama_model = "qwen3.5:4b"
    model = build_fast_model(cfg, vault=VAULT, router=object(), budget=None)
    assert isinstance(model, HttpFastModel) and model.local and model.name == "qwen3.5:4b"
    assert describe_setup(cfg) == "local"


def test_no_model_at_all_gets_none():
    assert build_fast_model(config(""), vault=None, router=None, budget=None) is None
    assert describe_setup(config("")) == "none"


def test_nothing_in_the_package_needs_the_sdk_at_import():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['anthropic'] = None\n"
        "import pilot.conversation, pilot.conversation.fast_model as f\n"
        "assert not f.sdk_available()\n"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_warming_opens_the_connection_with_the_free_token_count_at_most_every_45_s():
    client = FakeClaudeClient(FakeClaudeStream(["x"], usage()))
    counted = []

    async def count_tokens(**kwargs):
        counted.append(kwargs)

    client.messages.count_tokens = count_tokens
    now = [100.0]
    model = claude(client, clock=lambda: now[0])
    asyncio.run(model.warm())
    asyncio.run(model.warm())
    assert len(counted) == 1 and counted[0]["model"] == "claude-haiku-5-5"
    now[0] += 50
    asyncio.run(model.warm())
    assert len(counted) == 2

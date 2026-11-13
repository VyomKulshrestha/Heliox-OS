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

    async def post(self, url, json=None, headers=None, timeout=None):
        self.posted = getattr(self, "posted", []) + [(url, json, headers)]


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
    assert body["options"] == {"num_predict": 400} and body["think"] is False
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


@pytest.mark.parametrize(("status", "reason"), [(401, "no_key"), (429, "busy"), (402, "credit"), (500, "busy")])
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


CLAUDE_SSE = [
    "event: message_start",
    'data: {"type":"message_start","message":{"usage":{"input_tokens":321,"output_tokens":1}}}',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"there."}}',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}',
    'data: {"type":"message_stop"}',
]


def test_claude_over_http_sends_the_same_request_and_records_the_same_usage():
    budget = FakeBudget()
    model, client = http_model("claude", CLAUDE_SSE, name="claude-haiku-5-5", budget=budget)
    out = asyncio.run(collect(model.stream("sys", [{"role": "user", "content": "hi"}])))
    assert out == ["Hello ", "there."]
    _m, url, body, headers = client.sent
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "test-key"
    assert body == {
        "model": "claude-haiku-5-5",
        "max_tokens": 400,
        # The same words, with one cache breakpoint: the next turn reads the stable system prompt from the cache.
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},  # Haiku 5.5 accepts it at effort low, medium and high
        "output_config": {"effort": "low"},
        "stream": True,
    }
    assert not {"temperature", "top_p", "top_k"} & set(body)
    assert budget.checked == ["claude"]
    assert budget.records == [("claude", "claude-haiku-5-5", 321, 7, "reported")]


def test_a_claude_http_stream_cut_off_records_what_it_reported():
    budget = FakeBudget()
    model, _ = http_model("claude", CLAUDE_SSE, name="claude-haiku-5-5", budget=budget)
    asyncio.run(collect(model.stream("s", [{"role": "user", "content": "hi"}]), stop_after=1))
    provider, _name, input_tokens, output_tokens, source = budget.records[0]
    assert (provider, input_tokens, source) == ("claude", 321, "reported") and output_tokens >= 1


def test_claude_error_and_refusal_events_become_reasons():
    for line, reason in (
        ('data: {"type":"error","error":{"type":"overloaded_error"}}', "busy"),
        ('data: {"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":1}}', "refused"),
    ):
        model, _ = http_model("claude", [line])
        with pytest.raises(FastModelUnavailable) as caught:
            asyncio.run(collect(model.stream("s", [])))
        assert caught.value.reason == reason


@pytest.mark.parametrize(
    ("cloud", "planner", "expected"),
    [
        ("gemini", "", "gemini-2.5-flash-lite"),
        ("gemini", "gemini-4-carbon", "gemini-2.5-flash-lite"),
        ("openai", "gpt-6.1-sol", "gpt-6-luna"),
        ("openai", "gpt-5.5-pro", "gpt-6-luna"),
        ("openrouter", "", "openai/gpt-6-luna"),
        ("openrouter", "openrouter/auto", "openai/gpt-6-luna"),
        ("meta", "muse-wonder-1.0", "muse-spark-1.3"),
        ("claude", "claude-opus-5-5", "claude-haiku-5-5"),
    ],
)
def test_talk_never_inherits_the_planners_model(cloud, planner, expected):
    """A reasoning or large planner model (o3, a Pro model, openrouter/auto) is never the talking model: with only
    the planner's model set, talk takes the provider's fast default."""
    model = build_fast_model(config("cloud", cloud, planner), vault=VAULT, router=object(), budget=None)
    assert isinstance(model, HttpFastModel) and model.provider == cloud and model.name == expected


@pytest.mark.parametrize(
    ("provider", "name", "field", "value"),
    [
        ("openai", "gpt-6-luna", "reasoning_effort", "none"),
        ("openrouter", "openai/gpt-6-luna", "reasoning", {"effort": "none"}),
    ],
)
def test_the_talk_default_is_asked_for_no_reasoning(provider, name, field, value):
    model = HttpFastModel(provider=provider, name=name, client_factory=lambda: None, api_key=None)
    _url, _headers, body = model.request("s", [{"role": "user", "content": "hi"}], "k")
    assert body[field] == value


def test_a_model_the_person_chose_is_sent_as_it_is():
    model = HttpFastModel(provider="openai", name="gpt-6.1-sol", client_factory=lambda: None, api_key=None)
    _url, _headers, body = model.request("s", [{"role": "user", "content": "hi"}], "k")
    assert "reasoning_effort" not in body  # GPT-6.1 Sol rejects "none"; the person's own model is left as it is


def test_every_provider_in_settings_has_a_fast_default_with_a_current_price():
    from pilot.conversation.fast_model import FAST_DEFAULTS
    from pilot.models.cloud import PROVIDER_ENDPOINTS
    from pilot.models.prices import price_for

    assert set(FAST_DEFAULTS) == set(PROVIDER_ENDPOINTS)
    for provider, name in FAST_DEFAULTS.items():
        if provider != "claude":
            assert price_for(provider, name)[1], f"{provider}: no current price for its fast default {name}"


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


def test_a_claude_key_gets_haiku_streamed_over_http():
    model = build_fast_model(config("cloud", "claude", "claude-opus-5-5"), vault=VAULT, router=object(), budget=None)
    assert isinstance(model, HttpFastModel) and model.provider == "claude" and model.name == "claude-haiku-5-5"
    chosen = build_fast_model(
        config("cloud", "claude", claude_model="claude-sonnet-5-5"), vault=VAULT, router=object(), budget=None
    )
    assert chosen.name == "claude-sonnet-5-5"


def test_a_claude_stream_cut_before_any_usage_records_the_abandoned_estimate():
    budget = FakeBudget()
    model, _ = http_model("claude", [], name="claude-haiku-5-5", budget=budget, error=httpx.ReadTimeout("slow"))
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [{"role": "user", "content": "hi"}])))
    assert caught.value.reason == "offline"
    assert budget.records and budget.records[0][-1] == "abandoned_estimate"


def test_an_exhausted_budget_or_missing_key_never_calls_the_model():
    model, client = http_model("claude", CLAUDE_SSE, budget=FakeBudget(exceeded=True))
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [])))
    assert caught.value.reason == "budget" and client.sent is None

    async def no_key():
        return None

    model, client = http_model("claude", CLAUDE_SSE)
    model.api_key = no_key
    with pytest.raises(FastModelUnavailable) as caught:
        asyncio.run(collect(model.stream("s", [])))
    assert caught.value.reason == "no_key" and client.sent is None


def test_after_a_connection_failure_the_next_turn_falls_back_at_once():
    model, client = http_model("claude", [], error=httpx.ConnectError("down"))
    for _ in range(2):
        with pytest.raises(FastModelUnavailable) as caught:
            asyncio.run(collect(model.stream("s", [])))
        assert caught.value.reason == "offline"
    assert len(model.calls) == 1  # the second turn did not wait on the network


def test_nothing_in_the_package_imports_the_anthropic_sdk():
    import subprocess
    import sys

    code = (
        "import sys; import pilot.conversation, pilot.conversation.fast_model\nassert 'anthropic' not in sys.modules\n"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


# ── failures seen in the network drill ──


@pytest.mark.parametrize(
    ("provider", "status", "body", "reason"),
    [
        ("claude", 400, '{"error":{"message":"Your credit balance is too low to access the API"}}', "credit"),
        ("openai", 429, '{"error":{"code":"insufficient_quota"}}', "credit"),
        ("openrouter", 402, '{"error":{"message":"Insufficient credits"}}', "credit"),
        ("openai", 429, '{"error":{"code":"rate_limit_exceeded"}}', "busy"),
    ],
)
def test_an_account_out_of_credit_is_told_apart_from_a_busy_one(provider, status, body, reason):
    import httpx

    request = httpx.Request("POST", "http://x")
    error = httpx.HTTPStatusError("x", request=request, response=httpx.Response(status, request=request, text=body))
    assert fm._classify(error, {}) == reason


def test_out_of_credit_and_rate_limits_are_held_so_the_provider_is_not_asked_every_turn():
    import httpx

    now = [0.0]
    model, _ = http_model("openai", [], status=429)
    model.clock = lambda: now[0]
    request = httpx.Request("POST", "http://x")
    limited = httpx.HTTPStatusError(
        "x", request=request, response=httpx.Response(429, request=request, headers={"retry-after": "12"})
    )
    model._after_failure("busy", limited)
    with pytest.raises(FastModelUnavailable) as caught:
        model._gate()
    assert caught.value.reason == "busy"
    now[0] = 13.0
    model._gate()  # the hold is over
    model._after_failure("credit", limited)
    now[0] = 300.0
    with pytest.raises(FastModelUnavailable) as caught:
        model._gate()
    assert caught.value.reason == "credit"


def test_a_local_model_held_after_a_failure_still_says_it_is_the_local_one():
    model, _ = http_model("ollama", [], error=httpx.ConnectError("refused"))
    for _ in range(2):
        with pytest.raises(FastModelUnavailable) as caught:
            asyncio.run(collect(model.stream("s", [])))
        assert caught.value.reason == "local_down"


def test_warming_opens_the_connection_with_the_free_token_count_at_most_every_few_seconds():
    now = [100.0]
    model, client = http_model("claude", CLAUDE_SSE, name="claude-haiku-5-5")
    model.clock = lambda: now[0]
    asyncio.run(model.warm())
    asyncio.run(model.warm())
    assert len(client.posted) == 1 and client.posted[0][0].endswith("/v1/messages/count_tokens")
    assert client.posted[0][2]["x-api-key"] == "test-key"
    now[0] += 5
    asyncio.run(model.warm())
    assert len(client.posted) == 2


def test_the_model_is_asked_to_answer_in_the_persons_language():
    """Spoken and typed: the reply's language follows the person's, including Hindi that the recogniser wrote in
    Latin letters (docs/PLATFORM_SUPPORT.md, "Languages other than English")."""
    from pilot.conversation.style import SYSTEM_PROMPT, TYPED_SYSTEM_PROMPT

    assert "Answer in the language the person spoke to you in" in SYSTEM_PROMPT
    assert "Latin letters" in SYSTEM_PROMPT
    assert "Answer in the language the person wrote in" in TYPED_SYSTEM_PROMPT

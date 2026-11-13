"""Every cloud provider the Settings panel offers, over its own documented streaming format, through a real socket.

The panel offers Gemini, OpenAI, OpenRouter, Claude and Meta, and Ollama on this computer. For each, a stub HTTP
server on 127.0.0.1 answers in the shape that provider's public documentation gives for a streamed reply: Anthropic's
named events with ``message_start`` usage and ``ping``; OpenAI's ``chat.completion.chunk`` objects with a role-only
first chunk, ``usage: null`` until a last usage-only chunk and ``[DONE]``; OpenRouter's the same with its
``: OPENROUTER PROCESSING`` comment lines; Gemini's ``alt=sse`` events separated by CRLF, each with
``usageMetadata``; Ollama's JSON lines. Meta's Model API is documented as OpenAI chat-completions compatible, so it
gets OpenAI's shape. The bad-key answers are each provider's documented error body and status (Gemini says 400
``API_KEY_INVALID``, not 401).

The real ``HttpFastModel`` reads them over httpx and a real connection, with chunked transfer, and the real
conversation layer speaks: the first sentence must be heard while the server is still holding the rest, the usage
the provider reported must reach the budget, and a bad key must say where to fix it. Nothing leaves the computer.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.layer import FALLBACK, LOST_MID_REPLY, ConversationLayer

QUESTION = "How long does pasta take to cook?"
FIRST, SECOND = "About ten minutes. ", "Taste it at eight."
KEY = "test-key-not-real"
NAMES = {
    "claude": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "meta": "muse-spark-1.1",
    "ollama": "llama3.1:8b",
}
CLOUD = ["claude", "openai", "openrouter", "gemini", "meta"]


# ── Each provider's documented stream ──


def _sse(data: dict | str, event: str = "", sep: str = "\n") -> bytes:
    text = data if isinstance(data, str) else json.dumps(data)
    head = f"event: {event}{sep}" if event else ""
    return f"{head}data: {text}{sep}{sep}".encode()


def claude_stream(pieces: list[str]) -> tuple[list[bytes], bytes]:
    """Anthropic Messages streaming: named events; usage in message_start and message_delta."""
    start = [
        _sse(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": NAMES["claude"],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 25,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 410,
                        "output_tokens": 1,
                    },
                },
            },
            "message_start",
        ),
        _sse(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            "content_block_start",
        ),
        _sse({"type": "ping"}, "ping"),
    ]
    deltas = [
        _sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}},
            "content_block_delta",
        )
        for p in pieces
    ]
    end = b"".join(
        [
            _sse({"type": "content_block_stop", "index": 0}, "content_block_stop"),
            _sse(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 12},
                },
                "message_delta",
            ),
            _sse({"type": "message_stop"}, "message_stop"),
        ]
    )
    return [b"".join(start) + deltas[0], *deltas[1:]], end


def _chunk(model: str, delta: dict, finish: str | None = None, **extra) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1763000000,
        "model": model,
        "system_fingerprint": "fp_1",
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
        "usage": None,
        **extra,
    }


def openai_stream(pieces: list[str], provider: str) -> tuple[list[bytes], bytes]:
    """Chat completions streaming with stream_options.include_usage: a role-only first chunk, usage null on every
    content chunk, a finish chunk, one usage-only chunk with empty choices, then [DONE]. OpenRouter adds SSE comment
    lines while its upstream starts."""
    model = NAMES[provider]
    comment = b": OPENROUTER PROCESSING\n\n" if provider == "openrouter" else b""
    first = comment + _sse(_chunk(model, {"role": "assistant", "content": "", "refusal": None}))
    deltas = [_sse(_chunk(model, {"content": p})) for p in pieces]
    usage = {
        "prompt_tokens": 25,
        "completion_tokens": 12,
        "total_tokens": 37,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    last = {**_chunk(model, {}), "choices": [], "usage": usage}
    end = _sse(_chunk(model, {}, "stop")) + _sse(last) + _sse("[DONE]")
    return [first + deltas[0], *deltas[1:]], end


def gemini_stream(pieces: list[str]) -> tuple[list[bytes], bytes]:
    """streamGenerateContent?alt=sse: each event a whole GenerateContentResponse with usageMetadata so far, CRLF."""
    events = []
    for count, piece in enumerate(pieces, 1):
        events.append(
            _sse(
                {
                    "candidates": [{"content": {"parts": [{"text": piece}], "role": "model"}, "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": 25,
                        "candidatesTokenCount": count,
                        "totalTokenCount": 25 + count,
                    },
                    "modelVersion": NAMES["gemini"],
                    "responseId": "r1",
                },
                sep="\r\n",
            )
        )
    end = _sse(
        {
            "candidates": [{"content": {"parts": [{"text": ""}], "role": "model"}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 25, "candidatesTokenCount": 12, "totalTokenCount": 37},
            "modelVersion": NAMES["gemini"],
            "responseId": "r1",
        },
        sep="\r\n",
    )
    return events, end


def ollama_stream(pieces: list[str]) -> tuple[list[bytes], bytes]:
    """/api/chat with stream: one JSON object per line; counts on the last."""
    model = NAMES["ollama"]
    lines = [
        (
            json.dumps(
                {
                    "model": model,
                    "created_at": "2026-11-13T10:00:00Z",
                    "message": {"role": "assistant", "content": p},
                    "done": False,
                }
            )
            + "\n"
        ).encode()
        for p in pieces
    ]
    end = (
        json.dumps(
            {
                "model": model,
                "created_at": "2026-11-13T10:00:01Z",
                "message": {"role": "assistant", "content": ""},
                "done_reason": "stop",
                "done": True,
                "total_duration": 1,
                "load_duration": 1,
                "prompt_eval_count": 25,
                "prompt_eval_duration": 1,
                "eval_count": 12,
                "eval_duration": 1,
            }
        )
        + "\n"
    ).encode()
    return lines, end


def stream_for(provider: str, pieces: list[str]) -> tuple[list[bytes], bytes]:
    if provider == "claude":
        return claude_stream(pieces)
    if provider == "gemini":
        return gemini_stream(pieces)
    if provider == "ollama":
        return ollama_stream(pieces)
    return openai_stream(pieces, provider)


# The documented answer to a bad key: (status, body).
BAD_KEY = {
    "claude": (
        401,
        {
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            "request_id": "req_1",
        },
    ),
    "openai": (
        401,
        {
            "error": {
                "message": "Incorrect API key provided: test-key***real.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    ),
    "openrouter": (401, {"error": {"message": "No auth credentials found", "code": 401}}),
    "meta": (
        401,
        {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    ),
    "gemini": (
        400,
        {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                        "domain": "googleapis.com",
                        "metadata": {"service": "generativelanguage.googleapis.com"},
                    }
                ],
            }
        },
    ),
}


# ── A stub server on a real socket ──


@dataclass
class Seen:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: dict


@dataclass
class StubServer:
    """Answers every request with ``status``, ``content_type`` and the ``steps``: bytes are sent as one chunk each,
    an ``asyncio.Event`` holds the rest until it is set, a float waits that many seconds, an exception drops the
    connection."""

    status: int = 200
    content_type: str = "text/event-stream"
    steps: list = field(default_factory=list)
    seen: list[Seen] = field(default_factory=list)
    port: int = 0
    _server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> StubServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
            method, target, _ = head[0].split(" ", 2)
            headers = {k.strip().casefold(): v.strip() for k, v in (h.split(":", 1) for h in head[1:] if ":" in h)}
            raw = await reader.readexactly(int(headers.get("content-length", "0")))
            split = urlsplit(target)
            self.seen.append(Seen(method, split.path, parse_qs(split.query), headers, json.loads(raw) if raw else {}))
            reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized"}.get(self.status, "Error")
            if self.status != 200:
                body = json.dumps(self.steps[0]).encode()
                writer.write(
                    f"HTTP/1.1 {self.status} {reason}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
                return
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: {self.content_type}\r\nTransfer-Encoding: chunked\r\n"
                "Connection: close\r\n\r\n".encode()
            )
            await writer.drain()
            for step in self.steps:
                if isinstance(step, asyncio.Event):
                    await asyncio.wait_for(step.wait(), 5)
                elif isinstance(step, float):
                    await asyncio.sleep(step)
                elif isinstance(step, BaseException):
                    writer.transport.abort()  # the connection drops mid-reply
                    return
                else:
                    writer.write(f"{len(step):x}\r\n".encode() + step + b"\r\n")
                    await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()


class FakeBudget:
    def __init__(self) -> None:
        self.recorded: list[dict] = []

    def check_budget(self, provider: str) -> None:
        return None

    async def record_usage(self, provider, model, input_tokens, output_tokens, **kwargs) -> None:
        self.recorded.append({"provider": provider, "input": input_tokens, "output": output_tokens, **kwargs})


class Ear:
    def __init__(self) -> None:
        self.heard: list[str] = []
        self.first = asyncio.Event()

    async def __call__(self, text: str, channel: str, key: str) -> str:
        self.heard.append(text)
        if channel == "final_answer":
            self.first.set()
        return "spoken"


def model_for(provider: str, server: StubServer, monkeypatch, budget: FakeBudget | None = None) -> HttpFastModel:
    from pilot.models import cloud

    if provider != "ollama":
        path = "/v1beta" if provider == "gemini" else "/v1/messages" if provider == "claude" else "/v1/chat/completions"
        monkeypatch.setitem(cloud.PROVIDER_ENDPOINTS, provider, server.base + path)

    async def api_key() -> str:
        return KEY

    return HttpFastModel(
        provider=provider,
        name=NAMES[provider],
        client_factory=lambda: httpx.AsyncClient(trust_env=False),
        api_key=None if provider == "ollama" else api_key,
        base_url=server.base if provider == "ollama" else "",
        budget=budget,
    )


def layer_for(model: HttpFastModel, ear: Ear) -> ConversationLayer:
    return ConversationLayer(sink=ear, model=lambda: model, hold_after=60.0)


# ── The tests ──


@pytest.mark.parametrize("provider", [*CLOUD, "ollama"])
async def test_the_first_sentence_is_spoken_while_the_provider_is_still_sending(provider, monkeypatch):
    ear, budget = Ear(), FakeBudget()
    pieces, end = stream_for(provider, ["About ten", " minutes. ", "Taste it", " at eight."])
    # The server sends the first sentence, then holds everything else until the person has heard it.
    steps = [pieces[0], pieces[1], ear.first, pieces[2], pieces[3], end]
    kind = "application/x-ndjson" if provider == "ollama" else "text/event-stream"
    async with StubServer(content_type=kind, steps=steps) as server:
        model = model_for(provider, server, monkeypatch, budget)
        result = await layer_for(model, ear).take(QUESTION)
    assert result.done
    assert ear.heard == ["About ten minutes.", "Taste it at eight."]
    if provider == "ollama":
        assert budget.recorded == [] or all(r["provider"] == "ollama" for r in budget.recorded)
    else:
        # What the provider reported reached the budget, not an estimate.
        assert len(budget.recorded) == 1
        record = budget.recorded[0]
        assert record["usage_source"] == "reported"
        assert (record["input"], record["output"]) == (25, 12)
        if provider == "claude":
            assert record["cache_read_tokens"] == 410


@pytest.mark.parametrize("provider", [*CLOUD, "ollama"])
async def test_each_request_has_the_documented_path_key_and_stream_flag(provider, monkeypatch):
    ear = Ear()
    pieces, end = stream_for(provider, [FIRST, SECOND])
    async with StubServer(steps=[*pieces, end]) as server:
        await layer_for(model_for(provider, server, monkeypatch), ear).take(QUESTION)
    (seen,) = server.seen
    assert seen.method == "POST"
    if provider == "claude":
        assert seen.path == "/v1/messages"
        assert seen.headers["x-api-key"] == KEY and seen.headers["anthropic-version"]
        assert seen.body["stream"] is True
    elif provider == "gemini":
        assert seen.path == f"/v1beta/models/{NAMES['gemini']}:streamGenerateContent"
        assert seen.query["alt"] == ["sse"]
        assert seen.headers.get("x-goog-api-key") == KEY
        assert "key" not in seen.query  # the key never travels in the address
        assert seen.body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    elif provider == "ollama":
        assert seen.path == "/api/chat" and seen.body["stream"] is True
        assert "authorization" not in seen.headers
    else:
        assert seen.path == "/v1/chat/completions"
        assert seen.headers["authorization"] == f"Bearer {KEY}"
        assert seen.body["stream"] is True and seen.body["stream_options"] == {"include_usage": True}
    assert QUESTION in json.dumps(seen.body)


@pytest.mark.parametrize("provider", CLOUD)
async def test_a_bad_key_in_each_providers_own_words_says_where_to_fix_it(provider, monkeypatch):
    ear = Ear()
    status, body = BAD_KEY[provider]
    async with StubServer(status=status, steps=[body]) as server:
        result = await layer_for(model_for(provider, server, monkeypatch), ear).take(QUESTION)
    assert result.done
    assert ear.heard == [FALLBACK["no_key"]]
    assert "Settings, under Model" in ear.heard[0]


@pytest.mark.parametrize("provider", ["openai", "openrouter", "meta"])
async def test_an_error_sent_inside_an_openai_compatible_stream_is_not_a_finished_reply(provider, monkeypatch):
    """OpenRouter documents that a failure after the stream has started comes as a chunk with an ``error`` object
    and ``finish_reason: "error"``, with HTTP 200 already sent. The half sentence before it is never spoken."""
    ear = Ear()
    model_name = NAMES[provider]
    pieces, _ = openai_stream(["About ten minutes. ", "Then you"], provider)
    failed = _chunk(
        model_name, {"content": ""}, "error", error={"code": "server_error", "message": "Provider disconnected"}
    )
    async with StubServer(steps=[*pieces, _sse(failed)]) as server:
        result = await layer_for(model_for(provider, server, monkeypatch), ear).take(QUESTION)
    assert result.done
    assert ear.heard[0] == "About ten minutes."
    assert not any(line.startswith("Then you") for line in ear.heard)
    assert ear.heard[-1] == LOST_MID_REPLY


@pytest.mark.parametrize("provider", [*CLOUD, "ollama"])
async def test_a_connection_dropped_mid_reply_over_a_real_socket_ends_plainly(provider, monkeypatch):
    ear = Ear()
    pieces, _ = stream_for(provider, ["About ten minutes. ", "Then you"])
    # The drop comes once the first sentence was heard: a reset discards what the receiver has not read yet (Windows
    # does), and the test is about the end of a reply, not its start.
    steps = [pieces[0], ear.first, pieces[1], 0.2, ConnectionResetError()]
    async with StubServer(steps=steps) as server:
        result = await layer_for(model_for(provider, server, monkeypatch), ear).take(QUESTION)
    assert result.done
    assert ear.heard[0] == "About ten minutes."
    assert not any(line.startswith("Then you") for line in ear.heard)
    assert ear.heard[-1] == LOST_MID_REPLY


# The documented answer when the model named is not one the account can use: (status, body).
UNKNOWN_MODEL = {
    "claude": (404, {"type": "error", "error": {"type": "not_found_error", "message": "model: claude-nope"}}),
    "openai": (
        404,
        {
            "error": {
                "message": "The model `gpt-nope` does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        },
    ),
    "openrouter": (400, {"error": {"message": "nope/nope is not a valid model ID", "code": 400}}),
    "meta": (
        404,
        {"error": {"message": "The model does not exist", "type": "invalid_request_error", "code": "model_not_found"}},
    ),
    "gemini": (
        404,
        {
            "error": {
                "code": 404,
                "message": "models/gemini-nope is not found for API version v1beta, or is not supported for generateContent.",
                "status": "NOT_FOUND",
            }
        },
    ),
}


@pytest.mark.parametrize("provider", CLOUD)
async def test_a_model_the_account_does_not_have_says_what_to_change(provider, monkeypatch):
    """The first real answer from a provider whose model id this account cannot use: a sentence naming the
    provider and where to change it, never "something went wrong", and the next turn does not ask again."""
    from pilot.conversation.layer import PROVIDER_NAMES

    ear = Ear()
    status, body = UNKNOWN_MODEL[provider]
    async with StubServer(status=status, steps=[body]) as server:
        model = model_for(provider, server, monkeypatch)
        layer = layer_for(model, ear)
        result = await layer.take(QUESTION)
        assert result.done
        assert ear.heard == [FALLBACK["unknown_model"].replace("{provider}", PROVIDER_NAMES[provider])]
        await layer.take("And what about rice?")
    assert len(server.seen) == 1  # held: the same model is not asked again a moment later

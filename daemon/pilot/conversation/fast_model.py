"""The small, fast model that talks: streamed wherever the provider streams, counted in the budget every time.

Every cloud provider is reached the same way, over ``httpx`` with server-sent events, as ``pilot.models`` does:

* Claude: the Messages API with ``stream: true``. Default ``claude-haiku-5-5`` with thinking off and effort
  ``low`` (Haiku 5.5 accepts ``{"type": "disabled"}`` at effort low, medium and high; no sampling parameters).
  On 12 November this path gave a first token in 0.77 s median against 0.88 s through the Anthropic SDK, whose
  import took up to 30 s, so the SDK is not used.
* OpenAI-compatible (OpenAI, OpenRouter, Meta): chat completions with ``stream: true``, usage from the final chunk
  (``stream_options.include_usage``).
* Gemini: ``streamGenerateContent`` with ``alt=sse``, usage from ``usageMetadata``.
* A local model through Ollama: ``/api/chat`` with ``stream: true`` (one JSON object per line). Nothing leaves the
  machine; the first token can take seconds while the model loads.
* Anything else (a subscription CLI, llama-cpp in process): the model router's ordinary call, one piece.

The fast model for each provider is a setting (``conversation.<provider>_model``). Its default is the provider's
fast tier where one is known for certain (Claude: ``claude-haiku-5-5``), and otherwise the model the person
already configured for that provider, so nothing is hard-wired to a vendor.

Every call goes to the same ``BudgetTracker`` as the planner's, with the same provider names and
``usage_source`` values. A finished stream records what the provider reported. A stream cut off by the person
records what it had reported so far, with output estimated from the text received; a stream cut off before any
usage arrived records the existing abandoned-call estimate (``pilot.models.cloud.estimate_abandoned_call``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pilot.models.catalog import REASONING_OFF, TALK_DEFAULTS

logger = logging.getLogger("pilot.conversation.fast_model")

DEFAULT_CLAUDE_FAST_MODEL = "claude-haiku-5-5"
# The model that talks, per provider, when the person has not chosen one (conversation.<provider>_model). Talk never
# inherits the planner's model (model.cloud_model): a planner set to a reasoning or large model would make every
# spoken reply slow, and its reasoning tokens count against the reply's token limit. The list, with the pages it was
# checked against, is pilot.models.catalog.TALK_DEFAULTS; the Settings panel shows it beside the planner's model.
FAST_DEFAULTS: dict[str, str] = dict(TALK_DEFAULTS)
OFFLINE_HOLD_SECONDS = 30.0  # after a connection failure, answer at once from local rules for this long
CREDIT_HOLD_SECONDS = 600.0  # an account out of credit does not refill in seconds
BUSY_HOLD_SECONDS = 20.0  # a rate limit without a Retry-After header
# A warm-up opens the connection while the person is still talking; it is sent again at most this often (the
# listener asks at the wake word and again at the end of the sentence).
WARM_EVERY_SECONDS = 3.0
# A request that finds a warm-up still opening the connection waits this long for it before opening another: the
# warm-up is the connection only (a HEAD to the API's address), so its rest is shorter than a second handshake.
WARM_JOIN_SECONDS = 0.3
# How long an idle connection to a provider is kept for the next turn. httpx's default is 5 s, which made every
# sentence longer than a few seconds, and every first question after a pause, start on a new connection (the
# provider keeps an idle one at least 120 s: measured 13 November). Only connections a request opened are kept;
# nothing is opened for this.
KEEPALIVE_SECONDS = 120.0
LOCAL_PROVIDERS = frozenset({"ollama", "local"})
LOCAL_CONNECT_TIMEOUT = 0.5
OPENAI_COMPATIBLE = frozenset({"openai", "openrouter", "meta"})


class FastModelUnavailable(RuntimeError):
    """The fast model cannot answer now. ``reason`` is one of: offline, no_key, unknown_model, credit, budget, busy,
    refused, error (and for a local model local_down, local_missing)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


@dataclass(slots=True)
class CallRecord:
    """What one call cost and how long it took; kept for tests, measurements and the log."""

    model: str
    started: float
    first_text_at: float | None = None
    ended_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    usage_source: str = ""
    chars: int = 0
    cut_off: bool = False


class FastModel(Protocol):
    name: str
    provider: str
    streams: bool
    local: bool  # runs on this machine: nothing it is given leaves it
    calls: list[CallRecord]

    def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class Budget(Protocol):
    def check_budget(self, provider: str) -> None: ...

    async def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        usage_source: str = "estimate",
    ) -> None: ...


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reported: bool = False


# What a provider's streamed answer is sent as: server-sent events, or JSON lines (Ollama). A web page is never an
# answer (a Wi-Fi sign-in page, a proxy's own page). Any other type is read: some local and proxy servers stream
# events as text/plain or a type of their own, and an answer is whatever parses.
API_CONTENT_TYPES = frozenset({"text/event-stream", "application/json", "application/x-ndjson"})
PAGE_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})

# The rest after late first words in a row: none after one (the next turn tries again; the spoken sign at 2 s
# covers its wait), then a few seconds, then longer, never more than half a minute. A real outage (no network, a key
# refused, no credit) keeps its own holds.
SLOW_PAUSES = (0.0, 5.0, 15.0, 30.0)


class _StreamingBase:
    """Budget gate, offline hold, timing and usage recording shared by every streamed provider."""

    name: str
    provider: str
    budget: Budget | None
    clock: Callable[[], float]
    calls: list[CallRecord]
    _offline_until: float

    def _gate(self) -> None:
        if self.clock() < self._offline_until:
            reason = getattr(self, "_hold_reason", "") or (
                "local_down" if self.provider in LOCAL_PROVIDERS else "offline"
            )
            raise FastModelUnavailable(reason, "held after a failure a moment ago")
        if self.budget is not None and self.provider not in LOCAL_PROVIDERS:
            try:
                self.budget.check_budget(self.provider)
            except Exception as error:  # BudgetExceededError
                raise FastModelUnavailable("budget", str(error)) from None

    def _went_offline(self) -> None:
        self.hold(OFFLINE_HOLD_SECONDS, "local_down" if self.provider in LOCAL_PROVIDERS else "offline")

    def hold(self, seconds: float, reason: str) -> None:
        """Answer from local rules for ``seconds`` without asking the provider (no network, no credit, too many
        requests, or a provider that stopped answering)."""
        self._offline_until = max(self._offline_until, self.clock() + max(0.0, seconds))
        self._hold_reason = reason

    def note_slow(self) -> float:
        """A first word did not come in time. One late answer costs that turn only: the next turn asks again. Late
        answers in a row rest the provider a little longer each time (SLOW_PAUSES, 30 s at most); an answer resets
        it. Returns the rest given, in seconds."""
        in_row = getattr(self, "_slow_in_row", 0) + 1
        self._slow_in_row = in_row
        pause = SLOW_PAUSES[min(in_row, len(SLOW_PAUSES)) - 1]
        if pause:
            self.hold(pause, "slow")
        return pause

    def note_answered(self) -> None:
        """A first word came: late answers before it no longer count."""
        self._slow_in_row = 0

    def _after_failure(self, reason: str, error: BaseException) -> None:
        if reason in ("offline", "local_down"):
            self._went_offline()
        elif reason in ("credit", "unknown_model"):
            self.hold(CREDIT_HOLD_SECONDS, reason)  # neither is fixed by asking again in a moment
        elif reason == "busy":
            self.hold(_retry_after(error), "busy")

    async def _record(self, record: CallRecord, usage: _Usage, payload: dict[str, Any], finished: bool) -> None:
        if usage.reported and usage.input_tokens:
            record.input_tokens = usage.input_tokens
            record.output_tokens = (
                usage.output_tokens if finished else max(usage.output_tokens, math.ceil(record.chars / 4))
            )
            record.usage_source = "reported"
        elif finished:
            # A billed response without a usage object: the router's length-based estimate.
            body = json.dumps(payload, ensure_ascii=False, default=str)
            record.input_tokens = math.ceil(len(body) / 4)
            record.output_tokens = math.ceil(record.chars / 4)
            record.usage_source = "estimate"
        else:
            from pilot.models.cloud import estimate_abandoned_call

            estimate = estimate_abandoned_call(
                self.provider, self.name, payload, (record.ended_at or self.clock()) - record.started
            )
            record.input_tokens, record.output_tokens = estimate.input_tokens, estimate.output_tokens
            record.usage_source = "abandoned_estimate"
        record.cut_off = record.cut_off or not finished
        if self.budget is None:
            return
        try:
            await self.budget.record_usage(
                self.provider,
                self.name,
                record.input_tokens,
                record.output_tokens,
                cache_read_tokens=usage.cache_read if record.usage_source == "reported" else 0,
                cache_creation_tokens=usage.cache_write if record.usage_source == "reported" else 0,
                usage_source=record.usage_source,
            )
        except Exception:
            logger.warning("Could not record the fast model's usage", exc_info=True)


def claude_payload(
    model: str, max_tokens: int, system: str, messages: list[dict[str, str]], effort: str, thinking_off: bool
) -> dict[str, Any]:
    """The Messages API body for a spoken reply, the same for the SDK and the HTTP path."""
    # The system prompt is the same text on every turn and comes first: one cache breakpoint on it lets the next turn
    # read it from the cache (about a tenth of the input price) instead of paying for it again. Same words as before:
    # only the shape of the field changes. The context that varies by turn (the time, the profile, the screen) is in
    # the last user message, after the breakpoint. A prompt under the model's cacheable minimum is sent as usual.
    cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "system": cached_system, "messages": messages}
    if not model.startswith(("claude-3", "claude-haiku-4", "claude-sonnet-4-5")):
        # Older models reject output_config.effort (see pilot.models.cloud.claude_supports_effort).
        payload["output_config"] = {"effort": effort}
    if thinking_off:
        payload["thinking"] = {"type": "disabled"}  # Haiku 5.5 accepts it at effort low, medium and high
    return payload


def _claude_usage(source: Any, usage: _Usage) -> None:
    if source is None or not getattr(source, "input_tokens", None):
        return
    usage.input_tokens = int(source.input_tokens or 0)
    usage.output_tokens = int(getattr(source, "output_tokens", 0) or 0)
    usage.cache_read = int(getattr(source, "cache_read_input_tokens", 0) or 0)
    usage.cache_write = int(getattr(source, "cache_creation_input_tokens", 0) or 0)
    usage.reported = True


CLAUDE_API_VERSION = "2023-06-01"  # the version pilot.models.cloud sends


def _claude_headers(key: str) -> dict[str, str]:
    return {"x-api-key": key, "anthropic-version": CLAUDE_API_VERSION, "content-type": "application/json"}


class _Namespace:
    def __init__(self, data: dict[str, Any]) -> None:
        self.__dict__.update(data)


def _classify(error: BaseException, errors: dict[str, tuple[type[BaseException], ...]] | None = None) -> str:
    for reason, kinds in (errors or {}).items():
        if isinstance(error, kinds):
            return reason
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    body = _error_text(error)
    # The provider account out of credit, which waiting does not fix: Anthropic says 400 "credit balance is too
    # low", OpenAI 429 "insufficient_quota", OpenRouter 402. Heliox's own monthly budget is "budget".
    if status == 402 or "credit balance" in body or "insufficient_quota" in body or "insufficient credits" in body:
        return "credit"
    if status in (401, 403):
        return "no_key"
    # The model named in Settings is not one this account can use: OpenAI 404 model_not_found, Anthropic 404
    # not_found_error, Gemini 404 NOT_FOUND, OpenRouter 400 "is not a valid model ID". Waiting does not fix it.
    if status == 404 or "model_not_found" in body or "not a valid model" in body:
        return "unknown_model"
    if status == 400 and ("api_key_invalid" in body or "api key not valid" in body):
        return "no_key"  # Gemini says a bad key with 400 INVALID_ARGUMENT, reason API_KEY_INVALID
    if status == 429 or (isinstance(status, int) and status >= 500):
        return "busy"
    try:
        import httpx

        if isinstance(error, httpx.TransportError | httpx.TimeoutException):
            return "offline"
    except ImportError:  # pragma: no cover - httpx is a core dependency
        pass
    if isinstance(error, TimeoutError | ConnectionError | OSError):
        return "offline"
    return "error"


def _error_text(error: BaseException) -> str:
    parts = [str(error)]
    response = getattr(error, "response", None)
    try:
        parts.append(str(getattr(response, "text", "") or ""))
    except Exception:
        pass
    parts.append(str(getattr(error, "body", "") or ""))
    return " ".join(parts).casefold()


def _retry_after(error: BaseException) -> float:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        value = float(headers.get("retry-after", BUSY_HOLD_SECONDS))
    except (TypeError, ValueError, AttributeError):
        value = BUSY_HOLD_SECONDS
    return min(max(value, 1.0), 300.0)


# ── Providers reached over plain HTTP, as pilot.models does ──


@dataclass
class HttpFastModel(_StreamingBase):
    """OpenAI-compatible, Gemini or Ollama, streamed over ``httpx``. ``client`` is an ``httpx.AsyncClient``."""

    provider: str
    name: str
    client_factory: Callable[[], Any]
    api_key: Callable[[], Any] | None = None  # async; None for a local server
    budget: Budget | None = None
    base_url: str = ""
    max_tokens: int = 400
    keep_alive: int | None = None
    request_timeout: float = 30.0
    effort: str = "low"  # Claude only
    thinking_off: bool = True  # Claude only
    clock: Callable[[], float] = time.monotonic
    streams: bool = True
    calls: list[CallRecord] = field(default_factory=list)
    _client: Any = None
    _offline_until: float = 0.0
    _hold_reason: str = ""
    _warmed_at: float = float("-inf")
    _warm_done_at: float = float("-inf")
    _warming: Any = None  # the warm-up under way (an asyncio task), joined by a request that starts during it

    @property
    def local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    def connection_warm(self) -> bool:
        """A connection to the provider was likely open for the next call: the client exists and a warm-up or a call
        used it within KEEPALIVE_SECONDS (for the turn trace; nothing depends on it)."""
        if self.local or self._client is None:
            return False
        last = self._warm_done_at
        if self.calls:
            last = max(last, self.calls[-1].ended_at or self.calls[-1].started)
        return self.clock() - last < KEEPALIVE_SECONDS

    def request(self, system: str, messages: list[dict[str, str]], key: str) -> tuple[str, dict, dict]:
        """URL, headers and JSON body for one streamed call."""
        if self.provider in LOCAL_PROVIDERS:
            body: dict[str, Any] = {
                "model": self.name,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": True,
                "options": {"num_predict": self.max_tokens},
                "think": False,  # a spoken reply needs no reasoning trace (Ollama 0.9 and later)
            }
            if self.keep_alive is not None:
                body["keep_alive"] = self.keep_alive
            return f"{self.base_url.rstrip('/')}/api/chat", {"Content-Type": "application/json"}, body
        if self.provider == "claude":
            from pilot.models.cloud import PROVIDER_ENDPOINTS

            body = claude_payload(self.name, self.max_tokens, system, messages, self.effort, self.thinking_off)
            body["stream"] = True
            return PROVIDER_ENDPOINTS["claude"], _claude_headers(key), body
        if self.provider == "gemini":
            from pilot.models.cloud import PROVIDER_ENDPOINTS

            contents = [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ]
            body = {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {"maxOutputTokens": self.max_tokens},
            }
            if "flash" in self.name:
                # Flash models think by default; a spoken reply does not need it, and it delays the first word.
                body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
            # The key goes in Google's header, as the warm-up sends it, never in the address: an address ends up in
            # exception text and request logs.
            url = f"{PROVIDER_ENDPOINTS['gemini']}/models/{self.name}:streamGenerateContent?alt=sse"
            return url, {"Content-Type": "application/json", "x-goog-api-key": key}, body
        from pilot.models.cloud import PROVIDER_ENDPOINTS

        body = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.max_tokens
        if self.name in REASONING_OFF:
            # A spoken reply needs no reasoning phase: it delays the first word and spends the token limit. Only for
            # models whose own page says effort "none" is accepted (pilot.models.catalog).
            if self.provider == "openrouter":
                body["reasoning"] = {"effort": "none"}
            else:
                body["reasoning_effort"] = "none"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://www.helioxos.dev/"
            headers["X-Title"] = "Heliox OS"
        return PROVIDER_ENDPOINTS.get(self.provider, PROVIDER_ENDPOINTS["openai"]), headers, body

    def parse(self, line: str, usage: _Usage) -> str:
        """The text in one streamed line, updating ``usage`` when the line reports it."""
        line = line.strip()
        if self.provider not in LOCAL_PROVIDERS:
            if not line.startswith("data:"):
                return ""
            line = line[5:].strip()
            if line == "[DONE]":
                return ""
        if not line:
            return ""
        try:
            data = json.loads(line)
        except ValueError:
            return ""
        if not isinstance(data, dict):
            return ""
        if self.provider in LOCAL_PROVIDERS:
            if data.get("done"):
                usage.input_tokens = int(data.get("prompt_eval_count") or 0)
                usage.output_tokens = int(data.get("eval_count") or 0)
                usage.reported = True
            if data.get("error"):
                raise FastModelUnavailable("error", str(data["error"])[:120])
            return str((data.get("message") or {}).get("content") or "")
        if self.provider == "claude":
            kind = data.get("type")
            if kind == "message_start":
                _claude_usage(_Namespace((data.get("message") or {}).get("usage") or {}), usage)
            elif kind == "message_delta":
                out = (data.get("usage") or {}).get("output_tokens")
                if isinstance(out, int):
                    usage.output_tokens = out
                if (data.get("delta") or {}).get("stop_reason") == "refusal":
                    raise FastModelUnavailable("refused", "the model declined")
            elif kind == "error":
                error_type = str((data.get("error") or {}).get("type") or "")
                raise FastModelUnavailable("busy" if "overloaded" in error_type else "error", error_type)
            elif kind == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    return str(delta.get("text") or "")
            return ""
        if self.provider == "gemini":
            meta = data.get("usageMetadata")
            if isinstance(meta, dict) and "promptTokenCount" in meta:
                usage.input_tokens = int(meta.get("promptTokenCount") or 0)
                usage.output_tokens = int(meta.get("candidatesTokenCount") or 0) + int(
                    meta.get("thoughtsTokenCount") or 0
                )
                usage.reported = True
            parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            return "".join(str(part.get("text") or "") for part in parts if not part.get("thought"))
        failed = data.get("error")
        if failed:
            # OpenRouter (and some OpenAI-compatible servers) report a failure after the 200 has been sent as a chunk
            # with an error object and finish_reason "error": the reply stopped, it did not finish.
            detail = failed.get("code") if isinstance(failed, dict) else failed
            raise FastModelUnavailable("error", str(detail or "error in the stream")[:120])
        found = data.get("usage")
        if isinstance(found, dict) and "prompt_tokens" in found:
            usage.input_tokens = int(found.get("prompt_tokens") or 0)
            usage.output_tokens = int(found.get("completion_tokens") or 0)
            usage.reported = True
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("delta") or {}).get("content") or "")

    def warm_request(self) -> tuple[str, str]:
        """(method, URL) of the warm-up: a HEAD to the provider API's own address. It opens and secures the
        connection the reply will use (the same host, so httpx keeps it for the next request) and asks the provider
        for no work: no key, no model, nothing the person said, and any status will do. Until 13 November it was a
        token count (Claude) or the model list, a full authenticated call that was still running when the request
        went out on 12 of 12 measured turns, so every request opened a second connection beside it."""
        from urllib.parse import urlsplit

        from pilot.models.cloud import PROVIDER_ENDPOINTS

        parts = urlsplit(PROVIDER_ENDPOINTS.get(self.provider, PROVIDER_ENDPOINTS["openai"]))
        return "HEAD", f"{parts.scheme}://{parts.netloc}/"

    async def warm(self) -> None:
        """Open the connection to a cloud provider before the turn ends, at most every WARM_EVERY_SECONDS.

        Never for a model on this computer (no connection to open, nothing to gain), nor with no key stored (no
        request will follow). In the background: it never holds the turn, and a warm-up that fails (no network, a
        slow provider) is silent and is not a provider fault. A request that starts while it runs joins it for at
        most WARM_JOIN_SECONDS (stream)."""
        now = self.clock()
        if self.local or now - self._warmed_at < WARM_EVERY_SECONDS or now < self._offline_until:
            return
        self._warmed_at = now
        try:
            if self.api_key is not None and not await self.api_key():
                return
        except Exception:
            logger.debug("No key could be read to warm the %s connection", self.provider, exc_info=True)
            return
        self._warming = asyncio.ensure_future(self._open_connection())
        await asyncio.shield(self._warming)

    async def _open_connection(self) -> None:
        from pilot.conversation import trace

        started = self.clock()
        opened = False
        try:
            if self._client is None:
                self._client = await asyncio.to_thread(self.client_factory)
            method, url = self.warm_request()
            await self._client.request(method, url, timeout=5.0)
            opened = True
            self._warm_done_at = self.clock()
        except Exception:
            logger.debug("Warming the %s connection failed", self.provider, exc_info=True)
        finally:
            trace.mark("warm-up done", ms=round((self.clock() - started) * 1000), opened=opened)

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self._gate()
        key = ""
        if self.api_key is not None:
            key = await self.api_key() or ""
            if not key:
                raise FastModelUnavailable("no_key", f"no {self.provider} API key is stored")
        if self._client is None:
            # Building the client loads certificates and a TLS context: about 0.1 s of blocking work, done off the
            # event loop so nothing else (hearing the person, stopping a reply) waits for it.
            self._client = await asyncio.to_thread(self.client_factory)
        warming = self._warming
        if warming is not None and not warming.done():
            # The warm-up is opening the connection this request would use: finishing it is quicker than opening a
            # second one beside it, but never worth more than WARM_JOIN_SECONDS.
            with contextlib.suppress(Exception):
                async with asyncio.timeout(WARM_JOIN_SECONDS):  # not wait_for: it can lose a cancellation before 3.12
                    await asyncio.shield(warming)
        url, headers, body = self.request(system, messages, key)
        record = CallRecord(model=self.name, started=self.clock())
        self.calls.append(record)
        usage = _Usage()
        finished = False
        kind = ""
        try:
            import httpx

            # A server on this machine accepts at once when it runs; Windows takes about 2 s to report a refused
            # local connection, so the local wait for a connection is short.
            timeout = httpx.Timeout(self.request_timeout, connect=LOCAL_CONNECT_TIMEOUT if self.local else 10.0)
            async with self._client.stream("POST", url, json=body, headers=headers, timeout=timeout) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    resp.raise_for_status()
                kind = str(getattr(resp, "headers", {}).get("content-type", "") or "").split(";")[0].strip().casefold()
                if kind in PAGE_CONTENT_TYPES:
                    # A Wi-Fi sign-in page (or any proxy's own page) answers 200 with HTML: not the provider. Said
                    # as such, like a network that is down, instead of being read as an empty reply.
                    self.hold(OFFLINE_HOLD_SECONDS, "not_api")
                    raise FastModelUnavailable("not_api", f"not from the API ({kind})")
                async for line in resp.aiter_lines():
                    text = self.parse(line, usage)
                    if not text:
                        continue
                    if record.first_text_at is None:
                        record.first_text_at = self.clock()
                    record.chars += len(text)
                    yield text
            finished = True
            if not record.chars:
                if kind and kind not in API_CONTENT_TYPES:
                    # A type of its own and nothing in it parsed: not the provider's answer.
                    self.hold(OFFLINE_HOLD_SECONDS, "not_api")
                    raise FastModelUnavailable("not_api", f"not from the API ({kind})")
                # A provider's stream (or one with no type) that carried no text: said, never silence.
                raise FastModelUnavailable("empty", "the reply held no text")
        except FastModelUnavailable:
            raise
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            reason = _classify(error, {})
            if self.local:
                status = getattr(getattr(error, "response", None), "status_code", None)
                # No local server running, or the named model not pulled: say how to fix it, not "offline".
                reason = "local_down" if reason == "offline" else "local_missing" if status == 404 else reason
            self._after_failure(reason, error)
            raise FastModelUnavailable(reason, type(error).__name__) from None
        finally:
            record.ended_at = self.clock()
            if not self.local:
                await self._record(record, usage, body, finished)
            elif usage.reported or record.chars:
                # A local model costs nothing, but its calls are counted like any other (at zero).
                await self._record(record, usage, body, True)


# ── Any other backend: the router's ordinary call ──


@dataclass
class RouterFastModel:
    """Through ``ModelRouter.generate``: not streamed, the whole reply as one piece. The router counts usage."""

    router: Any
    name: str = "router"
    provider: str = ""
    local: bool = False
    clock: Callable[[], float] = time.monotonic
    streams: bool = False
    calls: list[CallRecord] = field(default_factory=list)

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        record = CallRecord(model=self.name, started=self.clock())
        self.calls.append(record)
        try:
            text = await self.router.generate(messages, system=system, temperature=0.3)
        except Exception as error:
            message = str(error).lower()
            if "budget" in message:
                reason = "budget"
            elif "api key" in message:
                reason = "no_key"
            elif any(word in message for word in ("timed out", "connect", "network", "unavailable", "resolve")):
                reason = "offline"
            else:
                reason = "error"
            raise FastModelUnavailable(reason, type(error).__name__) from None
        record.first_text_at = record.ended_at = self.clock()
        record.chars = len(text or "")
        record.usage_source = "router"
        if text:
            yield text


# ── Choosing the model ──


_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def fast_model_name(config: Any, provider: str) -> str:
    """The model that talks for ``provider``: the person's own choice for talk, else the provider's fast default
    (FAST_DEFAULTS), never the planner's model. A provider with no known fast model (none today) falls back to the
    planner's model, so it still answers. A local model is the one model on this computer."""
    from pilot.models.cloud import DEFAULT_MODELS

    conv = getattr(config, "conversation", None)
    model_cfg = getattr(config, "model", None)
    chosen = str(getattr(conv, f"{provider}_model", "") or "").strip()
    if chosen:
        return chosen
    if provider in LOCAL_PROVIDERS:
        return str(getattr(model_cfg, "ollama_model", "") or "")
    if provider in FAST_DEFAULTS:
        return FAST_DEFAULTS[provider]
    return str(getattr(model_cfg, "cloud_model", "") or DEFAULT_MODELS.get(provider, ""))


def describe_setup(config: Any) -> str:
    """Which kind of person this is, for the log and the design note: claude, cloud, local or none."""
    model_cfg = getattr(config, "model", None)
    provider = getattr(model_cfg, "provider", "")
    if provider == "cloud":
        return "claude" if getattr(model_cfg, "cloud_provider", "") == "claude" else "cloud"
    if provider in LOCAL_PROVIDERS:
        return "local"
    if provider == "subscription":
        return "subscription"
    return "none"


def build_fast_model(config: Any, *, vault: Any, router: Any, budget: Any, http_client: Any = None) -> FastModel | None:
    """The fast model for the configured provider, or None when no model is set up at all."""
    model_cfg = getattr(config, "model", None)
    conv = getattr(config, "conversation", None)
    provider = getattr(model_cfg, "provider", "")
    cloud = getattr(model_cfg, "cloud_provider", "")
    max_tokens = int(getattr(conv, "max_reply_tokens", 400) or 400)

    def client_factory() -> Any:
        if http_client is not None:
            return http_client
        import httpx

        from pilot.system.http_client import create_httpx_client

        return create_httpx_client(config, timeout=30.0, limits=httpx.Limits(keepalive_expiry=KEEPALIVE_SECONDS))

    if provider == "cloud" and cloud and vault is not None:

        async def key() -> Any:
            return await vault.get_key(cloud)

        name = fast_model_name(config, cloud)
        if cloud in ("claude", "gemini") or cloud in OPENAI_COMPATIBLE:
            return HttpFastModel(
                provider=cloud,
                name=name,
                client_factory=client_factory,
                api_key=key,
                budget=budget,
                max_tokens=max_tokens,
            )
    if provider in LOCAL_PROVIDERS:
        name = fast_model_name(config, "ollama")
        if name:
            return HttpFastModel(
                provider="ollama",
                name=name,
                client_factory=client_factory,
                budget=budget,
                base_url=str(getattr(model_cfg, "ollama_base_url", "") or "http://127.0.0.1:11434"),
                max_tokens=max_tokens,
                keep_alive=int(getattr(model_cfg, "idle_unload_seconds", 60) or 60) * 5,
                request_timeout=90.0,  # loading a local model into memory can take most of a minute
            )
    if router is None or provider not in ("cloud", "subscription", *LOCAL_PROVIDERS):
        return None
    return RouterFastModel(
        router=router,
        name=f"{provider}:{cloud or 'default'}",
        provider=provider or "",
        local=provider in LOCAL_PROVIDERS,
    )

"""The small, fast model that talks: streamed wherever the provider streams, counted in the budget every time.

One class per kind of provider the product supports (``pilot.models``):

* Claude: the Anthropic SDK's async ``messages.stream``. Default ``claude-haiku-5-5`` with thinking off and effort
  ``low`` (Haiku 5.5 accepts ``{"type": "disabled"}`` at effort low, medium and high; API version 2026-10-21
  rejects every sampling parameter, so none is sent). The SDK is imported only when this model is used. Without
  it, the ordinary provider call is used and the log says once that fast replies need the package.
* OpenAI-compatible (OpenAI, OpenRouter, Meta): chat completions with ``stream: true`` over server-sent events,
  usage from the final chunk (``stream_options.include_usage``).
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
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("pilot.conversation.fast_model")

DEFAULT_CLAUDE_FAST_MODEL = "claude-haiku-5-5"
OFFLINE_HOLD_SECONDS = 30.0
WARM_EVERY_SECONDS = 45.0  # an idle HTTPS connection is usually kept about this long  # after a connection failure, answer at once from local rules for this long
LOCAL_PROVIDERS = frozenset({"ollama", "local"})
OPENAI_COMPATIBLE = frozenset({"openai", "openrouter", "meta"})


class FastModelUnavailable(RuntimeError):
    """The fast model cannot answer now. ``reason`` is one of: offline, no_key, budget, busy, refused, error."""

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
            raise FastModelUnavailable("offline", "no connection a moment ago")
        if self.budget is not None and self.provider not in LOCAL_PROVIDERS:
            try:
                self.budget.check_budget(self.provider)
            except Exception as error:  # BudgetExceededError
                raise FastModelUnavailable("budget", str(error)) from None

    def _went_offline(self) -> None:
        self._offline_until = self.clock() + OFFLINE_HOLD_SECONDS

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


# ── Claude, through the Anthropic SDK ──


@dataclass
class ClaudeFastModel(_StreamingBase):
    api_key: Callable[[], Any]  # async callable returning the key, or None
    budget: Budget | None
    name: str = DEFAULT_CLAUDE_FAST_MODEL
    max_tokens: int = 400
    effort: str = "low"
    thinking_off: bool = True
    request_timeout: float = 20.0
    client_factory: Callable[[str], Any] | None = None
    clock: Callable[[], float] = time.monotonic
    provider: str = "claude"
    streams: bool = True
    local: bool = False
    calls: list[CallRecord] = field(default_factory=list)
    _client: Any = None
    _client_key: str = ""
    _offline_until: float = 0.0
    _warmed_at: float = float("-inf")

    def payload(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if not self.name.startswith(("claude-3", "claude-haiku-4", "claude-sonnet-4-5")):
            # Older models reject output_config.effort (see pilot.models.cloud.claude_supports_effort).
            payload["output_config"] = {"effort": self.effort}
        if self.thinking_off:
            payload["thinking"] = {"type": "disabled"}
        return payload

    async def _get_client(self, key: str) -> Any:
        if self._client is None or key != self._client_key:
            factory = self.client_factory or _default_claude_client
            # Built off the event loop: the first import of the SDK reads hundreds of files.
            self._client = await asyncio.to_thread(factory, key)  # kept: its connection stays open between turns
            self._client_key = key
        return self._client

    async def warm(self) -> None:
        """Build the client (importing the SDK) and open its connection, so the first reply skips both. Uses the
        token-counting endpoint, which is free; at most once every WARM_EVERY_SECONDS."""
        now = self.clock()
        if now - self._warmed_at < WARM_EVERY_SECONDS or now < self._offline_until:
            return
        self._warmed_at = now
        try:
            key = await self.api_key()
            if not key:
                return
            client = await self._get_client(key)
            await client.messages.count_tokens(
                model=self.name, messages=[{"role": "user", "content": "hi"}], timeout=5.0
            )
        except Exception:
            logger.debug("Warming the Claude connection failed", exc_info=True)

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self._gate()
        key = await self.api_key()
        if not key:
            raise FastModelUnavailable("no_key", "no Claude API key is stored")
        client = await self._get_client(key)
        payload = self.payload(system, messages)
        record = CallRecord(model=self.name, started=self.clock())
        self.calls.append(record)
        usage = _Usage()
        manager = None
        stream = None
        finished = False
        try:
            manager = client.messages.stream(**payload, timeout=self.request_timeout)
            stream = await manager.__aenter__()
            async for text in stream.text_stream:
                if not text:
                    continue
                if record.first_text_at is None:
                    record.first_text_at = self.clock()
                record.chars += len(text)
                yield text
            final = await stream.get_final_message()
            finished = True
            _claude_usage(getattr(final, "usage", None), usage)
            if getattr(final, "stop_reason", "") == "refusal":
                raise FastModelUnavailable("refused", "the model declined")
        except FastModelUnavailable:
            raise
        except BaseException as error:
            if not isinstance(error, Exception):  # cancelled or closed: the person talked over Heliox
                raise
            reason = _classify(error, _sdk_errors())
            if reason == "offline":
                self._went_offline()
            raise FastModelUnavailable(reason, type(error).__name__) from None
        finally:
            record.ended_at = self.clock()
            if not finished and stream is not None:
                try:
                    _claude_usage(getattr(stream.current_message_snapshot, "usage", None), usage)
                except Exception:
                    pass  # no message_start yet: nothing was reported
            await self._record(record, usage, payload, finished)
            if manager is not None:
                try:
                    await manager.__aexit__(None, None, None)
                except BaseException:  # closing a cancelled stream must never raise into the turn
                    logger.debug("Closing the Claude stream failed", exc_info=True)


def _claude_usage(source: Any, usage: _Usage) -> None:
    if source is None or not getattr(source, "input_tokens", None):
        return
    usage.input_tokens = int(source.input_tokens or 0)
    usage.output_tokens = int(getattr(source, "output_tokens", 0) or 0)
    usage.cache_read = int(getattr(source, "cache_read_input_tokens", 0) or 0)
    usage.cache_write = int(getattr(source, "cache_creation_input_tokens", 0) or 0)
    usage.reported = True


def _default_claude_client(key: str) -> Any:
    import anthropic

    # No SDK retries: a turn that fails says so in a sentence instead of waiting through backoff.
    return anthropic.AsyncAnthropic(api_key=key, max_retries=0)


def _sdk_errors() -> dict[str, tuple[type[BaseException], ...]]:
    try:
        import anthropic
    except ImportError:
        return {}
    return {
        "offline": (anthropic.APIConnectionError,),  # includes APITimeoutError
        "no_key": (anthropic.AuthenticationError, anthropic.PermissionDeniedError),
        "busy": (anthropic.RateLimitError, anthropic.InternalServerError),
    }


def _classify(error: BaseException, errors: dict[str, tuple[type[BaseException], ...]]) -> str:
    for reason, kinds in errors.items():
        if isinstance(error, kinds):
            return reason
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status in (401, 403):
        return "no_key"
    if status == 402:
        return "budget"
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
    clock: Callable[[], float] = time.monotonic
    streams: bool = True
    calls: list[CallRecord] = field(default_factory=list)
    _client: Any = None
    _offline_until: float = 0.0

    @property
    def local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    def request(self, system: str, messages: list[dict[str, str]], key: str) -> tuple[str, dict, dict]:
        """URL, headers and JSON body for one streamed call."""
        if self.provider in LOCAL_PROVIDERS:
            body: dict[str, Any] = {
                "model": self.name,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": True,
                "options": {"num_predict": self.max_tokens},
            }
            if self.keep_alive is not None:
                body["keep_alive"] = self.keep_alive
            return f"{self.base_url.rstrip('/')}/api/chat", {"Content-Type": "application/json"}, body
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
            url = f"{PROVIDER_ENDPOINTS['gemini']}/models/{self.name}:streamGenerateContent?alt=sse&key={key}"
            return url, {"Content-Type": "application/json"}, body
        from pilot.models.cloud import PROVIDER_ENDPOINTS

        body = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.max_tokens
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
        found = data.get("usage")
        if isinstance(found, dict) and "prompt_tokens" in found:
            usage.input_tokens = int(found.get("prompt_tokens") or 0)
            usage.output_tokens = int(found.get("completion_tokens") or 0)
            usage.reported = True
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("delta") or {}).get("content") or "")

    async def stream(self, system: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self._gate()
        key = ""
        if self.api_key is not None:
            key = await self.api_key() or ""
            if not key:
                raise FastModelUnavailable("no_key", f"no {self.provider} API key is stored")
        if self._client is None:
            self._client = self.client_factory()
        url, headers, body = self.request(system, messages, key)
        record = CallRecord(model=self.name, started=self.clock())
        self.calls.append(record)
        usage = _Usage()
        finished = False
        try:
            async with self._client.stream(
                "POST", url, json=body, headers=headers, timeout=self.request_timeout
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    resp.raise_for_status()
                async for line in resp.aiter_lines():
                    text = self.parse(line, usage)
                    if not text:
                        continue
                    if record.first_text_at is None:
                        record.first_text_at = self.clock()
                    record.chars += len(text)
                    yield text
            finished = True
        except FastModelUnavailable:
            raise
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            reason = _classify(error, {})
            if reason == "offline":
                self._went_offline()
            if self.local:
                status = getattr(getattr(error, "response", None), "status_code", None)
                # No local server running, or the named model not pulled: say how to fix it, not "offline".
                reason = "local_down" if reason == "offline" else "local_missing" if status == 404 else reason
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


def sdk_available() -> bool:
    """Whether the Anthropic SDK is installed, without importing it (a cold import can take many seconds)."""
    import importlib.util
    import sys

    if "anthropic" in sys.modules:
        return sys.modules["anthropic"] is not None
    try:
        return importlib.util.find_spec("anthropic") is not None
    except (ImportError, ValueError):
        return False


_sdk_import: asyncio.Task[Any] | None = None


def prewarm_sdk() -> None:
    """Import the SDK on a worker thread now, so the first spoken reply does not wait for it."""
    global _sdk_import
    if _sdk_import is not None or not sdk_available():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    import importlib

    _sdk_import = loop.create_task(asyncio.to_thread(importlib.import_module, "anthropic"))
    _sdk_import.add_done_callback(lambda task: task.cancelled() or task.exception())


_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def fast_model_name(config: Any, provider: str) -> str:
    """The fast model for ``provider``: its setting, else the known fast tier, else the configured model."""
    from pilot.models.cloud import DEFAULT_MODELS

    conv = getattr(config, "conversation", None)
    model_cfg = getattr(config, "model", None)
    chosen = str(getattr(conv, f"{provider}_model", "") or "").strip()
    if chosen:
        return chosen
    if provider == "claude":
        return DEFAULT_CLAUDE_FAST_MODEL
    if provider in LOCAL_PROVIDERS:
        return str(getattr(model_cfg, "ollama_model", "") or "")
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
        from pilot.system.http_client import create_httpx_client

        return create_httpx_client(config, timeout=30.0)

    if provider == "cloud" and cloud and vault is not None:

        async def key() -> Any:
            return await vault.get_key(cloud)

        name = fast_model_name(config, cloud)
        if cloud == "claude":
            if sdk_available():
                prewarm_sdk()
                return ClaudeFastModel(api_key=key, budget=budget, name=name, max_tokens=max_tokens)
            _warn_once(
                "sdk",
                "Fast spoken replies with Claude need the anthropic package; until it is installed, replies "
                "use the planner model through the ordinary call and arrive in one piece, seconds later",
            )
        elif cloud == "gemini" or cloud in OPENAI_COMPATIBLE:
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

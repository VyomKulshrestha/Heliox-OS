"""Cloud API client supporting OpenAI-compatible endpoints and native Gemini.

OpenAI, OpenRouter, and Meta's Muse Spark speak the OpenAI
chat-completions format, so they share `_call_openai_compat`; Gemini and
Claude each need their own native request/response shape.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from pilot.config import PilotConfig
from pilot.system.http_client import create_httpx_client

if TYPE_CHECKING:
    from pilot.security.vault import KeyVault

logger = logging.getLogger("pilot.models.cloud")

PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "claude": "https://api.anthropic.com/v1/messages",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    # Meta Model API (public preview as of July 2026) -- drop-in OpenAI
    # chat-completions compatible, so it needs no dispatch branch of its
    # own below; it falls through to _call_openai_compat like any other
    # OpenAI-format provider. Base URL/model id per Meta's own docs
    # (ai.developer.meta.com/docs/features/chat-completion); reverify
    # if Meta changes this during the public preview.
    "meta": "https://api.meta.ai/v1/chat/completions",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-opus-5-5",
    "openrouter": "openrouter/auto",
    "meta": "muse-spark-1.1",
}

PROVIDER_DISPLAY_NAMES = {
    "openrouter": "OpenRouter",
}

MAX_RETRIES = 3
# Non-streaming ceiling for Claude. Current models always think, and thinking
# tokens count against max_tokens, so 4096 truncated plans before any text.
CLAUDE_MAX_TOKENS = 16000
# Claude Opus 5.5 defaults to "medium" effort. Planning calls run under the
# 35 second generation budget below, so ask for less thinking.
CLAUDE_EFFORT = "low"
# Model ID prefixes where `output_config.effort` is rejected (claude-api skill
# thinking/effort table: Sonnet 4.5, Haiku 4.5 and older models). Anything
# else, including unknown or future claude-* IDs, is treated as current.
CLAUDE_NO_EFFORT_PREFIXES = (
    "claude-3",
    "claude-haiku-4",
    "claude-sonnet-4-0",
    "claude-sonnet-4-2",
    "claude-sonnet-4-5",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2",
)
RETRY_BACKOFF_BASE = 3.0  # seconds
# Planning requests carry Heliox's structured action contract and can take
# longer than short conversational turns even on a healthy provider.  The
# previous 18-second process-level deadline routinely cancelled Gemini before
# the HTTP client's own timeout and surfaced a false "provider unavailable"
# error.  This remains a hard bound, but gives complex plans enough time to
# finish; successful requests still return as soon as the provider responds.
CLOUD_GENERATION_BUDGET_SECONDS = 35.0
RATE_LIMIT_KEY_COOLDOWN_SECONDS = 60.0
INVALID_KEY_COOLDOWN_SECONDS = 3600.0
TRANSIENT_KEY_COOLDOWN_SECONDS = 15.0
_SECRET_VALUE = re.compile(r"(?i)(?P<prefix>(?:[?&](?:key|api[_-]?key|token)|authorization)\s*[=:]\s*)[^&\s]+")
_GOOGLE_API_KEY = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")


def safe_provider_error(error: Exception, provider: str) -> str:
    """Return an actionable provider failure that never includes secrets or URLs."""
    provider_name = PROVIDER_DISPLAY_NAMES.get(provider, provider.title()) or "Cloud"
    if isinstance(error, TimeoutError | httpx.TimeoutException):
        return f"{provider_name} API unavailable: the request timed out."
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        body = error.response.text.lower()
        if status == 429:
            detail = "quota or rate limit reached"
        elif status in {401, 403}:
            detail = "authentication was rejected"
        elif status == 402:
            detail = "the account or API key has insufficient credits"
        elif status == 400 and ("api_key_invalid" in body or "api key not valid" in body):
            detail = "the configured API key is invalid"
        elif status == 400:
            detail = "the request was rejected"
        elif status >= 500:
            detail = "the provider is temporarily unavailable"
        else:
            detail = "the provider rejected the request"
        return f"{provider_name} API unavailable ({status}): {detail}."

    message = str(error).strip() or error.__class__.__name__
    message = _SECRET_VALUE.sub(lambda match: f"{match.group('prefix')}[redacted]", message)
    message = _GOOGLE_API_KEY.sub("[redacted]", message)
    message = re.sub(r"https?://\S+", "[provider endpoint]", message)
    if "api unavailable" in message.lower():
        return message[:240]
    return f"{provider_name} API unavailable: {message[:200]}"


def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429
    err_str = str(error).lower()
    return any(
        keyword in err_str
        for keyword in (
            "429",
            "quota",
            "rate",
            "exceeded",
            "resource has been exhausted",
            "too many requests",
            "limit",
        )
    )


def _is_api_key_error(error: Exception) -> bool:
    """Return whether another configured key can repair this provider failure."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        body = error.response.text.lower()
        return status in {401, 403} or (status == 400 and ("api_key_invalid" in body or "api key not valid" in body))
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "api_key_invalid",
            "api key not valid",
            "invalid api key",
            "authentication was rejected",
        )
    )


def _is_transient_transport_error(error: Exception) -> bool:
    return isinstance(error, TimeoutError | httpx.TimeoutException | httpx.TransportError)


def claude_supports_effort(model: str) -> bool:
    return not model.lower().startswith(CLAUDE_NO_EFFORT_PREFIXES)


def _parse_anthropic_response(data: dict[str, Any]) -> str:
    """Return the text of a Messages API response or raise a clear error.

    Responses can start with a thinking block, so text blocks are selected
    by type. A refusal arrives as HTTP 200 and must be checked first.
    """
    stop_reason = data.get("stop_reason")
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    if stop_reason == "refusal":
        details = data.get("stop_details") or {}
        category = details.get("category")
        suffix = f" (category: {category})" if category else ""
        raise RuntimeError(f"Claude declined this request{suffix}")
    if stop_reason == "max_tokens":
        if not text:
            raise RuntimeError("Claude stopped at max_tokens before producing any text")
        logger.warning("Claude response truncated at max_tokens")
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={stop_reason})")
    return text


class CloudClient:
    """Unified cloud LLM client. API keys are fetched from the vault at call time."""

    def __init__(self, config: PilotConfig, vault: KeyVault) -> None:
        self._config = config
        self._vault = vault
        self._client = create_httpx_client(config, timeout=120.0)
        self._key_cooldowns: dict[bytes, float] = {}

    @staticmethod
    def _key_identity(api_key: str) -> bytes:
        """Create a non-reversible in-memory identity without retaining another key copy."""
        return hashlib.sha256(api_key.encode("utf-8")).digest()

    def _prefer_healthy_keys(self, api_keys: list[str]) -> list[str]:
        """Skip keys that already failed during this daemon session when possible."""
        now = asyncio.get_running_loop().time()
        healthy = [key for key in api_keys if self._key_cooldowns.get(self._key_identity(key), 0.0) <= now]
        if healthy:
            return healthy
        # If every key is cooling down, retry the one eligible soonest instead
        # of declaring the provider permanently unavailable.
        return [
            min(
                api_keys,
                key=lambda key: self._key_cooldowns.get(self._key_identity(key), 0.0),
            )
        ]

    def _cool_down_key(self, api_key: str, seconds: float) -> None:
        self._key_cooldowns[self._key_identity(api_key)] = asyncio.get_running_loop().time() + seconds

    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: callable | None = None,
    ) -> str:
        """Generate a completion. If stream_callback is provided, tokens are streamed via callback."""
        provider = self._config.model.cloud_provider
        model = self._config.model.cloud_model or DEFAULT_MODELS.get(provider, "")

        # Build list of API keys to try: primary + backups
        api_keys = []
        primary_key = await self._vault.get_key(provider)
        if primary_key:
            api_keys.append(primary_key)
        # Add backup keys
        for suffix in ("_backup_1", "_backup_2", "_backup_3", "_backup_4", "_backup_5"):
            backup = await self._vault.get_key(provider + suffix)
            if backup:
                api_keys.append(backup)

        if not api_keys:
            raise RuntimeError(f"No API key configured for {provider}")

        api_keys = self._prefer_healthy_keys(api_keys)
        has_backups = len(api_keys) > 1
        last_error = None
        deadline = asyncio.get_running_loop().time() + CLOUD_GENERATION_BUDGET_SECONDS
        for key_idx, api_key in enumerate(api_keys):
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    if provider == "gemini":
                        # Do not spend a backoff cycle on a key when another
                        # configured key can be tried immediately.
                        max_retries = 0 if (has_backups and key_idx < len(api_keys) - 1) else MAX_RETRIES
                        result = await self._call_gemini_native(
                            api_key,
                            model,
                            prompt,
                            system,
                            json_mode,
                            temperature,
                            max_retries=max_retries,
                        )
                    elif provider == "claude":
                        result = await self._call_anthropic(api_key, model, prompt, system, temperature)
                    else:
                        result = await self._call_openai_compat(
                            provider, api_key, model, prompt, system, json_mode, temperature
                        )
                if stream_callback:
                    await stream_callback(result)
                self._key_cooldowns.pop(self._key_identity(api_key), None)
                return result
            except Exception as e:
                last_error = e
                is_rate_limit = _is_rate_limit_error(e)
                is_api_key_error = _is_api_key_error(e)
                is_transport_error = _is_transient_transport_error(e)
                if is_api_key_error:
                    self._cool_down_key(api_key, INVALID_KEY_COOLDOWN_SECONDS)
                elif is_rate_limit:
                    self._cool_down_key(api_key, RATE_LIMIT_KEY_COOLDOWN_SECONDS)
                elif is_transport_error:
                    self._cool_down_key(api_key, TRANSIENT_KEY_COOLDOWN_SECONDS)
                can_try_another_key = is_rate_limit or is_api_key_error or is_transport_error
                if can_try_another_key and key_idx < len(api_keys) - 1:
                    if asyncio.get_running_loop().time() >= deadline:
                        break
                    logger.warning(
                        "API key %d/%d failed (%s), rotating to next key",
                        key_idx + 1,
                        len(api_keys),
                        safe_provider_error(e, provider),
                    )
                    continue
                raise RuntimeError(safe_provider_error(e, provider)) from None

        safe_error = safe_provider_error(
            last_error or RuntimeError("all configured keys were exhausted"),
            provider,
        )
        raise RuntimeError(safe_error) from None

    async def _call_gemini_native(
        self,
        api_key: str,
        model: str,
        prompt: str | list[dict[str, Any]],
        system: str,
        json_mode: bool,
        temperature: float,
        *,
        max_retries: int | None = None,
    ) -> str:
        """Call Gemini using the native REST API (most reliable)."""
        if max_retries is None:
            max_retries = MAX_RETRIES
        base_url = PROVIDER_ENDPOINTS["gemini"]
        endpoint = f"{base_url}/models/{model}:generateContent?key={api_key}"

        # Build contents
        contents = []
        system_content = system
        if isinstance(prompt, list):
            for msg in prompt:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    role = "model" if msg["role"] == "assistant" else "user"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        else:
            contents.append({"parts": [{"text": prompt}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        # Add system instruction
        if system_content:
            payload["systemInstruction"] = {"parts": [{"text": system_content}]}

        # Add JSON mode
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        for attempt in range(max_retries + 1):
            resp = await self._client.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code == 429:
                if attempt < max_retries:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Rate limited (429) by Gemini, retrying in %.1fs (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                else:
                    logger.error("Rate limited (429) by Gemini after %d retries", max_retries)

            if resp.status_code != 200:
                logger.error("Gemini API request failed with status %d", resp.status_code)
                resp.raise_for_status()

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {data}")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError(f"Gemini returned no parts: {data}")

            return parts[0].get("text", "")

        raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")

    async def _call_openai_compat(
        self,
        provider: str,
        api_key: str,
        model: str,
        prompt: str | list[dict[str, Any]],
        system: str,
        json_mode: bool,
        temperature: float,
    ) -> str:
        endpoint = PROVIDER_ENDPOINTS.get(provider, PROVIDER_ENDPOINTS["openai"])
        messages = []
        if isinstance(prompt, list):
            messages = prompt
        else:
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider == "openrouter":
            # Optional OpenRouter attribution headers. They do not grant any
            # additional authority and contain no local/user information.
            headers["HTTP-Referer"] = "https://www.helioxos.dev/"
            headers["X-Title"] = "Heliox OS"

        for attempt in range(MAX_RETRIES + 1):
            resp = await self._client.post(endpoint, json=payload, headers=headers)

            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Rate limited (429) by %s, retrying in %.1fs (attempt %d/%d)",
                        provider,
                        wait,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                else:
                    logger.error("Rate limited (429) by %s after %d retries", provider, MAX_RETRIES)

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to rate limiting")

    async def _call_anthropic(
        self,
        api_key: str,
        model: str,
        prompt: str | list[dict[str, Any]],
        system: str,
        temperature: float,
    ) -> str:
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        messages = []
        system_content = system
        if isinstance(prompt, list):
            for msg in prompt:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            messages = [{"role": "user", "content": prompt}]

        # No temperature/top_p/top_k: current Claude models reject any
        # non-default sampling parameter with HTTP 400, so the caller's
        # temperature is intentionally ignored for this provider.
        payload: dict = {
            "model": model,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": messages,
        }
        if claude_supports_effort(model):
            payload["output_config"] = {"effort": CLAUDE_EFFORT}
        if system_content:
            payload["system"] = system_content

        for attempt in range(MAX_RETRIES + 1):
            resp = await self._client.post(PROVIDER_ENDPOINTS["claude"], json=payload, headers=headers)

            if resp.status_code == 429 and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "Rate limited (429) by Claude, retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            return _parse_anthropic_response(resp.json())

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to rate limiting")

    async def close(self) -> None:
        await self._client.aclose()

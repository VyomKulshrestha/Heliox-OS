"""Cloud API client supporting OpenAI-compatible endpoints and native Gemini."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.security.vault import KeyVault

logger = logging.getLogger("pilot.models.cloud")

PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "claude": "https://api.anthropic.com/v1/messages",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-20250514",
}

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 3.0  # seconds


class CloudClient:
    """Unified cloud LLM client. API keys are fetched from the vault at call time."""

    def __init__(self, config: PilotConfig, vault: KeyVault) -> None:
        self._config = config
        self._vault = vault
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
    ) -> str:
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

        has_backups = len(api_keys) > 1
        last_error = None
        for key_idx, api_key in enumerate(api_keys):
            try:
                if provider == "gemini":
                    # When we have backup keys, reduce retries per key to rotate faster
                    max_retries = 1 if (has_backups and key_idx < len(api_keys) - 1) else MAX_RETRIES
                    return await self._call_gemini_native(
                        api_key,
                        model,
                        prompt,
                        system,
                        json_mode,
                        temperature,
                        max_retries=max_retries,
                    )
                elif provider == "claude":
                    return await self._call_anthropic(api_key, model, prompt, system, temperature)
                else:
                    return await self._call_openai_compat(
                        provider, api_key, model, prompt, system, json_mode, temperature
                    )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Rotate keys on rate limits, quota errors, or resource exhaustion
                is_rate_limit = any(
                    kw in err_str
                    for kw in [
                        "429",
                        "quota",
                        "rate",
                        "exceeded",
                        "resource has been exhausted",
                        "too many requests",
                        "limit",
                    ]
                )
                if is_rate_limit and key_idx < len(api_keys) - 1:
                    logger.warning(
                        "API key %d/%d failed (%s), rotating to next key",
                        key_idx + 1,
                        len(api_keys),
                        str(e)[:80],
                    )
                    continue
                raise  # Non-rate-limit errors or last key — propagate

        raise last_error or RuntimeError(f"All {len(api_keys)} API keys exhausted for {provider}")

    async def _call_gemini_native(
        self,
        api_key: str,
        model: str,
        prompt: str,
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
        if system:
            # Gemini uses systemInstruction for system prompts
            pass  # handled below
        contents.append({"parts": [{"text": prompt}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        # Add system instruction
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

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
                error_body = resp.text[:300]
                logger.error("Gemini API error %d: %s", resp.status_code, error_body)
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
        prompt: str,
        system: str,
        json_mode: bool,
        temperature: float,
    ) -> str:
        endpoint = PROVIDER_ENDPOINTS.get(provider, PROVIDER_ENDPOINTS["openai"])
        messages = []
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
        prompt: str,
        system: str,
        temperature: float,
    ) -> str:
        headers = {
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload: dict = {
            "model": model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

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
            data = resp.json()
            return data["content"][0]["text"]

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to rate limiting")

    async def close(self) -> None:
        await self._client.aclose()

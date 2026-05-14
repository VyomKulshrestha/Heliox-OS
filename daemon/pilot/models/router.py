"""Model router: selects and dispatches to the appropriate LLM backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.config import DATA_DIR
from pilot.models.cache import LLMCache
from pilot.models.cloud import CloudClient
from pilot.models.ollama import OllamaClient
from pilot.models.rate_limiter import TokenBucketRateLimiter

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.security.vault import KeyVault

logger = logging.getLogger("pilot.models.router")


class ModelRouter:
    """Routes inference requests to the appropriate model backend.

    Selection order:
    1. If cloud provider is configured and keys are available, use cloud
    2. Try Ollama (primary local backend)
    3. Fall back to llama-cpp-python if available
    """

    def __init__(self, config: PilotConfig, vault: KeyVault) -> None:
        self._config = config
        self._vault = vault
        self._ollama = OllamaClient(config.model.ollama_base_url)
        self._cloud: CloudClient | None = None
        self._llamacpp: object | None = None
        self._resolved_ollama_model: str | None = None
        self._cache = LLMCache(DATA_DIR / "llm_cache.db")

        if config.model.cloud_provider:
            self._cloud = CloudClient(config, vault)

        self._rate_limiter = TokenBucketRateLimiter(config.model)

    async def initialize(self) -> None:
        """Initialize the cache. Must be called before using generate()."""
        await self._cache.initialize()

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
    ) -> str:
        """Generate a completion from the best available model.
        
        Flow:
        1. Determine model and provider
        2. Check cache for exact match
        3. On cache miss, acquire rate limit token
        4. Call backend (cloud or local)
        5. Store successful response in cache
        """
        # Determine which backend and model we'll use, and check cache
        provider = self._config.model.provider
        model: str | None = None
        response: str | None = None

        # Try cloud backend first if configured
        if provider == "cloud" and self._cloud:
            model = self._config.model.cloud_model or "unknown"
            response = await self._cache.get(
                prompt, model, provider, temperature, json_mode, system
            )
            if response is not None:
                return response

            try:
                await self._rate_limiter.acquire()
                response = await self._cloud.generate(
                    prompt, system=system, json_mode=json_mode, temperature=temperature
                )
                await self._cache.set(
                    prompt, model, provider, temperature, json_mode, response, system
                )
                return response
            except Exception as e:
                logger.error("Cloud API failed: %s", e)
                raise RuntimeError(f"Cloud API Failed: {e}")

        # Try ollama or local backend
        if provider in ("ollama", "local"):
            if await self._ollama.is_available():
                model = await self._resolve_ollama_model()
                response = await self._cache.get(
                    prompt, model, provider, temperature, json_mode, system
                )
                if response is not None:
                    return response

                await self._rate_limiter.acquire()
                response = await self._ollama.generate(
                    model,
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    temperature=temperature,
                )
                await self._cache.set(
                    prompt, model, provider, temperature, json_mode, response, system
                )
                return response

            if self._try_llamacpp():
                model = "llamacpp"  # Use generic name for llamacpp
                response = await self._cache.get(
                    prompt, model, provider, temperature, json_mode, system
                )
                if response is not None:
                    return response

                await self._rate_limiter.acquire()
                response = await self._llamacpp_generate(
                    prompt, system=system, temperature=temperature
                )
                await self._cache.set(
                    prompt, model, provider, temperature, json_mode, response, system
                )
                return response

        # Fallback: try ollama if not already tried
        if provider != "ollama" and await self._ollama.is_available():
            model = await self._resolve_ollama_model()
            response = await self._cache.get(
                prompt, model, "ollama", temperature, json_mode, system
            )
            if response is not None:
                return response

            await self._rate_limiter.acquire()
            response = await self._ollama.generate(
                model,
                prompt,
                system=system,
                json_mode=json_mode,
                temperature=temperature,
            )
            await self._cache.set(
                prompt, model, "ollama", temperature, json_mode, response, system
            )
            return response

        # Final fallback: cloud API
        if self._cloud:
            model = self._config.model.cloud_model or "unknown"
            response = await self._cache.get(
                prompt, model, "cloud", temperature, json_mode, system
            )
            if response is not None:
                return response

            logger.warning("Falling back to cloud API — Ollama unavailable")
            await self._rate_limiter.acquire()
            response = await self._cloud.generate(
                prompt, system=system, json_mode=json_mode, temperature=temperature
            )
            await self._cache.set(
                prompt, model, "cloud", temperature, json_mode, response, system
            )
            return response

        raise RuntimeError("No model backend available. Start Ollama or configure a cloud API key.")

    async def _resolve_ollama_model(self) -> str:
        """Return a valid Ollama model name, falling back to an installed model if needed."""
        if self._resolved_ollama_model:
            return self._resolved_ollama_model

        configured = self._config.model.ollama_model
        available = await self._ollama.list_models()

        if not available:
            raise RuntimeError(
                "Ollama is running but has no models installed. Run 'ollama pull <model>' to install one."
            )

        for m in available:
            if m == configured or m.startswith(configured.split(":")[0]):
                self._resolved_ollama_model = m
                return m

        fallback = available[0]
        logger.warning(
            "Configured model '%s' not found in Ollama. Using '%s' instead. Available: %s",
            configured,
            fallback,
            ", ".join(available),
        )
        self._resolved_ollama_model = fallback
        self._config.model.ollama_model = fallback
        return fallback

    def _try_llamacpp(self) -> bool:
        if self._llamacpp is not None:
            return True
        try:
            from pilot.models.llamacpp import LlamaCppClient

            self._llamacpp = LlamaCppClient(self._config)
            return True
        except ImportError:
            return False

    async def _llamacpp_generate(self, prompt: str, *, system: str = "", temperature: float = 0.1) -> str:
        from pilot.models.llamacpp import LlamaCppClient

        client: LlamaCppClient = self._llamacpp  # type: ignore[assignment]
        return await client.generate(prompt, system=system, temperature=temperature)

    def rate_limit_stats(self) -> dict[str, Any]:
        """Return current rate limiter state and lifetime counters."""
        return self._rate_limiter.get_stats()

    async def check_health(self) -> dict[str, bool]:
        """Check which backends are available."""
        status: dict[str, bool] = {}
        status["ollama"] = await self._ollama.is_available()
        status["llamacpp"] = self._try_llamacpp()
        status["cloud"] = self._cloud is not None
        return status

    async def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return await self._cache.stats()

    async def cache_clear(self, provider: str | None = None, model: str | None = None) -> int:
        """Clear cache entries.

        Args:
            provider: If specified, only clear entries for this provider.
            model: If specified, only clear entries for this model.

        Returns:
            Number of entries deleted.
        """
        return await self._cache.clear(provider, model)

    async def close(self) -> None:
        """Close the cache connection."""
        await self._cache.close()

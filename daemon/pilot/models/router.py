"""Model router: selects and dispatches to the appropriate LLM backend."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from pilot.config import DATA_DIR
from pilot.db.redis_adapter import RedisCacheAdapter
from pilot.models.cache import LLMCache
from pilot.models.cloud import CloudClient, safe_provider_error
from pilot.models.ollama import OllamaClient
from pilot.models.rate_limiter import TokenBucketRateLimiter
from pilot.models.subscription_cli import SubscriptionCLIClient

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.models.budget_tracker import BudgetTracker
    from pilot.security.vault import KeyVault

from pilot.models.budget_tracker import ActionBudgetExceededError

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
        self._ollama = OllamaClient(config.model.ollama_base_url, config)
        self._cloud: CloudClient | None = None
        self._subscription = SubscriptionCLIClient(config)
        self._llamacpp: object | None = None
        self._llamacpp_active_calls = 0
        self._llamacpp_idle_task: asyncio.Task[None] | None = None
        self._llamacpp_reload_pending = False
        self._resolved_ollama_model: str | None = None
        self._cloud_unavailable_until = 0.0
        self._cloud_unavailable_reason = ""
        redis_adapter = RedisCacheAdapter.from_config(config.redis)
        self._cache = LLMCache(DATA_DIR / "llm_cache.db", redis=redis_adapter)
        self._budget_tracker: BudgetTracker | None = None

        if config.model.cloud_provider:
            self._cloud = CloudClient(config, vault)

        self._rate_limiter = TokenBucketRateLimiter(config.model)

        # Prevent local LLM overload from concurrent inference calls
        self._local_llm_semaphore = asyncio.Semaphore(2)
        self._local_llm_timeout = 300  # 5 min — local LLMs can be slow on large prompts

    async def initialize(self) -> None:
        """Initialize the cache. Must be called before using generate()."""
        await self._cache._redis.initialize()
        await self._cache.initialize()

    def set_budget_tracker(self, tracker: BudgetTracker) -> None:
        self._budget_tracker = tracker

    @staticmethod
    def _estimate_input_tokens(prompt: str | list[dict[str, Any]]) -> int:
        """Estimate input tokens with the same tokenizer as context compaction."""
        from pilot.memory.sliding_window import count_message_tokens, get_token_count

        if isinstance(prompt, list):
            return count_message_tokens(prompt)
        return get_token_count(prompt)

    def get_config(self) -> PilotConfig:
        return self._config

    def get_vault(self) -> KeyVault:
        return self._vault

    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: callable | None = None,
    ) -> str:
        """Generate a completion from the best available model.

        If stream_callback is provided, tokens will be streamed via the callback
        instead of waiting for the full response.

        Flow:
        1. Determine model and provider
        2. Check cache for exact match (skip cache if streaming)
        3. On cache miss, acquire rate limit token
        4. Call backend (cloud or local)
        5. Store successful response in cache (skip if streaming)
        """
        # Estimate input tokens once — used for both the per-action gate
        # and the post-call usage recording below.
        in_tokens_estimate = self._estimate_input_tokens(prompt)

        if self._budget_tracker:
            # Per-action gate: refuse calls whose estimated input alone
            # exceeds the cap, before they go out.
            max_per_action = self._config.model.max_tokens_per_action
            if max_per_action > 0 and in_tokens_estimate > max_per_action:
                raise ActionBudgetExceededError(
                    f"LLM call estimated at {in_tokens_estimate} input tokens, "
                    f"exceeding max_tokens_per_action={max_per_action}. "
                    f"Reduce prompt size or raise the cap."
                )
            # Monthly cumulative cap (existing behavior)
            self._budget_tracker.check_budget(self._config.model.provider)
            # Per-task cumulative cap (new)
            self._budget_tracker.check_task_budget()

        result = await self._generate_with_cache(
            prompt,
            system=system,
            json_mode=json_mode,
            temperature=temperature,
            stream_callback=stream_callback,
        )
        subscription_usage = (
            self._subscription.consume_request_usage() if self._config.model.provider == "subscription" else None
        )

        if self._budget_tracker:
            out_tokens = len(result) // 4
            usage_input_tokens = in_tokens_estimate
            if self._config.model.provider == "subscription":
                # Subscription CLIs report their real harness-inclusive usage.
                # Exact Heliox cache hits consume no provider quota at all.
                if subscription_usage is None:
                    return result
                usage_input_tokens = subscription_usage.get("input_tokens", in_tokens_estimate)
                out_tokens = subscription_usage.get("output_tokens", out_tokens)
            provider_key = (
                self._config.model.cloud_provider
                if self._config.model.provider == "cloud"
                else (
                    f"subscription:{self._config.model.subscription_provider}"
                    if self._config.model.provider == "subscription"
                    else self._config.model.provider
                )
            )
            if self._config.model.provider == "subscription":
                model_name = self._config.model.subscription_model or "cli-default"
            else:
                model_name = self._config.model.cloud_model or self._config.model.ollama_model

            await self._budget_tracker.record_usage(
                provider_key,
                model_name,
                usage_input_tokens,
                out_tokens,
            )

        return result

    async def _generate_with_cache(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: callable | None = None,
    ) -> str:
        """Dispatch to the appropriate backend with cache and rate limiting."""
        provider = self._config.model.provider
        model: str | None = None
        response: str | None = None

        # Skip cache when streaming - can't cache partial tokens
        if not stream_callback:
            if provider == "subscription":
                model = self._config.model.subscription_model or "cli-default"
                response = await self._cache.get(
                    prompt,
                    model,
                    f"subscription:{self._config.model.subscription_provider}",
                    temperature,
                    json_mode,
                    system,
                )
                if response is not None:
                    return response

            # Try cloud backend first if configured
            if provider == "cloud" and self._cloud:
                model = self._config.model.cloud_model or "unknown"

                response = await self._cache.get(
                    prompt,
                    model,
                    provider,
                    temperature,
                    json_mode,
                    system,
                )

                if response is not None:
                    return response

            # Try ollama or local backend
            if provider in ("ollama", "local"):
                if await self._ollama.is_available():
                    model = await self._resolve_ollama_model()

                    response = await self._cache.get(
                        prompt,
                        model,
                        provider,
                        temperature,
                        json_mode,
                        system,
                    )

                    if response is not None:
                        return response

                if self._try_llamacpp():
                    model = "llamacpp"

                    response = await self._cache.get(
                        prompt,
                        model,
                        provider,
                        temperature,
                        json_mode,
                        system,
                    )

                    if response is not None:
                        return response

            # Fallback: try ollama if not already tried
            if provider not in ("ollama", "subscription") and await self._ollama.is_available():
                model = await self._resolve_ollama_model()

                response = await self._cache.get(
                    prompt,
                    model,
                    "ollama",
                    temperature,
                    json_mode,
                    system,
                )

                if response is not None:
                    return response

            # Final fallback: cloud API
            if self._cloud and provider not in ("ollama", "local", "subscription"):
                model = self._config.model.cloud_model or "unknown"

                response = await self._cache.get(
                    prompt,
                    model,
                    "cloud",
                    temperature,
                    json_mode,
                    system,
                )

                if response is not None:
                    return response

        # Now do the actual generation with rate limiting
        await self._rate_limiter.acquire()

        if provider == "subscription":
            model = self._config.model.subscription_model or "cli-default"
            response = await self._subscription.generate(
                prompt,
                system=system,
                json_mode=json_mode,
                temperature=temperature,
                stream_callback=stream_callback,
            )
            if not stream_callback:
                await self._cache.set(
                    prompt,
                    model,
                    f"subscription:{self._config.model.subscription_provider}",
                    temperature,
                    json_mode,
                    response,
                    system,
                )
            return response

        # Cloud provider
        if provider == "cloud" and self._cloud:
            self._raise_if_cloud_circuit_open()
            try:
                response = await self._cloud.generate(
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    temperature=temperature,
                    stream_callback=stream_callback,
                )

                if not stream_callback and model:
                    await self._cache.set(
                        prompt,
                        model,
                        provider,
                        temperature,
                        json_mode,
                        response,
                        system,
                    )

                self._clear_cloud_circuit()
                return response

            except asyncio.CancelledError:
                self._open_cloud_circuit(
                    f"{self._config.model.cloud_provider.title()} API advisory request timed out.",
                    seconds=30.0,
                )
                raise
            except Exception as e:
                safe_error = safe_provider_error(e, self._config.model.cloud_provider)
                self._open_cloud_circuit(safe_error)
                logger.error("%s", safe_error)
                raise RuntimeError(safe_error) from None

        # Try ollama or local backend
        if provider in ("ollama", "local"):
            if await self._ollama.is_available():
                model = await self._resolve_ollama_model()

                logger.info("Waiting for local Ollama slot...")

                async with self._local_llm_semaphore:
                    logger.info("Acquired local Ollama slot")

                    try:
                        response = await asyncio.wait_for(
                            self._ollama.generate(
                                model,
                                prompt,
                                system=system,
                                json_mode=json_mode,
                                temperature=temperature,
                                stream_callback=stream_callback,
                            ),
                            timeout=self._local_llm_timeout,
                        )

                    except asyncio.TimeoutError:
                        logger.error("Ollama request timed out")
                        raise RuntimeError("Local Ollama inference timed out")

                if not stream_callback and model:
                    await self._cache.set(
                        prompt,
                        model,
                        provider,
                        temperature,
                        json_mode,
                        response,
                        system,
                    )

                return response

            if self._try_llamacpp():
                model = "llamacpp"

                logger.info("Waiting for llama.cpp slot...")

                async with self._local_llm_semaphore:
                    logger.info("Acquired llama.cpp slot")

                    try:
                        response = await asyncio.wait_for(
                            self._llamacpp_generate(
                                prompt,
                                system=system,
                                temperature=temperature,
                            ),
                            timeout=self._local_llm_timeout,
                        )

                    except asyncio.TimeoutError:
                        logger.error("llama.cpp request timed out")
                        raise RuntimeError("Local llama.cpp inference timed out")

                if not stream_callback and model:
                    await self._cache.set(
                        prompt,
                        model,
                        provider,
                        temperature,
                        json_mode,
                        response,
                        system,
                    )

                return response

        # Fallback: try ollama if not already tried
        if provider != "ollama" and await self._ollama.is_available():
            model = await self._resolve_ollama_model()

            logger.info("Waiting for fallback Ollama slot...")

            async with self._local_llm_semaphore:
                logger.info("Acquired fallback Ollama slot")

                try:
                    response = await asyncio.wait_for(
                        self._ollama.generate(
                            model,
                            prompt,
                            system=system,
                            json_mode=json_mode,
                            temperature=temperature,
                            stream_callback=stream_callback,
                        ),
                        timeout=self._local_llm_timeout,
                    )

                except asyncio.TimeoutError:
                    logger.error("Fallback Ollama request timed out")
                    raise RuntimeError("Fallback Ollama inference timed out")

            if not stream_callback and model:
                await self._cache.set(
                    prompt,
                    model,
                    "ollama",
                    temperature,
                    json_mode,
                    response,
                    system,
                )

            return response

        # Final fallback: cloud API
        if self._cloud:
            model = self._config.model.cloud_model or "unknown"

            logger.warning("Falling back to cloud API — Ollama unavailable")
            self._raise_if_cloud_circuit_open()

            try:
                response = await self._cloud.generate(
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    temperature=temperature,
                    stream_callback=stream_callback,
                )
                self._clear_cloud_circuit()
            except asyncio.CancelledError:
                self._open_cloud_circuit(
                    f"{self._config.model.cloud_provider.title()} API advisory request timed out.",
                    seconds=30.0,
                )
                raise
            except Exception as e:
                safe_error = safe_provider_error(e, self._config.model.cloud_provider)
                self._open_cloud_circuit(safe_error)
                logger.error("%s", safe_error)
                raise RuntimeError(safe_error) from None

            if not stream_callback and model:
                await self._cache.set(
                    prompt,
                    model,
                    "cloud",
                    temperature,
                    json_mode,
                    response,
                    system,
                )

            return response

        raise RuntimeError(
            "No model backend available. Start Ollama, configure a cloud API key, "
            "or connect a supported subscription CLI."
        )

    def _open_cloud_circuit(self, reason: str, *, seconds: float = 60.0) -> None:
        self._cloud_unavailable_reason = reason
        self._cloud_unavailable_until = time.monotonic() + seconds

    def _clear_cloud_circuit(self) -> None:
        self._cloud_unavailable_reason = ""
        self._cloud_unavailable_until = 0.0

    def _raise_if_cloud_circuit_open(self) -> None:
        if time.monotonic() < self._cloud_unavailable_until:
            reason = self._cloud_unavailable_reason or "Cloud provider is temporarily unavailable."
            raise RuntimeError(f"{reason} Retry shortly or configure a healthy provider.")

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
        # Use the fallback for this run only; the person's chosen model stays in config.toml.
        self._config.set_runtime_override("model", "ollama_model", fallback)

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

    async def _llamacpp_generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        from pilot.models.llamacpp import LlamaCppClient

        client: LlamaCppClient = self._llamacpp  # type: ignore[assignment]
        await self._cancel_llamacpp_idle_unload()
        self._llamacpp_active_calls += 1
        try:
            return await client.generate(
                prompt,
                system=system,
                temperature=temperature,
            )
        finally:
            self._llamacpp_active_calls -= 1
            if self._llamacpp_active_calls == 0:
                if self._llamacpp_reload_pending:
                    client.unload()
                    if self._llamacpp is client:
                        self._llamacpp = None
                    self._llamacpp_reload_pending = False
                else:
                    self._schedule_llamacpp_idle_unload(client)

    def _schedule_llamacpp_idle_unload(self, client: Any) -> None:
        self._llamacpp_idle_task = asyncio.create_task(
            self._unload_llamacpp_after_idle(client),
            name="llamacpp-idle-unload",
        )

    async def _cancel_llamacpp_idle_unload(self) -> None:
        task = self._llamacpp_idle_task
        self._llamacpp_idle_task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _unload_llamacpp_after_idle(self, client: Any) -> None:
        idle_seconds = max(0, int(self._config.model.idle_unload_seconds))
        try:
            await asyncio.sleep(idle_seconds)
        except asyncio.CancelledError:
            return
        if self._llamacpp_active_calls == 0 and self._llamacpp is client:
            client.unload()

    async def reconfigure(self, changed_keys: set[str]) -> None:
        """Apply validated model settings to live provider resources."""
        rate_limit_keys = {"rate_limit_enabled", "rate_limit_rpm", "rate_limit_burst"}
        if changed_keys & rate_limit_keys:
            self._rate_limiter.reconfigure(self._config.model)

        budget_keys = {
            "budget_enabled",
            "budget_monthly_limit_usd",
            "max_tokens_per_action",
            "max_tokens_per_task",
            "max_usd_per_task",
        }
        budget_tracker = getattr(self, "_budget_tracker", None)
        if budget_tracker is not None and changed_keys & budget_keys:
            budget_tracker.reconfigure(self._config.model)

        if "ollama_model" in changed_keys:
            self._resolved_ollama_model = None

        if "ollama_base_url" in changed_keys:
            replacement = OllamaClient(self._config.model.ollama_base_url, self._config)
            previous = self._ollama
            self._ollama = replacement
            self._retire_client(previous)

        if "cloud_provider" in changed_keys:
            replacement_cloud = CloudClient(self._config, self._vault) if self._config.model.cloud_provider else None
            previous_cloud = self._cloud
            self._cloud = replacement_cloud
            self._clear_cloud_circuit()
            self._retire_client(previous_cloud)

        subscription_keys = {
            "provider",
            "subscription_provider",
            "subscription_model",
            "subscription_timeout_seconds",
            "subscription_max_prompt_chars",
        }
        if changed_keys & subscription_keys:
            replacement_subscription = SubscriptionCLIClient(self._config)
            previous_subscription = self._subscription
            self._subscription = replacement_subscription
            self._retire_client(previous_subscription)

        if "idle_unload_seconds" in changed_keys:
            await self._cancel_llamacpp_idle_unload()
            if self._llamacpp is not None and self._llamacpp_active_calls == 0:
                self._schedule_llamacpp_idle_unload(self._llamacpp)

        if "gpu_memory_limit_mb" in changed_keys and self._llamacpp is not None:
            await self._cancel_llamacpp_idle_unload()
            if self._llamacpp_active_calls:
                self._llamacpp_reload_pending = True
            else:
                self._llamacpp.unload()
                self._llamacpp = None

    def _retire_client(self, client: Any) -> None:
        """Close a replaced provider client after calls already using it have had time to finish.

        Switching provider or model used to close the old client at once, failing the
        generations of background jobs and voice tasks mid-answer (and killing a
        subscription CLI's process). New calls use the replacement immediately.
        """
        if client is None:
            return
        retiring = getattr(self, "_retiring_clients", None)
        if retiring is None:
            retiring = self._retiring_clients = {}
        grace = max(60.0, float(getattr(self._config.model, "subscription_timeout_seconds", 120) or 120))
        task = asyncio.create_task(self._close_after(client, grace))
        retiring[task] = client

    async def _close_after(self, client: Any, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            pass  # shutting down: close now
        finally:
            getattr(self, "_retiring_clients", {}).pop(asyncio.current_task(), None)
            await self._close_quietly(client)

    @staticmethod
    async def _close_quietly(client: Any) -> None:
        try:
            await client.close()
        except Exception:
            logger.warning("A replaced model client did not close cleanly", exc_info=True)

    def rate_limit_stats(self) -> dict[str, Any]:
        """Return current rate limiter state and lifetime counters."""
        return self._rate_limiter.get_stats()

    async def budget_stats(self) -> dict:
        """Return current budget tracker stats, or empty dict if not configured."""
        if self._budget_tracker:
            return await self._budget_tracker.get_stats()

        return {}

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

    async def cache_clear(
        self,
        provider: str | None = None,
        model: str | None = None,
    ) -> int:
        """Clear cache entries.

        Args:
            provider: If specified, only clear entries for this provider.
            model: If specified, only clear entries for this model.

        Returns:
            Number of entries deleted.
        """
        return await self._cache.clear(provider, model)

    async def close(self) -> None:
        """Close provider subprocesses and the cache connection."""
        retiring = dict(getattr(self, "_retiring_clients", {}))
        for task in retiring:
            task.cancel()
        if retiring:
            await asyncio.gather(*retiring, return_exceptions=True)
        # A task cancelled before it ever ran never reached its own close; close those here.
        for task, client in retiring.items():
            if getattr(self, "_retiring_clients", {}).pop(task, None) is not None:
                await self._close_quietly(client)
        await self._cancel_llamacpp_idle_unload()
        if self._llamacpp is not None:
            self._llamacpp.unload()
            self._llamacpp = None
        self._llamacpp_reload_pending = False

        closers = [
            self._subscription.close(),
            self._cache.close(),
            self._ollama.close(),
        ]
        if self._cloud is not None:
            closers.append(self._cloud.close())
        results = await asyncio.gather(*closers, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Model provider shutdown failed: %s", result)

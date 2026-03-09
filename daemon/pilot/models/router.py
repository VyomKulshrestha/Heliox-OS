"""Model router: selects and dispatches to the appropriate LLM backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pilot.models.cloud import CloudClient
from pilot.models.ollama import OllamaClient

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

        if config.model.cloud_provider:
            self._cloud = CloudClient(config, vault)

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
    ) -> str:
        """Generate a completion from the best available model."""
        provider = self._config.model.provider

        if provider == "cloud" and self._cloud:
            try:
                return await self._cloud.generate(prompt, system=system, json_mode=json_mode, temperature=temperature)
            except Exception as e:
                logger.warning("Cloud API failed (%s), falling back to Ollama", e)
                # Fall through to Ollama

        if provider in ("ollama", "local") or provider == "cloud":
            if await self._ollama.is_available():
                model = await self._resolve_ollama_model()
                return await self._ollama.generate(
                    model,
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    temperature=temperature,
                )

            if self._try_llamacpp():
                return await self._llamacpp_generate(prompt, system=system, temperature=temperature)

        if await self._ollama.is_available():
            model = await self._resolve_ollama_model()
            return await self._ollama.generate(
                model,
                prompt,
                system=system,
                json_mode=json_mode,
                temperature=temperature,
            )

        if self._cloud:
            logger.warning("Falling back to cloud API — Ollama unavailable")
            return await self._cloud.generate(prompt, system=system, json_mode=json_mode, temperature=temperature)

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

    async def check_health(self) -> dict[str, bool]:
        """Check which backends are available."""
        status: dict[str, bool] = {}
        status["ollama"] = await self._ollama.is_available()
        status["llamacpp"] = self._try_llamacpp()
        status["cloud"] = self._cloud is not None
        return status

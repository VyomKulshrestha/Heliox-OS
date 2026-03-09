"""llama-cpp-python integration for direct GGUF model loading.

This module is optional — only used if Ollama is unavailable and
the llama-cpp-python package is installed.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.models.llamacpp")


class LlamaCppClient:
    """Wrapper around llama-cpp-python for local GGUF inference."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._llm = None
        self._model_path: str | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed. Install it with: pip install llama-cpp-python")

        model_path = self._find_model()
        if not model_path:
            raise RuntimeError("No GGUF model file found. Download one to ~/.local/share/pilot/models/")

        gpu_layers = -1
        if self._config.model.gpu_memory_limit_mb > 0:
            gpu_layers = max(1, self._config.model.gpu_memory_limit_mb // 200)

        logger.info("Loading GGUF model: %s (n_gpu_layers=%d)", model_path, gpu_layers)
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        self._model_path = str(model_path)

    def _find_model(self) -> Path | None:
        """Search for a GGUF model file in the standard location."""
        from pilot.config import DATA_DIR

        models_dir = DATA_DIR / "models"
        if not models_dir.exists():
            return None
        gguf_files = sorted(models_dir.glob("*.gguf"))
        return gguf_files[0] if gguf_files else None

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, prompt, system, temperature)

    def _generate_sync(self, prompt: str, system: str, temperature: float) -> str:
        self._ensure_loaded()
        full_prompt = prompt
        if system:
            full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"

        result = self._llm(
            full_prompt,
            max_tokens=2048,
            temperature=temperature,
            stop=["<|end|>", "<|user|>"],
        )
        return result["choices"][0]["text"].strip()

    def unload(self) -> None:
        """Release the model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("Model unloaded from memory")

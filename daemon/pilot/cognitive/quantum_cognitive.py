"""Quantum-Ready Cognitive Architecture — model-agnostic cognitive API layer.

- Model-agnostic cognitive pipeline
- Standard cognitive APIs for any model adapter
- Plugin architecture for new models
- Standard interface for developers

Architecture:
  Model Adapter → Cognitive Pipeline → Standard API → Plugins
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger("pilot.cognitive.quantum_ready")

# ── Model Types ──


class ModelType(StrEnum):
    GPT_NEURO = "gpt_neuro"
    CLAUDE_NEURO = "claude_neuro"
    GEMINI_NEURO = "gemini_neuro"
    CUSTOM = "custom"


class CognitiveCapability(StrEnum):
    ATTENTION_PREDICTION = "attention_prediction"
    STRESS_DETECTION = "stress_detection"
    LOAD_ESTIMATION = "load_estimation"
    INTENT_CLASSIFICATION = "intent_classification"
    EMOTION_RECOGNITION = "emotion_recognition"


# ── Data Structures ──


@dataclass
class CognitiveInput:
    """Standard input to cognitive pipeline."""

    stimulus: str = ""
    modality: str = "text"  # text, visual, audio, multimodal
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveOutput:
    """Standard output from cognitive pipeline."""

    attention_score: float = 0.5
    stress_level: float = 0.3
    cognitive_load: float = 0.4
    emotional_state: str = "neutral"
    confidence: float = 0.5

    # Raw model output
    raw_output: dict[str, Any] = field(default_factory=dict)

    # Timing
    latency_ms: float = 0.0
    model_used: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelInfo:
    """Information about a cognitive model."""

    model_type: str
    name: str
    version: str
    capabilities: list[str]
    is_available: bool
    avg_latency_ms: float = 0.0
    accuracy_estimate: float = 0.0


# ── Base Adapter ──


class CognitiveModelAdapter(ABC):
    """Abstract base class for cognitive model adapters."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available."""
        pass

    @abstractmethod
    async def predict(
        self,
        input_data: CognitiveInput,
    ) -> CognitiveOutput:
        """Run prediction on input."""
        pass

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Get model information."""
        pass

    @abstractmethod
    async def load_model(self) -> bool:
        """Load the model."""
        pass


# ── Fallback Adapter ──


class FallbackAdapter(CognitiveModelAdapter):
    """Fallback adapter using heuristics when no model available."""

    def __init__(self):
        self._total_predictions = 0
        self._total_latency = 0.0

    @property
    def model_type(self) -> str:
        return "fallback"

    @property
    def name(self) -> str:
        return "Heuristic Fallback"

    def is_available(self) -> bool:
        return True

    async def predict(self, input_data: CognitiveInput) -> CognitiveOutput:
        t0 = time.time()

        # Simple heuristic-based prediction
        stimulus = input_data.stimulus.lower()

        # Stress detection from keywords
        stress = 0.3
        stress_keywords = ["urgent", "asap", "critical", "emergency", "immediately"]
        for kw in stress_keywords:
            if kw in stimulus:
                stress = min(1.0, stress + 0.2)

        # Attention estimation from length
        attention = 0.5
        if len(stimulus) > 100:
            attention = 0.7
        elif len(stimulus) < 20:
            attention = 0.4

        # Load estimation
        load = min(1.0, len(stimulus) / 100.0)

        latency = (time.time() - t0) * 1000
        self._total_predictions += 1
        self._total_latency += latency

        return CognitiveOutput(
            attention_score=attention,
            stress_level=stress,
            cognitive_load=load,
            confidence=0.4,
            latency_ms=latency,
            model_used="fallback",
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            model_type=self.model_type,
            name=self.name,
            version="1.0",
            capabilities=[
                CognitiveCapability.LOAD_ESTIMATION,
            ],
            is_available=True,
            avg_latency_ms=1.0,
            accuracy_estimate=0.4,
        )

    async def load_model(self) -> bool:
        return True


# ── Quantum Cognitive Pipeline ──


class QuantumCognitivePipeline:
    """Model-agnostic cognitive pipeline with plugin architecture."""

    def __init__(self):
        # Available adapters
        self._adapters: dict[str, CognitiveModelAdapter] = {}
        self._active_adapter: CognitiveModelAdapter | None = None
        self._fallback = FallbackAdapter()

        # Plugin system
        self._preprocessors: list[Callable[[CognitiveInput], CognitiveInput]] = []
        self._postprocessors: list[Callable[[CognitiveOutput], CognitiveOutput]] = []

        # Initialize with the heuristic fallback as the default adapter
        self._initialize_adapters()

        # Stats
        self._total_requests = 0
        self._failed_requests = 0

    def _initialize_adapters(self) -> None:
        """Initialize available model adapters."""
        self._adapters[self._fallback.model_type] = self._fallback
        self._active_adapter = self._fallback
        logger.info("Using fallback heuristic adapter")

    # ── Adapter Management ──

    def register_adapter(self, adapter: CognitiveModelAdapter) -> None:
        """Register a new model adapter."""
        self._adapters[adapter.model_type] = adapter
        logger.info("Registered adapter: %s (%s)", adapter.name, adapter.model_type)

    def set_active_model(self, model_type: str) -> bool:
        """Switch the active model."""
        if model_type in self._adapters:
            self._active_adapter = self._adapters[model_type]
            logger.info("Switched to model: %s", model_type)
            return True
        return False

    def get_available_models(self) -> list[ModelInfo]:
        """Get list of available models."""
        return [adapter.get_info() for adapter in self._adapters.values()]

    # ── Pipeline ──

    def add_preprocessor(self, fn: Callable[[CognitiveInput], CognitiveInput]) -> None:
        """Add a preprocessor function."""
        self._preprocessors.append(fn)

    def add_postprocessor(self, fn: Callable[[CognitiveOutput], CognitiveOutput]) -> None:
        """Add a postprocessor function."""
        self._postprocessors.append(fn)

    async def predict(
        self,
        stimulus: str,
        modality: str = "text",
        source: str = "",
    ) -> CognitiveOutput:
        """Run cognitive prediction through the pipeline."""
        self._total_requests += 1

        # Build input
        input_data = CognitiveInput(
            stimulus=stimulus,
            modality=modality,
            source=source,
        )

        # Run preprocessors
        for preprocessor in self._preprocessors:
            input_data = preprocessor(input_data)

        # Get adapter
        adapter = self._active_adapter or self._fallback

        try:
            # Run prediction
            output = await adapter.predict(input_data)

            # Run postprocessors
            for postprocessor in self._postprocessors:
                output = postprocessor(output)

            return output

        except Exception as e:
            self._failed_requests += 1
            logger.warning("Prediction failed: %s", e)

            # Return fallback
            return await self._fallback.predict(input_data)

    # ── Batch Prediction ──

    async def predict_batch(
        self,
        inputs: list[CognitiveInput],
    ) -> list[CognitiveOutput]:
        """Run batch prediction."""
        outputs = []
        for input_data in inputs:
            output = await self.predict(
                input_data.stimulus,
                input_data.modality,
                input_data.source,
            )
            outputs.append(output)
        return outputs

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        active_model = self._active_adapter.get_info() if self._active_adapter else None

        return {
            "active_model": active_model.name if active_model else None,
            "active_model_type": active_model.model_type if active_model else None,
            "available_models": len(self._adapters),
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": ((self._total_requests - self._failed_requests) / max(1, self._total_requests)),
            "registered_adapters": list(self._adapters.keys()),
        }


# ── Factory ──


def create_pipeline() -> QuantumCognitivePipeline:
    """Factory function to create a cognitive pipeline."""
    return QuantumCognitivePipeline()


# ── Plugin Base ──


class CognitivePlugin(ABC):
    """Base class for cognitive pipeline plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def process(self, input_data: CognitiveInput) -> CognitiveInput:
        pass


# Example plugin interface usage:
#
# class MyPlugin(CognitivePlugin):
#     @property
#     def name(self) -> str:
#         return "my_plugin"
#
#     @property
#     def version(self) -> str:
#         return "1.0"
#
#     def process(self, input_data: CognitiveInput) -> CognitiveInput:
#         # Modify input
#         input_data.metadata["processed_by"] = self.name
#         return input_data
#
# pipeline = create_pipeline()
# pipeline.add_preprocessor(MyPlugin().process)

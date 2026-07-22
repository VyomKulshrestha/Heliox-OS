"""CognitiveEngine — lightweight, dependency-free cognitive state estimator.

Estimates attention/stress/cognitive-load from local interaction-history
heuristics only (event frequency, intensity, and diversity). No external
model, no network download, no GPU, no third-party license terms — every
line here is Heliox OS's own code.

This replaces the previous Meta TRIBE v2 integration (`tribev2`, loaded
from Hugging Face's `facebook/tribev2`), which was removed because both
its code (GitHub) and weights (Hugging Face) are licensed CC-BY-NC-4.0
(non-commercial), incompatible with a commercial product. There is no
comparable open, permissively-licensed model in the "predict brain
responses to stimuli" niche — that space is exclusively research-only
model releases — so this heuristic estimator is the practical
lightweight/open/free alternative, not a stopgap.

Usage across Heliox OS:
  ┌──────────────────────────────────────┐
  │        CognitiveEngine (singleton)   │
  │  ┌──────────┐ ┌──────────────────┐  │
  │  │ predict  │ │  attention_map   │  │
  │  └──────────┘ └──────────────────┘  │
  │  ┌──────────┐ ┌──────────────────┐  │
  │  │  stress  │ │ intent_affinity  │  │
  │  └──────────┘ └──────────────────┘  │
  └──────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pilot.cognitive.cognitive_engine")


@dataclass
class CognitiveSnapshot:
    """A point-in-time cognitive state estimate."""

    timestamp: float = field(default_factory=time.time)
    attention_score: float = 0.5  # 0=distracted, 1=fully focused
    stress_level: float = 0.3  # 0=calm, 1=high stress
    cognitive_load: float = 0.4  # 0=idle, 1=overloaded
    dominant_modality: str = "visual"  # visual | auditory | linguistic
    confidence: float = 0.0  # how confident the prediction is
    raw_activations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "attention_score": round(self.attention_score, 3),
            "stress_level": round(self.stress_level, 3),
            "cognitive_load": round(self.cognitive_load, 3),
            "dominant_modality": self.dominant_modality,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class AttentionRegion:
    """Predicted visual attention hotspot."""

    x: float  # normalized 0-1
    y: float  # normalized 0-1
    radius: float  # attention spread
    salience: float  # predicted attention strength


class CognitiveEngine:
    """Singleton heuristic cognitive-state estimator.

    Provides three prediction APIs, all computed from local interaction
    history (no model, no network, no license concerns):
      1. predict_cognitive_state() — overall attention/stress/load snapshot
      2. predict_attention_map()   — which UI elements are drawing attention
      3. predict_intent_affinity() — which intent candidate best matches
    """

    _instance: CognitiveEngine | None = None
    _init_lock = asyncio.Lock()

    def __init__(self) -> None:
        self._loaded = False
        self._prediction_cache: dict[str, Any] = {}
        self._cache_ttl_s = 2.0  # predictions valid for 2 seconds
        self._total_predictions = 0
        self._total_latency_ms = 0.0

        self._interaction_history: list[dict[str, Any]] = []
        self._max_history = 100

    @classmethod
    async def get_instance_async(cls) -> CognitiveEngine:
        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def get_instance(cls) -> CognitiveEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_available(self) -> bool:
        """Always True — heuristic estimation has no external dependency."""
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_fallback(self) -> bool:
        """Always True — heuristics are the only estimation mode."""
        return True

    # ── Lifecycle ──

    async def load_model(self) -> bool:
        """No-op: nothing to load. Kept for API parity with callers that
        `asyncio.create_task(engine.load_model())` on startup."""
        self._loaded = True
        return True

    def unload_model(self) -> None:
        self._loaded = False
        self._prediction_cache.clear()

    # ── Interaction Tracking ──

    def record_interaction(
        self,
        event_type: str,
        modality: str = "visual",
        intensity: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a user interaction for heuristic cognitive estimation."""
        self._interaction_history.append(
            {
                "timestamp": time.time(),
                "event_type": event_type,
                "modality": modality,
                "intensity": intensity,
                "metadata": metadata or {},
            }
        )
        if len(self._interaction_history) > self._max_history:
            self._interaction_history = self._interaction_history[-self._max_history :]

    # ── Core Prediction APIs ──

    async def predict_cognitive_state(
        self,
        stimulus_description: str = "",
        screen_region: str = "full",
    ) -> CognitiveSnapshot:
        """Estimate the user's current cognitive state from interaction history."""
        t0 = time.time()
        snapshot = self._predict_with_heuristics()
        latency_ms = (time.time() - t0) * 1000
        self._total_predictions += 1
        self._total_latency_ms += latency_ms
        return snapshot

    async def predict_attention_map(
        self,
        ui_elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score UI elements by predicted visual attention capture.

        Returns the same list enriched with `attention_score` field.
        """
        return self._attention_map_heuristic(ui_elements)

    async def predict_intent_affinity(
        self,
        candidates: list[dict[str, Any]],
        voice_transcript: str = "",
        gesture_name: str = "",
    ) -> list[dict[str, Any]]:
        """Score intent candidates by predicted alignment with voice/gesture.

        Returns candidates enriched with `neural_affinity` field (0-1).
        """
        return self._intent_affinity_heuristic(candidates, voice_transcript, gesture_name)

    # ── Heuristic predictions ──

    def _predict_with_heuristics(self) -> CognitiveSnapshot:
        """Estimate cognitive state from interaction history patterns."""
        now = time.time()
        recent = [e for e in self._interaction_history if now - e["timestamp"] < 30]

        if not recent:
            return CognitiveSnapshot(confidence=0.3)

        # Interaction frequency → cognitive load
        freq = len(recent) / 30.0  # events per second
        load = min(1.0, freq * 0.5)

        # High-intensity interactions → stress
        avg_intensity = sum(e["intensity"] for e in recent) / len(recent)
        stress = min(1.0, avg_intensity * 0.7)

        # Event diversity → attention scatter (less diverse = more focused)
        event_types = set(e["event_type"] for e in recent)
        attention = max(0.2, 1.0 - (len(event_types) - 1) * 0.15)

        # Dominant modality
        modalities = [e["modality"] for e in recent]
        dominant = max(set(modalities), key=modalities.count) if modalities else "visual"

        return CognitiveSnapshot(
            attention_score=attention,
            stress_level=stress,
            cognitive_load=load,
            dominant_modality=dominant,
            confidence=0.4,
        )

    def _attention_map_heuristic(
        self,
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score UI elements using basic salience heuristics."""
        type_weights = {
            "error": 1.0,
            "warning": 0.85,
            "alert": 0.9,
            "notification": 0.7,
            "progress": 0.6,
            "button": 0.5,
            "text": 0.3,
            "background": 0.1,
        }

        scored = []
        for el in elements:
            el_type = el.get("type", "text").lower()
            base_score = type_weights.get(el_type, 0.4)

            # Boost for recent/new elements
            age_s = time.time() - el.get("created_at", time.time())
            recency_boost = max(0.0, 0.2 * (1.0 - age_s / 10.0))

            # Boost for elements with motion/animation
            if el.get("animated", False):
                base_score = min(1.0, base_score + 0.15)

            el_copy = dict(el)
            el_copy["attention_score"] = min(1.0, base_score + recency_boost)
            el_copy["confidence"] = 0.35
            scored.append(el_copy)

        return scored

    def _intent_affinity_heuristic(
        self,
        candidates: list[dict[str, Any]],
        voice: str,
        gesture: str,
    ) -> list[dict[str, Any]]:
        """Score intent candidates using word-overlap/gesture heuristics."""
        scored = []
        voice_words = set(voice.lower().split()) if voice else set()

        for c in candidates:
            desc_words = set(c.get("description", c.get("command", "")).lower().split())

            if voice_words and desc_words:
                overlap = len(voice_words & desc_words) / max(len(voice_words), 1)
            else:
                overlap = 0.3

            gesture_bonus = 0.0
            gesture_map = c.get("gesture_match", "")
            if gesture and gesture_map and gesture.lower() == gesture_map.lower():
                gesture_bonus = 0.3

            c_copy = dict(c)
            c_copy["neural_affinity"] = min(1.0, overlap * 0.7 + gesture_bonus + 0.1)
            scored.append(c_copy)

        return sorted(scored, key=lambda x: x["neural_affinity"], reverse=True)

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_available": True,
            "model_loaded": self._loaded,
            "fallback_mode": True,
            "total_predictions": self._total_predictions,
            "avg_latency_ms": (
                round(self._total_latency_ms / self._total_predictions, 2) if self._total_predictions > 0 else 0
            ),
            "interaction_history_size": len(self._interaction_history),
        }

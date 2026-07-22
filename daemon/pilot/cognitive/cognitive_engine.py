"""CognitiveEngine — lightweight, dependency-free cognitive state estimator.

Estimates attention/stress/cognitive-load from local signals only: recency-
decayed interaction history (event frequency, intensity, and diversity),
real keyboard/mouse cadence (see `record_input_dynamics`), gaze region (see
`record_gaze`), and a small auditable keyword table applied to whatever
stimulus text a caller provides. No external model, no network download,
no GPU, no third-party license terms — every line here is Heliox OS's own
code.

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
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pilot.cognitive.cognitive_engine")

# Auditable, hardcoded keyword tables -- never a learned classifier, mirrors
# pilot.security.risk_patterns' "named, auditable rule, never persist the
# source text" contract. _text_signal() below returns only two bounded
# floats; the stimulus string itself is never stored or logged anywhere.
_STRESS_SIGNAL_KEYWORDS: tuple[str, ...] = (
    "error",
    "failed",
    "failure",
    "critical",
    "urgent",
    "warning",
    "crash",
    "exception",
    "denied",
    "timeout",
    "delete",
    "danger",
)
_ATTENTION_SIGNAL_KEYWORDS: tuple[str, ...] = (
    "click",
    "confirm",
    "required",
    "alert",
    "notification",
)


def _text_signal(stimulus: str) -> tuple[float, float]:
    """Derive (stress_bump, attention_bump) in [0, 1] from stimulus text via
    a small hardcoded keyword table -- auditable, not a learned classifier.
    Returns only numeric signals; never stores or logs the input text."""
    if not stimulus:
        return 0.0, 0.0
    lowered = stimulus.lower()
    stress_hits = sum(1 for kw in _STRESS_SIGNAL_KEYWORDS if kw in lowered)
    attention_hits = sum(1 for kw in _ATTENTION_SIGNAL_KEYWORDS if kw in lowered)
    return min(0.4, stress_hits * 0.15), min(0.3, attention_hits * 0.1)


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

    Provides three prediction APIs, all computed from local signals (no
    model, no network, no license concerns):
      1. predict_cognitive_state() — overall attention/stress/load snapshot
      2. predict_attention_map()   — which UI elements are drawing attention
      3. predict_intent_affinity() — which intent candidate best matches

    Three independent signal streams feed prediction #1, each recency-
    decayed (recent samples matter more, old ones fade smoothly instead of
    falling off a hard cutoff cliff):
      - `record_interaction()`     — UI/action event log (event frequency,
        per-event intensity, event-type diversity)
      - `record_input_dynamics()`  — real keyboard/mouse cadence from
        `pilot.system.input_hook.InputSupervisionHook`
      - `record_gaze()`            — gaze region from the frontend's
        webcam-based gaze tracking, via `multimodal.fusion`
    """

    _instance: CognitiveEngine | None = None
    _init_lock = asyncio.Lock()

    # Recency decay: a sample's weight is exp(-age_seconds / _DECAY_TAU_S).
    # At tau=20s a sample is ~60% relevant after 10s, ~14% after 40s, and
    # negligible (<2%) past _HISTORY_HORIZON_S -- smooth responsiveness
    # instead of a hard cliff where a 29s-old event counts fully and a
    # 31s-old one counts zero.
    _DECAY_TAU_S = 20.0
    _HISTORY_HORIZON_S = 90.0

    def __init__(self) -> None:
        self._loaded = False
        self._prediction_cache: dict[str, Any] = {}
        self._cache_ttl_s = 2.0  # predictions valid for 2 seconds
        self._total_predictions = 0
        self._total_latency_ms = 0.0

        self._interaction_history: list[dict[str, Any]] = []
        self._max_history = 100

        self._input_dynamics_history: list[dict[str, Any]] = []
        self._max_dynamics_history = 60

        self._gaze_history: list[dict[str, Any]] = []
        self._max_gaze_history = 60

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

    # ── Signal Tracking ──

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

    def record_input_dynamics(self, keystroke_rate_per_min: float, click_rate_per_min: float) -> None:
        """Feed a real keyboard/mouse cadence sample (e.g. from
        `InputSupervisionHook.snapshot()`) — an independent, objective
        engagement signal rather than the engine's own prior output."""
        self._input_dynamics_history.append(
            {
                "timestamp": time.time(),
                "keystroke_rate": max(0.0, keystroke_rate_per_min),
                "click_rate": max(0.0, click_rate_per_min),
            }
        )
        if len(self._input_dynamics_history) > self._max_dynamics_history:
            self._input_dynamics_history = self._input_dynamics_history[-self._max_dynamics_history :]

    def record_gaze(self, region: str, confidence: float = 0.5) -> None:
        """Feed a gaze-region sample (see `multimodal.fusion.on_gaze_event`).
        A non-"center" region is treated as a mild visual-distraction signal,
        scaled by the frontend's own confidence for that sample."""
        distraction = max(0.0, min(1.0, confidence)) if region and region != "center" else 0.0
        self._gaze_history.append({"timestamp": time.time(), "distraction": distraction})
        if len(self._gaze_history) > self._max_gaze_history:
            self._gaze_history = self._gaze_history[-self._max_gaze_history :]

    # ── Core Prediction APIs ──

    async def predict_cognitive_state(
        self,
        stimulus_description: str = "",
        screen_region: str = "full",
    ) -> CognitiveSnapshot:
        """Estimate the user's current cognitive state from interaction
        history, input dynamics, gaze, and (if provided) stimulus text."""
        t0 = time.time()
        snapshot = self._predict_with_heuristics(stimulus_description)
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

    def _decay_weight(self, age_s: float) -> float:
        """exp(-age/tau), or 0.0 once past the hard horizon cutoff."""
        if age_s > self._HISTORY_HORIZON_S:
            return 0.0
        return math.exp(-age_s / self._DECAY_TAU_S)

    def _predict_with_heuristics(self, stimulus_description: str = "") -> CognitiveSnapshot:
        """Estimate cognitive state by blending recency-decayed interaction
        history, real input-dynamics cadence, gaze distraction, and a
        keyword-derived stimulus signal. No component is a learned model;
        each is a small, auditable, independently-justified heuristic.

        Each history is walked exactly once (not once per derived field) so
        this stays cheap even though it now combines four signal streams."""
        now = time.time()

        event_weight = 0.0
        weighted_intensity = 0.0
        event_type_weight: dict[str, float] = {}
        modality_weight: dict[str, float] = {}
        for e in self._interaction_history:
            w = self._decay_weight(now - e["timestamp"])
            if w <= 0.0:
                continue
            event_weight += w
            weighted_intensity += w * e["intensity"]
            event_type_weight[e["event_type"]] = event_type_weight.get(e["event_type"], 0.0) + w
            modality_weight[e["modality"]] = modality_weight.get(e["modality"], 0.0) + w

        dynamics_weight = 0.0
        weighted_keystroke = 0.0
        weighted_click = 0.0
        for d in self._input_dynamics_history:
            w = self._decay_weight(now - d["timestamp"])
            if w <= 0.0:
                continue
            dynamics_weight += w
            weighted_keystroke += w * d["keystroke_rate"]
            weighted_click += w * d["click_rate"]

        gaze_weight = 0.0
        weighted_distraction = 0.0
        for g in self._gaze_history:
            w = self._decay_weight(now - g["timestamp"])
            if w <= 0.0:
                continue
            gaze_weight += w
            weighted_distraction += w * g["distraction"]

        text_stress, text_attention = _text_signal(stimulus_description)

        if event_weight < 1e-6 and dynamics_weight < 1e-6 and gaze_weight < 1e-6 and not stimulus_description:
            return CognitiveSnapshot(confidence=0.2)

        # Cognitive load: blend decayed event frequency with real keystroke/
        # click cadence when both are available, otherwise use whichever
        # signal exists. `_DECAY_TAU_S` normalizes event_weight to roughly
        # "events per decay window" for a steady stream of interactions.
        event_freq_load = min(1.0, (event_weight / self._DECAY_TAU_S) * 4.0)
        keystroke_rate = weighted_keystroke / dynamics_weight if dynamics_weight > 0 else 0.0
        click_rate = weighted_click / dynamics_weight if dynamics_weight > 0 else 0.0
        # Ceilings (60 keys/min, 10 clicks/min) match neural_bridge.py's
        # InputDynamicsMonitor engagement-score normalization.
        dynamics_load = min(1.0, (keystroke_rate / 60.0) * 0.7 + (click_rate / 10.0) * 0.3)

        if event_weight > 1e-6 and dynamics_weight > 1e-6:
            load = event_freq_load * 0.5 + dynamics_load * 0.5
        elif dynamics_weight > 1e-6:
            load = dynamics_load
        else:
            load = event_freq_load

        avg_intensity = weighted_intensity / event_weight if event_weight > 1e-6 else 0.0
        gaze_distraction = weighted_distraction / gaze_weight if gaze_weight > 1e-6 else 0.0
        stress = min(1.0, avg_intensity * 0.65 + text_stress + gaze_distraction * 0.15)

        diversity = len(event_type_weight)
        attention = 1.0 - max(0, diversity - 1) * 0.12 - gaze_distraction * 0.25 + text_attention
        attention = max(0.15, min(1.0, attention))

        dominant = max(modality_weight, key=modality_weight.get) if modality_weight else "visual"

        # Confidence reflects how much real, recent data actually backs this
        # estimate (more signal streams + more recent samples = higher, but
        # still capped well below what a real trained model would report).
        richness = (
            min(1.0, event_weight / 3.0) * 0.45
            + min(1.0, dynamics_weight / 3.0) * 0.25
            + min(1.0, gaze_weight / 3.0) * 0.1
            + (0.15 if stimulus_description else 0.0)
        )
        confidence = round(min(0.6, 0.2 + richness * 0.4), 3)

        return CognitiveSnapshot(
            attention_score=attention,
            stress_level=stress,
            cognitive_load=load,
            dominant_modality=dominant,
            confidence=confidence,
            raw_activations={
                "event_weight": round(event_weight, 3),
                "dynamics_weight": round(dynamics_weight, 3),
                "keystroke_rate": round(keystroke_rate, 1),
                "click_rate": round(click_rate, 1),
                "gaze_distraction": round(gaze_distraction, 3),
            },
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
        """Score intent candidates using Jaccard word-overlap + gesture
        cross-modal match. Jaccard (intersection / union) penalizes verbose
        candidate descriptions that merely contain the voice words as a
        small subset, unlike a plain recall-only overlap ratio."""
        scored = []
        voice_lower = voice.lower().strip() if voice else ""
        voice_words = set(voice_lower.split()) if voice_lower else set()

        for c in candidates:
            desc = c.get("description", c.get("command", ""))
            desc_lower = desc.lower().strip()
            desc_words = set(desc_lower.split())

            if voice_words and desc_words:
                union = voice_words | desc_words
                overlap = len(voice_words & desc_words) / len(union) if union else 0.0
                # A multi-word phrase match/containment is a much stronger
                # signal than word-set overlap alone -- don't let Jaccard
                # dilute it just because the candidate has a few extra words.
                # Restricted to multi-word transcripts: for a single word,
                # "contained in" is nearly always trivially true for any
                # candidate mentioning that word at all, which would defeat
                # Jaccard's whole point of penalizing verbose candidates.
                if len(voice_words) > 1 and (voice_lower in desc_lower or desc_lower in voice_lower):
                    overlap = max(overlap, 0.85)
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
            "input_dynamics_history_size": len(self._input_dynamics_history),
            "gaze_history_size": len(self._gaze_history),
        }

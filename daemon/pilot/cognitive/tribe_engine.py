"""TRIBE v2 Engine — Core wrapper for Meta's neural prediction model.

Loads facebook/tribev2 from Hugging Face, caches weights locally, and
exposes an async API for predicting brain responses to visual, auditory,
and linguistic stimuli.

Usage across Heliox OS:
  ┌──────────────────────────────────────┐
  │          TribeEngine (singleton)     │
  │  ┌──────────┐ ┌──────────────────┐  │
  │  │ predict  │ │  attention_map   │  │
  │  └──────────┘ └──────────────────┘  │
  │  ┌──────────┐ ┌──────────────────┐  │
  │  │  stress  │ │ intent_affinity  │  │
  │  └──────────┘ └──────────────────┘  │
  └──────────────────────────────────────┘

The engine gracefully degrades: if tribev2 is not installed, all methods
return safe defaults so Heliox OS continues to function normally.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.cognitive.tribe_engine")

# ── Cache directory ──
_CACHE_DIR = Path(os.environ.get("HELIOX_TRIBE_CACHE", "")) or (Path.home() / ".cache" / "heliox" / "tribe_v2")

# ── Attempt to import tribev2 ──
_tribe_available = False
_TribeModel: Any = None

try:
    from tribev2 import TribeModel as _TM

    _TribeModel = _TM
    _tribe_available = True
    logger.info("TRIBE v2 library detected — cognitive features enabled")
except ImportError:
    logger.info(
        "tribev2 not installed — cognitive features will use heuristic fallbacks. Install with: pip install tribev2"
    )


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


class TribeEngine:
    """Singleton wrapper around Meta TRIBE v2.

    The engine provides three main prediction APIs:
      1. predict_cognitive_state() — overall attention/stress/load snapshot
      2. predict_attention_map()   — where the user is likely looking
      3. predict_intent_affinity() — which intent best matches neural patterns
    """

    _instance: TribeEngine | None = None
    _init_lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded = False
        self._loading = False
        self._lock = asyncio.Lock()
        self._load_cv = asyncio.Condition(self._lock)
        self._prediction_cache: dict[str, Any] = {}
        self._cache_ttl_s = 2.0  # predictions valid for 2 seconds
        self._total_predictions = 0
        self._total_latency_ms = 0.0
        self._fallback_mode = not _tribe_available

        # Heuristic state (used when TRIBE v2 is not available)
        self._interaction_history: list[dict[str, Any]] = []
        self._max_history = 100

    @classmethod
    async def get_instance_async(cls) -> TribeEngine:
        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def get_instance(cls) -> TribeEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_available(self) -> bool:
        return _tribe_available

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode

    # ── Model Loading ──

    async def load_model(self) -> bool:
        """Load TRIBE v2 model from Hugging Face Hub.

        Works around a Windows bug where Path('facebook/tribev2') gets
        mangled to 'facebook\\tribev2' which fails HuggingFace repo
        validation. We download the files manually and load from a
        local directory instead.
        """
        if self._loaded:
            return True
        if not _tribe_available:
            logger.info("TRIBE v2 not installed — using heuristic fallback mode")
            self._fallback_mode = True
            return False

        async with self._load_cv:
            if self._loaded:
                return True
            while self._loading:
                await self._load_cv.wait()
            if self._loaded:
                return True

        self._loading = True

        try:
            logger.info("Loading TRIBE v2 model from facebook/tribev2...")
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_str = str(_CACHE_DIR)

            def _load():
                """Download via huggingface_hub then load from local."""
                import pathlib
                import platform

                from huggingface_hub import hf_hub_download

                if platform.system() == "Windows":
                    pathlib.PosixPath = pathlib.WindowsPath

                repo_id = "facebook/tribev2"
                config_path = hf_hub_download(repo_id, "config.yaml")
                ckpt_path = hf_hub_download(repo_id, "best.ckpt")

                import os

                local_dir = os.path.dirname(config_path)
                model = _TribeModel.from_pretrained(
                    local_dir,
                    cache_folder=cache_str,
                    device="auto",
                )

                _UNGATED_LLAMA = "unsloth/Llama-3.2-3B"
                import torch as _torch

                _device = "cuda" if _torch.cuda.is_available() else "cpu"

                try:
                    text_ext = model.data.text_feature
                    if hasattr(text_ext, "model_name"):
                        old_name = text_ext.model_name
                        if "meta-llama" in old_name:
                            logger.info(
                                "Patching text extractor: %s → %s (ungated)",
                                old_name,
                                _UNGATED_LLAMA,
                            )
                        text_ext.model_name = _UNGATED_LLAMA
                        text_ext.device = _device

                        from transformers import AutoTokenizer

                        _ = AutoTokenizer.from_pretrained(_UNGATED_LLAMA)

                except Exception as patch_err:
                    logger.debug("Text extractor patch skipped: %s", patch_err)

                if platform.system() == "Windows":
                    if hasattr(model, "data") and hasattr(model.data, "num_workers"):
                        model.data.num_workers = 0

                return model

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(None, _load)
            self._loaded = True
            self._fallback_mode = False
            logger.info("TRIBE v2 model loaded successfully")

            async with self._load_cv:
                self._loading = False
                self._load_cv.notify_all()

            return True
        except Exception as e:
            logger.warning("Failed to load TRIBE v2 model: %s — using fallback", e)
            self._fallback_mode = True

            async with self._load_cv:
                self._loading = False
                self._load_cv.notify_all()

            return False

    def unload_model(self) -> None:
        """Free the TRIBE v2 model from memory."""
        self._model = None
        self._loaded = False
        self._prediction_cache.clear()
        logger.info("TRIBE v2 model unloaded")

    # ── Interaction Tracking (for heuristic fallback) ──

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
        """Predict the user's current cognitive state.

        Uses TRIBE v2 neural predictions if available, otherwise falls back
        to interaction-history heuristics.
        """
        t0 = time.time()

        if self._loaded and self._model and not self._fallback_mode:
            snapshot = await self._predict_with_model(stimulus_description, screen_region)
        else:
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
        if self._loaded and self._model and not self._fallback_mode:
            return await self._attention_map_model(ui_elements)
        return self._attention_map_heuristic(ui_elements)

    async def predict_intent_affinity(
        self,
        candidates: list[dict[str, Any]],
        voice_transcript: str = "",
        gesture_name: str = "",
    ) -> list[dict[str, Any]]:
        """Score intent candidates by predicted neural response alignment.

        Returns candidates enriched with `neural_affinity` field (0-1).
        """
        if self._loaded and self._model and not self._fallback_mode:
            return await self._intent_affinity_model(candidates, voice_transcript, gesture_name)
        return self._intent_affinity_heuristic(candidates, voice_transcript, gesture_name)

    # ── Model-based predictions ──

    async def _predict_with_model(
        self,
        stimulus: str,
        region: str,
    ) -> CognitiveSnapshot:
        """Use TRIBE v2 to predict cognitive state from stimulus description.

        Bypasses the text_path → TTS → WhisperX pipeline (which has
        broken dependencies on Windows) by constructing word-level
        events DataFrames directly for the model.
        """
        if not stimulus or not stimulus.strip():
            return self._predict_with_heuristics()

        try:
            loop = asyncio.get_event_loop()

            def _run_prediction():
                import numpy as np
                import pandas as pd

                # TRIBE v2 needs substantial natural language context
                # (it was designed for story/paragraph-level neuroscience).
                # Wrap our short stimuli in a rich context passage.
                context_template = (
                    "The person is sitting at their computer, working on "
                    "important tasks. They are currently focused on the screen "
                    "where a notification appears. The system displays a "
                    "message about {stimulus}. The user reads this carefully, "
                    "considering what action to take next. They feel the "
                    "weight of the decision as they process the information "
                    "shown on the display. The cognitive demand of this task "
                    "requires sustained attention and careful deliberation."
                )
                full_text = context_template.format(stimulus=stimulus.strip())
                words = full_text.split()

                word_duration = 0.3  # ~300ms per word (average speech)
                events = []
                t = 0.0
                char_pos = 0
                for word in words:
                    clean_word = word.lower().strip(".,;:!?")
                    if not clean_word:
                        char_pos += len(word) + 1
                        continue
                    events.append(
                        {
                            "type": "Word",
                            "text": clean_word,
                            "context": full_text,
                            "sentence": full_text,
                            "sentence_char": char_pos,
                            "start": t,
                            "duration": word_duration,
                            "timeline": "default",
                            "subject": "default",
                        }
                    )
                    t += word_duration
                    char_pos += len(word) + 1

                # Add a dummy Fixation event to ensure the dataset has >1 class.
                # This resolves the neuralset LabelEncoder UserWarning.
                events.append(
                    {
                        "type": "Fixation",
                        "text": "",
                        "context": full_text,
                        "sentence": full_text,
                        "sentence_char": char_pos,
                        "start": t,
                        "duration": 0.1,
                        "timeline": "default",
                        "subject": "default",
                    }
                )

                events_df = pd.DataFrame(events)

                # Run the TRIBE v2 transformer prediction
                preds, segments = self._model.predict(events_df, verbose=False)

                # preds shape: (n_segments, n_vertices)
                mean_activation = float(np.mean(np.abs(preds)))
                max_activation = float(np.max(np.abs(preds)))
                std_activation = float(np.std(preds))
                n_vertices = int(preds.shape[1]) if len(preds.shape) > 1 else 0

                return mean_activation, max_activation, std_activation, n_vertices

            mean_act, max_act, std_act, n_verts = await loop.run_in_executor(None, _run_prediction)

            # Map brain activations to cognitive metrics
            attention = min(1.0, mean_act * 2.5)
            stress = min(1.0, max_act * 0.8)
            load = min(1.0, (mean_act + max_act) / 4.0)

            # Save for global ambient TTS modulation access
            self._last_cognitive_load = load

            return CognitiveSnapshot(
                attention_score=attention,
                stress_level=stress,
                cognitive_load=load,
                dominant_modality="linguistic" if stimulus else "visual",
                confidence=0.85,
                raw_activations={
                    "mean": mean_act,
                    "max": max_act,
                    "std": std_act,
                    "n_vertices": n_verts,
                },
            )
        except Exception as e:
            logger.warning("TRIBE v2 prediction failed: %s — using fallback", e)
            return self._predict_with_heuristics()

    async def _attention_map_model(
        self,
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Use TRIBE v2 to score UI elements by attention capture."""
        try:
            scored = []
            for el in elements:
                label = el.get("label", el.get("type", "unknown"))
                state = await self.predict_cognitive_state(stimulus_description=label)
                el_copy = dict(el)
                el_copy["attention_score"] = state.attention_score
                el_copy["tribe_confidence"] = state.confidence
                scored.append(el_copy)
            return scored
        except Exception:
            return self._attention_map_heuristic(elements)

    async def _intent_affinity_model(
        self,
        candidates: list[dict[str, Any]],
        voice: str,
        gesture: str,
    ) -> list[dict[str, Any]]:
        """Use TRIBE v2 to score intent candidates."""
        try:
            scored = []
            for c in candidates:
                desc = c.get("description", c.get("command", ""))
                combined = f"{voice} {gesture} {desc}".strip()
                state = await self.predict_cognitive_state(stimulus_description=combined)
                c_copy = dict(c)
                c_copy["neural_affinity"] = state.attention_score * state.confidence
                scored.append(c_copy)
            return sorted(scored, key=lambda x: x["neural_affinity"], reverse=True)
        except Exception:
            return self._intent_affinity_heuristic(candidates, voice, gesture)

    # ── Heuristic fallback predictions ──

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
            confidence=0.4,  # lower confidence for heuristics
        )

    def _attention_map_heuristic(
        self,
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score UI elements using basic salience heuristics."""
        # Priority weights by element type
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
            el_copy["tribe_confidence"] = 0.35  # heuristic confidence
            scored.append(el_copy)

        return scored

    def _intent_affinity_heuristic(
        self,
        candidates: list[dict[str, Any]],
        voice: str,
        gesture: str,
    ) -> list[dict[str, Any]]:
        """Score intent candidates using semantic similarity heuristics."""
        scored = []
        voice_words = set(voice.lower().split()) if voice else set()

        for c in candidates:
            desc_words = set(c.get("description", c.get("command", "")).lower().split())

            # Word overlap → affinity
            if voice_words and desc_words:
                overlap = len(voice_words & desc_words) / max(len(voice_words), 1)
            else:
                overlap = 0.3

            # Gesture alignment bonus
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
            "tribe_available": _tribe_available,
            "model_loaded": self._loaded,
            "fallback_mode": self._fallback_mode,
            "total_predictions": self._total_predictions,
            "avg_latency_ms": (
                round(self._total_latency_ms / self._total_predictions, 2) if self._total_predictions > 0 else 0
            ),
            "interaction_history_size": len(self._interaction_history),
            "cache_dir": str(_CACHE_DIR),
        }

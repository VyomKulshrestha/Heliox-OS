"""Intent Predictor — JARVIS mode boost via neural command disambiguation.

Feature 3 of the TRIBE v2 integration. Uses brain-response predictions to
improve voice/gesture → command mapping accuracy. Reduces false positives
by scoring each candidate intent against predicted neural patterns.

How it works:
  1. When MultimodalFusionEngine produces multiple candidate intents,
     this module scores each candidate by predicted neural affinity.
  2. TRIBE v2 predicts which stimulus (command description) would elicit
     the strongest neural response — i.e., which command the user
     most likely "meant" given their cognitive context.
  3. The highest-affinity candidate is boosted; low-affinity candidates
     are suppressed.

Integration point: MultimodalFusionEngine in multimodal/fusion.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pilot.cognitive.tribe_engine import TribeEngine

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.intent_predictor")


# ── Configuration ──

# Minimum neural affinity to boost an intent
MIN_AFFINITY_BOOST = 0.45

# Maximum affinity bonus added to confidence score
MAX_AFFINITY_BONUS = 0.25

# If top candidate's affinity is this much higher than second, auto-select
AUTO_SELECT_DELTA = 0.3

# Gesture → typical associated voice commands (for cross-modal validation)
GESTURE_VOICE_AFFINITY: dict[str, list[str]] = {
    "thumbs_up": ["yes", "confirm", "do it", "ok", "approve"],
    "thumbs_down": ["no", "cancel", "deny", "stop", "reject"],
    "fist": ["execute", "run", "do it", "go", "start"],
    "open_palm": ["stop", "cancel", "wait", "hold", "pause"],
    "point_up": ["scroll", "up", "top", "look", "check"],
    "peace": ["voice", "toggle", "switch", "mode", "mic"],
    "ok": ["good", "fine", "acknowledge", "got it", "sure"],
    "finger_gun": ["screenshot", "capture", "snap", "photo", "shoot"],
    "call_me": ["settings", "config", "options", "preferences", "setup"],
    "rock": ["system", "info", "status", "diagnostics", "health"],
    "swipe_left": ["back", "previous", "left", "return", "undo"],
    "swipe_right": ["next", "forward", "right", "continue", "more"],
}


@dataclass
class IntentCandidate:
    """A candidate intent being evaluated."""

    command: str
    source: str = ""  # "voice", "gesture", "fused"
    original_confidence: float = 0.0
    neural_affinity: float = 0.0
    boosted_confidence: float = 0.0
    cross_modal_match: bool = False
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "source": self.source,
            "original_confidence": round(self.original_confidence, 3),
            "neural_affinity": round(self.neural_affinity, 3),
            "boosted_confidence": round(self.boosted_confidence, 3),
            "cross_modal_match": self.cross_modal_match,
            "selected": self.selected,
        }


@dataclass
class IntentPrediction:
    """Result of intent prediction with neural disambiguation."""

    selected_command: str
    candidates: list[IntentCandidate]
    disambiguation_used: bool = False
    auto_selected: bool = False
    confidence_boost: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_command": self.selected_command,
            "candidates": [c.to_dict() for c in self.candidates],
            "disambiguation_used": self.disambiguation_used,
            "auto_selected": self.auto_selected,
            "confidence_boost": round(self.confidence_boost, 3),
        }


class IntentPredictor:
    """Disambiguates voice/gesture intents using TRIBE v2 predictions.

    Integrated into the MultimodalFusionEngine to boost JARVIS-mode
    accuracy. When multiple interpretations of a voice+gesture combo
    exist, this predictor uses neural affinity scoring to pick the
    most likely intended command.
    """

    def __init__(self, tribe_engine: TribeEngine | None = None) -> None:
        self._tribe = tribe_engine or TribeEngine.get_instance()
        self._enabled = True

        # Stats
        self._total_predictions = 0
        self._total_disambiguations = 0
        self._total_auto_selects = 0
        self._prediction_history: list[IntentPrediction] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self, enabled: bool | None = None) -> bool:
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = not self._enabled
        return self._enabled

    async def predict_best_intent(
        self,
        voice_transcript: str,
        gesture_name: str,
        gesture_confidence: float,
        voice_confidence: float,
        candidate_commands: list[str] | None = None,
    ) -> IntentPrediction:
        """Score and select the best intent from voice+gesture input.

        Args:
            voice_transcript: The transcribed voice command.
            gesture_name: The detected gesture name.
            gesture_confidence: Confidence score of gesture detection.
            voice_confidence: Confidence score of voice transcription.
            candidate_commands: Optional list of candidate command strings.
                If not provided, generates candidates from voice + gesture.

        Returns:
            IntentPrediction with the selected command and scored candidates.
        """
        self._total_predictions += 1

        # Generate candidates if not provided
        if not candidate_commands:
            candidate_commands = self._generate_candidates(voice_transcript, gesture_name)

        if not candidate_commands:
            return IntentPrediction(
                selected_command=voice_transcript or gesture_name,
                candidates=[],
            )

        # Build candidate objects
        candidates: list[IntentCandidate] = []
        for cmd in candidate_commands:
            source = "fused"
            if cmd == voice_transcript:
                source = "voice"
            elif cmd in (
                gesture_name,
                GESTURE_VOICE_AFFINITY.get(gesture_name, [""])[0] if gesture_name else "",
            ):
                source = "gesture"

            orig_conf = voice_confidence if source == "voice" else gesture_confidence
            cross_match = self._check_cross_modal(cmd, gesture_name, voice_transcript)

            candidates.append(
                IntentCandidate(
                    command=cmd,
                    source=source,
                    original_confidence=orig_conf,
                    cross_modal_match=cross_match,
                )
            )

        if not self._enabled:
            # No prediction — just pick highest original confidence
            best = max(candidates, key=lambda c: c.original_confidence)
            best.selected = True
            best.boosted_confidence = best.original_confidence
            return IntentPrediction(
                selected_command=best.command,
                candidates=candidates,
            )

        # Score with TRIBE v2 neural affinity
        tribe_candidates = [
            {
                "command": c.command,
                "description": c.command,
                "gesture_match": gesture_name,
            }
            for c in candidates
        ]

        scored = await self._tribe.predict_intent_affinity(
            candidates=tribe_candidates,
            voice_transcript=voice_transcript,
            gesture_name=gesture_name,
        )

        # Map scores back to candidates
        for i, c in enumerate(candidates):
            if i < len(scored):
                c.neural_affinity = scored[i].get("neural_affinity", 0.3)
            else:
                c.neural_affinity = 0.3

            # Compute boosted confidence
            affinity_bonus = 0.0
            if c.neural_affinity > MIN_AFFINITY_BOOST:
                affinity_bonus = (c.neural_affinity - MIN_AFFINITY_BOOST) * MAX_AFFINITY_BONUS
            if c.cross_modal_match:
                affinity_bonus += 0.1  # Cross-modal consistency bonus

            c.boosted_confidence = min(1.0, c.original_confidence + affinity_bonus)

        # Sort by boosted confidence
        candidates.sort(key=lambda c: c.boosted_confidence, reverse=True)

        # Determine if we auto-select
        auto_selected = False
        disambiguation_used = False

        if len(candidates) >= 2:
            delta = candidates[0].boosted_confidence - candidates[1].boosted_confidence
            if delta > AUTO_SELECT_DELTA:
                auto_selected = True
                self._total_auto_selects += 1

            if candidates[0].neural_affinity != candidates[1].neural_affinity:
                disambiguation_used = True
                self._total_disambiguations += 1

        # Mark winner
        candidates[0].selected = True
        confidence_boost = candidates[0].boosted_confidence - candidates[0].original_confidence

        prediction = IntentPrediction(
            selected_command=candidates[0].command,
            candidates=candidates,
            disambiguation_used=disambiguation_used,
            auto_selected=auto_selected,
            confidence_boost=confidence_boost,
        )

        # Keep history
        self._prediction_history.append(prediction)
        if len(self._prediction_history) > 30:
            self._prediction_history = self._prediction_history[-30:]

        if disambiguation_used:
            logger.info(
                "JARVIS boost: selected '%s' (affinity=%.2f) over '%s' (affinity=%.2f)",
                candidates[0].command,
                candidates[0].neural_affinity,
                candidates[1].command if len(candidates) > 1 else "n/a",
                candidates[1].neural_affinity if len(candidates) > 1 else 0,
            )

        return prediction

    def _generate_candidates(
        self,
        voice: str,
        gesture: str,
    ) -> list[str]:
        """Generate candidate commands from voice and gesture inputs."""
        candidates = []

        if voice:
            candidates.append(voice)

        if gesture:
            # Add gesture standalone commands
            from pilot.multimodal.fusion import GESTURE_STANDALONE_COMMANDS

            standalone = GESTURE_STANDALONE_COMMANDS.get(gesture)
            if standalone and standalone not in candidates:
                candidates.append(standalone)

            # Add gesture-associated voice commands
            associated = GESTURE_VOICE_AFFINITY.get(gesture, [])
            for cmd in associated[:2]:
                if cmd not in candidates:
                    candidates.append(cmd)

        return candidates

    def _check_cross_modal(
        self,
        command: str,
        gesture: str,
        voice: str,
    ) -> bool:
        """Check if a command has cross-modal support."""
        if not gesture or not voice:
            return False

        cmd_lower = command.lower()
        associated = GESTURE_VOICE_AFFINITY.get(gesture, [])

        for keyword in associated:
            if keyword in cmd_lower or keyword in voice.lower():
                return True

        return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "total_predictions": self._total_predictions,
            "total_disambiguations": self._total_disambiguations,
            "total_auto_selects": self._total_auto_selects,
            "disambiguation_rate": (
                round(self._total_disambiguations / self._total_predictions, 3) if self._total_predictions > 0 else 0.0
            ),
            "recent_predictions": [p.to_dict() for p in self._prediction_history[-3:]],
        }

"""Advisory lexical diagnostics for multimodal intent candidates.

The current scorer is lexical overlap only. It can explain candidate
similarity, but is not a learned intent model and cannot select or authorize
actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pilot.cognitive.cognitive_engine import CognitiveEngine


@dataclass
class IntentCandidate:
    """A candidate intent being evaluated."""

    command: str
    source: str = ""  # "voice", "gesture", "fused"
    original_confidence: float = 0.0
    heuristic_affinity: float = 0.0
    boosted_confidence: float = 0.0
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "source": self.source,
            "original_confidence": round(self.original_confidence, 3),
            "heuristic_affinity": round(self.heuristic_affinity, 3),
            "boosted_confidence": round(self.boosted_confidence, 3),
            "selected": self.selected,
        }


@dataclass
class IntentPrediction:
    """Advisory lexical scores; the fused command remains authoritative."""

    selected_command: str
    candidates: list[IntentCandidate]
    disambiguation_used: bool = False
    auto_selected: bool = False
    confidence_boost: float = 0.0
    scoring_mode: str = "lexical_heuristic"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_command": self.selected_command,
            "candidates": [c.to_dict() for c in self.candidates],
            "disambiguation_used": self.disambiguation_used,
            "auto_selected": self.auto_selected,
            "confidence_boost": round(self.confidence_boost, 3),
            "scoring_mode": self.scoring_mode,
            "action_authority": False,
        }


class IntentPredictor:
    """Report lexical candidate similarity without changing the fused intent."""

    def __init__(self, cognitive_engine: CognitiveEngine | None = None) -> None:
        self._engine = cognitive_engine or CognitiveEngine.get_instance()
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
        """Score candidate diagnostics while preserving fusion's first choice.

        Args:
            voice_transcript: The transcribed voice command.
            gesture_name: The detected gesture name.
            gesture_confidence: Confidence score of gesture detection.
            voice_confidence: Confidence score of voice transcription.
            candidate_commands: Ordered candidates. The first is the canonical
                fusion result and remains selected by this advisory fallback.

        Returns:
            IntentPrediction with the canonical command and advisory scores.
        """
        self._total_predictions += 1

        # Generate candidates if not provided, then preserve caller ordering.
        if not candidate_commands:
            candidate_commands = self._generate_candidates(voice_transcript, gesture_name)
        candidate_commands = list(
            dict.fromkeys(command for command in candidate_commands if isinstance(command, str) and command.strip())
        )

        if not candidate_commands:
            return IntentPrediction(
                selected_command=voice_transcript or gesture_name,
                candidates=[],
            )

        # Build candidate objects. Only exact source matches inherit source
        # confidence; the heuristic must not invent confidence for a paraphrase.
        candidates: list[IntentCandidate] = []
        for cmd in candidate_commands:
            source = "fused"
            if cmd == voice_transcript:
                source = "voice"
            elif cmd == gesture_name:
                source = "gesture"

            orig_conf = voice_confidence if source == "voice" else gesture_confidence if source == "gesture" else 0.0

            candidates.append(
                IntentCandidate(
                    command=cmd,
                    source=source,
                    original_confidence=orig_conf,
                )
            )

        primary_command = candidate_commands[0]

        if not self._enabled:
            candidates[0].selected = True
            for candidate in candidates:
                candidate.boosted_confidence = candidate.original_confidence
            return IntentPrediction(
                selected_command=primary_command,
                candidates=candidates,
            )

        # Score diagnostics only. Gesture semantics have already been resolved
        # by the fusion engine; no gesture-to-phrase vocabulary is inferred here.
        engine_candidates = [
            {
                "command": c.command,
                "description": c.command,
            }
            for c in candidates
        ]

        scored = await self._engine.predict_intent_affinity(
            candidates=engine_candidates,
            voice_transcript=voice_transcript,
            gesture_name=gesture_name,
        )

        # The cognitive engine sorts diagnostics; join by candidate content,
        # never by list position. Candidate text is deduplicated above.
        scores_by_command = {item.get("command"): item.get("heuristic_affinity", 0.0) for item in scored}
        for candidate in candidates:
            candidate.heuristic_affinity = float(scores_by_command.get(candidate.command, 0.0))
            candidate.boosted_confidence = candidate.original_confidence
            candidate.selected = candidate.command == primary_command

        prediction = IntentPrediction(
            selected_command=primary_command,
            candidates=candidates,
        )

        # Keep history
        self._prediction_history.append(prediction)
        if len(self._prediction_history) > 30:
            self._prediction_history = self._prediction_history[-30:]

        return prediction

    def _generate_candidates(
        self,
        voice: str,
        gesture: str,
    ) -> list[str]:
        """Use only spoken text; the fusion engine resolves gesture semantics."""
        return [voice] if voice else []

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "scoring_mode": "lexical_heuristic",
            "action_authority": False,
            "total_predictions": self._total_predictions,
            "total_disambiguations": self._total_disambiguations,
            "total_auto_selects": self._total_auto_selects,
            "disambiguation_rate": (
                round(self._total_disambiguations / self._total_predictions, 3) if self._total_predictions > 0 else 0.0
            ),
            "recent_predictions": [p.to_dict() for p in self._prediction_history[-3:]],
        }

from __future__ import annotations

import pytest

from pilot.cognitive.intent_predictor import IntentCandidate, IntentPrediction, IntentPredictor
from pilot.multimodal.fusion import FusedIntent, InputEvent, ModalityType, MultimodalFusionEngine


class SortedDiagnosticEngine:
    async def predict_intent_affinity(self, candidates, voice_transcript="", gesture_name=""):
        scores = {
            "launch selected item": 0.1,
            "open report": 0.9,
        }
        return sorted(
            [{**candidate, "heuristic_affinity": scores[candidate["command"]]} for candidate in candidates],
            key=lambda candidate: candidate["heuristic_affinity"],
            reverse=True,
        )


@pytest.mark.asyncio
async def test_lexical_scores_are_joined_by_candidate_but_do_not_select() -> None:
    predictor = IntentPredictor(SortedDiagnosticEngine())

    result = await predictor.predict_best_intent(
        voice_transcript="open report",
        gesture_name="custom_gesture",
        gesture_confidence=0.98,
        voice_confidence=0.91,
        candidate_commands=["launch selected item", "open report"],
    )

    assert result.selected_command == "launch selected item"
    assert [candidate.heuristic_affinity for candidate in result.candidates] == [0.1, 0.9]
    assert [candidate.selected for candidate in result.candidates] == [True, False]
    assert result.auto_selected is False
    assert result.disambiguation_used is False
    assert result.confidence_boost == 0.0
    assert result.to_dict()["action_authority"] is False


def test_unknown_gesture_does_not_invent_voice_candidates() -> None:
    predictor = IntentPredictor(SortedDiagnosticEngine())

    assert predictor._generate_candidates("open a page", "custom_gesture") == ["open a page"]


@pytest.mark.asyncio
async def test_fusion_keeps_canonical_command_and_confidence_when_advisor_disagrees() -> None:
    class DisagreeingAdvisor:
        enabled = True

        async def predict_best_intent(self, **_kwargs):
            return IntentPrediction(
                selected_command="untrusted alternate",
                candidates=[
                    IntentCandidate(
                        command="untrusted alternate",
                        boosted_confidence=1.0,
                        selected=True,
                    )
                ],
                auto_selected=True,
                disambiguation_used=True,
                confidence_boost=0.5,
            )

    fusion = MultimodalFusionEngine()
    fusion._intent_predictor = DisagreeingAdvisor()
    intent = FusedIntent(command="canonical fused command", confidence=0.42)
    voice = InputEvent(
        modality=ModalityType.VOICE,
        transcript="do not perform the alternate",
        voice_confidence=0.9,
    )

    result = await fusion._apply_cognitive_intent(intent, voice=voice, gesture=None)

    assert result.command == "canonical fused command"
    assert result.confidence == 0.42
    assert result.metadata["intent_assessment"]["selected_command"] == "untrusted alternate"

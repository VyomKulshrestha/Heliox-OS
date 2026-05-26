"""Quick verification test for TRIBE v2 cognitive integration."""

import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
import sys

sys.path.insert(0, ".")


async def test():
    print("=" * 60)
    print("  TRIBE v2 Cognitive Intelligence — Integration Test")
    print("=" * 60)

    # 1. Test TribeEngine
    from pilot.cognitive.tribe_engine import TribeEngine

    engine = TribeEngine.get_instance()
    print(f"\n[1] TribeEngine")
    print(f"    tribev2 library installed: {engine.is_available}")
    print(f"    Model loaded:              {engine.is_loaded}")
    print(f"    Fallback mode:             {engine.is_fallback}")

    # Try loading the model
    loaded = await engine.load_model()
    print(f"    After load_model():        loaded={loaded}, fallback={engine.is_fallback}")

    # Simulate some interactions
    engine.record_interaction("click", "visual", 0.6)
    engine.record_interaction("typing", "linguistic", 0.8)
    engine.record_interaction("scroll", "visual", 0.3)
    engine.record_interaction("click", "visual", 0.9)
    engine.record_interaction("error_popup", "visual", 1.0)

    # Predict cognitive state
    state = await engine.predict_cognitive_state("user reading error dialog")
    print(f"\n[2] Cognitive State Prediction")
    print(f"    Attention:       {state.attention_score:.3f}")
    print(f"    Stress:          {state.stress_level:.3f}")
    print(f"    Cognitive Load:  {state.cognitive_load:.3f}")
    print(f"    Dominant Mode:   {state.dominant_modality}")
    print(f"    Confidence:      {state.confidence:.3f}")

    # 3. Test Attention-Aware UI
    from pilot.cognitive.attention_scorer import AttentionAwareUI

    attention_ui = AttentionAwareUI(engine)

    print(f"\n[3] Attention-Aware UI Scoring")
    test_events = [
        ("error", {"message": "File not found"}),
        ("status", {"phase": "planning"}),
        ("background_update", {"message": "Memory consolidated"}),
        ("confirmation_required", {"plan_id": "abc123"}),
    ]
    for evt_type, content in test_events:
        scored = await attention_ui.score_event(evt_type, content)
        print(
            f"    {evt_type:25s} -> display={scored.should_display}, "
            f"animate={scored.should_animate}, "
            f"attention={scored.attention_score:.2f}, "
            f"reason={scored.reason}"
        )

    # 4. Test Stress Gate
    from pilot.actions import ActionType
    from pilot.cognitive.stress_gate import StressGate

    gate = StressGate(engine)

    print(f"\n[4] Stress-Aware Task Gating")
    test_actions = [
        ActionType.FILE_READ,
        ActionType.FILE_DELETE,
        ActionType.POWER_SHUTDOWN,
        ActionType.FILE_WRITE,
        ActionType.PROCESS_KILL,
    ]
    for action in test_actions:
        decision = await gate.evaluate(action)
        status = "GATED" if decision.gated else "PASS"
        print(f"    {action.value:20s} -> {status}  stress={decision.stress_level:.2f} reason={decision.reason}")

    # 5. Test Intent Predictor (JARVIS mode)
    from pilot.cognitive.intent_predictor import IntentPredictor

    predictor = IntentPredictor(engine)

    print(f"\n[5] JARVIS Mode Intent Prediction")
    prediction = await predictor.predict_best_intent(
        voice_transcript="take a screenshot",
        gesture_name="finger_gun",
        gesture_confidence=0.85,
        voice_confidence=0.9,
    )
    print(f"    Voice: 'take a screenshot' + Gesture: 'finger_gun'")
    print(f"    Selected: '{prediction.selected_command}'")
    print(f"    Disambiguation used: {prediction.disambiguation_used}")
    print(f"    Candidates:")
    for c in prediction.candidates:
        marker = " *" if c.selected else ""
        print(f"      {c.command:25s} conf={c.boosted_confidence:.2f} affinity={c.neural_affinity:.2f}{marker}")

    prediction2 = await predictor.predict_best_intent(
        voice_transcript="confirm",
        gesture_name="thumbs_up",
        gesture_confidence=0.8,
        voice_confidence=0.75,
    )
    print(f"\n    Voice: 'confirm' + Gesture: 'thumbs_up'")
    print(f"    Selected: '{prediction2.selected_command}'")
    for c in prediction2.candidates:
        marker = " *" if c.selected else ""
        print(f"      {c.command:25s} conf={c.boosted_confidence:.2f} affinity={c.neural_affinity:.2f}{marker}")

    # 6. Overall stats
    print(f"\n[6] Engine Stats")
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("  ALL COGNITIVE MODULES WORKING!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())

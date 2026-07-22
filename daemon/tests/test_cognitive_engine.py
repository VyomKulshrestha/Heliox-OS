"""Tests for pilot.cognitive.cognitive_engine.CognitiveEngine.

Covers the enhanced heuristic engine: recency-decayed weighting (vs. the
old hard 30s cutoff), the three independent signal streams
(record_interaction/record_input_dynamics/record_gaze), the auditable
keyword-derived text signal, confidence scaling with real data richness,
and the Jaccard-based intent-affinity scoring.

Each test constructs its own `CognitiveEngine()` directly (bypassing the
`get_instance()` singleton) so tests never share history state.
"""

from __future__ import annotations

import pytest

from pilot.cognitive.cognitive_engine import CognitiveEngine, _text_signal


class TestNoData:
    @pytest.mark.asyncio
    async def test_no_signal_at_all_returns_low_confidence_default(self):
        engine = CognitiveEngine()
        snap = await engine.predict_cognitive_state()
        assert snap.confidence == 0.2
        assert snap.attention_score == 0.5
        assert snap.stress_level == 0.3
        assert snap.cognitive_load == 0.4

    @pytest.mark.asyncio
    async def test_gaze_only_data_is_not_silently_dropped(self):
        """Regression: the "no data" early-return must consider gaze_weight
        too, not just event/dynamics weight, or gaze-only samples would be
        silently ignored."""
        engine = CognitiveEngine()
        engine.record_gaze("left", confidence=0.9)
        snap = await engine.predict_cognitive_state()
        assert snap.confidence > 0.2


class TestDecayWeighting:
    @pytest.mark.asyncio
    async def test_sample_past_old_hard_cutoff_still_contributes(self):
        """The old implementation hard-filtered to `age < 30`, so a 35s-old
        event would be invisible. The new exp-decay weighting should still
        register it (with reduced influence), not treat it as absent."""
        engine = CognitiveEngine()
        engine.record_interaction("test_event", intensity=1.0)
        engine._interaction_history[-1]["timestamp"] -= 35

        snap = await engine.predict_cognitive_state()

        assert snap.confidence > 0.2
        assert snap.stress_level > 0.0

    @pytest.mark.asyncio
    async def test_recent_sample_weighs_more_than_an_older_one(self):
        """A single sample's *average* intensity doesn't change with age
        (decay cancels out of weighted_intensity/event_weight for one
        sample) -- what decays is event_weight itself, which feeds
        cognitive_load (event frequency) and confidence (data richness).
        That's the actually-decayed part of the estimate."""
        fresh = CognitiveEngine()
        fresh.record_interaction("x", intensity=1.0)

        stale = CognitiveEngine()
        stale.record_interaction("x", intensity=1.0)
        stale._interaction_history[-1]["timestamp"] -= 60

        fresh_snap = await fresh.predict_cognitive_state()
        stale_snap = await stale.predict_cognitive_state()

        assert fresh_snap.cognitive_load > stale_snap.cognitive_load
        assert fresh_snap.confidence > stale_snap.confidence

    @pytest.mark.asyncio
    async def test_sample_past_horizon_is_fully_ignored(self):
        engine = CognitiveEngine()
        engine.record_interaction("x", intensity=1.0)
        engine._interaction_history[-1]["timestamp"] -= 500  # well past _HISTORY_HORIZON_S

        snap = await engine.predict_cognitive_state()

        assert snap.confidence == 0.2  # falls back to the "no data" default


class TestInputDynamics:
    @pytest.mark.asyncio
    async def test_high_keystroke_and_click_rate_drives_up_cognitive_load(self):
        engine = CognitiveEngine()
        engine.record_input_dynamics(keystroke_rate_per_min=120.0, click_rate_per_min=20.0)

        snap = await engine.predict_cognitive_state()

        assert snap.cognitive_load > 0.5
        assert snap.raw_activations["keystroke_rate"] == pytest.approx(120.0, abs=0.5)
        assert snap.raw_activations["click_rate"] == pytest.approx(20.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_negative_rates_are_clamped_to_zero(self):
        engine = CognitiveEngine()
        engine.record_input_dynamics(keystroke_rate_per_min=-5.0, click_rate_per_min=-1.0)

        assert engine._input_dynamics_history[-1]["keystroke_rate"] == 0.0
        assert engine._input_dynamics_history[-1]["click_rate"] == 0.0

    def test_history_is_bounded(self):
        engine = CognitiveEngine()
        for _ in range(engine._max_dynamics_history + 20):
            engine.record_input_dynamics(10.0, 1.0)
        assert len(engine._input_dynamics_history) == engine._max_dynamics_history


class TestGaze:
    @pytest.mark.asyncio
    async def test_off_center_gaze_reduces_attention_vs_centered_gaze(self):
        centered = CognitiveEngine()
        centered.record_gaze("center", confidence=0.9)
        centered_snap = await centered.predict_cognitive_state()

        distracted = CognitiveEngine()
        distracted.record_gaze("left", confidence=0.9)
        distracted_snap = await distracted.predict_cognitive_state()

        assert distracted_snap.attention_score < centered_snap.attention_score
        assert distracted_snap.stress_level > centered_snap.stress_level

    @pytest.mark.asyncio
    async def test_low_confidence_off_center_gaze_has_muted_effect(self):
        confident = CognitiveEngine()
        confident.record_gaze("left", confidence=0.9)
        confident_snap = await confident.predict_cognitive_state()

        unsure = CognitiveEngine()
        unsure.record_gaze("left", confidence=0.1)
        unsure_snap = await unsure.predict_cognitive_state()

        assert unsure_snap.attention_score > confident_snap.attention_score


class TestTextSignal:
    def test_empty_stimulus_has_no_signal(self):
        assert _text_signal("") == (0.0, 0.0)

    def test_stress_keywords_raise_stress_bump(self):
        stress_bump, attention_bump = _text_signal("Critical error: task failed with timeout")
        assert stress_bump > 0.0

    def test_attention_keywords_raise_attention_bump(self):
        stress_bump, attention_bump = _text_signal("Please confirm this alert notification")
        assert attention_bump > 0.0

    def test_signal_is_bounded(self):
        # Every keyword in both tables, repeated -- must still stay bounded.
        loud_text = " ".join(
            [
                "error failed failure critical urgent warning crash exception denied timeout delete danger",
                "click confirm required alert notification",
            ]
            * 5
        )
        stress_bump, attention_bump = _text_signal(loud_text)
        assert 0.0 <= stress_bump <= 0.4
        assert 0.0 <= attention_bump <= 0.3

    @pytest.mark.asyncio
    async def test_stimulus_description_is_actually_used(self):
        """Regression: predict_cognitive_state's stimulus_description used to
        be silently discarded by the heuristic entirely."""
        engine = CognitiveEngine()
        calm_snap = await engine.predict_cognitive_state(stimulus_description="everything is fine")
        stressful_snap = await engine.predict_cognitive_state(
            stimulus_description="critical error: urgent failure detected"
        )
        assert stressful_snap.stress_level > calm_snap.stress_level


class TestConfidenceScaling:
    @pytest.mark.asyncio
    async def test_confidence_grows_with_more_signal_streams(self):
        one_stream = CognitiveEngine()
        one_stream.record_interaction("x", intensity=0.5)
        one_snap = await one_stream.predict_cognitive_state()

        two_streams = CognitiveEngine()
        two_streams.record_interaction("x", intensity=0.5)
        two_streams.record_input_dynamics(60.0, 5.0)
        two_snap = await two_streams.predict_cognitive_state(stimulus_description="status update")

        assert two_snap.confidence > one_snap.confidence

    @pytest.mark.asyncio
    async def test_confidence_never_exceeds_cap(self):
        engine = CognitiveEngine()
        for _ in range(20):
            engine.record_interaction("x", intensity=1.0)
        engine.record_input_dynamics(200.0, 50.0)
        engine.record_gaze("left", confidence=1.0)

        snap = await engine.predict_cognitive_state(stimulus_description="critical urgent error")

        assert snap.confidence <= 0.6


class TestIntentAffinityJaccard:
    @pytest.mark.asyncio
    async def test_exact_match_outscores_verbose_candidate_with_same_word(self):
        engine = CognitiveEngine()
        candidates = [
            {"command": "run", "description": "run"},
            {
                "command": "run_verbose",
                "description": "please run this task for me right now urgently",
            },
        ]

        scored = await engine.predict_intent_affinity(candidates, voice_transcript="run")

        assert scored[0]["command"] == "run"
        assert scored[0]["neural_affinity"] > scored[1]["neural_affinity"]

    @pytest.mark.asyncio
    async def test_multiword_phrase_containment_is_boosted(self):
        engine = CognitiveEngine()
        candidates = [
            {"command": "a", "description": "please run it now for me"},
            {"command": "b", "description": "cancel the operation entirely"},
        ]

        scored = await engine.predict_intent_affinity(candidates, voice_transcript="run it now")

        assert scored[0]["command"] == "a"

    @pytest.mark.asyncio
    async def test_gesture_cross_modal_match_adds_bonus(self):
        engine = CognitiveEngine()
        candidates = [
            {"command": "confirm", "description": "confirm", "gesture_match": "thumbs_up"},
        ]

        without_gesture = await engine.predict_intent_affinity(candidates, voice_transcript="confirm")
        with_gesture = await engine.predict_intent_affinity(
            candidates, voice_transcript="confirm", gesture_name="thumbs_up"
        )

        assert with_gesture[0]["neural_affinity"] >= without_gesture[0]["neural_affinity"]


class TestGetStats:
    def test_stats_report_all_three_history_sizes(self):
        engine = CognitiveEngine()
        engine.record_interaction("x")
        engine.record_input_dynamics(10.0, 1.0)
        engine.record_gaze("left", 0.9)

        stats = engine.get_stats()

        assert stats["interaction_history_size"] == 1
        assert stats["input_dynamics_history_size"] == 1
        assert stats["gaze_history_size"] == 1

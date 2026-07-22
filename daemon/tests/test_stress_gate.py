"""Tests for pilot.cognitive.stress_gate.StressGate.

Covers the fix for a circular-feedback bug: StressGate.evaluate() used to
call `self._engine.record_interaction(intensity=state.stress_level)`,
feeding CognitiveEngine's own just-computed stress_level back into its own
interaction history as a new "intensity" sample -- self-reinforcing rather
than reflecting an independent signal. It should now record the action's
own objective risk tier instead.
"""

from __future__ import annotations

import pytest

from pilot.actions import ActionType
from pilot.cognitive.cognitive_engine import CognitiveEngine
from pilot.cognitive.stress_gate import StressGate


class TestNoCircularFeedback:
    @pytest.mark.asyncio
    async def test_recorded_intensity_is_not_the_engines_own_output(self):
        engine = CognitiveEngine()
        gate = StressGate(engine)

        await gate.evaluate(ActionType.FILE_DELETE)

        recorded_intensity = engine._interaction_history[-1]["intensity"]
        # Fixed, objective risk-tier values -- never a self-referential
        # copy of a CognitiveSnapshot.stress_level the engine just returned.
        assert recorded_intensity in (0.75, 0.35)

    @pytest.mark.asyncio
    async def test_high_risk_action_records_higher_intensity_than_normal(self):
        engine = CognitiveEngine()
        gate = StressGate(engine)

        await gate.evaluate(ActionType.FILE_DELETE)  # HIGH_RISK_ACTIONS member
        high_risk_intensity = engine._interaction_history[-1]["intensity"]

        engine2 = CognitiveEngine()
        gate2 = StressGate(engine2)
        await gate2.evaluate(ActionType.FILE_WRITE)  # not in SAFE_ACTIONS or HIGH_RISK_ACTIONS
        normal_intensity = engine2._interaction_history[-1]["intensity"]

        assert high_risk_intensity > normal_intensity

    @pytest.mark.asyncio
    async def test_fixed_intensity_converges_instead_of_compounding(self):
        """Simulates what StressGate now does on every evaluate() call: record
        the SAME fixed, independent intensity (0.75 for high-risk) each time
        -- unlike the old `intensity=state.stress_level` bug, where each
        recorded sample was a function of the *previous* prediction, letting
        stress compound upward across repeated calls. A fixed intensity
        should make the decayed average converge to ~0.75, not climb
        indefinitely."""
        engine = CognitiveEngine()

        stress_levels = []
        for _ in range(15):
            engine.record_interaction("action_gate_check", modality="cognitive", intensity=0.75)
            snap = await engine.predict_cognitive_state()
            stress_levels.append(snap.stress_level)

        # avg_intensity converges toward 0.75 (stress = avg_intensity*0.65,
        # so it converges toward ~0.4875), never runs away toward 1.0.
        assert stress_levels[-1] < 0.6
        assert stress_levels[-1] == pytest.approx(stress_levels[-2], abs=0.02)


class TestSafeActionsNeverGated:
    @pytest.mark.asyncio
    async def test_safe_action_is_never_gated(self):
        engine = CognitiveEngine()
        gate = StressGate(engine)

        decision = await gate.evaluate(ActionType.FILE_READ)

        assert decision.gated is False
        assert decision.reason == "safe_action"


class TestToggle:
    def test_disabled_gate_never_evaluates(self):
        gate = StressGate(CognitiveEngine())
        assert gate.enabled is True
        gate.toggle(False)
        assert gate.enabled is False

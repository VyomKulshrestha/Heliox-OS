"""Tests for pilot.multimodal.fusion.MultimodalFusionEngine.

No prior test coverage existed for this engine before gaze was added as a
third modality — these cover both the pre-existing voice+gesture behavior
(regression safety, since adding gaze touched several shared methods) and
the new gaze corroboration/passive-ingestion behavior.

Timestamps use real time.time()-based offsets, not arbitrary small
numbers: _prune_buffers() compares event timestamps against a real
wall-clock cutoff (time.time() - window*2), called on every single
ingestion — a synthetic timestamp like `1000.0` would be pruned
immediately since it's nowhere near the real current epoch time.
"""

from __future__ import annotations

import time

import pytest

from pilot.multimodal.fusion import InputEvent, ModalityType, MultimodalFusionEngine


def _voice(
    transcript: str, confidence: float = 0.8, is_final: bool = True, timestamp: float | None = None
) -> InputEvent:
    return InputEvent(
        modality=ModalityType.VOICE,
        transcript=transcript,
        voice_confidence=confidence,
        is_final=is_final,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


def _gesture(name: str, confidence: float = 0.9, timestamp: float | None = None) -> InputEvent:
    return InputEvent(
        modality=ModalityType.GESTURE,
        gesture_name=name,
        gesture_confidence=confidence,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


def _gaze(region: str, confidence: float = 0.7, timestamp: float | None = None) -> InputEvent:
    return InputEvent(
        modality=ModalityType.GAZE,
        gaze_region=region,
        gaze_confidence=confidence,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


class TestVoiceOnly:
    @pytest.mark.asyncio
    async def test_final_voice_produces_voice_only_intent(self):
        engine = MultimodalFusionEngine()
        intent = await engine.on_voice_event(_voice("open settings"))
        assert intent is not None
        assert intent.fusion_type == "voice_only"
        assert intent.command == "open settings"

    @pytest.mark.asyncio
    async def test_non_final_voice_produces_no_intent(self):
        engine = MultimodalFusionEngine()
        intent = await engine.on_voice_event(_voice("open sett", is_final=False))
        assert intent is None

    @pytest.mark.asyncio
    async def test_low_confidence_voice_produces_no_intent(self):
        engine = MultimodalFusionEngine(min_voice_confidence=0.5)
        intent = await engine.on_voice_event(_voice("open settings", confidence=0.2))
        assert intent is None


class TestGestureOnly:
    @pytest.mark.asyncio
    async def test_standalone_gesture_produces_gesture_only_intent(self):
        engine = MultimodalFusionEngine()
        intent = await engine.on_gesture_event(_gesture("thumbs_up"))
        assert intent is not None
        assert intent.fusion_type == "gesture_only"
        assert intent.command == "confirm"

    @pytest.mark.asyncio
    async def test_low_confidence_gesture_produces_no_intent(self):
        engine = MultimodalFusionEngine(min_gesture_confidence=0.8)
        intent = await engine.on_gesture_event(_gesture("thumbs_up", confidence=0.3))
        assert intent is None

    @pytest.mark.asyncio
    async def test_repeated_gesture_within_cooldown_is_suppressed(self):
        engine = MultimodalFusionEngine()
        now = time.time()
        first = await engine.on_gesture_event(_gesture("thumbs_up", timestamp=now))
        second = await engine.on_gesture_event(_gesture("thumbs_up", timestamp=now + 0.1))
        assert first is not None
        assert second is None


class TestVoiceGestureFusion:
    @pytest.mark.asyncio
    async def test_voice_then_gesture_within_window_fuses(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_voice_event(_voice("delete", timestamp=now))
        intent = await engine.on_gesture_event(_gesture("point_up", timestamp=now + 0.5))
        assert intent is not None
        assert intent.fusion_type == "voice_gesture"
        assert intent.voice_component == "delete"
        assert intent.gesture_component == "point_up"
        assert intent.gesture_modifier == "target"

    @pytest.mark.asyncio
    async def test_gesture_outside_window_does_not_fuse(self):
        engine = MultimodalFusionEngine(time_window_ms=500)
        now = time.time()
        await engine.on_voice_event(_voice("delete", timestamp=now))
        intent = await engine.on_gesture_event(_gesture("point_up", timestamp=now + 2.0))
        assert intent is not None
        assert intent.fusion_type == "gesture_only"

    @pytest.mark.asyncio
    async def test_combined_confidence_is_weighted_average(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_voice_event(_voice("delete", confidence=1.0, timestamp=now))
        intent = await engine.on_gesture_event(_gesture("point_up", confidence=1.0, timestamp=now + 0.5))
        assert intent is not None
        assert intent.confidence == pytest.approx(1.0)


class TestGaze:
    @pytest.mark.asyncio
    async def test_gaze_event_never_emits_an_intent_on_its_own(self):
        engine = MultimodalFusionEngine()
        result = await engine.on_gaze_event(_gaze("left"))
        assert result is None  # on_gaze_event returns None (no FusedIntent), unlike voice/gesture

    @pytest.mark.asyncio
    async def test_low_confidence_gaze_is_not_buffered(self):
        engine = MultimodalFusionEngine(min_gaze_confidence=0.5)
        await engine.on_gaze_event(_gaze("left", confidence=0.1))
        assert engine.get_stats()["gaze_buffer_size"] == 0

    @pytest.mark.asyncio
    async def test_recent_gaze_attached_as_metadata_on_voice_gesture_fusion(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_gaze_event(_gaze("left", timestamp=now))
        await engine.on_voice_event(_voice("delete", timestamp=now + 0.2))
        intent = await engine.on_gesture_event(_gesture("point_up", timestamp=now + 0.4))
        assert intent is not None
        assert intent.metadata["gaze_region"] == "left"

    @pytest.mark.asyncio
    async def test_gaze_corroboration_boosts_confidence_for_target_modifier(self):
        # Confidence deliberately left below the 1.0 cap (0.8, not 1.0) so
        # the gaze bonus has headroom to actually show a difference --
        # min(1.0, x + bonus) would clamp identically otherwise.
        now = time.time()

        engine_no_gaze = MultimodalFusionEngine(time_window_ms=1500)
        await engine_no_gaze.on_voice_event(_voice("delete", confidence=0.8, timestamp=now))
        intent_no_gaze = await engine_no_gaze.on_gesture_event(
            _gesture("point_up", confidence=0.8, timestamp=now + 0.2)
        )

        engine_with_gaze = MultimodalFusionEngine(time_window_ms=1500)
        await engine_with_gaze.on_gaze_event(_gaze("left", timestamp=now))
        await engine_with_gaze.on_voice_event(_voice("delete", confidence=0.8, timestamp=now + 0.1))
        intent_with_gaze = await engine_with_gaze.on_gesture_event(
            _gesture("point_up", confidence=0.8, timestamp=now + 0.2)
        )

        assert intent_no_gaze is not None
        assert intent_with_gaze is not None
        assert intent_with_gaze.confidence > intent_no_gaze.confidence

    @pytest.mark.asyncio
    async def test_gaze_does_not_boost_confidence_for_non_target_modifier(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_gaze_event(_gaze("left", timestamp=now))
        await engine.on_voice_event(_voice("confirm", confidence=0.8, timestamp=now + 0.1))
        intent = await engine.on_gesture_event(_gesture("thumbs_up", confidence=0.8, timestamp=now + 0.2))
        assert intent is not None
        expected = 0.8 * 0.6 + 0.8 * 0.4
        assert intent.confidence == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_gaze_center_does_not_boost_confidence(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_gaze_event(_gaze("center", timestamp=now))
        await engine.on_voice_event(_voice("delete", confidence=0.8, timestamp=now + 0.1))
        intent = await engine.on_gesture_event(_gesture("point_up", confidence=0.8, timestamp=now + 0.2))
        assert intent is not None
        expected = 0.8 * 0.6 + 0.8 * 0.4
        assert intent.confidence == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_gaze_metadata_attached_to_voice_only_intent(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_gaze_event(_gaze("up", timestamp=now))
        intent = await engine.on_voice_event(_voice("what is this", timestamp=now + 0.1))
        assert intent is not None
        assert intent.metadata["gaze_region"] == "up"

    @pytest.mark.asyncio
    async def test_gaze_metadata_attached_to_gesture_only_intent(self):
        engine = MultimodalFusionEngine(time_window_ms=1500)
        now = time.time()
        await engine.on_gaze_event(_gaze("right", timestamp=now))
        intent = await engine.on_gesture_event(_gesture("point_up", timestamp=now + 0.1))
        assert intent is not None
        assert intent.metadata["gaze_region"] == "right"

    @pytest.mark.asyncio
    async def test_no_gaze_means_no_gaze_metadata_key(self):
        engine = MultimodalFusionEngine()
        intent = await engine.on_voice_event(_voice("open settings"))
        assert intent is not None
        assert "gaze_region" not in intent.metadata


class TestStats:
    @pytest.mark.asyncio
    async def test_get_stats_reports_all_three_buffer_sizes(self):
        engine = MultimodalFusionEngine()
        await engine.on_gaze_event(_gaze("left"))
        stats = engine.get_stats()
        assert "voice_buffer_size" in stats
        assert "gesture_buffer_size" in stats
        assert "gaze_buffer_size" in stats
        assert stats["gaze_buffer_size"] == 1

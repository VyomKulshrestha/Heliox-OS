from __future__ import annotations

import numpy as np
import pytest

from pilot.system.vad import EndpointEvent, UtteranceEndpointer, frame_rms


def test_frame_rms_of_silence_is_zero():
    silence = np.zeros(160, dtype=np.int16)
    assert frame_rms(silence) == 0.0


def test_frame_rms_of_empty_frame_is_zero():
    assert frame_rms(np.array([], dtype=np.int16)) == 0.0


def test_frame_rms_of_full_scale_tone_is_close_to_one():
    loud = np.full(160, 32767, dtype=np.int16)
    assert frame_rms(loud) == pytest.approx(1.0, abs=1e-3)


def test_frame_rms_scales_with_amplitude():
    quiet = np.full(160, 1000, dtype=np.int16)
    loud = np.full(160, 10000, dtype=np.int16)
    assert frame_rms(loud) > frame_rms(quiet)


class TestUtteranceEndpointer:
    def test_silence_produces_no_events(self):
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=2, silence_frames=3)
        for _ in range(20):
            assert ep.push(0.0) == EndpointEvent.NONE
        assert ep.is_speaking is False

    def test_single_loud_frame_does_not_start_speech(self):
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=3, silence_frames=3)
        assert ep.push(0.5) == EndpointEvent.NONE
        assert ep.push(0.0) == EndpointEvent.NONE  # resets the above-count streak
        assert ep.is_speaking is False

    def test_sustained_speech_triggers_started(self):
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=3, silence_frames=3)
        assert ep.push(0.5) == EndpointEvent.NONE
        assert ep.push(0.5) == EndpointEvent.NONE
        assert ep.push(0.5) == EndpointEvent.STARTED
        assert ep.is_speaking is True

    def test_brief_pause_mid_utterance_does_not_end_it(self):
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=2, silence_frames=5)
        ep.push(0.5)
        ep.push(0.5)  # STARTED
        # Two quiet frames, short of the 5-frame silence_frames threshold.
        assert ep.push(0.0) == EndpointEvent.NONE
        assert ep.push(0.0) == EndpointEvent.NONE
        # Speech resumes -- should still be "speaking", not ended.
        assert ep.push(0.5) == EndpointEvent.NONE
        assert ep.is_speaking is True

    def test_sustained_silence_after_speech_triggers_ended(self):
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=2, silence_frames=3)
        ep.push(0.5)
        ep.push(0.5)  # STARTED
        assert ep.push(0.0) == EndpointEvent.NONE
        assert ep.push(0.0) == EndpointEvent.NONE
        assert ep.push(0.0) == EndpointEvent.ENDED
        assert ep.is_speaking is False

    def test_max_frames_forces_end_even_without_silence(self):
        # silence_frames is set absurdly high so only max_frames can end this.
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=1, silence_frames=100, max_frames=5)
        assert ep.push(0.5) == EndpointEvent.STARTED
        events = [ep.push(0.5) for _ in range(5)]
        # Sustained loud audio forces an end at max_frames, without ever
        # going quiet -- confirmed by MAX_DURATION appearing before any
        # ENDED would have been possible from silence_frames=100.
        assert events[-1] == EndpointEvent.MAX_DURATION
        assert EndpointEvent.ENDED not in events

    def test_reset_clears_in_progress_state(self):
        ep = UtteranceEndpointer(energy_threshold=0.02, start_frames=2, silence_frames=3)
        ep.push(0.5)
        ep.push(0.5)  # STARTED
        ep.reset()
        assert ep.is_speaking is False
        # After reset, needs a fresh full start_frames streak again.
        assert ep.push(0.5) == EndpointEvent.NONE

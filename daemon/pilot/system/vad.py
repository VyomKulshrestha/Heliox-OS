"""Lightweight energy-based voice activity detection (VAD) and utterance
endpointing for the continuous voice listener (see voice.py).

No ML model, no new dependency — a simple RMS-over-threshold classifier
with hysteresis (separate frame counts for confirming speech has STARTED
vs. confirming it has ENDED), so it doesn't chatter on a single loud click
or cut off mid-word during a natural pause between words. Pure/stateful
classes with no sounddevice/hardware dependency, so this is fully
unit-testable without a real microphone — only the actual audio capture in
voice.py's _ContinuousRecorder needs real hardware.

This replaces ContinuousVoiceListener's previous approach of recording a
BLIND fixed-duration window (3s to catch a wake word, 8s for the command
that follows) regardless of how long the person actually spoke — the fixed
window either cuts someone off mid-sentence or wastes time waiting out a
window after they've already finished talking. Endpointing on actual
silence removes both problems and is the mechanical basis for the
barge-in/interrupt behavior in voice.py: the same energy-threshold check
that detects "utterance started" during listening also detects "user
started talking" while Heliox is mid-speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


def frame_rms(frame: np.ndarray) -> float:
    """Root-mean-square energy of one int16 PCM frame, normalized to
    roughly [0, 1] (0 = silence, 1 = full-scale). Returns 0.0 for an empty
    frame rather than raising, since a dropped/short frame from the audio
    callback shouldn't crash the endpointer."""
    if frame.size == 0:
        return 0.0
    normalized = frame.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(normalized * normalized)))


class EndpointEvent(StrEnum):
    NONE = "none"
    STARTED = "started"
    ENDED = "ended"
    MAX_DURATION = "max_duration"


@dataclass
class UtteranceEndpointer:
    """Tracks speech-start/speech-end across a stream of per-frame RMS
    energy values.

    - `start_frames` consecutive above-threshold frames confirm speech has
      STARTED (a single loud click or pop doesn't count).
    - `silence_frames` consecutive below-threshold frames after speech has
      started confirm it has ENDED (a brief pause mid-sentence doesn't end
      the utterance early).
    - `max_frames` is a hard safety cap — if speech never naturally ends
      (e.g. sustained background noise crossing the threshold), the
      utterance is forcibly ended so the mic doesn't stay "open" forever.

    One instance covers one utterance lifecycle; call `reset()` (or use a
    fresh instance) to start tracking the next one.
    """

    energy_threshold: float = 0.02
    start_frames: int = 2
    silence_frames: int = 12
    max_frames: int = 500

    is_speaking: bool = False
    _above_count: int = 0
    _below_count: int = 0
    _frames_since_start: int = 0

    def push(self, rms: float) -> EndpointEvent:
        """Feed one frame's RMS energy; returns the endpoint event (if any)
        this frame triggered."""
        above = rms >= self.energy_threshold

        if not self.is_speaking:
            self._above_count = self._above_count + 1 if above else 0
            if self._above_count >= self.start_frames:
                self.is_speaking = True
                self._above_count = 0
                self._below_count = 0
                self._frames_since_start = 0
                return EndpointEvent.STARTED
            return EndpointEvent.NONE

        # Already speaking — track toward ENDED or MAX_DURATION.
        self._frames_since_start += 1
        self._below_count = self._below_count + 1 if not above else 0

        if self._below_count >= self.silence_frames:
            self.reset()
            return EndpointEvent.ENDED
        if self._frames_since_start >= self.max_frames:
            self.reset()
            return EndpointEvent.MAX_DURATION
        return EndpointEvent.NONE

    def reset(self) -> None:
        self.is_speaking = False
        self._above_count = 0
        self._below_count = 0
        self._frames_since_start = 0

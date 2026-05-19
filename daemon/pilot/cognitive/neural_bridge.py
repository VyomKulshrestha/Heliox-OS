"""Multi-Modal Neural Bridge — unifies webcam, audio, and input dynamics into a neural workspace.

This module implements the third revolutionary feature:
- Webcam eye-tracking for attention mapping
- Audio tone analysis for emotional state detection
- Keyboard/mouse dynamics for engagement detection
- Unified "neural workspace" that predicts user needs
- Fuses all modalities into a single cognitive context

Architecture:
  Eye Tracker ─┬─→ Attention Map ─┐
  Audio Analyzer ─→ Emotion State ──┼─→ Neural Workspace ──→ Prediction Engine
  Input Dynamics ─→ Engagement ─────┘
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.neural_bridge")

# ── Configuration ──

EYE_TRACKING_ENABLED = True  # Requires webcam access (stub for now)
AUDIO_TONE_ENABLED = True  # Requires microphone access
INPUT_DYNAMICS_ENABLED = True  # Always available

# Thresholds
EYE_MOVEMENT_SMOOTHING = 0.3
GAZE_SHIFT_THRESHOLD = 0.4  # Significant gaze shift
DWELL_TIME_THRESHOLD_SECONDS = 3.0

AUDIO_ENERGY_THRESHOLD = 0.3  # Voice activity detection
EMOTION_CONFIDENCE_THRESHOLD = 0.6

KEYBOARD_CLUSTER_THRESHOLD_MS = 500  # Clustering keystrokes
MOUSE_VELOCITY_SMOOTHING = 0.2
IDLE_MOUSE_THRESHOLD_SECONDS = 5.0

# Prediction weights
WEIGHT_EYE = 0.35
WEIGHT_AUDIO = 0.30
WEIGHT_INPUT = 0.35


# ── Data Structures ──


@dataclass
class GazeData:
    """Eye tracking data."""

    x: float = 0.5  # Normalized 0-1
    y: float = 0.5
    dwell_time: float = 0.0
    is_on_screen: bool = True
    shift_magnitude: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioToneData:
    """Audio tone and emotional state."""

    energy: float = 0.0  # 0-1
    pitch_avg: float = 0.0
    speech_likelihood: float = 0.0
    emotion: str = "neutral"  # neutral, happy, sad, angry, stressed, excited
    emotion_confidence: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class InputDynamicsData:
    """Keyboard and mouse dynamics."""

    keystroke_rate: float = 0.0  # keys per minute
    mouse_movement: float = 0.0  # pixels per second
    idle_time: float = 0.0  # seconds since last input
    click_frequency: float = 0.0
    scroll_frequency: float = 0.0
    engagement_score: float = 0.5  # Derived engagement
    timestamp: float = field(default_factory=time.time)


@dataclass
class NeuralWorkspace:
    """Unified neural workspace combining all modalities."""

    gaze: GazeData | None = None
    audio: AudioToneData | None = None
    input_dynamics: InputDynamicsData | None = None

    # Fused metrics
    attention_focus: float = 0.5  # Where is user looking
    cognitive_state: str = "unknown"  # derived state
    emotional_state: str = "neutral"
    engagement_level: float = 0.5

    # Predictions
    predicted_need: str = ""  # "help", "break", "continue", "switch"
    prediction_confidence: float = 0.0

    timestamp: float = field(default_factory=time.time)


# ── Eye Tracking Module ──


class EyeTracker:
    """Webcam-based eye tracking for attention mapping."""

    def __init__(self):
        self._enabled = EYE_TRACKING_ENABLED
        self._last_gaze: GazeData | None = None
        self._gaze_history: list[GazeData] = []
        self._dwell_start: float = 0.0
        self._dwell_region: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def update_gaze(self, x: float, y: float) -> GazeData:
        """Update gaze position from webcam."""
        if not self._enabled:
            return GazeData()

        # Apply smoothing
        if self._last_gaze:
            x = EYE_MOVEMENT_SMOOTHING * x + (1 - EYE_MOVEMENT_SMOOTHING) * self._last_gaze.x
            y = EYE_MOVEMENT_SMOOTHING * y + (1 - EYE_MOVEMENT_SMOOTHING) * self._last_gaze.y

        # Calculate shift
        shift = 0.0
        if self._last_gaze:
            shift = ((x - self._last_gaze.x) ** 2 + (y - self._last_gaze.y) ** 2) ** 0.5

        # Update dwell time
        in_region = self._is_in_dwell_region(x, y)
        now = time.time()

        if in_region and self._dwell_start > 0:
            dwell = now - self._dwell_start
        else:
            self._dwell_start = now if in_region else 0.0
            dwell = 0.0

        gaze = GazeData(
            x=x,
            y=y,
            dwell_time=dwell,
            is_on_screen=True,
            shift_magnitude=shift,
        )

        self._last_gaze = gaze
        self._gaze_history.append(gaze)

        if len(self._gaze_history) > 100:
            self._gaze_history = self._gaze_history[-100:]

        return gaze

    def _is_in_dwell_region(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self._dwell_region
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def set_dwell_region(self, x: float, y: float, w: float, h: float) -> None:
        self._dwell_region = (x, y, w, h)

    def get_attention_map(self) -> dict[str, float]:
        """Generate attention heat map from gaze history."""
        if not self._gaze_history:
            return {"center": 0.5, "periphery": 0.3, "corners": 0.2}

        # Simple zone analysis
        center_count = sum(1 for g in self._gaze_history if 0.3 <= g.x <= 0.7 and 0.3 <= g.y <= 0.7)
        total = len(self._gaze_history)

        return {
            "center": center_count / total,
            "periphery": (total - center_count) / total * 0.5,
            "corners": (total - center_count) / total * 0.5,
        }


# ── Audio Tone Module ──


class AudioToneAnalyzer:
    """Microphone-based audio analysis for emotional state."""

    def __init__(self):
        self._enabled = AUDIO_TONE_ENABLED
        self._audio_history: list[AudioToneData] = []
        self._last_voice_time: float = 0.0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def analyze_tone(self, audio_chunk: bytes | None = None) -> AudioToneData:
        """Analyze audio chunk for emotional state."""
        if not self._enabled or audio_chunk is None:
            return AudioToneData()

        # Placeholder: In production, would use a real audio analysis model
        # For now, return simulated data based on basic heuristics
        energy = self._estimate_energy(audio_chunk)

        # Simulate emotion detection
        if energy > 0.7:
            emotion = "excited"
            confidence = 0.6
        elif energy > 0.5:
            emotion = "happy"
            confidence = 0.5
        elif energy < 0.2:
            emotion = "calm"
            confidence = 0.7
        else:
            emotion = "neutral"
            confidence = 0.6

        audio = AudioToneData(
            energy=energy,
            pitch_avg=0.5,
            speech_likelihood=energy,
            emotion=emotion,
            emotion_confidence=confidence,
            volume=energy,
        )

        self._audio_history.append(audio)

        if len(self._audio_history) > 100:
            self._audio_history = self._audio_history[-100:]

        if energy > AUDIO_ENERGY_THRESHOLD:
            self._last_voice_time = time.time()

        return audio

    def _estimate_energy(self, audio_chunk: bytes) -> float:
        """Estimate audio energy from chunk."""
        # Simple RMS-like calculation
        if not audio_chunk:
            return 0.0
        import struct

        try:
            samples = struct.unpack(f"{len(audio_chunk) // 2}h", audio_chunk)
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            return min(1.0, rms / 32768.0 * 2)
        except Exception:
            return 0.5

    def get_emotion_trend(self) -> tuple[str, float]:
        """Get dominant emotion over recent history."""
        if not self._audio_history:
            return "neutral", 0.0

        recent = self._audio_history[-10:]
        emotions = [a.emotion for a in recent]

        # Most common
        emotion = max(set(emotions), key=emotions.count)
        confidence = sum(1 for e in recent if e.emotion == emotion) / len(recent)

        return emotion, confidence


# ── Input Dynamics Module ──


class InputDynamicsMonitor:
    """Keyboard/mouse dynamics for engagement detection."""

    def __init__(self):
        self._enabled = INPUT_DYNAMICS_ENABLED
        self._keystrokes: list[float] = []
        self._mouse_movements: list[tuple[float, float, float]] = []  # x, y, timestamp
        self._last_input_time: float = time.time()
        self._click_times: list[float] = []
        self._scroll_times: list[float] = []

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def record_keystroke(self, timestamp: float | None = None) -> None:
        """Record a keystroke."""
        if not self._enabled:
            return
        t = timestamp or time.time()
        self._keystrokes.append(t)
        self._last_input_time = t
        self._cleanup_old_events()

    def record_mouse_move(self, x: float, y: float, timestamp: float | None = None) -> None:
        """Record mouse movement."""
        if not self._enabled:
            return
        t = timestamp or time.time()
        self._mouse_movements.append((x, y, t))
        self._last_input_time = t
        self._cleanup_old_events()

    def record_click(self, timestamp: float | None = None) -> None:
        """Record a mouse click."""
        if not self._enabled:
            return
        t = timestamp or time.time()
        self._click_times.append(t)
        self._last_input_time = t

    def record_scroll(self, timestamp: float | None = None) -> None:
        """Record a scroll event."""
        if not self._enabled:
            return
        t = timestamp or time.time()
        self._scroll_times.append(t)
        self._last_input_time = t

    def _cleanup_old_events(self) -> None:
        """Remove events older than 60 seconds."""
        now = time.time()
        cutoff = now - 60

        self._keystrokes = [t for t in self._keystrokes if t > cutoff]
        self._mouse_movements = [(x, y, t) for x, y, t in self._mouse_movements if t > cutoff]
        self._click_times = [t for t in self._click_times if t > cutoff]
        self._scroll_times = [t for t in self._scroll_times if t > cutoff]

    def get_dynamics(self) -> InputDynamicsData:
        """Get current input dynamics metrics."""
        now = time.time()

        # Keystroke rate (keys per minute in last 60s)
        recent_keys = [t for t in self._keystrokes if now - t < 60]
        keystroke_rate = len(recent_keys) if recent_keys else 0.0

        # Mouse velocity
        recent_mouse = [(x, y, t) for x, y, t in self._mouse_movements if now - t < 10]
        if len(recent_mouse) >= 2:
            total_dist = sum(
                (
                    (recent_mouse[i][0] - recent_mouse[i - 1][0]) ** 2
                    + (recent_mouse[i][1] - recent_mouse[i - 1][1]) ** 2
                )
                ** 0.5
                for i in range(1, len(recent_mouse))
            )
            time_span = recent_mouse[-1][2] - recent_mouse[0][2]
            mouse_velocity = total_dist / time_span if time_span > 0 else 0.0
        else:
            mouse_velocity = 0.0

        # Idle time
        idle_time = now - self._last_input_time

        # Click frequency
        recent_clicks = [t for t in self._click_times if now - t < 60]
        click_frequency = len(recent_clicks)

        # Scroll frequency
        recent_scrolls = [t for t in self._scroll_times if now - t < 60]
        scroll_frequency = len(recent_scrolls)

        # Engagement score (derived)
        engagement = self._calculate_engagement(keystroke_rate, mouse_velocity, idle_time, click_frequency)

        return InputDynamicsData(
            keystroke_rate=keystroke_rate,
            mouse_movement=mouse_velocity,
            idle_time=idle_time,
            click_frequency=click_frequency,
            scroll_frequency=scroll_frequency,
            engagement_score=engagement,
        )

    def _calculate_engagement(
        self,
        keystroke_rate: float,
        mouse_velocity: float,
        idle_time: float,
        click_frequency: float,
    ) -> float:
        """Calculate engagement score from input metrics."""
        # High engagement: active typing, mouse movement, low idle time
        keystroke_score = min(1.0, keystroke_rate / 60.0)  # 60 keys/min = max
        mouse_score = min(1.0, mouse_velocity / 500.0)  # 500 px/s = max
        idle_score = max(0.0, 1.0 - idle_time / 30.0)  # 30s idle = 0
        click_score = min(1.0, click_frequency / 10.0)  # 10 clicks/min = max

        return keystroke_score * 0.3 + mouse_score * 0.2 + idle_score * 0.3 + click_score * 0.2


# ── Neural Bridge Core ──


class NeuralBridge:
    """Unified multi-modal neural workspace."""

    def __init__(self):
        self._eye = EyeTracker()
        self._audio = AudioToneAnalyzer()
        self._input = InputDynamicsMonitor()

        self._workspace = NeuralWorkspace()
        self._prediction_history: list[dict[str, Any]] = []

    # ── Update Methods ──

    def update_from_webcam(self, gaze_x: float, gaze_y: float) -> None:
        """Update gaze data from webcam."""
        self._workspace.gaze = self._eye.update_gaze(gaze_x, gaze_y)

    def update_from_audio(self, audio_chunk: bytes | None = None) -> None:
        """Update audio tone data."""
        self._workspace.audio = self._audio.analyze_tone(audio_chunk)

    def update_from_input(self) -> None:
        """Update input dynamics from current state."""
        self._workspace.input_dynamics = self._input.get_dynamics()

    # ── Recording Methods (for external integration) ──

    def record_keystroke(self) -> None:
        """Record a keystroke event."""
        self._input.record_keystroke()

    def record_mouse_move(self, x: float, y: float) -> None:
        """Record a mouse movement."""
        self._input.record_mouse_move(x, y)

    def record_click(self) -> None:
        """Record a mouse click."""
        self._input.record_click()

    def record_scroll(self) -> None:
        """Record a scroll event."""
        self._input.record_scroll()

    # ── Fusion & Prediction ──

    def compute_workspace(self) -> NeuralWorkspace:
        """Compute unified neural workspace from all modalities."""
        now = time.time()

        # Get individual states
        gaze = self._workspace.gaze
        audio = self._workspace.audio
        input_dyn = self._workspace.input_dynamics

        # Calculate fused metrics
        attention_focus = 0.5
        if gaze:
            # Focus on center of screen
            attention_focus = 1.0 - ((gaze.x - 0.5) ** 2 + (gaze.y - 0.5) ** 2) ** 0.5 / 0.7

        # Cognitive state derivation
        cognitive_state = "focused"
        if input_dyn:
            if input_dyn.idle_time > IDLE_MOUSE_THRESHOLD_SECONDS:
                cognitive_state = "idle"
            elif input_dyn.engagement_score > 0.7:
                cognitive_state = "highly_engaged"
            elif input_dyn.engagement_score < 0.3:
                cognitive_state = "disengaged"

        # Emotional state
        emotional_state = "neutral"
        if audio:
            emotional_state = audio.emotion

        # Engagement level
        engagement = 0.5
        if input_dyn:
            engagement = input_dyn.engagement_score
        if audio and audio.energy > 0.3:
            engagement = min(1.0, engagement + 0.1)

        # Predict user need
        predicted_need, confidence = self._predict_user_need(
            attention_focus, cognitive_state, emotional_state, engagement
        )

        # Update workspace
        self._workspace.attention_focus = attention_focus
        self._workspace.cognitive_state = cognitive_state
        self._workspace.emotional_state = emotional_state
        self._workspace.engagement_level = engagement
        self._workspace.predicted_need = predicted_need
        self._workspace.prediction_confidence = confidence
        self._workspace.timestamp = now

        return self._workspace

    def _predict_user_need(
        self,
        attention_focus: float,
        cognitive_state: str,
        emotional_state: str,
        engagement: float,
    ) -> tuple[str, float]:
        """Predict what the user needs based on workspace state."""

        # High stress + low engagement = need help
        if emotional_state in ("stressed", "angry") and engagement < 0.5:
            return "help", 0.75

        # High engagement + high focus = continue
        if engagement > 0.7 and attention_focus > 0.7:
            return "continue", 0.8

        # Idle for long time = might need switch
        if cognitive_state == "idle":
            return "switch", 0.6

        # Low engagement = might need break
        if engagement < 0.3:
            return "break", 0.6

        # Emotional distress
        if emotional_state in ("sad", "frustrated"):
            return "support", 0.7

        return "continue", 0.5

    # ── Getters ──

    def get_workspace(self) -> NeuralWorkspace:
        return self._workspace

    def get_attention_map(self) -> dict[str, float]:
        return self._eye.get_attention_map()

    def get_emotion_trend(self) -> tuple[str, float]:
        return self._audio.get_emotion_trend()

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "eye_tracking_enabled": self._eye.is_enabled,
            "audio_analysis_enabled": self._audio.is_enabled,
            "input_dynamics_enabled": self._input.is_enabled,
            "workspace": {
                "cognitive_state": self._workspace.cognitive_state,
                "emotional_state": self._workspace.emotional_state,
                "engagement_level": round(self._workspace.engagement_level, 2),
                "predicted_need": self._workspace.predicted_need,
            },
        }

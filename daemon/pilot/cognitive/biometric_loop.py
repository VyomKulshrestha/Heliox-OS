"""Adaptive Biometric Learning Loop — tracks patterns over weeks, learns cognitive fingerprints.

This module implements the first revolutionary feature:
- Track time-of-day productivity patterns
- Detect stress cycles and energy levels
- Build personalized cognitive fingerprints
- Predict optimal interaction times
- Closed-loop feedback: user responses refine predictions

Architecture:
  Weekly Pattern Analysis → Cognitive Fingerprint → Interaction Optimizer → User Feedback → Refinement
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.biometric_loop")

# ── Configuration ──

_PATTERN_WINDOW_DAYS = 7  # Look back 7 days for patterns
_MIN_SAMPLES_FOR_PATTERN = 20  # Need at least 20 data points to establish a pattern
_CONFIDENCE_DECAY = 0.95  # Older predictions fade in confidence
_HOURS_TO_ANALYZE = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Hours to track
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# ── Data Structures ──


@dataclass
class HourlyPattern:
    """Productivity pattern for a specific hour."""

    hour: int
    avg_attention: float = 0.5
    avg_stress: float = 0.3
    avg_load: float = 0.4
    sample_count: int = 0
    confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class CognitiveFingerprint:
    """Personalized cognitive fingerprint for a user."""

    user_id: str
    weekly_patterns: dict[int, HourlyPattern] = field(default_factory=dict)
    weekday_patterns: dict[str, HourlyPattern] = field(default_factory=dict)
    optimal_interaction_hours: list[int] = field(default_factory=list)
    peak_productivity_hours: list[int] = field(default_factory=list)
    recovery_hours: list[int] = field(default_factory=list)
    avg_cognitive_baseline: float = 0.4
    stress_sensitivity: float = 0.5  # How quickly stress accumulates
    last_sync: float = field(default_factory=time.time)
    total_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "optimal_interaction_hours": self.optimal_interaction_hours,
            "peak_productivity_hours": self.peak_productivity_hours,
            "recovery_hours": self.recovery_hours,
            "avg_cognitive_baseline": round(self.avg_cognitive_baseline, 3),
            "stress_sensitivity": round(self.stress_sensitivity, 3),
            "total_samples": self.total_samples,
            "last_sync": datetime.fromtimestamp(self.last_sync).isoformat(),
        }


@dataclass
class InteractionRecommendation:
    """Recommendation for optimal interaction."""

    recommended: bool
    interaction_type: str  # "proactive", "reactive", "wait"
    confidence: float
    reason: str
    suggested_action: str = ""
    optimal_time_estimate: str = ""


# ── Storage ──


class BiometricStore:
    """Persists biometric data to disk."""

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or (Path.home() / ".cache" / "heliox" / "biometric")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _get_fingerprint_path(self, user_id: str) -> Path:
        return self._data_dir / f"fingerprint_{user_id}.json"

    def _get_samples_path(self, user_id: str) -> Path:
        return self._data_dir / f"samples_{user_id}.json"

    def load_fingerprint(self, user_id: str) -> CognitiveFingerprint | None:
        path = self._get_fingerprint_path(user_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Convert hourly patterns
            patterns = {}
            for h, p in data.get("weekly_patterns", {}).items():
                patterns[int(h)] = HourlyPattern(**p)
            data["weekly_patterns"] = patterns
            # Weekday patterns
            wd_patterns = {}
            for d, p in data.get("weekday_patterns", {}).items():
                wd_patterns[d] = HourlyPattern(**p)
            data["weekday_patterns"] = wd_patterns
            return CognitiveFingerprint(**data)
        except Exception as e:
            logger.warning("Failed to load fingerprint: %s", e)
            return None

    def save_fingerprint(self, fp: CognitiveFingerprint) -> None:
        path = self._get_fingerprint_path(fp.user_id)
        data = {
            "user_id": fp.user_id,
            "weekly_patterns": {
                h: {
                    "hour": p.hour,
                    "avg_attention": p.avg_attention,
                    "avg_stress": p.avg_stress,
                    "avg_load": p.avg_load,
                    "sample_count": p.sample_count,
                    "confidence": p.confidence,
                    "last_updated": p.last_updated,
                }
                for h, p in fp.weekly_patterns.items()
            },
            "weekday_patterns": {
                d: {
                    "hour": p.hour,
                    "avg_attention": p.avg_attention,
                    "avg_stress": p.avg_stress,
                    "avg_load": p.avg_load,
                    "sample_count": p.sample_count,
                    "confidence": p.confidence,
                    "last_updated": p.last_updated,
                }
                for d, p in fp.weekday_patterns.items()
            },
            "optimal_interaction_hours": fp.optimal_interaction_hours,
            "peak_productivity_hours": fp.peak_productivity_hours,
            "recovery_hours": fp.recovery_hours,
            "avg_cognitive_baseline": fp.avg_cognitive_baseline,
            "stress_sensitivity": fp.stress_sensitivity,
            "last_sync": fp.last_sync,
            "total_samples": fp.total_samples,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_samples(self, user_id: str) -> list[dict[str, Any]]:
        path = self._get_samples_path(user_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_samples(self, user_id: str, samples: list[dict[str, Any]]) -> None:
        path = self._get_samples_path(user_id)
        # Keep last 10000 samples
        samples = samples[-10000:]
        path.write_text(json.dumps(samples, indent=2), encoding="utf-8")


# ── Core Engine ──


class BiometricLearningLoop:
    """Adaptive biometric learning loop that tracks patterns and refines predictions."""

    def __init__(self, user_id: str = "default", store: BiometricStore | None = None):
        self._user_id = user_id
        self._store = store or BiometricStore()
        self._fingerprint = self._load_or_create_fingerprint()
        self._samples: list[dict[str, Any]] = self._store.load_samples(user_id)

        # Real-time feedback buffer
        self._pending_feedback: list[dict[str, Any]] = []
        self._feedback_weights: dict[str, float] = {
            "ignored": -0.1,
            "accepted": 0.15,
            "modified": 0.05,
            "dismissed": -0.15,
        }

    def _load_or_create_fingerprint(self) -> CognitiveFingerprint:
        fp = self._store.load_fingerprint(self._user_id)
        if fp:
            return fp
        return CognitiveFingerprint(user_id=self._user_id)

    # ── Data Recording ──

    def record_cognitive_sample(
        self,
        attention: float,
        stress: float,
        load: float,
        context: str = "",
    ) -> None:
        """Record a cognitive state sample with timestamp."""
        hour = datetime.now().hour
        weekday = _WEEKDAYS[datetime.now().weekday()]

        sample = {
            "timestamp": time.time(),
            "hour": hour,
            "weekday": weekday,
            "attention": attention,
            "stress": stress,
            "load": load,
            "context": context,
        }

        self._samples.append(sample)

        # Update patterns incrementally
        self._update_hourly_pattern(hour, attention, stress, load)
        self._update_weekday_pattern(weekday, hour, attention, stress, load)

        # Recalculate fingerprint periodically
        if len(self._samples) % 10 == 0:
            self._recalculate_fingerprint()

        # Persist
        self._store.save_samples(self._user_id, self._samples)
        self._store.save_fingerprint(self._fingerprint)

    def _update_hourly_pattern(self, hour: int, attention: float, stress: float, load: float) -> None:
        """Incrementally update hourly pattern with exponential moving average."""
        if hour not in self._fingerprint.weekly_patterns:
            self._fingerprint.weekly_patterns[hour] = HourlyPattern(hour=hour)

        p = self._fingerprint.weekly_patterns[hour]
        alpha = 0.1  # Learning rate

        p.avg_attention = alpha * attention + (1 - alpha) * p.avg_attention
        p.avg_stress = alpha * stress + (1 - alpha) * p.avg_stress
        p.avg_load = alpha * load + (1 - alpha) * p.avg_load
        p.sample_count += 1
        p.confidence = min(1.0, p.sample_count / _MIN_SAMPLES_FOR_PATTERN)
        p.last_updated = time.time()

    def _update_weekday_pattern(self, weekday: str, hour: int, attention: float, stress: float, load: float) -> None:
        """Update weekday-specific patterns."""
        key = f"{weekday}_{hour}"
        if key not in self._fingerprint.weekday_patterns:
            self._fingerprint.weekday_patterns[key] = HourlyPattern(hour=hour)

        p = self._fingerprint.weekday_patterns[key]
        alpha = 0.1

        p.avg_attention = alpha * attention + (1 - alpha) * p.avg_attention
        p.avg_stress = alpha * stress + (1 - alpha) * p.avg_stress
        p.avg_load = alpha * load + (1 - alpha) * p.avg_load
        p.sample_count += 1
        p.confidence = min(1.0, p.sample_count / (_MIN_SAMPLES_FOR_PATTERN // 2))
        p.last_updated = time.time()

    def _recalculate_fingerprint(self) -> None:
        """Recalculate optimal hours, peak hours, and recovery hours."""
        patterns = self._fingerprint.weekly_patterns

        if len(patterns) < 3:
            return

        # Find optimal interaction hours (high attention, low stress)
        optimal = []
        peak = []
        recovery = []

        for hour, p in patterns.items():
            if p.confidence < 0.3:
                continue
            # Optimal: high attention, low stress, moderate load
            if p.avg_attention > 0.6 and p.avg_stress < 0.4 and p.avg_load < 0.6:
                optimal.append((hour, p.avg_attention - p.avg_stress))
            # Peak productivity: very high attention
            if p.avg_attention > 0.75:
                peak.append((hour, p.avg_attention))
            # Recovery: low load, low stress
            if p.avg_load < 0.4 and p.avg_stress < 0.3:
                recovery.append((hour, 1.0 - p.avg_load))

        self._fingerprint.optimal_interaction_hours = [
            h for h, _ in sorted(optimal, key=lambda x: x[1], reverse=True)[:4]
        ]
        self._fingerprint.peak_productivity_hours = [h for h, _ in sorted(peak, key=lambda x: x[1], reverse=True)[:3]]
        self._fingerprint.recovery_hours = [h for h, _ in sorted(recovery, key=lambda x: x[1], reverse=True)[:3]]

        # Calculate baseline and sensitivity
        total_attention = sum(p.avg_attention for p in patterns.values()) / len(patterns)
        total_stress = sum(p.avg_stress for p in patterns.values()) / len(patterns)
        self._fingerprint.avg_cognitive_baseline = total_attention
        self._fingerprint.stress_sensitivity = min(1.0, total_stress / 0.5)

        self._fingerprint.total_samples = len(self._samples)
        self._fingerprint.last_sync = time.time()

        logger.info(
            "Biometric fingerprint updated: optimal_hours=%s, peak_hours=%s",
            self._fingerprint.optimal_interaction_hours,
            self._fingerprint.peak_productivity_hours,
        )

    # ── Feedback Loop ──

    def record_interaction_feedback(
        self,
        interaction_type: str,
        user_response: str,  # "ignored", "accepted", "modified", "dismissed"
        suggested_action: str = "",
    ) -> None:
        """Record user response to refine future predictions."""
        weight = self._feedback_weights.get(user_response, 0.0)

        # Adjust the confidence of optimal hours based on feedback
        if interaction_type == "proactive" and user_response == "ignored":
            # Reduce confidence for proactive suggestions
            for h in self._fingerprint.optimal_interaction_hours:
                if h in self._fingerprint.weekly_patterns:
                    self._fingerprint.weekly_patterns[h].confidence *= 0.9

        elif interaction_type == "proactive" and user_response == "accepted":
            # Increase confidence for accepted suggestions
            current_hour = datetime.now().hour
            if current_hour in self._fingerprint.weekly_patterns:
                self._fingerprint.weekly_patterns[current_hour].confidence = min(
                    1.0, self._fingerprint.weekly_patterns[current_hour].confidence + 0.1
                )

        self._store.save_fingerprint(self._fingerprint)

        logger.info(
            "Feedback recorded: type=%s, response=%s, weight=%.2f",
            interaction_type,
            user_response,
            weight,
        )

    # ── Prediction API ──

    def get_interaction_recommendation(self) -> InteractionRecommendation:
        """Get recommendation for whether to interact now."""
        current_hour = datetime.now().hour
        weekday = _WEEKDAYS[datetime.now().weekday()]

        # Check hourly pattern
        hour_key = f"{weekday}_{current_hour}"
        pattern = self._fingerprint.weekday_patterns.get(hour_key) or self._fingerprint.weekly_patterns.get(
            current_hour
        )

        if not pattern or pattern.confidence < 0.3:
            return InteractionRecommendation(
                recommended=True,
                interaction_type="reactive",
                confidence=0.3,
                reason="Insufficient data for pattern prediction",
            )

        attention = pattern.avg_attention
        stress = pattern.avg_stress
        load = pattern.avg_load

        # Decision logic
        if current_hour in self._fingerprint.optimal_interaction_hours:
            if stress < 0.4 and load < 0.7:
                return InteractionRecommendation(
                    recommended=True,
                    interaction_type="proactive",
                    confidence=pattern.confidence,
                    reason=f"Optimal interaction time (hour {current_hour})",
                    suggested_action="Proactive assistance appropriate",
                )

        if current_hour in self._fingerprint.recovery_hours:
            return InteractionRecommendation(
                recommended=False,
                interaction_type="wait",
                confidence=pattern.confidence,
                reason="User is in recovery mode",
                optimal_time_estimate=f"Try again in {60 - datetime.now().minute} minutes",
            )

        if load > 0.8 or stress > 0.7:
            return InteractionRecommendation(
                recommended=False,
                interaction_type="wait",
                confidence=pattern.confidence,
                reason="High cognitive load detected",
                optimal_time_estimate="Wait for cognitive load to decrease",
            )

        # Default: reactive mode
        return InteractionRecommendation(
            recommended=True,
            interaction_type="reactive",
            confidence=pattern.confidence,
            reason="Normal operational state",
        )

    def get_optimal_window(self, minutes_ahead: int = 60) -> list[tuple[int, float]]:
        """Get optimal interaction windows for the next N minutes."""
        now = datetime.now()
        windows = []

        for offset in range(0, minutes_ahead, 15):
            check_time = now + timedelta(minutes=offset)
            hour = check_time.hour

            pattern = self._fingerprint.weekly_patterns.get(hour)
            if pattern and pattern.confidence > 0.3:
                score = pattern.avg_attention - pattern.avg_stress * 0.5 - pattern.avg_load * 0.3
                windows.append((hour, score))

        return sorted(windows, key=lambda x: x[1], reverse=True)[:4]

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "user_id": self._user_id,
            "total_samples": len(self._samples),
            "fingerprint": self._fingerprint.to_dict(),
            "optimal_hours": self._fingerprint.optimal_interaction_hours,
            "peak_hours": self._fingerprint.peak_productivity_hours,
            "recovery_hours": self._fingerprint.recovery_hours,
            "patterns_loaded": len(self._fingerprint.weekly_patterns),
        }

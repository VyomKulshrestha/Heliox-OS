"""Ambient Intelligence Mode — proactive suggestions based on cognitive state.

This module implements the second revolutionary feature:
- Monitors cognitive state continuously
- Detects patterns that suggest user needs help
- Proactively suggests actions BEFORE the user asks
- Context-aware: considers time, task complexity, stress levels
- "What-If" predictions: anticipates next steps

Architecture:
  Cognitive Monitor → Pattern Detector → Need Predictor → Proactive Suggestion Engine → User Response
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine

logger = logging.getLogger("pilot.cognitive.ambient_intelligence")

# ── Configuration ──

# Thresholds for proactive intervention
STRESS_ESCALATION_THRESHOLD = 0.6  # Stress rising over time
TASK_DURATION_THRESHOLD_MINUTES = 120  # 2 hours on single task
MONOTONY_THRESHOLD = 0.8  # Same app/window for too long
LOAD_SPIKE_THRESHOLD = 0.75  # Sudden cognitive load spike

# Timing
IDLE_DETECTION_SECONDS = 30.0  # No interaction = idle
SUGGESTION_COOLDOWN_SECONDS = 300.0  # 5 min between suggestions
MAX_SUGGESTIONS_PER_HOUR = 4

# Context windows for pattern detection
CONTEXT_WINDOW_MINUTES = 15  # Look back 15 minutes for patterns
TASK_CHANGE_THRESHOLD = 0.4  # Significant change in task focus


# ── Data Structures ──


@dataclass
class CognitiveTrend:
    """Trend analysis of cognitive state over time."""

    direction: str  # "rising", "falling", "stable"
    slope_attention: float = 0.0
    slope_stress: float = 0.0
    slope_load: float = 0.0
    confidence: float = 0.0
    duration_minutes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "slope_attention": self.slope_attention,
            "slope_stress": self.slope_stress,
            "slope_load": self.slope_load,
            "confidence": self.confidence,
            "duration_minutes": self.duration_minutes,
        }


@dataclass
class ProactiveSuggestion:
    """A proactive suggestion to present to the user."""

    suggestion_id: str
    suggestion_type: str  # "break", "help", "simplify", "delegate", "reminder", "context"
    message: str
    confidence: float
    urgency: str  # "low", "medium", "high"
    action_options: list[str] = field(default_factory=list)
    context_data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    dismissed: bool = False
    accepted: bool = False


@dataclass
class AmbientState:
    """Current ambient intelligence state."""

    is_active: bool = True
    last_suggestion_time: float = 0.0
    suggestions_today: int = 0
    current_task_duration_minutes: float = 0.0
    monotony_score: float = 0.0
    last_task_change: float = field(default_factory=time.time)
    active_app: str = ""
    cognitive_trend: CognitiveTrend | None = None


# ── Suggestion Templates ──

SUGGESTION_TEMPLATES = {
    "break": [
        "You've been focused for a while. Want to take a short break?",
        "Your stress level has been rising. Consider taking 5 minutes to reset?",
        "You've been on this task for {duration} minutes. A brief break might help.",
    ],
    "help": [
        "This seems complex. Would you like me to help break it down?",
        "I can handle some of this work if you'd like to delegate.",
        "This task has multiple steps. Want me to automate some of it?",
    ],
    "simplify": [
        "There might be a simpler way to do this. Interested?",
        "Your cognitive load is high. Want me to simplify the interface?",
        "I can reduce visual noise to help you focus. Should I?",
    ],
    "delegate": [
        "This is repetitive. I can handle similar tasks automatically.",
        "Want me to schedule this for later when things are calmer?",
        "I can take care of this in the background while you continue.",
    ],
    "reminder": [
        "You've been working on this for {duration} minutes. Don't forget to take breaks!",
        "Remember to stay hydrated and stretch occasionally.",
        "You've reached a good pause point. Want to save your progress?",
    ],
    "context": [
        "I notice you've switched contexts. Here's a quick summary of where you left off.",
        "You've been in this app for a while. Need any relevant information from earlier?",
        "Based on your recent work, here's some context that might help.",
    ],
    "celebration": [
        "Great progress! You've completed {count} tasks today.",
        "Nice work! Your focus has been excellent lately.",
        "You're on a roll! Keep it up.",
    ],
}


# ── Core Engine ──


class AmbientIntelligenceEngine:
    """Ambient intelligence that proactively assists based on cognitive state."""

    def __init__(self, biometric_loop: Any | None = None):
        self._biometric = biometric_loop
        self._state = AmbientState()
        self._history: list[dict[str, Any]] = []
        self._max_history = 200
        self._last_date = datetime.now().date()

        # Callback for when suggestions are accepted/dismissed
        self._on_suggestion: Callable[[ProactiveSuggestion, str], Coroutine] | None = None

        # Task tracking
        self._task_start_time: float = 0.0
        self._task_contexts: list[dict[str, Any]] = []
        self._current_app: str = ""

    def set_suggestion_handler(self, handler: Callable[[ProactiveSuggestion, str], Coroutine]) -> None:
        self._on_suggestion = handler

    # ── Cognitive State Monitoring ──

    async def update_cognitive_state(
        self,
        attention: float,
        stress: float,
        load: float,
        app_name: str = "",
    ) -> None:
        """Update ambient state with current cognitive metrics."""
        now = time.time()

        # Check if day changed
        current_date = datetime.now().date()
        if current_date != self._last_date:
            self.reset_daily_counters()
            self._last_date = current_date

        # Update cognitive trend
        self._update_trend(attention, stress, load)

        # Track app changes for monotony detection
        if app_name and app_name != self._state.active_app:
            self._state.last_task_change = now
            self._state.active_app = app_name
            self._state.monotony_score = 0.0
        elif app_name:
            # Increase monotony score over time
            time_on_task = (now - self._state.last_task_change) / 60.0
            self._state.monotony_score = min(1.0, time_on_task / 60.0)  # Max at 60 min

        # Update task duration
        if self._task_start_time == 0.0:
            self._task_start_time = now
        self._state.current_task_duration_minutes = (now - self._task_start_time) / 60.0

        # Record in history for pattern analysis
        self._history.append(
            {
                "timestamp": now,
                "attention": attention,
                "stress": stress,
                "load": load,
                "app": app_name,
            }
        )

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Check if we should generate a suggestion
        await self._check_for_suggestions(attention, stress, load)

    def _update_trend(self, attention: float, stress: float, load: float) -> None:
        """Calculate cognitive trend over recent history."""
        if len(self._history) < 5:
            self._state.cognitive_trend = None
            return

        recent = self._history[-10:]
        n = len(recent)

        # Simple linear regression for slope
        x = list(range(n))
        avg_attention = sum(e["attention"] for e in recent) / n
        avg_stress = sum(e["stress"] for e in recent) / n
        avg_load = sum(e["load"] for e in recent) / n

        # Calculate slopes
        def calc_slope(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            mean_y = sum(values) / n
            mean_x = (n - 1) / 2.0
            num = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
            den = sum((x[i] - mean_x) ** 2 for i in range(n))
            return num / den if den != 0 else 0.0

        slope_att = calc_slope([e["attention"] for e in recent])
        slope_stress = calc_slope([e["stress"] for e in recent])
        slope_load = calc_slope([e["load"] for e in recent])

        # Determine direction
        avg_recent = sum(e["stress"] for e in recent[-3:]) / 3
        avg_older = sum(e["stress"] for e in recent[:3]) / 3

        if abs(slope_stress) < 0.01:
            direction = "stable"
        elif slope_stress > 0:
            direction = "rising"
        else:
            direction = "falling"

        # Confidence based on consistency
        std_stress = (sum((e["stress"] - avg_stress) ** 2 for e in recent) / n) ** 0.5
        confidence = max(0.0, 1.0 - std_stress)

        self._state.cognitive_trend = CognitiveTrend(
            direction=direction,
            slope_attention=slope_att,
            slope_stress=slope_stress,
            slope_load=slope_load,
            confidence=confidence,
            duration_minutes=self._state.current_task_duration_minutes,
        )

    # ── Suggestion Generation ──

    async def _check_for_suggestions(
        self,
        attention: float,
        stress: float,
        load: float,
    ) -> None:
        """Check if conditions warrant a proactive suggestion."""
        now = time.time()

        # Check cooldown
        if now - self._state.last_suggestion_time < SUGGESTION_COOLDOWN_SECONDS:
            return

        # Check daily limit
        if self._state.suggestions_today >= MAX_SUGGESTIONS_PER_HOUR:
            return

        # Generate suggestion based on conditions
        suggestion = self._evaluate_conditions(attention, stress, load)

        if suggestion:
            self._state.last_suggestion_time = now
            self._state.suggestions_today += 1

            logger.info(
                "Ambient intelligence suggestion: type=%s, message='%s'",
                suggestion.suggestion_type,
                suggestion.message,
            )

            # Execute callback
            if self._on_suggestion:
                await self._on_suggestion(suggestion, "presented")

    def _evaluate_conditions(
        self,
        attention: float,
        stress: float,
        load: float,
    ) -> ProactiveSuggestion | None:
        """Evaluate current conditions to determine if a suggestion is warranted."""

        trend = self._state.cognitive_trend

        # Condition 1: Stress escalation
        if trend and trend.direction == "rising" and trend.slope_stress > STRESS_ESCALATION_THRESHOLD:
            return self._create_suggestion(
                "break",
                f"Your stress level has been rising over the last few minutes. Take a moment to breathe?",
                urgency="high",
                action_options=["Take 5 min break", "Continue but slow down", "Dismiss"],
            )

        # Condition 2: Long task duration
        if self._state.current_task_duration_minutes > TASK_DURATION_THRESHOLD_MINUTES:
            duration = int(self._state.current_task_duration_minutes)
            return self._create_suggestion(
                "break",
                f"You've been working on this task for {duration} minutes. Want to take a break?",
                urgency="medium",
                action_options=["Take a break", "Continue working", "Save progress"],
                context_data={"task_duration": duration},
            )

        # Condition 3: High cognitive load
        if load > LOAD_SPIKE_THRESHOLD:
            return self._create_suggestion(
                "simplify",
                "Your cognitive load is very high right now. Want me to simplify the interface?",
                urgency="high",
                action_options=["Simplify UI", "Show only essentials", "Dismiss"],
                context_data={"current_load": load},
            )

        # Condition 4: Monotony (same app too long)
        if self._state.monotony_score > MONOTONY_THRESHOLD:
            return self._create_suggestion(
                "context",
                f"You've been in {self._state.active_app} for a while. Need any context from earlier?",
                urgency="low",
                action_options=["Show context", "Continue", "Switch task"],
                context_data={"app": self._state.active_app, "monotony": self._state.monotony_score},
            )

        # Condition 5: Low attention (likely distracted)
        if attention < 0.3 and load < 0.4:
            return self._create_suggestion(
                "reminder",
                "I notice your attention seems low. Want me to wait until you're ready?",
                urgency="low",
                action_options=["I'm ready now", "Wait and retry", "Cancel"],
            )

        # Condition 6: Biometric-based (from learning loop)
        if self._biometric:
            rec = self._biometric.get_interaction_recommendation()
            if rec.recommended and rec.interaction_type == "proactive":
                return self._create_suggestion(
                    "context",
                    rec.suggested_action,
                    urgency="low",
                    confidence=rec.confidence,
                )

        return None

    def _create_suggestion(
        self,
        suggestion_type: str,
        message: str,
        urgency: str = "low",
        confidence: float = 0.7,
        action_options: list[str] | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> ProactiveSuggestion:
        return ProactiveSuggestion(
            suggestion_id=f"suggest_{int(time.time() * 1000)}",
            suggestion_type=suggestion_type,
            message=message,
            confidence=confidence,
            urgency=urgency,
            action_options=action_options or [],
            context_data=context_data or {},
        )

    # ── User Response Handling ──

    async def handle_suggestion_response(
        self,
        suggestion: ProactiveSuggestion,
        response: str,  # "accepted", "dismissed", "modified"
    ) -> None:
        """Process user's response to a suggestion."""
        if response == "accepted":
            suggestion.accepted = True
            logger.info("User accepted suggestion: %s", suggestion.suggestion_type)
        elif response == "dismissed":
            suggestion.dismissed = True
            logger.info("User dismissed suggestion: %s", suggestion.suggestion_type)

        # If biometric loop available, provide feedback
        if self._biometric:
            self._biometric.record_interaction_feedback(
                interaction_type="proactive",
                user_response=response,
                suggested_action=suggestion.suggestion_type,
            )

    # ── Reset ──

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of new day)."""
        self._state.suggestions_today = 0
        logger.info("Ambient intelligence daily counters reset")

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "is_active": self._state.is_active,
            "suggestions_today": self._state.suggestions_today,
            "current_task_duration_minutes": round(self._state.current_task_duration_minutes, 1),
            "monotony_score": round(self._state.monotony_score, 2),
            "cognitive_trend": self._state.cognitive_trend.to_dict() if self._state.cognitive_trend else None,
            "history_size": len(self._history),
        }

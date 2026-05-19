"""Attention-Aware UI — Scores UI events by predicted visual attention capture.

Feature 1 of the TRIBE v2 integration. Uses neural attention predictions
to reduce cognitive overload by:
  1. Scoring every UI notification/event by its predicted attention demand
  2. Suppressing low-priority notifications when cognitive load is high
  3. Batching non-urgent updates during focus periods
  4. Broadcasting attention-aware hints to the Svelte frontend

Integration point: WebSocket notification pipeline in server.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from pilot.cognitive.tribe_engine import CognitiveSnapshot, TribeEngine

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.attention_scorer")


# ── Thresholds ──

# When cognitive load exceeds this, suppress low-priority notifications
LOAD_SUPPRESS_THRESHOLD = 0.7

# When attention is below this, batch non-urgent updates
ATTENTION_BATCH_THRESHOLD = 0.4

# Minimum attention score for a notification to pass through during high load
MIN_ATTENTION_PASS = 0.6

# Priority levels
PRIORITY_CRITICAL = "critical"  # Always passes
PRIORITY_HIGH = "high"  # Passes unless extreme overload
PRIORITY_MEDIUM = "medium"  # May be deferred
PRIORITY_LOW = "low"  # Likely deferred during high load


@dataclass
class ScoredUIEvent:
    """A UI event enriched with cognitive attention scoring."""

    event_type: str
    content: dict[str, Any]
    priority: str = PRIORITY_MEDIUM
    attention_score: float = 0.5
    should_display: bool = True
    should_animate: bool = True
    display_duration_ms: int = 5000
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "content": self.content,
            "priority": self.priority,
            "attention_score": round(self.attention_score, 3),
            "should_display": self.should_display,
            "should_animate": self.should_animate,
            "display_duration_ms": self.display_duration_ms,
            "reason": self.reason,
        }


# ── Event type → base priority mapping ──

EVENT_PRIORITIES: dict[str, str] = {
    # Critical — always show
    "error": PRIORITY_CRITICAL,
    "security_alert": PRIORITY_CRITICAL,
    "confirmation_required": PRIORITY_CRITICAL,
    # High — show unless extreme stress
    "execution_complete": PRIORITY_HIGH,
    "plan_ready": PRIORITY_HIGH,
    "verification_result": PRIORITY_HIGH,
    # Medium — may defer
    "status": PRIORITY_MEDIUM,
    "agent_routing": PRIORITY_MEDIUM,
    "reasoning_step": PRIORITY_MEDIUM,
    "multimodal_intent": PRIORITY_MEDIUM,
    # Low — defer during focus
    "background_update": PRIORITY_LOW,
    "plugin_event": PRIORITY_LOW,
    "memory_consolidation": PRIORITY_LOW,
    "screen_vision_update": PRIORITY_LOW,
}


class AttentionAwareUI:
    """Filters and scores UI events based on predicted cognitive state.

    Sits between the server's notification pipeline and the WebSocket
    broadcast to intelligently manage what the user sees and when.
    """

    def __init__(self, tribe_engine: TribeEngine | None = None) -> None:
        self._tribe = tribe_engine or TribeEngine.get_instance()
        self._enabled = True
        self._last_state: CognitiveSnapshot | None = None
        self._state_refresh_interval_s = 5.0
        self._last_state_time = 0.0
        self._deferred_events: list[ScoredUIEvent] = []
        self._flush_task: asyncio.Task | None = None
        self._broadcast_fn: Callable[..., Coroutine] | None = None

        # Stats
        self._total_scored = 0
        self._total_suppressed = 0
        self._total_deferred = 0

    def set_broadcast(self, fn: Callable[..., Coroutine]) -> None:
        self._broadcast_fn = fn

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self, enabled: bool | None = None) -> bool:
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = not self._enabled
        return self._enabled

    # ── Core scoring method ──

    async def score_event(
        self,
        event_type: str,
        content: dict[str, Any],
    ) -> ScoredUIEvent:
        """Score a UI event and determine if/how it should be displayed.

        This is the main entry point. Called from the server's notification
        pipeline BEFORE broadcasting to WebSocket clients.
        """
        self._total_scored += 1

        # Determine base priority
        priority = EVENT_PRIORITIES.get(event_type, PRIORITY_MEDIUM)

        # Critical events always pass through unmodified
        if priority == PRIORITY_CRITICAL:
            return ScoredUIEvent(
                event_type=event_type,
                content=content,
                priority=priority,
                attention_score=1.0,
                should_display=True,
                should_animate=True,
                display_duration_ms=10000,
                reason="critical_always_show",
            )

        if not self._enabled:
            return ScoredUIEvent(
                event_type=event_type,
                content=content,
                priority=priority,
                attention_score=0.5,
                reason="attention_scoring_disabled",
            )

        # Refresh cognitive state if stale
        state = await self._get_current_state()

        # Score the event's attention demand
        attention_score = await self._compute_attention_score(event_type, content)

        # Determine display behavior based on cognitive state
        should_display = True
        should_animate = True
        display_duration_ms = 5000
        reason = "normal"

        if state.cognitive_load > LOAD_SUPPRESS_THRESHOLD:
            if attention_score < MIN_ATTENTION_PASS:
                if priority == PRIORITY_LOW:
                    should_display = False
                    reason = "suppressed_high_load"
                    self._total_suppressed += 1
                elif priority == PRIORITY_MEDIUM:
                    should_animate = False
                    display_duration_ms = 2000
                    reason = "reduced_high_load"

        if state.attention_score < ATTENTION_BATCH_THRESHOLD:
            if priority in (PRIORITY_LOW, PRIORITY_MEDIUM):
                should_animate = False
                display_duration_ms = 3000
                reason = "batched_low_attention"

        # During high stress, reduce animation for everything non-critical
        if state.stress_level > 0.7:
            should_animate = False
            reason = f"{reason}|stress_reduced"

        event = ScoredUIEvent(
            event_type=event_type,
            content=content,
            priority=priority,
            attention_score=attention_score,
            should_display=should_display,
            should_animate=should_animate,
            display_duration_ms=display_duration_ms,
            reason=reason,
        )

        # Track interaction for future predictions
        self._tribe.record_interaction(
            event_type=event_type,
            modality="visual",
            intensity=attention_score,
        )

        return event

    async def _compute_attention_score(
        self,
        event_type: str,
        content: dict[str, Any],
    ) -> float:
        """Compute how much attention this event will demand."""
        # Build a list of UI elements for TRIBE v2 scoring
        elements = [
            {
                "type": event_type,
                "label": str(content.get("message", content.get("phase", event_type))),
                "animated": event_type in ("error", "confirmation_required"),
                "created_at": time.time(),
            }
        ]

        scored = await self._tribe.predict_attention_map(elements)
        if scored:
            return scored[0].get("attention_score", 0.5)
        return 0.5

    async def _get_current_state(self) -> CognitiveSnapshot:
        """Get or refresh the current cognitive state."""
        now = time.time()
        if self._last_state is None or now - self._last_state_time > self._state_refresh_interval_s:
            self._last_state = await self._tribe.predict_cognitive_state()
            self._last_state_time = now
        return self._last_state

    # ── Deferred event flushing ──

    async def flush_deferred(self) -> list[ScoredUIEvent]:
        """Flush all deferred events (called when cognitive load drops)."""
        events = list(self._deferred_events)
        self._deferred_events.clear()
        return events

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "total_scored": self._total_scored,
            "total_suppressed": self._total_suppressed,
            "total_deferred": self._total_deferred,
            "deferred_queue_size": len(self._deferred_events),
            "last_cognitive_state": (self._last_state.to_dict() if self._last_state else None),
        }

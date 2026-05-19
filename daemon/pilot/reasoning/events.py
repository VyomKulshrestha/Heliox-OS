"""Thought Visualization Event System â€” granular reasoning telemetry.

Defines the event schema, emitter, and stream manager for broadcasting
real-time AI reasoning events to the frontend.

Event Flow:
  Agent Runtime â†’ ReasoningEmitter â†’ WebSocket â†’ ThoughtGraph UI

Each reasoning event carries:
  - event_type: Categorized phase (memory, planning, routing, etc.)
  - event_name: Specific event identifier
  - timestamp: When the event was produced
  - stage: Which pipeline stage produced it
  - data: Event-specific payload
  - duration_ms: How long the phase took (for completed events)
  - parent_id: Links sub-events to their parent phase

This enables the frontend to render a live execution graph where nodes
light up, edges animate, and reasoning details stream in real-time.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Coroutine

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.reasoning.events")


# â”€â”€ Event Schema â”€â”€


class ReasoningStage(StrEnum):
    """Pipeline stages in the ReAct loop."""

    USER_INPUT = "user_input"
    MEMORY_RECALL = "memory_recall"
    AGENT_ROUTING = "agent_routing"
    PLANNING = "planning"
    CONFIRMATION = "confirmation"
    ORCHESTRATION = "orchestration"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    REFLECTION = "reflection"
    MEMORY_UPDATE = "memory_update"
    CRITIC_REVIEW = "critic_review"


class EventType(StrEnum):
    """The type/severity of reasoning event."""

    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    PHASE_ERROR = "phase_error"
    THOUGHT = "thought"  # LLM inner reasoning text
    DECISION = "decision"  # A decision point
    DATA = "data"  # Data payload (plan, results, etc.)
    PROGRESS = "progress"  # Progress update (% or step count)
    METRIC = "metric"  # Performance metric


@dataclass
class ReasoningEvent:
    """A single reasoning event in the thought visualization stream."""

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: str = EventType.THOUGHT
    event_name: str = ""
    stage: str = ReasoningStage.USER_INPUT
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0
    data: dict[str, Any] = field(default_factory=dict)
    parent_id: str = ""  # For linking sub-events to parent phases
    sequence: int = 0  # Global ordering

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_name": self.event_name,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 1),
            "data": self.data,
            "parent_id": self.parent_id,
            "sequence": self.sequence,
        }


# â”€â”€ Predefined event names â”€â”€

# Memory stage
MEMORY_SEARCH_STARTED = "memory_search_started"
MEMORY_SEARCH_COMPLETE = "memory_search_complete"
MEMORY_CONTEXT_LOADED = "memory_context_loaded"
MEMORY_STORE_STARTED = "memory_store_started"
MEMORY_STORE_COMPLETE = "memory_store_complete"

# Agent routing stage
ROUTING_ANALYSIS_STARTED = "routing_analysis_started"
ROUTING_AGENTS_ASSIGNED = "routing_agents_assigned"
ROUTING_MULTI_AGENT_DETECTED = "routing_multi_agent_detected"

# Planning stage
PLANNER_STARTED = "planner_started"
PLANNER_LLM_CALL = "planner_llm_call"
PLANNER_GENERATED_PLAN = "planner_generated_plan"
PLANNER_REPLANNING = "planner_replanning"
PLANNER_ERROR = "planner_error"

# Confirmation stage
CONFIRMATION_REQUIRED = "confirmation_required"
CONFIRMATION_APPROVED = "confirmation_approved"
CONFIRMATION_DENIED = "confirmation_denied"
CONFIRMATION_TIMEOUT = "confirmation_timeout"

# Orchestration stage
ORCHESTRATOR_ROUTING = "orchestrator_routing"
ORCHESTRATOR_BATCH_STARTED = "orchestrator_batch_started"
ORCHESTRATOR_BATCH_COMPLETE = "orchestrator_batch_complete"
ORCHESTRATOR_AGENT_DELEGATED = "orchestrator_agent_delegated"

# Execution stage
EXECUTOR_STARTED = "executor_started"
EXECUTOR_ACTION_STARTED = "executor_action_started"
EXECUTOR_ACTION_COMPLETE = "executor_action_complete"
EXECUTOR_ALL_COMPLETE = "executor_all_complete"
EXECUTOR_ERROR = "executor_error"

# Verification stage
VERIFICATION_STARTED = "verification_started"
VERIFICATION_CHECK = "verification_check"
VERIFICATION_PASSED = "verification_passed"
VERIFICATION_FAILED = "verification_failed"

# Reflection stage
REFLECTION_STARTED = "reflection_started"
REFLECTION_INSIGHT = "reflection_insight"
REFLECTION_COMPLETE = "reflection_complete"

# Destructive critic stage â€” Tier 4 safety review
CRITIC_REVIEW_STARTED = "critic_review_started"
CRITIC_REVIEW_APPROVED = "critic_review_approved"
CRITIC_REVIEW_WARNED = "critic_review_warned"
CRITIC_REVIEW_BLOCKED = "critic_review_blocked"


class ReasoningEmitter:
    """Emits structured reasoning events to the WebSocket stream.

    Usage in the ReAct pipeline:
        emitter = ReasoningEmitter(broadcast_fn)

        # Start a phase and get its ID for sub-events
        phase_id = await emitter.phase_start("planning", "planner_started", {...})
        await emitter.thought("planning", "Analyzing user intent...", parent=phase_id)
        await emitter.phase_complete("planning", "planner_generated_plan", {...}, parent=phase_id)
    """

    def __init__(self) -> None:
        self._broadcast_fn: Callable[..., Coroutine] | None = None
        self._sequence = 0
        self._phase_timers: dict[str, float] = {}
        self._session_id = uuid.uuid4().hex[:8]
        self._event_log: list[ReasoningEvent] = []

    def set_broadcast(self, fn: Callable[..., Coroutine]) -> None:
        """Set the WebSocket broadcast function."""
        self._broadcast_fn = fn

    def _next_seq(self) -> int:
        self._sequence += 1
        return self._sequence

    def reset(self) -> None:
        """Reset for a new reasoning session."""
        self._sequence = 0
        self._phase_timers.clear()
        self._session_id = uuid.uuid4().hex[:8]
        self._event_log.clear()

    # â”€â”€ High-level emitters â”€â”€

    async def phase_start(
        self,
        stage: str,
        event_name: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Emit a phase start event and begin timing. Returns the phase event ID."""
        event = ReasoningEvent(
            event_type=EventType.PHASE_START,
            event_name=event_name,
            stage=stage,
            data=data or {},
            sequence=self._next_seq(),
        )
        self._phase_timers[event.event_id] = time.time()
        await self._emit(event)
        return event.event_id

    async def phase_complete(
        self,
        stage: str,
        event_name: str,
        data: dict[str, Any] | None = None,
        parent_id: str = "",
    ) -> None:
        """Emit a phase completion event with duration."""
        start_time = self._phase_timers.pop(parent_id, 0)
        duration = (time.time() - start_time) * 1000 if start_time else 0

        event = ReasoningEvent(
            event_type=EventType.PHASE_COMPLETE,
            event_name=event_name,
            stage=stage,
            duration_ms=duration,
            data=data or {},
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    async def phase_error(
        self,
        stage: str,
        event_name: str,
        error: str,
        parent_id: str = "",
    ) -> None:
        """Emit a phase error event."""
        start_time = self._phase_timers.pop(parent_id, 0)
        duration = (time.time() - start_time) * 1000 if start_time else 0

        event = ReasoningEvent(
            event_type=EventType.PHASE_ERROR,
            event_name=event_name,
            stage=stage,
            duration_ms=duration,
            data={"error": error},
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    async def thought(
        self,
        stage: str,
        text: str,
        parent_id: str = "",
    ) -> None:
        """Emit an inner reasoning thought."""
        event = ReasoningEvent(
            event_type=EventType.THOUGHT,
            event_name="thought",
            stage=stage,
            data={"text": text},
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    async def decision(
        self,
        stage: str,
        description: str,
        options: list[str] | None = None,
        chosen: str = "",
        parent_id: str = "",
    ) -> None:
        """Emit a decision point event."""
        event = ReasoningEvent(
            event_type=EventType.DECISION,
            event_name="decision",
            stage=stage,
            data={
                "description": description,
                "options": options or [],
                "chosen": chosen,
            },
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    async def data_event(
        self,
        stage: str,
        event_name: str,
        data: dict[str, Any],
        parent_id: str = "",
    ) -> None:
        """Emit a data payload event (plan, results, etc.)."""
        event = ReasoningEvent(
            event_type=EventType.DATA,
            event_name=event_name,
            stage=stage,
            data=data,
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    async def progress(
        self,
        stage: str,
        current: int,
        total: int,
        label: str = "",
        parent_id: str = "",
    ) -> None:
        """Emit a progress update."""
        event = ReasoningEvent(
            event_type=EventType.PROGRESS,
            event_name="progress",
            stage=stage,
            data={
                "current": current,
                "total": total,
                "percent": round((current / total) * 100) if total > 0 else 0,
                "label": label,
            },
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    async def metric(
        self,
        stage: str,
        name: str,
        value: float,
        unit: str = "",
        parent_id: str = "",
    ) -> None:
        """Emit a performance metric."""
        event = ReasoningEvent(
            event_type=EventType.METRIC,
            event_name="metric",
            stage=stage,
            data={"name": name, "value": value, "unit": unit},
            parent_id=parent_id,
            sequence=self._next_seq(),
        )
        await self._emit(event)

    # â”€â”€ Internal â”€â”€

    async def _emit(self, event: ReasoningEvent) -> None:
        """Broadcast the event to all connected frontends."""
        self._event_log.append(event)
        # Cap log at 200 events per session
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]

        logger.debug(
            "reasoning_event: seq=%d stage=%s type=%s name=%s",
            event.sequence,
            event.stage,
            event.event_type,
            event.event_name,
        )

        if self._broadcast_fn:
            await self._broadcast_fn("reasoning_event", event.to_dict())

    # â”€â”€ Stats â”€â”€

    def get_session_log(self) -> list[dict[str, Any]]:
        """Return the full event log for the current session."""
        return [e.to_dict() for e in self._event_log]

    def get_stats(self) -> dict[str, Any]:
        """Return emitter statistics."""
        type_counts: dict[str, int] = {}
        stage_counts: dict[str, int] = {}
        for e in self._event_log:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            stage_counts[e.stage] = stage_counts.get(e.stage, 0) + 1

        return {
            "session_id": self._session_id,
            "total_events": len(self._event_log),
            "current_sequence": self._sequence,
            "event_types": type_counts,
            "stage_counts": stage_counts,
        }

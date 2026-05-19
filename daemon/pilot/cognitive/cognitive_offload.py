"""Cognitive Offloading — absorbs cognitive burden by remembering complex workflows.

This module implements the fourth revolutionary feature:
- Monitors cognitive load in real-time
- When load > 80%, automatically surfaces "memory anchors"
- Remembers complex multi-step workflows
- Provides context recall for forgotten details
- Acts as an external cognitive buffer

Architecture:
  Cognitive Load Monitor → Memory Anchor Generator → Context Surface → Workflow Memory → User Recall
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.cognitive_offload")

# ── Configuration ──

LOAD_THRESHOLD_HIGH = 0.80  # Trigger offloading at 80% load
LOAD_THRESHOLD_MEDIUM = 0.60  # Pre-warning at 60%

ANCHOR_RETENTION_HOURS = 24  # Keep anchors for 24 hours
MAX_ANCHORS = 50  # Maximum memory anchors to keep

CONTEXT_SUMMARY_LENGTH = 100  # Characters per context summary

WORKFLOW_MIN_STEPS = 3  # Minimum steps to constitute a workflow


# ── Data Structures ──


@dataclass
class MemoryAnchor:
    """A memory anchor - key information to recall when overloaded."""

    anchor_id: str
    anchor_type: str  # "context", "decision", "action", "result", "workflow"
    title: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0-1
    load_at_creation: float = 0.0
    timestamp: float = field(default_factory=time.time)
    expires_at: float = 0.0
    accessed_count: int = 0
    last_accessed: float = 0.0

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "title": self.title,
            "summary": self.summary[:CONTEXT_SUMMARY_LENGTH],
            "importance": round(self.importance, 2),
            "load_at_creation": round(self.load_at_creation, 2),
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "accessed_count": self.accessed_count,
        }


@dataclass
class Workflow:
    """A multi-step workflow that can be recalled."""

    workflow_id: str
    name: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    completed: bool = False
    total_steps: int = 0
    current_step: int = 0
    created_at: float = field(default_factory=time.time)
    last_step_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "step_count": len(self.steps),
            "completed": self.completed,
            "progress": f"{self.current_step}/{self.total_steps}",
            "context": self.context[:50],
        }


@dataclass
class OffloadState:
    """Current offloading state."""

    current_load: float = 0.0
    is_overloaded: bool = False
    active_anchors: int = 0
    active_workflows: int = 0
    last_offload_time: float = 0.0
    offload_count_today: int = 0


# ── Core Engine ──


class CognitiveOffloader:
    """Cognitive offloading engine that absorbs mental burden."""

    def __init__(self):
        self._anchors: list[MemoryAnchor] = []
        self._workflows: list[Workflow] = []
        self._current_workflow: Workflow | None = None
        self._state = OffloadState()
        self._action_history: list[dict[str, Any]] = []

        # Settings
        self._enabled = True
        self._auto_surface = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self, enabled: bool | None = None) -> bool:
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = not self._enabled
        return self._enabled

    # ── Cognitive Load Monitoring ──

    def update_load(self, load: float) -> bool:
        """Update current cognitive load. Returns True if offloading triggered."""
        self._state.current_load = load
        self._state.is_overloaded = load > LOAD_THRESHOLD_HIGH

        if self._state.is_overloaded and self._auto_surface:
            self._state.last_offload_time = time.time()
            self._state.offload_count_today += 1
            return True

        return False

    # ── Memory Anchor Management ──

    def create_anchor(
        self,
        anchor_type: str,
        title: str,
        summary: str,
        details: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryAnchor:
        """Create a new memory anchor."""
        now = time.time()

        anchor = MemoryAnchor(
            anchor_id=f"anchor_{int(now * 1000)}",
            anchor_type=anchor_type,
            title=title,
            summary=summary,
            details=details or {},
            importance=importance,
            load_at_creation=self._state.current_load,
            timestamp=now,
            expires_at=now + (ANCHOR_RETENTION_HOURS * 3600),
        )

        self._anchors.append(anchor)
        self._cleanup_anchors()

        logger.info("Created memory anchor: %s (%s)", title, anchor_type)

        return anchor

    def get_relevant_anchors(self, max_count: int = 5) -> list[MemoryAnchor]:
        """Get anchors relevant to current cognitive state."""
        if not self._state.is_overloaded:
            return []

        # Sort by importance and recency
        relevant = [a for a in self._anchors if not a.is_expired()]
        relevant.sort(key=lambda a: (a.importance, a.timestamp), reverse=True)

        return relevant[:max_count]

    def recall_anchor(self, anchor_id: str) -> MemoryAnchor | None:
        """Recall a specific anchor."""
        for anchor in self._anchors:
            if anchor.anchor_id == anchor_id:
                anchor.accessed_count += 1
                anchor.last_accessed = time.time()
                return anchor
        return None

    def _cleanup_anchors(self) -> None:
        """Remove expired and excess anchors."""
        # Remove expired
        self._anchors = [a for a in self._anchors if not a.is_expired()]

        # Keep only MAX_ANCHORS most important
        if len(self._anchors) > MAX_ANCHORS:
            self._anchors.sort(key=lambda a: (a.importance, a.timestamp), reverse=True)
            self._anchors = self._anchors[:MAX_ANCHORS]

    # ── Workflow Tracking ──

    def start_workflow(self, name: str, context: str = "") -> str:
        """Start tracking a new multi-step workflow."""
        now = time.time()

        workflow = Workflow(
            workflow_id=f"wf_{int(now * 1000)}",
            name=name,
            context=context,
            created_at=now,
            last_step_time=now,
        )

        self._workflows.append(workflow)
        self._current_workflow = workflow

        logger.info("Started workflow: %s", name)

        return workflow.workflow_id

    def add_workflow_step(self, step_name: str, step_data: dict[str, Any] | None = None) -> None:
        """Add a step to the current workflow."""
        if not self._current_workflow:
            return

        now = time.time()

        step = {
            "step": len(self._current_workflow.steps) + 1,
            "name": step_name,
            "data": step_data or {},
            "timestamp": now,
        }

        self._current_workflow.steps.append(step)
        self._current_workflow.current_step = len(self._current_workflow.steps)
        self._current_workflow.last_step_time = now

        # Create anchor for important steps
        if len(self._current_workflow.steps) == WORKFLOW_MIN_STEPS:
            self.create_anchor(
                anchor_type="workflow",
                title=f"Workflow: {self._current_workflow.name}",
                summary=f"Step {step['step']}: {step_name}",
                details={"workflow_id": self._current_workflow.workflow_id},
                importance=0.6,
            )

    def complete_workflow(self) -> Workflow | None:
        """Mark the current workflow as complete."""
        if not self._current_workflow:
            return None

        self._current_workflow.completed = True
        self._current_workflow.total_steps = len(self._current_workflow.steps)

        # Create completion anchor
        self.create_anchor(
            anchor_type="result",
            title=f"Completed: {self._current_workflow.name}",
            summary=f"Completed {self._current_workflow.total_steps} steps",
            details={"workflow_id": self._current_workflow.workflow_id},
            importance=0.7,
        )

        completed = self._current_workflow
        self._current_workflow = None

        logger.info("Completed workflow: %s", completed.name)

        return completed

    def get_active_workflows(self) -> list[Workflow]:
        """Get all incomplete workflows."""
        return [w for w in self._workflows if not w.completed]

    def resume_workflow(self, workflow_id: str) -> Workflow | None:
        """Resume a specific workflow."""
        for wf in self._workflows:
            if wf.workflow_id == workflow_id and not wf.completed:
                self._current_workflow = wf
                return wf
        return None

    # ── Action History ──

    def record_action(self, action_type: str, details: dict[str, Any]) -> None:
        """Record an action for context."""
        now = time.time()

        self._action_history.append(
            {
                "timestamp": now,
                "action_type": action_type,
                "details": details,
                "load": self._state.current_load,
            }
        )

        # Keep last 100 actions
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]

        # Create anchors for high-load actions
        if self._state.current_load > LOAD_THRESHOLD_MEDIUM:
            self.create_anchor(
                anchor_type="action",
                title=f"Action: {action_type}",
                summary=f"Performed {action_type} during high load",
                details=details,
                importance=0.5,
            )

    def get_recent_context(self, minutes: int = 30) -> list[dict[str, Any]]:
        """Get context from recent actions."""
        cutoff = time.time() - (minutes * 60)
        return [a for a in self._action_history if a["timestamp"] > cutoff]

    # ── Offload Surface ──

    def get_offload_surface(self) -> dict[str, Any]:
        """Get the complete offload surface for the UI."""
        if not self._state.is_overloaded:
            return {"active": False, "anchors": [], "workflows": []}

        anchors = self.get_relevant_anchors(max_count=8)
        workflows = self.get_active_workflows()

        # Build context summary
        recent_context = self.get_recent_context(minutes=60)
        context_summary = self._build_context_summary(recent_context)

        return {
            "active": True,
            "current_load": round(self._state.current_load, 2),
            "anchors": [a.to_dict() for a in anchors],
            "workflows": [w.to_dict() for w in workflows[:3]],
            "context_summary": context_summary,
            "suggestions": self._generate_suggestions(anchors, workflows),
        }

    def _build_context_summary(self, recent_actions: list[dict[str, Any]]) -> str:
        """Build a summary of recent context."""
        if not recent_actions:
            return "No recent context available."

        action_types = [a["action_type"] for a in recent_actions[-10:]]
        summary = f"Recent: {', '.join(set(action_types))}"

        return summary[:200]

    def _generate_suggestions(
        self,
        anchors: list[MemoryAnchor],
        workflows: list[Workflow],
    ) -> list[str]:
        """Generate suggestions based on current state."""
        suggestions = []

        if anchors:
            suggestions.append(f"Recall {len(anchors)} memory anchors from recent work")

        if workflows:
            suggestions.append(f"Continue {len(workflows)} incomplete workflow(s)")

        if self._state.current_load > 0.9:
            suggestions.append("Take a short break to reset cognitive load")

        return suggestions

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "current_load": round(self._state.current_load, 2),
            "is_overloaded": self._state.is_overloaded,
            "total_anchors": len(self._anchors),
            "active_anchors": len([a for a in self._anchors if not a.is_expired()]),
            "total_workflows": len(self._workflows),
            "active_workflows": len(self.get_active_workflows()),
            "offload_count_today": self._state.offload_count_today,
            "current_workflow": self._current_workflow.name if self._current_workflow else None,
        }

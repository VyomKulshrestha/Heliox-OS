"""Per-task circuit breaker — halts retries after N consecutive failures.

The breaker tracks consecutive action-level failures per task. When the
threshold is hit, the breaker trips OPEN for that task and subsequent check()
calls raise CircuitBreakerOpenError until the task ends and reset() is called.

This is intentionally separate from self_heal.py's per-action retry counter.
self_heal counts retry attempts within a single action; the circuit breaker
counts final action outcomes across a task. Both can fire independently —
self_heal exhausting retries on one action is a "failure" the breaker sees,
and the breaker tripping halts further actions in the task.
"""

from __future__ import annotations

import logging
from enum import Enum

from pilot.models.budget_tracker import current_task_id

logger = logging.getLogger("pilot.agents.circuit_breaker")


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit breaker has tripped for the active task."""


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"


class CircuitBreaker:
    """Per-task circuit breaker keyed off the current_task_id ContextVar.

    Lifecycle:
      - record_success() / record_failure() update the per-task counter
      - On consecutive_failures >= threshold the breaker trips OPEN
      - check() raises CircuitBreakerOpenError while OPEN
      - reset() clears state (called by the orchestrator at task end)

    Once OPEN, the breaker stays OPEN for that task until reset — no
    automatic recovery within a task. A new task starts CLOSED.
    """

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold
        self._counts: dict[str, int] = {}
        self._state: dict[str, CircuitBreakerState] = {}

    def _resolve_task_id(self, task_id: str | None) -> str | None:
        return task_id if task_id is not None else current_task_id.get()

    def record_success(self, task_id: str | None = None) -> None:
        """A successful action resets the consecutive-failure counter."""
        tid = self._resolve_task_id(task_id)
        if tid:
            self._counts[tid] = 0

    def record_failure(self, task_id: str | None = None) -> None:
        """A failed action increments the counter; trips OPEN at threshold."""
        tid = self._resolve_task_id(task_id)
        if not tid:
            return
        self._counts[tid] = self._counts.get(tid, 0) + 1
        if (
            self._counts[tid] >= self._threshold
            and self._state.get(tid) != CircuitBreakerState.OPEN
        ):
            self._state[tid] = CircuitBreakerState.OPEN
            logger.warning(
                "CircuitBreaker tripped OPEN for task %s "
                "(%d consecutive failures >= threshold %d)",
                tid, self._counts[tid], self._threshold,
            )

    def check(self, task_id: str | None = None) -> None:
        """Raise if the breaker is OPEN for this task. No-op if CLOSED."""
        tid = self._resolve_task_id(task_id)
        if not tid:
            return
        if self._state.get(tid) == CircuitBreakerState.OPEN:
            count = self._counts.get(tid, 0)
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN for task {tid} "
                f"({count} consecutive failures >= threshold {self._threshold}). "
                f"Halting further retries to prevent runaway behavior. "
                f"User intervention required."
            )

    def reset(self, task_id: str | None = None) -> None:
        """Clear breaker state for a task. Called at task end."""
        tid = self._resolve_task_id(task_id)
        if tid:
            self._counts.pop(tid, None)
            self._state.pop(tid, None)

    def get_state(self, task_id: str | None = None) -> CircuitBreakerState:
        """Inspect breaker state for a task."""
        tid = self._resolve_task_id(task_id)
        if not tid:
            return CircuitBreakerState.CLOSED
        return self._state.get(tid, CircuitBreakerState.CLOSED)

    def get_failure_count(self, task_id: str | None = None) -> int:
        """Current consecutive-failure count for a task."""
        tid = self._resolve_task_id(task_id)
        if not tid:
            return 0
        return self._counts.get(tid, 0)
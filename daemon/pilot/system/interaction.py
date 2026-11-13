"""Observable state machine for Heliox's continuous interaction loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable


class InteractionPhase(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    ACTING = "acting"
    VERIFYING = "verifying"
    CORRECTING = "correcting"
    SPEAKING = "speaking"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(slots=True)
class InteractionSnapshot:
    interaction_id: str = ""
    source: str = "system"
    phase: InteractionPhase = InteractionPhase.IDLE
    user_input: str = ""
    message: str = "Ready"
    started_at: float = 0.0
    updated_at: float = field(default_factory=time.time)
    sequence: int = 0

    @property
    def active(self) -> bool:
        return self.phase not in {
            InteractionPhase.IDLE,
            InteractionPhase.COMPLETED,
            InteractionPhase.INTERRUPTED,
            InteractionPhase.FAILED,
        }

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {
            "interaction_id": self.interaction_id,
            "source": self.source,
            "phase": self.phase.value,
            "user_input": self.user_input,
            "message": self.message,
            "active": self.active,
            "sequence": self.sequence,
            "elapsed_ms": (int(max(0.0, now - self.started_at) * 1000) if self.started_at else 0),
            "updated_at": self.updated_at,
        }


Emitter = Callable[[str, dict[str, Any]], Awaitable[None]]


def acknowledgement_for(user_input: str) -> str:
    """Return a short intent-aware acknowledgement, never a fake result."""
    normalized = " ".join(user_input.lower().split())
    if any(token in normalized for token in ("find ", "search ", "look up", "research")):
        return "I’ll look into that."
    if any(token in normalized for token in ("show ", "check ", "inspect ", "what is", "what’s")):
        return "I’ll check."
    if any(token in normalized for token in ("open ", "launch ", "start ")):
        return "I’ll open it."
    if any(token in normalized for token in ("create ", "write ", "build ", "make ")):
        return "I’ll put that together."
    return "I’m on it."


class InteractionRuntime:
    """Own the one user-visible state shared by voice, text, and tools."""

    def __init__(self, emitter: Emitter) -> None:
        self._emitter = emitter
        self._snapshot = InteractionSnapshot()
        self._lock = asyncio.Lock()

    def status(self) -> dict[str, Any]:
        return self._snapshot.to_dict()

    async def start(self, user_input: str, *, source: str) -> dict[str, Any]:
        async with self._lock:
            now = time.time()
            self._snapshot = InteractionSnapshot(
                interaction_id=str(uuid.uuid4()),
                source=source,
                phase=InteractionPhase.UNDERSTANDING,
                user_input=" ".join(user_input.split())[:500],
                message="Understanding your request",
                started_at=now,
                updated_at=now,
                sequence=self._snapshot.sequence + 1,
            )
            payload = self._snapshot.to_dict()
        await self._emitter("interaction_state", payload)
        return payload

    async def transition(
        self,
        phase: InteractionPhase | str,
        *,
        message: str = "",
        interaction_id: str = "",
    ) -> dict[str, Any]:
        resolved = InteractionPhase(phase)
        async with self._lock:
            if interaction_id and interaction_id != self._snapshot.interaction_id:
                return self._snapshot.to_dict()
            self._snapshot.phase = resolved
            self._snapshot.message = message.strip() or resolved.value.replace("_", " ").capitalize()
            self._snapshot.updated_at = time.time()
            self._snapshot.sequence += 1
            payload = self._snapshot.to_dict()
        await self._emitter("interaction_state", payload)
        return payload

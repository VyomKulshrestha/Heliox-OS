"""Deterministic JSONL replay utilities for agent workflow debugging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReplayEventType(StrEnum):
    """Supported replay event families."""

    LLM_COMPLETION = "llm_completion"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_REQUEST = "system_request"
    SYSTEM_RESPONSE = "system_response"


class ReplayMismatchError(RuntimeError):
    """Raised when a replay fixture cannot satisfy the requested event."""


@dataclass(frozen=True)
class ReplayEvent:
    """One deterministic replay event persisted as a JSONL record."""

    session_id: str
    sequence: int
    event_type: ReplayEventType
    name: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> ReplayEvent:
        data = json.loads(line)
        data["event_type"] = ReplayEventType(data["event_type"])
        return cls(**data)


class ReplayRedactor:
    """Recursively redacts common secret fields before replay data is persisted."""

    DEFAULT_SECRET_MARKERS = (
        "api_key",
        "apikey",
        "authorization",
        "auth_header",
        "bearer",
        "client_secret",
        "cookie",
        "password",
        "refresh_token",
        "secret",
        "session",
        "token",
    )

    def __init__(
        self,
        secret_markers: tuple[str, ...] = DEFAULT_SECRET_MARKERS,
        replacement: str = "[REDACTED]",
    ) -> None:
        self._secret_markers = tuple(marker.lower() for marker in secret_markers)
        self._replacement = replacement

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._replacement if self._is_secret_key(str(key)) else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        return value

    def _is_secret_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in self._secret_markers)


class ReplayRecorder:
    """Captures deterministic workflow events and optionally flushes them to JSONL."""

    def __init__(
        self,
        session_id: str,
        path: str | Path | None = None,
        *,
        redactor: ReplayRedactor | None = None,
        auto_flush: bool = True,
    ) -> None:
        self.session_id = session_id
        self.path = Path(path) if path is not None else None
        self.redactor = redactor or ReplayRedactor()
        self.auto_flush = auto_flush
        self.events: list[ReplayEvent] = []

    def record_event(
        self,
        event_type: ReplayEventType | str,
        name: str,
        payload: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ReplayEvent:
        event = ReplayEvent(
            session_id=self.session_id,
            sequence=len(self.events),
            event_type=ReplayEventType(event_type),
            name=name,
            payload=self.redactor.redact(payload),
            metadata=self.redactor.redact(metadata or {}),
        )
        self.events.append(event)
        if self.auto_flush and self.path is not None:
            self.flush()
        return event

    def record_llm_completion(
        self,
        *,
        prompt: str,
        output: str,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        provider: str | None = None,
        model: str | None = None,
        name: str = "model.generate",
    ) -> ReplayEvent:
        return self.record_event(
            ReplayEventType.LLM_COMPLETION,
            name,
            {
                "request": {
                    "prompt": prompt,
                    "system": system,
                    "json_mode": json_mode,
                    "temperature": temperature,
                    "provider": provider,
                    "model": model,
                },
                "output": output,
            },
        )

    def record_tool_call(self, name: str, arguments: dict[str, Any]) -> ReplayEvent:
        return self.record_event(ReplayEventType.TOOL_CALL, name, {"arguments": arguments})

    def record_tool_result(self, name: str, result: dict[str, Any]) -> ReplayEvent:
        return self.record_event(ReplayEventType.TOOL_RESULT, name, {"result": result})

    def record_system_response(self, name: str, response: dict[str, Any]) -> ReplayEvent:
        return self.record_event(ReplayEventType.SYSTEM_RESPONSE, name, {"response": response})

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.to_jsonl(), encoding="utf-8")

    def to_jsonl(self) -> str:
        if not self.events:
            return ""
        return "\n".join(event.to_json() for event in self.events) + "\n"


class ReplaySession:
    """Consumes a captured replay fixture in sequence and validates requested events."""

    def __init__(self, events: list[ReplayEvent]) -> None:
        self._events = sorted(events, key=lambda event: event.sequence)
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, data: str) -> ReplaySession:
        events = [ReplayEvent.from_json(line) for line in data.splitlines() if line.strip()]
        return cls(events)

    @classmethod
    def from_path(cls, path: str | Path) -> ReplaySession:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))

    @property
    def remaining(self) -> int:
        return len(self._events) - self._cursor

    def next_event(
        self,
        event_type: ReplayEventType | str,
        *,
        name: str | None = None,
    ) -> ReplayEvent:
        if self._cursor >= len(self._events):
            raise ReplayMismatchError(f"No replay events remain for {ReplayEventType(event_type).value}.")

        event = self._events[self._cursor]
        expected_type = ReplayEventType(event_type)
        if event.event_type != expected_type:
            raise ReplayMismatchError(
                f"Expected replay event {expected_type.value}, found {event.event_type.value} "
                f"at sequence {event.sequence}."
            )
        if name is not None and event.name != name:
            raise ReplayMismatchError(
                f"Expected replay event named {name!r}, found {event.name!r} at sequence {event.sequence}."
            )

        self._cursor += 1
        return event

    def next_llm_completion(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        json_mode: bool | None = None,
        temperature: float | None = None,
        name: str = "model.generate",
    ) -> str:
        event = self.next_event(ReplayEventType.LLM_COMPLETION, name=name)
        request = event.payload.get("request", {})
        expected = {
            "prompt": prompt,
            "system": system,
            "json_mode": json_mode,
            "temperature": temperature,
        }
        mismatches = [key for key, value in expected.items() if value is not None and request.get(key) != value]
        if mismatches:
            details = ", ".join(mismatches)
            raise ReplayMismatchError(f"Replay LLM request mismatch for: {details}.")

        output = event.payload.get("output")
        if not isinstance(output, str):
            raise ReplayMismatchError("Replay LLM completion is missing a string output.")
        return output

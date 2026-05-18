"""Deterministic replay harness for multi-agent workflows.

The recorder captures the two non-deterministic seams in the ReAct loop:
LLM generations and system action results. A later test can load the same
recording and use replay-only wrappers to return identical responses without
calling a model provider or touching the operating system.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from pilot.actions import Action, ActionPlan, ActionResult

EventPayload = dict[str, Any]


@dataclass
class ReplayEvent:
    """One recorded interaction in a deterministic session."""

    event_type: str
    payload: EventPayload
    sequence: int
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "payload": self.payload,
                "sequence": self.sequence,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> ReplayEvent:
        data = json.loads(line)
        return cls(
            event_type=data["event_type"],
            payload=data["payload"],
            sequence=data["sequence"],
            timestamp=data.get("timestamp", time.time()),
        )


class ReplayMismatchError(AssertionError):
    """Raised when the live workflow diverges from a replay recording."""


class ReplayRecorder:
    """Append-only recorder for LLM and system action events."""

    def __init__(self) -> None:
        self._events: list[ReplayEvent] = []
        self._next_sequence = 0

    @property
    def events(self) -> list[ReplayEvent]:
        return list(self._events)

    def _append(self, event_type: str, payload: EventPayload) -> ReplayEvent:
        event = ReplayEvent(
            event_type=event_type,
            payload=payload,
            sequence=self._next_sequence,
        )
        self._next_sequence += 1
        self._events.append(event)
        return event

    def record_llm_call(
        self,
        *,
        prompt: str,
        response: str,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
    ) -> ReplayEvent:
        return self._append(
            "llm_call",
            {
                "prompt": prompt,
                "system": system,
                "json_mode": json_mode,
                "temperature": temperature,
                "response": response,
            },
        )

    def record_action_result(self, result: ActionResult) -> ReplayEvent:
        return self._append(
            "action_result",
            result.model_dump(mode="json"),
        )

    def record_system_response(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        body: Any,
    ) -> ReplayEvent:
        return self._append(
            "system_response",
            {
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "body": body,
            },
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "\n".join(event.to_json() for event in self._events) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayRecorder:
        recorder = cls()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = ReplayEvent.from_json(line)
            recorder._events.append(event)
            recorder._next_sequence = max(
                recorder._next_sequence,
                event.sequence + 1,
            )
        return recorder


class ReplayCursor:
    """Consumes a recording in sequence and validates deterministic inputs."""

    def __init__(self, events: list[ReplayEvent]) -> None:
        self._events = events
        self._index = 0

    @property
    def remaining(self) -> int:
        return len(self._events) - self._index

    def pop(self, expected_type: str) -> ReplayEvent:
        if self._index >= len(self._events):
            raise ReplayMismatchError(f"Expected {expected_type}, but the recording is exhausted.")

        event = self._events[self._index]
        self._index += 1
        if event.event_type != expected_type:
            raise ReplayMismatchError(
                f"Expected {expected_type}, found {event.event_type} at sequence {event.sequence}."
            )
        return event


class RecordingModelRouter:
    """ModelRouter-compatible wrapper that records every generation."""

    def __init__(self, delegate: Any, recorder: ReplayRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        response = await self._delegate.generate(
            prompt,
            system=system,
            json_mode=json_mode,
            temperature=temperature,
            stream_callback=stream_callback,
        )
        self._recorder.record_llm_call(
            prompt=prompt,
            system=system,
            json_mode=json_mode,
            temperature=temperature,
            response=response,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class ReplayModelRouter:
    """ModelRouter-compatible replay double for deterministic tests."""

    def __init__(self, cursor: ReplayCursor) -> None:
        self._cursor = cursor

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        stream_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        event = self._cursor.pop("llm_call")
        payload = event.payload
        expected = {
            "prompt": prompt,
            "system": system,
            "json_mode": json_mode,
            "temperature": temperature,
        }
        actual = {
            "prompt": payload["prompt"],
            "system": payload.get("system", ""),
            "json_mode": payload.get("json_mode", False),
            "temperature": payload.get("temperature", 0.1),
        }
        if actual != expected:
            raise ReplayMismatchError(
                f"LLM replay mismatch at sequence {event.sequence}: expected {expected!r}, recorded {actual!r}."
            )

        response = payload["response"]
        if stream_callback:
            await stream_callback(response)
        return response


class RecordingExecutor:
    """Executor-compatible wrapper that records every completed action."""

    def __init__(self, delegate: Any, recorder: ReplayRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    async def execute(
        self,
        plan: ActionPlan,
        on_action_start: Callable[[Action], Awaitable[None]] | None = None,
        on_action_complete: Callable[[ActionResult], Awaitable[None]] | None = None,
        cancel_event: asyncio.Event | None = None,
        plan_id: str | None = None,
        initial_last_output: str = "",
    ) -> list[ActionResult]:
        existing_action_events = sum(1 for event in self._recorder.events if event.event_type == "action_result")

        async def _recording_complete(result: ActionResult) -> None:
            self._recorder.record_action_result(result)
            if on_action_complete:
                await on_action_complete(result)

        results = await self._delegate.execute(
            plan,
            on_action_start=on_action_start,
            on_action_complete=_recording_complete,
            cancel_event=cancel_event,
            plan_id=plan_id,
            initial_last_output=initial_last_output,
        )

        # Some lightweight executor doubles do not call on_action_complete.
        current_action_events = sum(1 for event in self._recorder.events if event.event_type == "action_result")
        recorded_during_call = current_action_events - existing_action_events
        if recorded_during_call < len(results):
            for result in results[recorded_during_call:]:
                self._recorder.record_action_result(result)
        return results

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class ReplayExecutor:
    """Executor-compatible replay double that never touches the OS."""

    def __init__(self, cursor: ReplayCursor) -> None:
        self._cursor = cursor

    async def execute(
        self,
        plan: ActionPlan,
        on_action_start: Callable[[Action], Awaitable[None]] | None = None,
        on_action_complete: Callable[[ActionResult], Awaitable[None]] | None = None,
        cancel_event: asyncio.Event | None = None,
        plan_id: str | None = None,
        initial_last_output: str = "",
    ) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in plan.actions:
            if cancel_event and cancel_event.is_set():
                break
            if on_action_start:
                await on_action_start(action)

            event = self._cursor.pop("action_result")
            result = ActionResult.model_validate(event.payload)
            if result.action.model_dump(mode="json") != action.model_dump(mode="json"):
                raise ReplayMismatchError(
                    f"Action replay mismatch at sequence {event.sequence}: "
                    f"expected {action.action_type}, recorded {result.action.action_type}."
                )

            if on_action_complete:
                await on_action_complete(result)
            results.append(result)
        return results


@dataclass
class ReplayHarness:
    """Convenience factory for recording and replaying a workflow."""

    recorder: ReplayRecorder = field(default_factory=ReplayRecorder)

    def wrap_model_router(self, model_router: Any) -> RecordingModelRouter:
        return RecordingModelRouter(model_router, self.recorder)

    def wrap_executor(self, executor: Any) -> RecordingExecutor:
        return RecordingExecutor(executor, self.recorder)

    def save(self, path: str | Path) -> None:
        self.recorder.save(path)

    @classmethod
    def from_recording(cls, path: str | Path) -> ReplayHarness:
        return cls(ReplayRecorder.load(path))

    def replay_model_router(self) -> ReplayModelRouter:
        return ReplayModelRouter(ReplayCursor(self.recorder.events))

    def replay_executor(self) -> ReplayExecutor:
        return ReplayExecutor(ReplayCursor(self.recorder.events))

    def replay_components(self) -> tuple[ReplayModelRouter, ReplayExecutor]:
        cursor = ReplayCursor(self.recorder.events)
        return ReplayModelRouter(cursor), ReplayExecutor(cursor)

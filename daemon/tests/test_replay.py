from __future__ import annotations

import pytest

from pilot.workflows.replay import (
    ReplayEventType,
    ReplayMismatchError,
    ReplayRecorder,
    ReplaySession,
)


def test_replay_recorder_redacts_secrets_and_round_trips_jsonl(tmp_path):
    path = tmp_path / "session.jsonl"
    recorder = ReplayRecorder("session-1", path)

    recorder.record_tool_call(
        "fetch-profile",
        {
            "user_id": "u_123",
            "api_key": "sk-live",
            "nested": {"Authorization": "Bearer secret"},
        },
    )
    recorder.record_llm_completion(
        prompt="Summarize the profile",
        system="Be concise",
        json_mode=True,
        temperature=0,
        output='{"summary":"ok"}',
        provider="ollama",
        model="llama3",
    )

    session = ReplaySession.from_path(path)
    tool_event = session.next_event(ReplayEventType.TOOL_CALL, name="fetch-profile")

    assert tool_event.payload["arguments"]["user_id"] == "u_123"
    assert tool_event.payload["arguments"]["api_key"] == "[REDACTED]"
    assert tool_event.payload["arguments"]["nested"]["Authorization"] == "[REDACTED]"
    assert (
        session.next_llm_completion(
            prompt="Summarize the profile",
            system="Be concise",
            json_mode=True,
            temperature=0,
        )
        == '{"summary":"ok"}'
    )
    assert session.remaining == 0


def test_replay_session_enforces_event_order_and_name():
    recorder = ReplayRecorder("session-1", auto_flush=False)
    recorder.record_tool_result("search", {"items": [1]})
    session = ReplaySession.from_jsonl(recorder.to_jsonl())

    with pytest.raises(ReplayMismatchError, match="Expected replay event llm_completion"):
        session.next_llm_completion(prompt="anything")

    assert session.next_event(ReplayEventType.TOOL_RESULT, name="search").payload == {"result": {"items": [1]}}
    assert session.remaining == 0


def test_replay_session_detects_llm_request_drift():
    recorder = ReplayRecorder("session-1", auto_flush=False)
    recorder.record_llm_completion(
        prompt="original prompt",
        system="system",
        json_mode=False,
        temperature=0.2,
        output="original output",
    )
    session = ReplaySession.from_jsonl(recorder.to_jsonl())

    with pytest.raises(ReplayMismatchError, match="prompt"):
        session.next_llm_completion(prompt="changed prompt", system="system")


@pytest.mark.asyncio
async def test_model_router_replays_without_calling_backend():
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter

    recorder = ReplayRecorder("session-1", auto_flush=False)
    recorder.record_llm_completion(
        prompt="Plan my day",
        system="Return JSON",
        json_mode=True,
        temperature=0.3,
        output='{"tasks":[]}',
    )

    router = ModelRouter(PilotConfig(), vault=object())  # type: ignore[arg-type]
    router.set_replay_session(ReplaySession.from_jsonl(recorder.to_jsonl()))

    assert (
        await router.generate(
            "Plan my day",
            system="Return JSON",
            json_mode=True,
            temperature=0.3,
        )
        == '{"tasks":[]}'
    )

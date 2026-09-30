from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.agents.proactive import ProactiveSuggestionEngine, Suggestion
from pilot.config import PilotConfig
from pilot.server import PilotServer


def _suggestion(pattern_id: str, suffix: int) -> Suggestion:
    return Suggestion(
        suggestion_id=f"{pattern_id}_{suffix}",
        title="Can I help?",
        description="A context-aware suggestion",
        action_command="Inspect the current screen",
        trigger_reason="test",
        pattern_id=pattern_id,
    )


class _ScreenVisionStub:
    def __init__(self, *, running: bool, paused: bool = False) -> None:
        self.is_running = running
        self._paused = paused
        self.state = SimpleNamespace(active_app="PowerShell", active_window_title="Build error")

    def is_paused(self) -> bool:
        return self._paused

    def get_context(self):
        return SimpleNamespace(current=lambda: self.state)


def test_proactive_context_requires_live_unpaused_screen_monitor(tmp_path):
    stopped = _ScreenVisionStub(running=False)
    engine = ProactiveSuggestionEngine(
        screen_vision=stopped,
        feedback_path=tmp_path / "feedback.json",
    )

    assert engine._current_live_screen_state() is None

    stopped.is_running = True
    assert engine._current_live_screen_state() is stopped.state

    stopped._paused = True
    assert engine._current_live_screen_state() is None


@pytest.mark.asyncio
async def test_feedback_persists_across_engine_instances(tmp_path):
    feedback_path = tmp_path / "proactive-feedback.json"
    engine = ProactiveSuggestionEngine(feedback_path=feedback_path)

    accepted = _suggestion("terminal_error", 1)
    dismissed = _suggestion("terminal_error", 2)
    engine._pending_suggestions.extend([accepted, dismissed])

    assert await engine.accept_suggestion(accepted.suggestion_id) == accepted.action_command
    assert await engine.dismiss_suggestion(dismissed.suggestion_id) is True

    reloaded = ProactiveSuggestionEngine(feedback_path=feedback_path)
    learned = reloaded.get_learning_status()["patterns"]["terminal_error"]
    assert learned["accepted"] == 1
    assert learned["dismissed"] == 1


@pytest.mark.asyncio
async def test_spoken_yes_accepts_the_one_visible_suggestion(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    engine._pending_suggestions.append(suggestion)

    decision = await engine.resolve_spoken_response("Yes, do it!")

    assert decision == {
        "decision": "accepted",
        "suggestion_id": suggestion.suggestion_id,
        "title": suggestion.title,
        "action_command": suggestion.action_command,
    }
    assert suggestion not in engine._pending_suggestions


@pytest.mark.asyncio
async def test_ambiguous_speech_does_not_claim_a_suggestion(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    engine._pending_suggestions.append(suggestion)

    assert await engine.resolve_spoken_response("maybe later this afternoon") is None
    assert suggestion in engine._pending_suggestions


@pytest.mark.asyncio
async def test_repeated_dismissals_temporarily_suppress_pattern(tmp_path):
    feedback_path = tmp_path / "proactive-feedback.json"
    engine = ProactiveSuggestionEngine(feedback_path=feedback_path)

    for index in range(3):
        suggestion = _suggestion("terminal_error", index)
        engine._pending_suggestions.append(suggestion)
        assert await engine.dismiss_suggestion(suggestion.suggestion_id) is True

    reloaded = ProactiveSuggestionEngine(feedback_path=feedback_path)
    current = SimpleNamespace(
        active_app="PowerShell",
        active_window_title="Fatal error traceback",
    )
    dwell_key = "powershell:fatal error traceback"
    reloaded._app_dwell_tracker[dwell_key] = time.time() - 60

    await reloaded._check_patterns(current)

    assert reloaded._pending_suggestions == []


@pytest.mark.asyncio
async def test_unanswered_suggestion_expires_as_ignored_feedback(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    suggestion.timestamp = time.time() - engine._ignore_after_seconds - 1
    engine._pending_suggestions.append(suggestion)

    await engine._expire_ignored_suggestions(time.time())

    assert suggestion not in engine._pending_suggestions
    assert suggestion in engine._suggestion_history
    pattern = engine.get_learning_status()["patterns"]["terminal_error"]
    assert pattern["ignored"] == 1
    assert pattern["learned_relevance"] == 0.5
    assert engine._feedback["terminal_error"]["last_feedback"] == 0


def test_only_explicit_feedback_changes_proactive_relevance(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")

    for _ in range(5):
        engine._record_feedback("terminal_error", "shown")
        engine._record_feedback("terminal_error", "ignored")
    assert engine.get_learning_status()["patterns"]["terminal_error"]["learned_relevance"] == 0.5

    engine._record_feedback("terminal_error", "accepted")
    assert engine.get_learning_status()["patterns"]["terminal_error"]["learned_relevance"] == 0.667
    engine._record_feedback("terminal_error", "dismissed")
    assert engine.get_learning_status()["patterns"]["terminal_error"]["learned_relevance"] == 0.5


@pytest.mark.asyncio
async def test_accept_handler_fails_closed_without_guarded_executor(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    engine._pending_suggestions.append(suggestion)

    server = PilotServer(PilotConfig())
    server._proactive = engine
    server._autonomous = None

    result = await server._handle_proactive_accept(
        {"suggestion_id": suggestion.suggestion_id},
        ws=None,
    )

    assert result["status"] == "error"
    assert "not executed" in result["message"]
    assert suggestion in engine._pending_suggestions


@pytest.mark.asyncio
async def test_learning_rpc_reports_and_resets_persisted_feedback(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    engine._pending_suggestions.append(suggestion)
    await engine.dismiss_suggestion(suggestion.suggestion_id)

    server = PilotServer(PilotConfig())
    server._proactive = engine

    status = await server._handle_proactive_learning_status({}, ws=None)
    assert status["status"] == "ok"
    assert status["patterns"]["terminal_error"]["dismissed"] == 1

    reset = await server._handle_proactive_learning_reset({}, ws=None)
    assert reset == {"enabled": True, "patterns": {}}
    assert not engine._feedback_path.exists()


@pytest.mark.asyncio
async def test_accept_handler_uses_guarded_autonomous_pipeline(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    engine._pending_suggestions.append(suggestion)

    job = SimpleNamespace(to_dict=lambda: {"job_id": "job-1"})
    server = PilotServer(PilotConfig())
    server._proactive = engine
    server._autonomous = AsyncMock()
    server._autonomous.submit.return_value = job

    result = await server._handle_proactive_accept(
        {"suggestion_id": suggestion.suggestion_id},
        ws=None,
    )

    server._autonomous.submit.assert_awaited_once_with(
        suggestion.action_command,
        source="proactive",
    )
    assert result["status"] == "executing"


@pytest.mark.asyncio
async def test_dismiss_handler_reports_exact_outcome(tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    suggestion = _suggestion("terminal_error", 1)
    engine._pending_suggestions.append(suggestion)

    server = PilotServer(PilotConfig())
    server._proactive = engine

    missing = await server._handle_proactive_dismiss({"suggestion_id": "missing"}, ws=None)
    assert missing == {"status": "error", "message": "Suggestion not found: missing"}
    assert suggestion in engine._pending_suggestions

    dismissed = await server._handle_proactive_dismiss(
        {"suggestion_id": suggestion.suggestion_id},
        ws=None,
    )
    assert dismissed == {
        "status": "ok",
        "dismissed": True,
        "suggestion_id": suggestion.suggestion_id,
    }
    assert suggestion not in engine._pending_suggestions

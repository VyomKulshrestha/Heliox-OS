"""Regression + routing tests for PilotServer._voice_command_dispatch.

Before this fix, _voice_command_dispatch called self._executor.execute_plan(plan)
— a method Executor never defines (only execute()) — so every voice command
threw AttributeError, silently caught by the method's own broad except and
reported as "something went wrong." These tests cover both the crash fix and
the new specialist-orchestrator routing with a voice-derived scope_override.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.intelligence.experience import ExperienceContext
from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, InvocationSource
from pilot.server import PendingConfirmation, PilotServer
from pilot.system.companion_speech import SpeechChannel


def _bare_server() -> PilotServer:
    server = PilotServer(PilotConfig())
    server._voice_listener = None
    server._broadcast_notification = AsyncMock()
    server._handle_execute = AsyncMock(
        return_value={"status": "success", "message": "The notes are ready."},
    )
    server._speak_voice_response = AsyncMock(return_value=False)
    server._spawn_interaction_speech = MagicMock()
    return server


class _RunningVoiceListener:
    is_running = True
    last_detected_language = "en"

    def __init__(self) -> None:
        self.arm_follow_up_window = MagicMock()


@pytest.mark.asyncio
async def test_voice_runs_through_unified_interactive_handler():
    server = _bare_server()

    await server._voice_command_dispatch("read my notes")

    server._handle_execute.assert_awaited_once()
    params = server._handle_execute.await_args.args[0]
    assert params["input"] == "read my notes"
    assert params["source"] == "voice"
    server._spawn_interaction_speech.assert_called_once()
    server._speak_voice_response.assert_awaited_once_with(
        "The notes are ready.",
        channel=SpeechChannel.FINAL_ANSWER,
        dedupe_key="voice-result:read my notes",
    )


@pytest.mark.asyncio
async def test_quick_voice_command_cancels_delayed_acknowledgement():
    server = _bare_server()
    server._spawn_interaction_speech = PilotServer._spawn_interaction_speech.__get__(server)
    server._speak_companion_text = AsyncMock()

    await server._voice_command_dispatch("open Openscreen")
    await asyncio.sleep(0)

    server._speak_companion_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_voice_response_arms_bounded_conversational_follow_up():
    server = _bare_server()
    listener = _RunningVoiceListener()
    server._voice_listener = listener

    await server._voice_command_dispatch("open the world monitor")

    listener.arm_follow_up_window.assert_called_once_with()
    follow_up_status = next(
        call.args[1]
        for call in server._broadcast_notification.await_args_list
        if call.args[0] == "voice_status" and call.args[1].get("status") == "follow_up_ready"
    )
    assert follow_up_status["seconds"] == 30.0


@pytest.mark.asyncio
async def test_spoken_acceptance_runs_proactive_action_through_interactive_safety():
    server = _bare_server()
    server._proactive = AsyncMock()
    server._proactive.resolve_spoken_response.return_value = {
        "decision": "accepted",
        "suggestion_id": "suggestion-1",
        "title": "Open the game trailer?",
        "action_command": "Open the Crimson Desert trailer in the browser.",
    }

    await server._voice_command_dispatch("yes do it")

    params = server._handle_execute.await_args.args[0]
    assert params["source"] == "voice"
    assert "accepted the proactive suggestion" in params["input"]
    assert "Crimson Desert trailer" in params["input"]
    voice_command = next(
        call.args[1] for call in server._broadcast_notification.await_args_list if call.args[0] == "voice_command"
    )
    assert voice_command["command"] == "yes do it"


@pytest.mark.asyncio
async def test_spoken_rejection_dismisses_without_planning_an_action():
    server = _bare_server()
    server._proactive = AsyncMock()
    server._proactive.resolve_spoken_response.return_value = {
        "decision": "dismissed",
        "suggestion_id": "suggestion-1",
        "title": "Open the game trailer?",
        "action_command": "",
    }

    await server._voice_command_dispatch("no thanks")

    server._handle_execute.assert_not_awaited()
    server._speak_voice_response.assert_awaited_once()


def test_voice_source_resolves_to_voice_scope_override():
    server = PilotServer(PilotConfig())

    invocation_source, override = server._execution_scope_for_source("voice")

    assert invocation_source == InvocationSource.VOICE
    voice_profile = DEFAULT_SOURCE_PROFILES["voice"]
    assert override.max_tier == voice_profile.max_tier
    assert override.deny_action_types == voice_profile.deny_action_types
    assert override.allow_root == voice_profile.allow_root


def test_voice_scope_override_is_never_wider_than_voice_profile():
    """A user-configured 'voice' profile in gateway.source_profiles (not just
    the hardcoded default) must be what's used to build the override."""
    server = PilotServer(PilotConfig())
    server.config.gateway.source_profiles["voice"].allow_root = False
    _, override = server._execution_scope_for_source("voice")
    assert override.allow_root is False


@pytest.mark.asyncio
async def test_voice_command_becomes_live_correction_during_active_task():
    server = PilotServer(PilotConfig())
    server._interactive_request_active = True
    server._handle_interject = AsyncMock(
        return_value={"status": "revising", "message": "Applying correction"},
    )
    server._broadcast_notification = AsyncMock()
    server._planner = AsyncMock()

    await server._voice_command_dispatch("use the other folder instead")

    server._handle_interject.assert_awaited_once_with(
        {"input": "use the other folder instead"},
        None,
    )
    server._planner.plan.assert_not_called()
    payload = server._broadcast_notification.call_args.args[1]
    assert payload["coordinated_correction"] is True


@pytest.mark.asyncio
async def test_voice_question_during_task_answers_without_correction():
    server = _bare_server()
    server._interactive_request_active = True
    server._handle_interject = AsyncMock(
        return_value={"status": "answered", "message": "The task is still running."},
    )

    await server._voice_command_dispatch("Quick question, what is this plan for?")

    server._handle_interject.assert_awaited_once_with({"input": "what is this plan for?", "mode": "ask"}, None)
    server._handle_execute.assert_not_awaited()
    server._speak_voice_response.assert_awaited_once()
    payload = next(
        call.args[1] for call in server._broadcast_notification.await_args_list if call.args[0] == "voice_result"
    )
    assert payload["coordinated_correction"] is False


@pytest.mark.asyncio
async def test_voice_can_queue_next_task_without_correcting_current_one():
    server = _bare_server()
    server._interactive_request_active = True
    server._queue_interactive_request = AsyncMock(
        return_value={"status": "queued", "task_id": "queued-1", "message": "Task queued."}
    )
    server._handle_interject = AsyncMock()
    server._active_experience_context = ExperienceContext(session_id="current-chat")

    await server._voice_command_dispatch("Next task, summarize the report")

    server._queue_interactive_request.assert_awaited_once_with(
        {
            "input": "summarize the report",
            "session_id": "current-chat",
            "user_id": "local",
            "source": "voice",
        }
    )
    server._handle_interject.assert_not_awaited()
    server._speak_voice_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_cannot_approve_a_pending_visible_confirmation():
    server = _bare_server()
    server._interactive_request_active = True
    server._active_plan_id = "plan-visible"
    pending = PendingConfirmation(plan_id="plan-visible", event=asyncio.Event())
    server._pending_confirms["plan-visible"] = pending
    server._handle_interject = AsyncMock()

    await server._voice_command_dispatch("Yes, approve!")

    server._handle_interject.assert_not_awaited()
    server._handle_execute.assert_not_awaited()
    assert pending.event.is_set() is False
    assert pending.confirmed is False
    result = next(
        call.args[1] for call in server._broadcast_notification.await_args_list if call.args[0] == "voice_result"
    )
    assert result["status"] == "approval_required"
    assert result["plan_id"] == "plan-visible"
    server._speak_voice_response.assert_awaited_once_with(
        "For safety, approve this request in the visible Heliox confirmation dialog.",
        channel=SpeechChannel.APPROVAL_RISK,
        dedupe_key="voice-approval-boundary:plan-visible",
    )

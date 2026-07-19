"""Regression + routing tests for PilotServer._voice_command_dispatch.

Before this fix, _voice_command_dispatch called self._executor.execute_plan(plan)
— a method Executor never defines (only execute()) — so every voice command
threw AttributeError, silently caught by the method's own broad except and
reported as "something went wrong." These tests cover both the crash fix and
the new specialist-orchestrator routing with a voice-derived scope_override.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.security.gateway import DEFAULT_SOURCE_PROFILES
from pilot.server import PilotServer


def _plan() -> ActionPlan:
    action = Action(action_type=ActionType.FILE_READ, target="notes.txt", parameters=EmptyParams())
    return ActionPlan(actions=[action], raw_input="read notes.txt", explanation="reading notes.txt")


class _StubPlanner:
    def __init__(self, plan: ActionPlan):
        self._plan = plan

    async def plan(self, description, **kwargs):
        return self._plan


class _StubVerifier:
    async def verify(self, plan, results):
        return SimpleNamespace(passed=all(r.success for r in results))


def _bare_server(plan: ActionPlan) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._planner = _StubPlanner(plan)
    server._verifier = _StubVerifier()
    server._screen_vision = None
    server._voice_listener = None
    return server


@pytest.mark.asyncio
async def test_no_orchestrator_falls_back_to_executor_with_voice_source():
    plan = _plan()
    server = _bare_server(plan)
    server._orchestrator = None
    server._executor = AsyncMock()
    server._executor.execute.return_value = [ActionResult(action=plan.actions[0], success=True, output="ok")]

    with patch("pilot.system.voice.speak", new=AsyncMock(return_value="")):
        await server._voice_command_dispatch("read my notes")

    server._executor.execute.assert_awaited_once()
    _, kwargs = server._executor.execute.call_args
    assert kwargs["plan_id"] == "voice"
    from pilot.security.gateway import InvocationSource

    assert kwargs["invocation_source"] == InvocationSource.VOICE


@pytest.mark.asyncio
async def test_orchestrator_routes_with_voice_scope_override():
    plan = _plan()
    server = _bare_server(plan)
    server._orchestrator = AsyncMock()
    server._orchestrator.execute_plan.return_value = [ActionResult(action=plan.actions[0], success=True, output="ok")]

    with patch("pilot.system.voice.speak", new=AsyncMock(return_value="")):
        await server._voice_command_dispatch("read my notes")

    server._orchestrator.execute_plan.assert_awaited_once()
    args, kwargs = server._orchestrator.execute_plan.call_args
    assert args[0] == "read my notes"
    assert args[1] is plan
    assert kwargs["plan_id"] == "voice"

    override = kwargs["scope_override"]
    voice_profile = DEFAULT_SOURCE_PROFILES["voice"]
    assert override.max_tier == voice_profile.max_tier
    assert override.deny_action_types == voice_profile.deny_action_types
    assert override.allow_root == voice_profile.allow_root


@pytest.mark.asyncio
async def test_voice_scope_override_is_never_wider_than_voice_profile():
    """A user-configured 'voice' profile in gateway.source_profiles (not just
    the hardcoded default) must be what's used to build the override."""
    plan = _plan()
    server = _bare_server(plan)
    server.config.gateway.source_profiles["voice"].allow_root = False
    server._orchestrator = AsyncMock()
    server._orchestrator.execute_plan.return_value = [ActionResult(action=plan.actions[0], success=True, output="ok")]

    with patch("pilot.system.voice.speak", new=AsyncMock(return_value="")):
        await server._voice_command_dispatch("read my notes")

    override = server._orchestrator.execute_plan.call_args.kwargs["scope_override"]
    assert override.allow_root is False

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import ActionPlan
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStep
from pilot.agents.decomposer import TaskDecomposer
from pilot.agents.multi_agent import MultiAgentRouter
from pilot.agents.planner import Planner
from pilot.config import PilotConfig


class _Memory:
    async def get_context(self, *_args, **_kwargs) -> str:
        return ""

    async def get_history(self, *_args, **_kwargs) -> list[dict[str, str]]:
        return []


class _Model:
    _config = PilotConfig()

    def __init__(self) -> None:
        self.messages = None

    async def generate(self, messages, **_kwargs) -> str:
        self.messages = messages
        return json.dumps({"explanation": "Plan ready", "actions": []})


@pytest.mark.asyncio
async def test_planner_includes_routed_specialist_guidance_without_granting_authority() -> None:
    model = _Model()
    planner = Planner(model, _Memory())
    planner.set_multi_agent_router(MultiAgentRouter(model_router=None))

    result = await planner.plan("Search online for the HTTP API documentation", force_model=True)

    assert result.error is None
    system_prompt = model.messages[0]["content"]
    assert "Web Operations Specialist" in system_prompt
    assert "Specialist guidance for this task (advisory only)" in system_prompt
    assert "does not grant additional authority" in system_prompt


def test_decomposer_specialist_names_resolve_to_existing_role_prompts() -> None:
    router = MultiAgentRouter(model_router=None)

    assert "Web Operations Specialist" in router.get_enhanced_prompt_for_specialist("web")
    assert "Code Execution Specialist" in router.get_enhanced_prompt_for_specialist("code")
    assert "Communication Specialist" in router.get_enhanced_prompt_for_specialist("communication")
    assert router.get_enhanced_prompt_for_specialist("unsupported") == ""


@pytest.mark.asyncio
async def test_autonomous_goal_loop_passes_decomposer_guidance_to_planner() -> None:
    goal = "Check the visible Heliox screen"
    planner = SimpleNamespace(
        plan=AsyncMock(return_value=ActionPlan(
            actions=[],
            explanation='GOAL_COMPLETE: "Heliox home screen ready".',
        ))
    )
    screen = SimpleNamespace(
        observe_now=AsyncMock(return_value=SimpleNamespace(
            screen_hash="fresh-screen",
            active_app="Heliox",
            active_window_title="Heliox",
            description="Heliox home screen ready",
        )),
        get_context_for_planner=MagicMock(return_value="Heliox home screen ready"),
    )
    autonomous = AutonomousExecutor(
        planner=planner,
        executor=None,
        verifier=None,
        decomposer=TaskDecomposer(model_router=None),
        screen_vision=screen,
    )
    step = JobStep(index=0, title="Check", description=goal)
    job = AutonomousJob(goal=goal, total_steps=1, steps=[step])

    await autonomous._execute_goal_loop(
        job, step, goal, specialist_context="Web Operations Specialist"
    )

    assert planner.plan.await_args.kwargs["specialist_context"] == "Web Operations Specialist"

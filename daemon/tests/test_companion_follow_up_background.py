import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    EmptyParams,
    OpenApplicationParams,
    ScreenVisionParams,
    SystemInfoParams,
    VerificationResult,
)
from pilot.agents.execution_companion import CompanionFollowUp
from pilot.config import PilotConfig
from pilot.server import PilotServer


class _DelayedCompanion:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def follow_up(self, user_input, plan, results, verification):
        await self.release.wait()
        return CompanionFollowUp(
            message="The verified result is ready.",
            suggestions=["Compare it", "Save it"],
        )


@pytest.mark.asyncio
async def test_companion_follow_up_never_blocks_terminal_result_delivery():
    server = PilotServer(PilotConfig())
    companion = _DelayedCompanion()
    server._execution_companion = companion
    server._broadcast_notification = AsyncMock()
    plan = ActionPlan(actions=[], explanation="Done")
    verification = VerificationResult(passed=True, details=["verified"])

    started = time.perf_counter()
    server._spawn_companion_follow_up(
        user_input="show system information",
        plan=plan,
        results=[],
        verification=verification,
        result_text="Windows 11",
        chat_session_id="chat-1",
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    server._broadcast_notification.assert_not_awaited()

    companion.release.set()
    await asyncio.gather(*tuple(server._companion_follow_up_tasks))

    server._broadcast_notification.assert_awaited_once_with(
        "companion_follow_up",
        {
            "session_id": "chat-1",
            "message": "The verified result is ready.",
            "suggestions": ["Compare it", "Save it"],
            "source": "model",
        },
    )


def test_bounded_telemetry_plan_gets_immediate_local_companion_review():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="system",
                parameters=SystemInfoParams(),
            )
        ],
        explanation="Display current system information",
    )

    review = PilotServer._deterministic_companion_review(plan)

    assert review is not None
    assert review.decision == "CONTINUE"
    assert "read-only" in review.reason


def test_battery_only_request_revises_unrelated_system_telemetry():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="battery",
                parameters=SystemInfoParams(categories=["os", "cpu", "memory", "disk", "network"]),
            )
        ],
        explanation="Read battery status",
    )

    review = PilotServer._deterministic_companion_review(plan, "Show current battery status without changing anything.")

    assert review is not None
    assert review.decision == "REVISE"
    assert "battery_info" in review.planner_feedback


def test_mixed_battery_and_cpu_request_is_not_auto_approved():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="system",
                parameters=SystemInfoParams(categories=["battery", "cpu"]),
            )
        ],
        explanation="Read battery and CPU",
    )

    assert PilotServer._deterministic_companion_review(plan, "Show battery and CPU status") is None


def test_close_request_revises_open_application_plan():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.OPEN_APPLICATION,
                target="ChatGPT",
                parameters=OpenApplicationParams(name="ChatGPT"),
            )
        ],
        explanation="Open ChatGPT",
    )

    review = PilotServer._deterministic_companion_review(plan, "Close this application")

    assert review is not None
    assert review.decision == "REVISE"
    assert "not open_application" in review.planner_feedback
    assert PilotServer._deterministic_companion_review(plan, "Close the current window and open ChatGPT") is None


def test_system_health_review_gets_immediate_local_companion_review():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_HEALTH_REVIEW,
                target="system health",
                parameters=EmptyParams(),
            )
        ],
        explanation="Collect a read-only system health review",
    )

    review = PilotServer._deterministic_companion_review(plan)

    assert review is not None
    assert review.decision == "CONTINUE"
    assert "read-only" in review.reason


def test_screen_analysis_gets_immediate_local_companion_review():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SCREEN_ANALYZE,
                target="screen",
                parameters=ScreenVisionParams(prompt="What is visible?"),
            )
        ],
        explanation="Analyze the visible screen",
    )

    review = PilotServer._deterministic_companion_review(plan)

    assert review is not None
    assert review.decision == "CONTINUE"


def test_direct_application_launch_gets_immediate_local_companion_review():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.OPEN_APPLICATION,
                target="Openscreen",
                parameters=OpenApplicationParams(name="Openscreen"),
            )
        ],
        explanation="Launch Openscreen",
    )

    review = PilotServer._deterministic_companion_review(plan)

    assert review is not None
    assert review.decision == "CONTINUE"
    assert "bounded" in review.reason

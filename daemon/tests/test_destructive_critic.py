import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.agents.destructive_critic import DestructiveCriticAgent


@pytest.mark.asyncio
async def test_tier4_plan_gets_blocked():
    model_router = AsyncMock()

    model_router.generate.return_value = json.dumps(
        {
            "verdict": "BLOCK",
            "risk_score": 0.95,
            "issues": ["Dangerous destructive action"],
            "safe_actions": [],
            "flagged_actions": ["delete_root"],
            "recommendation": "Do not execute",
        }
    )

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "ROOT_CRITICAL"
    fake_plan.explanation = "Delete system files"

    verdict = await critic.review(
        "Delete everything",
        fake_plan,
    )

    assert verdict.verdict == "BLOCK"
    assert verdict.is_blocked is True


@pytest.mark.asyncio
async def test_tier3_plan_gets_warned():
    model_router = AsyncMock()

    model_router.generate.return_value = json.dumps(
        {
            "verdict": "WARN",
            "risk_score": 0.60,
            "issues": ["Potential risk detected"],
            "safe_actions": [],
            "flagged_actions": ["delete_files"],
            "recommendation": "Proceed carefully",
        }
    )

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "DESTRUCTIVE"
    fake_plan.explanation = "Remove user files"

    verdict = await critic.review(
        "Delete selected files",
        fake_plan,
    )

    assert verdict.verdict == "WARN"
    assert verdict.has_warnings is True


@pytest.mark.asyncio
async def test_safe_plan_approved():
    model_router = AsyncMock()

    model_router.generate.return_value = json.dumps(
        {
            "verdict": "APPROVE",
            "risk_score": 0.10,
            "issues": [],
            "safe_actions": ["read_file"],
            "flagged_actions": [],
            "recommendation": "Safe to continue",
        }
    )

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "SAFE"
    fake_plan.explanation = "Read a file"

    verdict = await critic.review(
        "Open a text file",
        fake_plan,
    )

    assert verdict.verdict == "APPROVE"
    assert verdict.is_blocked is False


@pytest.mark.asyncio
async def test_critic_error_falls_back_to_warn():
    model_router = AsyncMock()

    model_router.generate.side_effect = Exception("LLM unavailable")

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "ROOT_CRITICAL"
    fake_plan.explanation = "Delete files"

    verdict = await critic.review(
        "Delete files",
        fake_plan,
    )

    assert verdict.verdict == "WARN"
    assert verdict.has_warnings is True

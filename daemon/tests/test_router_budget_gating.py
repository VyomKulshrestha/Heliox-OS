"""Phase 2 tests: ModelRouter per-action gate + per-task budget integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.config import ModelConfig, PilotConfig
from pilot.models.budget_tracker import (
    ActionBudgetExceededError,
    BudgetTracker,
    TaskBudgetExceededError,
    current_task_id,
)
from pilot.models.router import ModelRouter


@pytest.fixture
def model_config():
    return ModelConfig(
        provider="ollama",
        budget_enabled=True,
        budget_monthly_limit_usd=100.0,
        max_tokens_per_action=100,        # very small cap so we can test it easily
        max_tokens_per_task=1000,
        max_usd_per_task=0.10,
    )


@pytest.fixture
def pilot_config(model_config):
    cfg = PilotConfig()
    cfg.model = model_config
    return cfg


@pytest.fixture
async def tracker(model_config, tmp_path):
    t = BudgetTracker(model_config, str(tmp_path / "b.db"))
    await t.initialize()
    yield t
    await t.close()


@pytest.fixture
def router(pilot_config, tracker):
    """ModelRouter with mocked cache + backends so generate() never actually calls a model."""
    # Patch heavy dependencies that ModelRouter creates in __init__
    with patch("pilot.models.router.OllamaClient"), \
    patch("pilot.models.router.CloudClient"), \
    patch("pilot.models.router.LLMCache"), \
    patch("pilot.models.router.RedisCacheAdapter"):
        r = ModelRouter(pilot_config, MagicMock())
    r._generate_with_cache = AsyncMock(return_value="mocked-response")
    r._rate_limiter = MagicMock()
    r._rate_limiter.acquire = AsyncMock()
    r.set_budget_tracker(tracker)
    return r


def test_estimate_input_tokens_uses_len_over_4():
    text = "a" * 400  # 400 chars -> 100 tokens estimated
    assert ModelRouter._estimate_input_tokens(text) == 100


def test_estimate_input_tokens_handles_list_prompts():
    prompts = [{"role": "user", "content": "hi"}]
    # Serializes to JSON, then len // 4
    count = ModelRouter._estimate_input_tokens(prompts)
    assert count > 0


@pytest.mark.asyncio
async def test_short_prompt_passes_action_gate(router):
    # "hello" is well under 100 tokens
    result = await router.generate("hello")
    assert result == "mocked-response"


@pytest.mark.asyncio
async def test_oversized_prompt_raises_action_budget_error(router):
    huge_prompt = "x" * 10_000  # ~2500 tokens, far above 100 cap
    with pytest.raises(ActionBudgetExceededError, match=r"max_tokens_per_action"):
        await router.generate(huge_prompt)


@pytest.mark.asyncio
async def test_task_budget_exceeded_blocks_call(router, tracker):
    tracker.start_task("task-x")
    budget = tracker.get_task_budget("task-x")
    budget.tokens_used = 1001  # over the 1000 cap

    token = current_task_id.set("task-x")
    try:
        with pytest.raises(TaskBudgetExceededError):
            await router.generate("short prompt")
    finally:
        current_task_id.reset(token)


@pytest.mark.asyncio
async def test_no_task_context_skips_task_check(router, tracker):
    # No task started, no contextvar set
    result = await router.generate("hello")
    assert result == "mocked-response"
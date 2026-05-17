from __future__ import annotations

import pytest

from pilot.config import ModelConfig
from pilot.models.budget_tracker import BudgetExceededError, BudgetTracker


@pytest.mark.asyncio
async def test_budget_tracker_uses_pool_for_usage_stats_and_reset(tmp_path):
    config = ModelConfig()
    config.budget_enabled = True
    config.budget_monthly_limit_usd = 1.0
    tracker = BudgetTracker(config, str(tmp_path / "budget.db"))
    await tracker.initialize()
    try:
        await tracker.record_usage("openai", "gpt-test", 100, 50)
        stats = await tracker.get_stats()

        assert stats["calls"] == 1
        assert stats["input_tokens"] == 100
        assert stats["output_tokens"] == 50
        assert stats["cost_usd"] > 0

        await tracker.reset_current_month()
        reset_stats = await tracker.get_stats()
        assert reset_stats["calls"] == 0
        assert reset_stats["cost_usd"] == 0
    finally:
        await tracker.close()


@pytest.mark.asyncio
async def test_budget_tracker_budget_gate_uses_cached_monthly_total(tmp_path):
    config = ModelConfig()
    config.budget_enabled = True
    config.budget_monthly_limit_usd = 0.0001
    tracker = BudgetTracker(config, str(tmp_path / "budget.db"))
    await tracker.initialize()
    try:
        await tracker.record_usage("openai", "gpt-test", 1000, 1000)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget("openai")
    finally:
        await tracker.close()

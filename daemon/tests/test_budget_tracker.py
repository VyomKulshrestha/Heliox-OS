"""Unit tests for pilot.models.budget_tracker.BudgetTracker.

Run with:
    cd daemon
    pytest tests/test_budget_tracker.py -v --tb=short
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pilot.models.budget_tracker import BudgetExceededError, BudgetTracker

# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------


@dataclass
class FakeModelConfig:
    budget_enabled: bool = True
    budget_monthly_limit_usd: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def make_tracker(tmp_path, **kwargs) -> BudgetTracker:
    cfg = FakeModelConfig(**kwargs)
    tracker = BudgetTracker(cfg, str(tmp_path / "budget.db"))
    await tracker.initialize()
    return tracker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_usage_accumulates(tmp_path):
    """record_usage() persists rows and accumulates the monthly cost."""
    tracker = await make_tracker(tmp_path)
    await tracker.record_usage("openai", "gpt-4", 1000, 500)
    await tracker.record_usage("openai", "gpt-4", 1000, 500)
    stats = await tracker.get_stats()
    assert stats["calls"] == 2
    assert stats["input_tokens"] == 2000
    assert stats["output_tokens"] == 1000
    # 2 × ((1000×0.005 + 500×0.015) / 1000) = 2 × 0.0125 = 0.025
    assert abs(stats["cost_usd"] - 0.025) < 1e-6
    await tracker.close()


@pytest.mark.asyncio
async def test_check_budget_raises_when_exceeded(tmp_path):
    """check_budget() raises BudgetExceededError once the limit is hit."""
    tracker = await make_tracker(tmp_path, budget_monthly_limit_usd=0.001)
    # Record enough usage to exceed the $0.001 limit
    await tracker.record_usage("openai", "gpt-4", 1000, 1000)
    with pytest.raises(BudgetExceededError):
        tracker.check_budget("cloud")
    await tracker.close()


@pytest.mark.asyncio
async def test_check_budget_noop_when_disabled(tmp_path):
    """check_budget() never raises when budget_enabled=False."""
    tracker = await make_tracker(tmp_path, budget_enabled=False, budget_monthly_limit_usd=0.0)
    await tracker.record_usage("openai", "gpt-4", 100_000, 100_000)
    tracker.check_budget("cloud")  # must not raise
    await tracker.close()


@pytest.mark.asyncio
async def test_check_budget_noop_for_free_providers(tmp_path):
    """check_budget() never blocks ollama or local providers."""
    tracker = await make_tracker(tmp_path, budget_monthly_limit_usd=0.0)
    tracker.check_budget("ollama")  # must not raise
    tracker.check_budget("local")  # must not raise
    await tracker.close()


@pytest.mark.asyncio
async def test_get_stats_shape(tmp_path):
    """get_stats() returns all expected keys."""
    tracker = await make_tracker(tmp_path)
    stats = await tracker.get_stats()
    for key in ("enabled", "month", "calls", "input_tokens", "output_tokens", "cost_usd", "limit_usd", "remaining_usd"):
        assert key in stats, f"Missing key: {key}"
    await tracker.close()


@pytest.mark.asyncio
async def test_get_stats_remaining_decreases(tmp_path):
    """remaining_usd decreases after recording usage."""
    tracker = await make_tracker(tmp_path, budget_monthly_limit_usd=1.0)
    before = (await tracker.get_stats())["remaining_usd"]
    await tracker.record_usage("openai", "gpt-4", 1000, 500)
    after = (await tracker.get_stats())["remaining_usd"]
    assert after < before
    await tracker.close()


@pytest.mark.asyncio
async def test_reset_current_month(tmp_path):
    """reset_current_month() zeroes the cached total and removes DB rows."""
    tracker = await make_tracker(tmp_path, budget_monthly_limit_usd=0.001)
    await tracker.record_usage("openai", "gpt-4", 1000, 1000)
    # Verify limit is exceeded before reset
    with pytest.raises(BudgetExceededError):
        tracker.check_budget("cloud")

    await tracker.reset_current_month()

    stats = await tracker.get_stats()
    assert stats["calls"] == 0
    assert stats["cost_usd"] == 0.0
    # After reset the budget check must pass again
    tracker.check_budget("cloud")
    await tracker.close()


@pytest.mark.asyncio
async def test_cross_month_isolation(tmp_path):
    """Usage from a different month is not counted against the current month."""
    from datetime import UTC, datetime

    import aiosqlite

    tracker = await make_tracker(tmp_path)
    db_path = str(tmp_path / "budget.db")

    # Directly insert a row for a past month to simulate historical data
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO token_usage
               (timestamp, month, provider, model, input_tokens, output_tokens, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(UTC).isoformat(), "2000-01", "openai", "gpt-4", 100000, 100000, 999.0),
        )
        await db.commit()

    # Re-initialize a fresh tracker — it should only load the current month
    tracker2 = await make_tracker(tmp_path, budget_monthly_limit_usd=1.0)
    stats = await tracker2.get_stats()
    assert stats["cost_usd"] == 0.0
    tracker2.check_budget("cloud")  # must not raise despite $999 in past month
    await tracker2.close()
    await tracker.close()


@pytest.mark.asyncio
async def test_ollama_cost_is_zero(tmp_path):
    """Ollama usage records $0 cost."""
    tracker = await make_tracker(tmp_path)
    await tracker.record_usage("ollama", "llama3", 10000, 10000)
    stats = await tracker.get_stats()
    assert stats["cost_usd"] == 0.0
    await tracker.close()

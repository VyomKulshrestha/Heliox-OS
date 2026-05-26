"""Phase 1 tests: BudgetTracker per-task tracking."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from pilot.config import ModelConfig
from pilot.models.budget_tracker import (
    ActionBudgetExceededError,
    BudgetTracker,
    TaskBudget,
    TaskBudgetExceededError,
    current_task_id,
)


@pytest.fixture
def config():
    return ModelConfig(
        budget_enabled=True,
        budget_monthly_limit_usd=100.0,
        max_tokens_per_task=1000,
        max_usd_per_task=0.05,
    )


@pytest.fixture
async def tracker(config, tmp_path):
    t = BudgetTracker(config, str(tmp_path / "test_budget.db"))
    await t.initialize()
    yield t
    await t.close()


def test_task_budget_dataclass_initializes_with_caps():
    b = TaskBudget(task_id="t1", token_cap=5000, usd_cap=0.10)
    assert b.task_id == "t1"
    assert b.tokens_used == 0
    assert b.usd_spent == 0.0
    assert b.exceeded is False


@pytest.mark.asyncio
async def test_start_task_creates_budget(tracker):
    budget = tracker.start_task("task-abc")
    assert budget.task_id == "task-abc"
    assert budget.token_cap == 1000
    assert budget.usd_cap == 0.05
    assert tracker.get_task_budget("task-abc") is budget


@pytest.mark.asyncio
async def test_end_task_removes_budget(tracker):
    tracker.start_task("task-abc")
    final = tracker.end_task("task-abc")
    assert final is not None
    assert tracker.get_task_budget("task-abc") is None


@pytest.mark.asyncio
async def test_check_task_budget_silent_below_limits(tracker):
    tracker.start_task("task-abc")
    # No usage recorded yet — should not raise
    tracker.check_task_budget("task-abc")


@pytest.mark.asyncio
async def test_check_task_budget_raises_on_token_limit(tracker):
    budget = tracker.start_task("task-abc")
    budget.tokens_used = 1000  # exactly at cap
    with pytest.raises(TaskBudgetExceededError, match=r"token budget"):
        tracker.check_task_budget("task-abc")


@pytest.mark.asyncio
async def test_check_task_budget_raises_on_usd_limit(tracker):
    budget = tracker.start_task("task-abc")
    budget.usd_spent = 0.05  # exactly at cap
    with pytest.raises(TaskBudgetExceededError, match=r"USD budget"):
        tracker.check_task_budget("task-abc")


@pytest.mark.asyncio
async def test_record_usage_updates_task_totals(tracker):
    tracker.start_task("task-abc")
    token = current_task_id.set("task-abc")
    try:
        await tracker.record_usage("openai", "gpt-4", input_tokens=500, output_tokens=200)
    finally:
        current_task_id.reset(token)

    budget = tracker.get_task_budget("task-abc")
    assert budget.tokens_used == 700
    assert budget.usd_spent > 0  # openai is non-free


@pytest.mark.asyncio
async def test_record_usage_without_active_task_skips_task_tracking(tracker):
    # No task started, contextvar unset
    await tracker.record_usage("openai", "gpt-4", input_tokens=500, output_tokens=200)
    # Monthly cost should still update, but no task budget was touched
    assert tracker._monthly_cost > 0


@pytest.mark.asyncio
async def test_concurrent_tasks_isolated(tracker):
    tracker.start_task("task-a")
    tracker.start_task("task-b")

    # Record against task-a
    token = current_task_id.set("task-a")
    try:
        await tracker.record_usage("openai", "gpt-4", 100, 50)
    finally:
        current_task_id.reset(token)

    a = tracker.get_task_budget("task-a")
    b = tracker.get_task_budget("task-b")
    assert a.tokens_used == 150
    assert b.tokens_used == 0  # task-b untouched


@pytest.mark.asyncio
async def test_check_task_budget_disabled_when_budget_disabled(tmp_path):
    cfg = ModelConfig(budget_enabled=False, max_tokens_per_task=10)
    t = BudgetTracker(cfg, str(tmp_path / "b.db"))
    await t.initialize()
    try:
        budget = t.start_task("task-abc")
        budget.tokens_used = 9999  # way over
        # Should not raise — budget disabled
        t.check_task_budget("task-abc")
    finally:
        await t.close()
"""Global API budget tracker — records token usage and enforces monthly spend limits."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pilot.db.sqlite_pool import AsyncSqlitePool

if TYPE_CHECKING:
    from pilot.config import ModelConfig

logger = logging.getLogger("pilot.models.budget_tracker")

# (input_usd, output_usd) per 1 000 tokens
COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "openai": (0.005, 0.015),
    "claude": (0.003, 0.015),
    "gemini": (0.000075, 0.0003),
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS token_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    month         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    task_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_token_usage_month ON token_usage(month);
CREATE INDEX IF NOT EXISTS idx_token_usage_task ON token_usage(task_id);
"""

# Existing installations won't have the task_id column. ALTER TABLE handles
# the migration; it's wrapped in try/except since SQLite < 3.35 doesn't
# support IF NOT EXISTS on ADD COLUMN and the column may already be there.
TASK_ID_MIGRATION_SQL = "ALTER TABLE token_usage ADD COLUMN task_id TEXT"


class BudgetExceededError(RuntimeError):
    """Raised when the monthly spend limit has been reached."""


def _current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(provider, (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1000.0


class TaskBudgetExceededError(BudgetExceededError):
    """Raised when a single task's cumulative budget is exceeded."""


class ActionBudgetExceededError(BudgetExceededError):
    """Raised when a single action's estimated token cost exceeds the per-action cap."""


# Threads the active task id through to record_usage / check_task_budget
# without requiring every call site (router, agents) to pass it explicitly.
# asyncio.create_task() inherits contextvars by default, so fire-and-forget
# record_usage tasks still see the right task_id.
current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)


@dataclass
class TaskBudget:
    """In-memory state for a single in-flight task's budget.

    Tracks cumulative tokens and USD against the task's caps. Stored in
    BudgetTracker._tasks under the task_id while the task is active; removed
    by end_task() when the task completes.
    """

    task_id: str
    token_cap: int
    usd_cap: float
    tokens_used: int = 0
    usd_spent: float = 0.0
    exceeded: bool = False


class BudgetTracker:
    """Tracks cumulative LLM token spend and enforces a monthly USD limit."""

    def __init__(self, config: ModelConfig, db_path: str) -> None:
        self._enabled: bool = config.budget_enabled
        self._monthly_limit: float = config.budget_monthly_limit_usd
        self._token_cap_per_task: int = config.max_tokens_per_task
        self._usd_cap_per_task: float = config.max_usd_per_task
        self._db_path = db_path
        self._pool: AsyncSqlitePool | None = None
        self._monthly_cost: float = 0.0
        self._tasks: dict[str, TaskBudget] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self._pool = AsyncSqlitePool(self._db_path, read_pool_size=2)
        await self._pool.start()
        async with self._pool.write() as db:
            await db.executescript(SCHEMA_SQL)
            # Migrate existing installations that predate the task_id column.
            # Duplicate-column errors here are expected and ignored.
            try:
                await db.execute(TASK_ID_MIGRATION_SQL)
            except Exception as exc:
                if "duplicate column" not in str(exc).lower():
                    logger.warning("task_id column migration: %s", exc)
            await db.commit()
        self._monthly_cost = await self._load_monthly_cost()
        logger.info(
            "BudgetTracker ready — month=%s spent=%.4f limit=%.2f enabled=%s",
            _current_month(),
            self._monthly_cost,
            self._monthly_limit,
            self._enabled,
        )

    async def _load_monthly_cost(self) -> float:
        if not self._pool:
            return 0.0
        async with self._pool.read() as db:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM token_usage WHERE month = ?",
                (_current_month(),),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return float(row[0]) if row else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_budget(self, provider: str) -> None:
        """Synchronous budget gate — raises BudgetExceededError if limit reached.

        Uses the in-memory cached monthly total so there is zero I/O on the
        hot path.  Free providers (ollama/local) are never blocked.
        """
        if not self._enabled:
            return
        if provider in ("ollama", "local"):
            return
        if self._monthly_limit > 0 and self._monthly_cost >= self._monthly_limit:
            raise BudgetExceededError(
                f"Monthly API budget of ${self._monthly_limit:.2f} exceeded "
                f"(spent ${self._monthly_cost:.4f}). "
                "Increase budget_monthly_limit_usd or reset via budget_reset."
            )

    def start_task(self, task_id: str) -> TaskBudget:
        """Begin tracking a new task. Returns the freshly-created TaskBudget.

        Called by the orchestrator at the start of execute_plan(). The task_id
        should also be set on the current_task_id contextvar so downstream
        record_usage / check_task_budget calls find this task.
        """
        budget = TaskBudget(
            task_id=task_id,
            token_cap=self._token_cap_per_task,
            usd_cap=self._usd_cap_per_task,
        )
        self._tasks[task_id] = budget
        logger.info(
            "BudgetTracker: started task %s (token_cap=%d, usd_cap=%.4f)",
            task_id,
            budget.token_cap,
            budget.usd_cap,
        )
        return budget

    def end_task(self, task_id: str) -> TaskBudget | None:
        """Remove a task from active tracking. Returns the final TaskBudget."""
        budget = self._tasks.pop(task_id, None)
        if budget:
            logger.info(
                "BudgetTracker: ended task %s (tokens=%d, usd=%.4f, exceeded=%s)",
                task_id,
                budget.tokens_used,
                budget.usd_spent,
                budget.exceeded,
            )
        return budget

    def check_task_budget(self, task_id: str | None = None) -> None:
        """Synchronous gate raising TaskBudgetExceededError if task limits hit.

        If task_id is None, reads from the contextvar. Returns silently if
        budgets are disabled, no task is active, or the task isn't tracked.
        """
        if not self._enabled:
            return
        if task_id is None:
            task_id = current_task_id.get()
        if not task_id:
            return
        task = self._tasks.get(task_id)
        if task is None:
            return

        if task.tokens_used >= task.token_cap:
            task.exceeded = True
            raise TaskBudgetExceededError(
                f"Task {task_id} exceeded token budget "
                f"({task.tokens_used} >= {task.token_cap}). "
                f"Increase max_tokens_per_task or split the task."
            )
        if task.usd_spent >= task.usd_cap:
            task.exceeded = True
            raise TaskBudgetExceededError(
                f"Task {task_id} exceeded USD budget "
                f"(${task.usd_spent:.4f} >= ${task.usd_cap:.4f}). "
                f"Increase max_usd_per_task or split the task."
            )

    def get_task_budget(self, task_id: str | None = None) -> TaskBudget | None:
        """Return the live TaskBudget for a task, or None if not tracked."""
        if task_id is None:
            task_id = current_task_id.get()
        if not task_id:
            return None
        return self._tasks.get(task_id)

    async def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Persist one call's token usage and update in-memory totals.

        Reads the active task_id from the contextvar; if a task is in flight,
        the call's tokens and cost are also added to the in-memory TaskBudget
        so subsequent check_task_budget() calls see the latest state.
        """
        if not self._pool:
            return
        cost = _estimate_cost(provider, input_tokens, output_tokens)
        now = datetime.now(UTC).isoformat()
        month = _current_month()
        task_id = current_task_id.get()

        async with self._lock:
            async with self._pool.write() as db:
                await db.execute(
                    """INSERT INTO token_usage
                       (timestamp, month, provider, model, input_tokens,
                        output_tokens, cost_usd, task_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (now, month, provider, model, input_tokens, output_tokens, cost, task_id),
                )
                await db.commit()
            self._monthly_cost += cost

            # Update per-task running totals if a task is active and tracked
            if task_id and task_id in self._tasks:
                task = self._tasks[task_id]
                task.tokens_used += input_tokens + output_tokens
                task.usd_spent += cost

    async def get_stats(self) -> dict:
        """Return current-month usage summary."""
        if not self._pool:
            return {}
        month = _current_month()
        async with self._pool.read() as db:
            cursor = await db.execute(
                """SELECT
                       COUNT(*) AS calls,
                       COALESCE(SUM(input_tokens), 0) AS total_input,
                       COALESCE(SUM(output_tokens), 0) AS total_output,
                       COALESCE(SUM(cost_usd), 0.0) AS total_cost
                   FROM token_usage WHERE month = ?""",
                (month,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        calls, total_input, total_output, total_cost = row if row else (0, 0, 0, 0.0)
        remaining = max(0.0, self._monthly_limit - float(total_cost)) if self._monthly_limit > 0 else None
        return {
            "enabled": self._enabled,
            "month": month,
            "calls": calls,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(float(total_cost), 6),
            "limit_usd": self._monthly_limit,
            "remaining_usd": round(remaining, 6) if remaining is not None else None,
        }

    async def reset_current_month(self) -> None:
        """Delete all records for the current month and reset the in-memory cache."""
        if not self._pool:
            return
        async with self._lock:
            async with self._pool.write() as db:
                await db.execute("DELETE FROM token_usage WHERE month = ?", (_current_month(),))
                await db.commit()
            self._monthly_cost = 0.0
        logger.info("BudgetTracker: current month reset")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

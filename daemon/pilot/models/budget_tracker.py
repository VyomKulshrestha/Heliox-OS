"""Global API budget tracker — records token usage and enforces monthly spend limits."""

from __future__ import annotations

import asyncio
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pilot.db.sqlite_pool import AsyncSqlitePool

if TYPE_CHECKING:
    from pilot.config import ModelConfig

logger = logging.getLogger("pilot.models.budget_tracker")


# Each provider's highest current rate, (input_usd, output_usd) per 1 000 tokens: what a call is costed at when
# its model is not known. Calls name their model and are priced per model (pilot.models.prices, every table with
# the page and date it was read from; Claude below).
def _highest_per_1k(provider: str) -> tuple[float, float]:
    from pilot.models import prices

    if provider in prices.FREE_PROVIDERS:
        return (0.0, 0.0)
    found = prices.price_for(provider, "")[0]
    return (found.input / 1000.0, found.output / 1000.0)


class _HighestRates(dict):
    """COST_PER_1K_TOKENS[provider]: that provider's highest current rate, read when asked (the tables may change)."""

    def __missing__(self, provider: str) -> tuple[float, float]:
        return _highest_per_1k(provider)

    def get(self, provider: str, default: object = None) -> tuple[float, float]:  # type: ignore[override]
        return _highest_per_1k(provider)


COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = _HighestRates()


@dataclass(frozen=True)
class ClaudePrice:
    """USD per million tokens for one Claude model."""

    input: float
    output: float
    # None means "use the default 0.1x of the input rate".
    cache_read: float | None = None
    # Beyond this many prompt tokens a higher (input, output) rate applies.
    long_prompt_threshold: int = 0
    long_input: float = 0.0
    long_output: float = 0.0


# Source: https://platform.claude.com/docs/en/about-claude/pricing, read 2026-11-13 (pilot.models.prices.SOURCES),
# Anthropic first-party API rates. Cache reads use the page's value where it differs from 0.1x input (0.025x for
# Fable and Mythos 5.2 and 5.1, 0.05x for Opus 5.5 and Sonnet 5.5) and 0.1x input otherwise. Cache writes are billed
# at 1.25x input (5 minute TTL, the only TTL Heliox requests). Retired models stay so old usage keeps its price.
CLAUDE_MODEL_PRICES: dict[str, ClaudePrice] = {
    "claude-fable-5-2": ClaudePrice(8.0, 40.0, 0.20),
    "claude-mythos-5-2": ClaudePrice(8.0, 40.0, 0.20),
    "claude-fable-5-1": ClaudePrice(10.0, 50.0, 0.25),
    "claude-mythos-5-1": ClaudePrice(10.0, 50.0, 0.25),
    "claude-fable-5": ClaudePrice(10.0, 50.0),
    "claude-mythos-5": ClaudePrice(10.0, 50.0),
    "claude-opus-5-5": ClaudePrice(4.0, 20.0, 0.20),
    "claude-opus-5": ClaudePrice(5.0, 25.0),
    "claude-opus-4-8": ClaudePrice(5.0, 25.0),
    "claude-opus-4-7": ClaudePrice(5.0, 25.0),
    "claude-opus-4-6": ClaudePrice(5.0, 25.0),
    "claude-opus-4-5": ClaudePrice(5.0, 25.0),
    "claude-opus-4-1": ClaudePrice(15.0, 75.0),
    "claude-opus-4": ClaudePrice(15.0, 75.0),
    "claude-sonnet-5-5": ClaudePrice(2.0, 10.0, 0.10),
    "claude-sonnet-5": ClaudePrice(2.0, 10.0),
    "claude-sonnet-4-6": ClaudePrice(3.0, 15.0),
    "claude-sonnet-4-5": ClaudePrice(3.0, 15.0),
    "claude-sonnet-4": ClaudePrice(3.0, 15.0),
    # Haiku 5.5: $0.10/$0.50 up to 100K prompt tokens, $0.50/$2.50 beyond.
    "claude-haiku-5-5": ClaudePrice(0.10, 0.50, None, 100_000, 0.50, 2.50),
    "claude-haiku-4-5": ClaudePrice(1.0, 5.0),
}

# An unknown Claude model is billed at the most expensive known rate so the
# cap errs on the safe side.
# Retired models keep their price for old usage, but do not set the price of an unknown one.
CLAUDE_RETIRED = frozenset({"claude-opus-4-1", "claude-opus-4", "claude-sonnet-4"})
CLAUDE_UNKNOWN_MODEL_PRICE = max(
    (price for name, price in CLAUDE_MODEL_PRICES.items() if name not in CLAUDE_RETIRED),
    key=lambda p: (p.output, p.input),
)

CLAUDE_CACHE_WRITE_MULTIPLIER = 1.25
_DATED_SUFFIX = re.compile(r"^-\d{8}$")


def claude_price_for(model: str) -> tuple[ClaudePrice, bool]:
    """Return (price, known). Accepts a dated snapshot suffix like -20260401."""
    key = (model or "").strip().lower()
    price = CLAUDE_MODEL_PRICES.get(key)
    if price is not None:
        return price, True
    for known, candidate in CLAUDE_MODEL_PRICES.items():
        if key.startswith(known) and _DATED_SUFFIX.match(key[len(known) :]):
            return candidate, True
    return CLAUDE_UNKNOWN_MODEL_PRICE, False


def claude_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Dollar cost of one Messages API call from the usage Anthropic reported.

    `output_tokens` already includes thinking tokens. The three input fields
    are disjoint in the API's usage object, so they are summed, not subtracted.
    """
    price, _ = claude_price_for(model)
    in_rate, out_rate = price.input, price.output
    prompt_total = input_tokens + cache_read_tokens + cache_creation_tokens
    long_prompt = bool(price.long_prompt_threshold) and prompt_total > price.long_prompt_threshold
    if long_prompt:
        in_rate, out_rate = price.long_input, price.long_output
    read_rate = in_rate * 0.1 if (long_prompt or price.cache_read is None) else price.cache_read
    return (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_read_tokens * read_rate
        + cache_creation_tokens * in_rate * CLAUDE_CACHE_WRITE_MULTIPLIER
    ) / 1_000_000.0


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
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    usage_source  TEXT NOT NULL DEFAULT 'estimate',
    task_id       TEXT,
    agent_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_token_usage_month ON token_usage(month);
CREATE INDEX IF NOT EXISTS idx_token_usage_task ON token_usage(task_id);
"""

# Existing installations won't have the task_id column. ALTER TABLE handles
# the migration; it's wrapped in try/except since SQLite < 3.35 doesn't
# support IF NOT EXISTS on ADD COLUMN and the column may already be there.
TASK_ID_MIGRATION_SQL = "ALTER TABLE token_usage ADD COLUMN task_id TEXT"
AGENT_ID_MIGRATION_SQL = "ALTER TABLE token_usage ADD COLUMN agent_id TEXT"
USAGE_COLUMN_MIGRATIONS = (
    "ALTER TABLE token_usage ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE token_usage ADD COLUMN cache_creation_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE token_usage ADD COLUMN usage_source TEXT NOT NULL DEFAULT 'estimate'",
)


class BudgetExceededError(RuntimeError):
    """Raised when the monthly spend limit has been reached."""


def _current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _estimate_cost(provider: str, input_tokens: int, output_tokens: int, model: str = "") -> float:
    """One call's cost for a provider other than Claude, priced per model (pilot.models.prices). A model with no
    current price is costed at its provider's highest rate, never at zero."""
    # Subscription CLI calls consume plan quota, not metered API dollars.
    # Keep their token evidence while preventing API-price estimates from
    # being attached to (for example) a Claude Pro/Max invocation.
    if provider.startswith("subscription:"):
        return 0.0
    from pilot.models import prices

    return prices.cost_usd(provider, model, input_tokens, output_tokens)


class TaskBudgetExceededError(BudgetExceededError):
    """Raised when a single task's cumulative budget is exceeded."""


class ActionBudgetExceededError(BudgetExceededError):
    """Raised when a single action's estimated token cost exceeds the per-action cap."""


# Threads the active task id through to record_usage / check_task_budget
# without requiring every call site (router, agents) to pass it explicitly.
# ModelRouter awaits record_usage before returning so the active task remains
# present until its final usage record and in-memory cap update are complete.
current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)
current_agent_id: ContextVar[str | None] = ContextVar("current_agent_id", default=None)


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


@dataclass
class AgentTokenBudget:
    """Per-specialist token allowance within one active task."""

    task_id: str
    agent_id: str
    token_cap: int
    tokens_used: int = 0
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
        self._cost_month: str = _current_month()
        self._tasks: dict[str, TaskBudget] = {}
        self._agent_tasks: dict[tuple[str, str], AgentTokenBudget] = {}
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
            try:
                await db.execute(AGENT_ID_MIGRATION_SQL)
            except Exception as exc:
                if "duplicate column" not in str(exc).lower():
                    logger.warning("agent_id column migration: %s", exc)
            for migration in USAGE_COLUMN_MIGRATIONS:
                try:
                    await db.execute(migration)
                except Exception as exc:
                    if "duplicate column" not in str(exc).lower():
                        logger.warning("usage column migration: %s", exc)
            await db.commit()
        self._monthly_cost = await self._load_monthly_cost()
        self._cost_month: str = _current_month()
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

    def reconfigure(self, config: ModelConfig) -> None:
        """Apply validated budget limits to new and active tasks."""
        self._enabled = config.budget_enabled
        self._monthly_limit = config.budget_monthly_limit_usd
        self._token_cap_per_task = config.max_tokens_per_task
        self._usd_cap_per_task = config.max_usd_per_task
        for budget in self._tasks.values():
            budget.token_cap = self._token_cap_per_task
            budget.usd_cap = self._usd_cap_per_task

    def check_budget(self, provider: str) -> None:
        """Synchronous budget gate — raises BudgetExceededError if limit reached.

        Uses the in-memory cached monthly total so there is zero I/O on the
        hot path.  Free providers (ollama/local) are never blocked.
        """
        if not self._enabled:
            return
        if provider in ("ollama", "local", "subscription"):
            return

        # If the month rolled over since the cache was last updated, the new
        # month starts fresh (sync gate, no I/O — record_usage keeps it
        # accurate from the next call onward).
        current_month = _current_month()
        if current_month != self._cost_month:
            self._cost_month = current_month
            self._monthly_cost = 0.0

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
        for key in [key for key in self._agent_tasks if key[0] == task_id]:
            del self._agent_tasks[key]
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

        if task.token_cap > 0 and task.tokens_used >= task.token_cap:
            task.exceeded = True
            raise TaskBudgetExceededError(
                f"Task {task_id} exceeded token budget "
                f"({task.tokens_used} >= {task.token_cap}). "
                "No more model calls will be made for this task. "
                "Increase max_tokens_per_task or split the task."
            )
        if task.usd_cap > 0 and task.usd_spent >= task.usd_cap:
            task.exceeded = True
            raise TaskBudgetExceededError(
                f"Task {task_id} exceeded USD budget "
                f"(${task.usd_spent:.4f} spent, per-task limit ${task.usd_cap:.4f}). "
                "No more model calls will be made for this task. "
                "Increase max_usd_per_task or split the task."
            )
        agent_id = current_agent_id.get()
        if agent_id:
            agent_budget = self._agent_tasks.get((task_id, agent_id))
            if (
                agent_budget is not None
                and agent_budget.token_cap > 0
                and agent_budget.tokens_used >= agent_budget.token_cap
            ):
                agent_budget.exceeded = True
                raise TaskBudgetExceededError(
                    f"Specialist {agent_id} exceeded its token budget "
                    f"({agent_budget.tokens_used} >= {agent_budget.token_cap}). "
                    "Use a smaller handoff or a different specialist."
                )

    def get_task_budget(self, task_id: str | None = None) -> TaskBudget | None:
        """Return the live TaskBudget for a task, or None if not tracked."""
        if task_id is None:
            task_id = current_task_id.get()
        if not task_id:
            return None
        return self._tasks.get(task_id)

    def start_agent_budget(
        self,
        task_id: str,
        agent_id: str,
        token_cap: int,
    ) -> AgentTokenBudget:
        key = (task_id, agent_id)
        budget = self._agent_tasks.get(key)
        if budget is None:
            budget = AgentTokenBudget(
                task_id=task_id,
                agent_id=agent_id,
                token_cap=max(0, int(token_cap)),
            )
            self._agent_tasks[key] = budget
        return budget

    def get_agent_budget(
        self,
        task_id: str,
        agent_id: str,
    ) -> AgentTokenBudget | None:
        return self._agent_tasks.get((task_id, agent_id))

    async def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        usage_source: str = "estimate",
    ) -> None:
        """Persist one call's token usage and update in-memory totals.

        `usage_source` is "reported" when the numbers came from the provider's
        own usage object and "estimate" when they were guessed from text length.

        Reads the active task_id from the contextvar; if a task is in flight,
        the call's tokens and cost are also added to the in-memory TaskBudget
        so subsequent check_task_budget() calls see the latest state.
        """
        if not self._pool:
            return
        if provider == "claude":
            cost = claude_cost_usd(model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens)
        else:
            cost = _estimate_cost(provider, input_tokens, output_tokens, model)
        call_tokens = input_tokens + output_tokens + cache_read_tokens + cache_creation_tokens
        now = datetime.now(UTC).isoformat()
        month = _current_month()
        task_id = current_task_id.get()
        agent_id = current_agent_id.get()

        async with self._lock:
            async with self._pool.write() as db:
                await db.execute(
                    """INSERT INTO token_usage
                       (timestamp, month, provider, model, input_tokens,
                        output_tokens, cost_usd, task_id, agent_id,
                        cache_read_tokens, cache_creation_tokens, usage_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now,
                        month,
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        cost,
                        task_id,
                        agent_id,
                        cache_read_tokens,
                        cache_creation_tokens,
                        usage_source,
                    ),
                )
                await db.commit()

            # Reset the cached monthly total when the calendar month rolls over,
            # so the budget gate compares against the new month's spend only.
            if month != self._cost_month:
                self._cost_month = month
                self._monthly_cost = 0.0
            self._monthly_cost += cost

            # Update per-task running totals if a task is active and tracked
            if task_id and task_id in self._tasks:
                task = self._tasks[task_id]
                task.tokens_used += call_tokens
                task.usd_spent += cost
            if task_id and agent_id and (task_id, agent_id) in self._agent_tasks:
                self._agent_tasks[(task_id, agent_id)].tokens_used += call_tokens

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
            cursor = await db.execute(
                "SELECT DISTINCT provider, model FROM token_usage WHERE month = ? ORDER BY provider, model", (month,)
            )
            used = await cursor.fetchall()
            await cursor.close()
        from pilot.models.prices import estimated_high

        # Calls this month costed without a current price: said in the budget view, so a high figure is explained.
        notes = sorted({note for provider, model in used if (note := estimated_high(str(provider), str(model)))})
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
            "estimated_high": notes,
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

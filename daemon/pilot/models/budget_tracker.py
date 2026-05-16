"""Global API budget tracker — records token usage and enforces monthly spend limits."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import aiosqlite

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
    cost_usd      REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_token_usage_month ON token_usage(month);
"""


class BudgetExceededError(RuntimeError):
    """Raised when the monthly spend limit has been reached."""


def _current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(provider, (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1000.0


class BudgetTracker:
    """Tracks cumulative LLM token spend and enforces a monthly USD limit."""

    def __init__(self, config: ModelConfig, db_path: str) -> None:
        self._enabled: bool = config.budget_enabled
        self._monthly_limit: float = config.budget_monthly_limit_usd
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._monthly_cost: float = 0.0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        self._monthly_cost = await self._load_monthly_cost()
        logger.info(
            "BudgetTracker ready — month=%s spent=%.4f limit=%.2f enabled=%s",
            _current_month(),
            self._monthly_cost,
            self._monthly_limit,
            self._enabled,
        )

    async def _load_monthly_cost(self) -> float:
        if not self._db:
            return 0.0
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM token_usage WHERE month = ?",
            (_current_month(),),
        )
        row = await cursor.fetchone()
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

    async def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Persist one call's token usage and update the in-memory monthly total."""
        if not self._db:
            return
        cost = _estimate_cost(provider, input_tokens, output_tokens)
        now = datetime.now(UTC).isoformat()
        month = _current_month()
        async with self._lock:
            await self._db.execute(
                """INSERT INTO token_usage
                   (timestamp, month, provider, model, input_tokens, output_tokens, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (now, month, provider, model, input_tokens, output_tokens, cost),
            )
            await self._db.commit()
            self._monthly_cost += cost

    async def get_stats(self) -> dict:
        """Return current-month usage summary."""
        if not self._db:
            return {}
        month = _current_month()
        cursor = await self._db.execute(
            """SELECT
                   COUNT(*) AS calls,
                   COALESCE(SUM(input_tokens), 0) AS total_input,
                   COALESCE(SUM(output_tokens), 0) AS total_output,
                   COALESCE(SUM(cost_usd), 0.0) AS total_cost
               FROM token_usage WHERE month = ?""",
            (month,),
        )
        row = await cursor.fetchone()
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
        if not self._db:
            return
        async with self._lock:
            await self._db.execute("DELETE FROM token_usage WHERE month = ?", (_current_month(),))
            await self._db.commit()
            self._monthly_cost = 0.0
        logger.info("BudgetTracker: current month reset")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

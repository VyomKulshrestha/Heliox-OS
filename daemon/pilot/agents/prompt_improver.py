"""Self-Improving Prompt System — stores and reuses successful reasoning chains.

When a task succeeds, the prompt strategy (system prompt + plan structure)
is stored as a "prompt template". For future similar tasks, the system
retrieves the best matching template and uses it as few-shot context,
dramatically improving reliability on repeated task patterns.

Schema:
  prompt_templates:
    - pattern:     regex or keyword signature of the task
    - strategy:    the prompt/plan structure that worked
    - success_rate: rolling success rate
    - usage_count: how many times reused
    - last_used:   timestamp

  prompt_chains:
    - task_input:  original user request
    - chain:       full reasoning chain (plan → result → reflection)
    - outcome:     success/failure
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.agents.prompt_improver")

PROMPT_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,
    keywords TEXT NOT NULL,
    strategy TEXT NOT NULL,
    example_input TEXT DEFAULT '',
    example_plan TEXT DEFAULT '',
    success_count INTEGER DEFAULT 1,
    failure_count INTEGER DEFAULT 0,
    usage_count INTEGER DEFAULT 1,
    last_used TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_input TEXT NOT NULL,
    keywords TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    result_summary TEXT DEFAULT '',
    reflection TEXT DEFAULT '',
    success INTEGER NOT NULL,
    duration_ms INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_templates_keywords ON prompt_templates(keywords);
CREATE INDEX IF NOT EXISTS idx_chains_success ON prompt_chains(success);
CREATE INDEX IF NOT EXISTS idx_chains_keywords ON prompt_chains(keywords);
"""


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a task description."""
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "and",
        "but",
        "or",
        "not",
        "no",
        "so",
        "if",
        "my",
        "me",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "our",
        "your",
        "please",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return sorted(set(w for w in words if w not in stop_words and len(w) > 2))


class PromptImprover:
    """Self-improving prompt system that learns from successful executions."""

    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None

    async def initialize(self, db_path: str) -> None:
        self._db = await aiosqlite.connect(db_path)
        await self._db.executescript(PROMPT_SCHEMA)
        await self._db.commit()
        logger.info("PromptImprover initialized")

    async def record_chain(
        self,
        task_input: str,
        plan_json: str,
        result_summary: str,
        reflection: str,
        success: bool,
        duration_ms: int = 0,
    ) -> None:
        """Store a complete reasoning chain for future reference."""
        if not self._db:
            return

        keywords = ",".join(_extract_keywords(task_input))
        now = datetime.now(UTC).isoformat()

        await self._db.execute(
            """INSERT INTO prompt_chains
               (task_input, keywords, plan_json, result_summary, reflection,
                success, duration_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_input, keywords, plan_json, result_summary, reflection, int(success), duration_ms, now),
        )

        # If successful, try to create or update a template
        if success:
            await self._upsert_template(task_input, keywords, plan_json, now)

        await self._db.commit()

    async def _upsert_template(
        self,
        task_input: str,
        keywords: str,
        plan_json: str,
        now: str,
    ) -> None:
        """Create or update a prompt template from a successful execution."""
        if not self._db:
            return

        # Check for existing template with similar keywords
        cursor = await self._db.execute(
            """SELECT id, keywords, success_count, usage_count
               FROM prompt_templates
               WHERE keywords = ?""",
            (keywords,),
        )
        existing = await cursor.fetchone()

        if existing:
            await self._db.execute(
                """UPDATE prompt_templates
                   SET success_count = success_count + 1,
                       usage_count = usage_count + 1,
                       last_used = ?,
                       example_plan = ?
                   WHERE id = ?""",
                (now, plan_json, existing[0]),
            )
        else:
            # Extract a strategy pattern from the plan
            strategy = self._extract_strategy(plan_json)

            await self._db.execute(
                """INSERT INTO prompt_templates
                   (pattern, keywords, strategy, example_input, example_plan,
                    last_used, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    keywords,
                    keywords,
                    strategy,
                    task_input,
                    plan_json,
                    now,
                    now,
                ),
            )

    async def record_failure(self, task_input: str) -> None:
        """Record a failure to update template stats."""
        if not self._db:
            return

        keywords = ",".join(_extract_keywords(task_input))
        await self._db.execute(
            """UPDATE prompt_templates
               SET failure_count = failure_count + 1
               WHERE keywords = ?""",
            (keywords,),
        )
        await self._db.commit()

    async def get_relevant_strategies(
        self,
        task_input: str,
        max_results: int = 3,
    ) -> str:
        """Retrieve successful prompt strategies similar to the current task."""
        if not self._db:
            return ""

        keywords = _extract_keywords(task_input)
        if not keywords:
            return ""

        # Search for templates with matching keywords
        parts: list[str] = []

        # Match by keyword overlap
        for kw in list(keywords)[:5]:
            cursor = await self._db.execute(
                """SELECT example_input, strategy, success_count, failure_count
                   FROM prompt_templates
                   WHERE keywords LIKE ?
                   AND success_count > failure_count
                   ORDER BY success_count DESC
                   LIMIT ?""",
                (f"%{kw}%", max_results),
            )
            rows = await cursor.fetchall()
            for row in rows:
                success_rate = row[2] / max(row[2] + row[3], 1) * 100
                parts.append(f'  Strategy (success: {success_rate:.0f}%): "{row[0]}" → {row[1]}')

        # Also get recent successful chains as examples
        cursor = await self._db.execute(
            """SELECT task_input, plan_json
               FROM prompt_chains
               WHERE success = 1
               AND keywords LIKE ?
               ORDER BY id DESC
               LIMIT 2""",
            (f"%{keywords[0]}%",),
        )
        chains = await cursor.fetchall()
        for row in chains:
            parts.append(f'  Proven approach: "{row[0]}"')

        if parts:
            # De-duplicate
            unique_parts = list(dict.fromkeys(parts))
            return "Past successful strategies:\n" + "\n".join(list(unique_parts)[:max_results])

        return ""

    def _extract_strategy(self, plan_json: str) -> str:
        """Extract a reusable strategy description from a plan."""
        try:
            plan = json.loads(plan_json)
            if isinstance(plan, dict):
                actions = plan.get("actions", [])
                if actions:
                    types = [a.get("action_type", "unknown") for a in actions]
                    return f"Action sequence: {' → '.join(types)}"
                return plan.get("explanation", "")[:200]
        except Exception:
            pass
        return "direct execution"

    async def get_stats(self) -> dict[str, Any]:
        """Return prompt improvement statistics."""
        if not self._db:
            return {}

        cursor = await self._db.execute("SELECT COUNT(*) FROM prompt_templates")
        templates = (await cursor.fetchone())[0]

        cursor = await self._db.execute("SELECT COUNT(*), SUM(success) FROM prompt_chains")
        row = await cursor.fetchone()
        total_chains = row[0] if row else 0
        success_chains = row[1] if row else 0

        return {
            "total_templates": templates,
            "total_chains": total_chains,
            "chain_success_rate": (float(int(success_chains / total_chains * 1000)) / 10 if total_chains > 0 else 0),
        }

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

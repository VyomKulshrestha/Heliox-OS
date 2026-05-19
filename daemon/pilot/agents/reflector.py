"""Reflection agent — post-task self-evaluation and learning.

After each task completes, the Reflector analyzes performance,
identifies failure patterns, and generates improvement insights
that feed back into future planning sessions.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiosqlite

from pilot.config import DB_FILE

if TYPE_CHECKING:
    from pilot.actions import ActionPlan, ActionResult, VerificationResult
    from pilot.models.router import ModelRouter

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.agents.reflector")

REFLECTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_input TEXT NOT NULL,
    success INTEGER NOT NULL,
    duration_ms INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    reflection TEXT NOT NULL,
    lessons_learned TEXT DEFAULT '',
    difficulty_score REAL DEFAULT 0.5,
    action_types TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS skill_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    action_template TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS task_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_task TEXT NOT NULL,
    child_task TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflections_success ON task_reflections(success);
CREATE INDEX IF NOT EXISTS idx_reflections_difficulty ON task_reflections(difficulty_score);
CREATE INDEX IF NOT EXISTS idx_skills_name ON skill_registry(skill_name);
CREATE INDEX IF NOT EXISTS idx_task_rels_parent ON task_relationships(parent_task);
"""

REFLECTION_PROMPT = """\
You are a performance analyst for an AI agent called Heliox OS.
Analyze the following completed task execution and produce a structured reflection.

User Request: {user_input}
Plan Explanation: {explanation}
Actions Executed: {action_summary}
Success: {success}
Errors: {errors}
Retry Count: {retry_count}

Produce a JSON response with these fields:
{{
  "reflection": "A brief analysis of how the task went",
  "lessons_learned": "What could be done better next time",
  "difficulty_score": 0.0-1.0 (how hard was this task),
  "discovered_skills": ["list of reusable skill names discovered"],
  "related_task_patterns": ["list of task patterns this is similar to"]
}}
"""


class Reflector:
    """Post-task reflection and self-improvement engine."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(str(DB_FILE))
        await self._db.executescript(REFLECTION_SCHEMA)
        await self._db.commit()
        logger.info("Reflector initialized")

    async def reflect(
        self,
        user_input: str,
        plan: ActionPlan,
        results: list[ActionResult],
        verification: VerificationResult | None,
        retry_count: int = 0,
        duration_ms: int = 0,
    ) -> dict[str, Any]:
        """Analyze a completed task and store the reflection."""
        success = all(r.success for r in results)
        errors = [r.error for r in results if r.error]
        action_types = [a.action_type.value for a in plan.actions]

        action_summary = ", ".join(
            f"{a.action_type.value}({'OK' if r.success else 'FAIL'})"
            for a, r in zip(plan.actions, results, strict=False)
        )

        # Generate reflection via LLM
        reflection_data = await self._generate_reflection(
            user_input=user_input,
            explanation=plan.explanation,
            action_summary=action_summary,
            success=success,
            errors=errors,
            retry_count=retry_count,
        )

        # Store reflection
        if self._db:
            now = datetime.now(UTC).isoformat()
            await self._db.execute(
                """INSERT INTO task_reflections
                   (timestamp, user_input, success, duration_ms, retry_count,
                    reflection, lessons_learned, difficulty_score, action_types)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    user_input,
                    int(success),
                    duration_ms,
                    retry_count,
                    reflection_data.get("reflection", ""),
                    reflection_data.get("lessons_learned", ""),
                    reflection_data.get("difficulty_score", 0.5),
                    json.dumps(action_types),
                ),
            )

            # Register discovered skills
            for skill_name in reflection_data.get("discovered_skills", []):
                await self._register_skill(skill_name, action_summary)

            # Record task relationships
            for pattern in reflection_data.get("related_task_patterns", []):
                await self._record_relationship(user_input, pattern)

            await self._db.commit()

        return reflection_data

    async def _generate_reflection(self, **kwargs: Any) -> dict[str, Any]:
        """Use LLM to analyze task performance."""
        prompt = REFLECTION_PROMPT.format(**kwargs)
        try:
            response = await self._model.generate(prompt)
            return json.loads(response)
        except Exception:
            logger.debug("LLM reflection failed, using basic analysis", exc_info=True)
            return {
                "reflection": f"Task {'succeeded' if kwargs['success'] else 'failed'}",
                "lessons_learned": "No automated analysis available",
                "difficulty_score": 0.5 if kwargs["success"] else 0.8,
                "discovered_skills": [],
                "related_task_patterns": [],
            }

    async def _register_skill(self, skill_name: str, description: str) -> None:
        """Register a newly discovered reusable skill."""
        if not self._db:
            return
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """INSERT INTO skill_registry (skill_name, description, discovered_at)
               VALUES (?, ?, ?)
               ON CONFLICT(skill_name) DO UPDATE SET
               usage_count = usage_count + 1""",
            (skill_name, description, now),
        )
        logger.info("Skill registered/updated: %s", skill_name)

    async def _record_relationship(self, parent: str, child: str) -> None:
        """Record a relationship between tasks."""
        if not self._db:
            return
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """INSERT INTO task_relationships
               (parent_task, child_task, relationship_type, strength, created_at)
               VALUES (?, ?, 'similar', 0.5, ?)""",
            (parent, child, now),
        )

    async def get_improvement_context(self, query: str) -> str:
        """Get lessons and patterns relevant to a new task."""
        if not self._db:
            return ""

        parts: list[str] = []

        # Recent failures and lessons
        cursor = await self._db.execute(
            """SELECT user_input, lessons_learned, difficulty_score
               FROM task_reflections
               WHERE success = 0
               ORDER BY id DESC LIMIT 5"""
        )
        failures = await cursor.fetchall()
        if failures:
            parts.append("Recent failure lessons:")
            for row in failures:
                parts.append(f'  - Task: "{row[0]}" | Lesson: {row[1]}')

        # Available skills
        cursor = await self._db.execute(
            """SELECT skill_name, description, success_rate
               FROM skill_registry
               ORDER BY usage_count DESC LIMIT 10"""
        )
        skills = await cursor.fetchall()
        if skills:
            parts.append("Discovered skills:")
            for row in skills:
                parts.append(f"  - {row[0]}: {row[1]} (success: {row[2]:.0%})")

        return "\n".join(parts) if parts else ""

    async def get_stats(self) -> dict[str, Any]:
        """Get reflection statistics."""
        if not self._db:
            return {}
        cursor = await self._db.execute("SELECT COUNT(*), SUM(success), AVG(difficulty_score) FROM task_reflections")
        row = await cursor.fetchone()
        cursor2 = await self._db.execute("SELECT COUNT(*) FROM skill_registry")
        skills_row = await cursor2.fetchone()

        total = row[0] if row else 0
        successes = row[1] if row else 0
        avg_difficulty = row[2] if row else 0

        return {
            "total_tasks": total,
            "success_rate": (successes / total * 100) if total > 0 else 0,
            "avg_difficulty": round(avg_difficulty or 0, 2),
            "discovered_skills": skills_row[0] if skills_row else 0,
        }

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

"""Reflection agent — post-task self-evaluation and learning.

After each task completes, the Reflector analyzes performance,
identifies failure patterns, and generates improvement insights
that feed back into future planning sessions.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiosqlite

from pilot.config import DB_FILE
from pilot.intelligence.experience import redact_for_persistence

if TYPE_CHECKING:
    from pilot.actions import ActionPlan, ActionResult, VerificationResult
    from pilot.memory.store import MemoryStore
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.reflector")

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
    action_types TEXT DEFAULT '[]',
    verification_state TEXT NOT NULL DEFAULT 'legacy_unverified'
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

MAX_REFLECTION_CANDIDATES = 256
MAX_REFLECTION_CONTEXT_ITEMS = 3

REFLECTION_PROMPT = """\
You are a performance analyst for an AI agent called Heliox OS.
Analyze the following completed task execution and produce a structured reflection.

User Request: {user_input}
Plan Explanation: {explanation}
Actions Executed: {action_summary}
Verifier Outcome: {verification_state}
Verifier Details: {verification_details}
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

Treat the verifier outcome and details as the evidence boundary. Do not infer
that an unobserved postcondition succeeded. Write one concise, reusable lesson
grounded in this trace; distinguish observed facts from a possible cause.
"""


def _reflection_terms(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]{2,}", value.casefold(), flags=re.UNICODE))


def _rank_reflection_rows(query: str, rows: list[tuple]) -> list[tuple[float, tuple]]:
    """Rank verifier-backed reflections by IDF-weighted query-term coverage."""
    query_terms = _reflection_terms(query)
    if not query_terms or not rows:
        return []

    documents: list[set[str]] = []
    for row in rows:
        action_types = str(row[4] or "")
        documents.append(_reflection_terms(" ".join((str(row[1] or ""), str(row[2] or ""), action_types))))
    document_frequency: dict[str, int] = {}
    for terms in documents:
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    weights = {
        term: math.log1p((len(rows) - document_frequency.get(term, 0) + 0.5) / (document_frequency.get(term, 0) + 0.5))
        for term in query_terms
    }
    denominator = sum(weights.values())
    if denominator <= 0:
        return []

    ranked = [
        (sum(weights[term] for term in query_terms & terms) / denominator, row)
        for row, terms in zip(rows, documents, strict=True)
    ]
    return sorted(
        ((score, row) for score, row in ranked if score > 0),
        key=lambda item: (-item[0], -int(item[1][0])),
    )[:MAX_REFLECTION_CONTEXT_ITEMS]


def _safe_reflection_text(value: object, limit: int) -> str:
    return str(redact_for_persistence(str(value or ""))).strip()[:limit]


class Reflector:
    """Post-task reflection and self-improvement engine."""

    def __init__(self, model_router: ModelRouter, memory_store: MemoryStore | None = None) -> None:
        self._model = model_router
        self._memory_store = memory_store
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(str(DB_FILE))
        await self._db.executescript(REFLECTION_SCHEMA)
        cursor = await self._db.execute("PRAGMA table_info(task_reflections)")
        columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if "verification_state" not in columns:
            await self._db.execute(
                "ALTER TABLE task_reflections ADD COLUMN verification_state TEXT NOT NULL DEFAULT 'legacy_unverified'"
            )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_reflections_verification_state ON task_reflections(verification_state, id)"
        )
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
        dry_run: bool = False,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """Store only a real task outcome confirmed by the independent verifier."""
        if self._db is None:
            return {"stored": False, "reason": "not_initialized"}
        if dry_run:
            return {"stored": False, "reason": "dry_run"}
        verified = getattr(verification, "passed", None)
        if not isinstance(verified, bool) or not results or not plan.actions:
            return {"stored": False, "reason": "missing_verified_action_outcome"}
        failed_indices = getattr(verification, "failed_actions", [])
        if not isinstance(failed_indices, (list, tuple)):
            failed_indices = []
        if verified and (
            failed_indices
            or len(results) != len(plan.actions)
            or not all(bool(getattr(result, "success", False)) for result in results)
        ):
            return {"stored": False, "reason": "inconsistent_verifier_and_action_results"}
        if verified:
            observed_indices = getattr(verification, "independently_observed_actions", [])
            if not isinstance(observed_indices, (list, tuple)) or set(observed_indices) != set(
                range(len(plan.actions))
            ):
                return {"stored": False, "reason": "missing_independent_postcondition_evidence"}
        elif (
            not failed_indices
            and not bool(getattr(verification, "rollback_triggered", False))
            and all(bool(getattr(result, "success", False)) for result in results)
        ):
            return {"stored": False, "reason": "missing_verified_failure_evidence"}

        verification_state = "passed" if verified else "failed"
        success = verified
        errors = [_safe_reflection_text(result.error, 500) for result in results if getattr(result, "error", None)]
        raw_details = getattr(verification, "details", [])
        if not isinstance(raw_details, (list, tuple)):
            raw_details = []
        verification_details = [_safe_reflection_text(detail, 500) for detail in raw_details if detail]
        if not verified:
            errors.extend(verification_details)
        action_types = [a.action_type.value for a in plan.actions]

        action_summary = ", ".join(
            f"{a.action_type.value}({'OK' if r.success else 'FAIL'})"
            for a, r in zip(plan.actions, results, strict=False)
        )

        # Generate reflection via LLM
        reflection_data = await self._generate_reflection(
            user_input=_safe_reflection_text(user_input, 2000),
            explanation=_safe_reflection_text(plan.explanation, 2000),
            action_summary=_safe_reflection_text(action_summary, 1000),
            success=success,
            verification_state=verification_state,
            verification_details=json.dumps(verification_details, ensure_ascii=False),
            errors=errors,
            retry_count=retry_count,
        )
        if not isinstance(reflection_data, dict):
            return {"stored": False, "reason": "invalid_reflection"}

        # Store reflection
        if self._db:
            now = datetime.now(UTC).isoformat()
            reflection_text = _safe_reflection_text(reflection_data.get("reflection"), 2000)
            lessons_text = _safe_reflection_text(reflection_data.get("lessons_learned"), 1200)
            try:
                difficulty_score = float(reflection_data.get("difficulty_score", 0.5))
            except (TypeError, ValueError):
                difficulty_score = 0.5
            if not math.isfinite(difficulty_score):
                difficulty_score = 0.5
            difficulty_score = max(0.0, min(1.0, difficulty_score))
            await self._db.execute(
                """INSERT INTO task_reflections
                   (timestamp, user_input, success, duration_ms, retry_count,
                    reflection, lessons_learned, difficulty_score, action_types, verification_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    _safe_reflection_text(user_input, 2000),
                    int(success),
                    duration_ms,
                    retry_count,
                    reflection_text,
                    lessons_text,
                    difficulty_score,
                    json.dumps(action_types),
                    verification_state,
                ),
            )

            # Register discovered skills
            discovered_skills = reflection_data.get("discovered_skills", [])
            if success and isinstance(discovered_skills, list):
                for skill_name in discovered_skills:
                    name = _safe_reflection_text(skill_name, 120)
                    if name:
                        await self._register_skill(name, action_summary)

            # Record task relationships
            related_patterns = reflection_data.get("related_task_patterns", [])
            if isinstance(related_patterns, list):
                for pattern in related_patterns:
                    if isinstance(pattern, str) and pattern.strip():
                        await self._record_relationship(
                            _safe_reflection_text(user_input, 2000),
                            _safe_reflection_text(pattern, 500),
                        )

            await self._db.commit()
            if self._memory_store is not None and lessons_text:
                try:
                    await self._memory_store.record_verified_reflection(
                        lessons_text,
                        outcome=verification_state,
                        session_id=session_id,
                        occurred_at=now,
                    )
                except Exception:
                    logger.debug("Could not index verified reflection for semantic retrieval", exc_info=True)

        return {"stored": True, "verification_state": verification_state}

    async def _generate_reflection(self, **kwargs: Any) -> dict[str, Any]:
        """Use LLM to analyze task performance."""
        prompt = REFLECTION_PROMPT.format(**kwargs)
        try:
            response = await self._model.generate(prompt)
            parsed = json.loads(response)
            if not isinstance(parsed, dict):
                raise ValueError("Reflection response must be a JSON object")
            return parsed
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
        """Retrieve only relevant, verifier-backed lessons without exposing past task text."""
        if not self._db or not _reflection_terms(query):
            return ""
        if self._memory_store is not None:
            try:
                semantic_cases = await self._memory_store.search_verified_reflections(
                    query,
                    limit=MAX_REFLECTION_CONTEXT_ITEMS,
                )
            except Exception:
                logger.debug("Could not retrieve semantic verified reflections", exc_info=True)
                semantic_cases = []
            if semantic_cases:
                semantic_lessons = [
                    f"- [verified {'success' if case['outcome'] == 'passed' else 'failure'}] "
                    f"{_safe_reflection_text(case['lesson'], 600)}"
                    for case in semantic_cases
                    if case.get("outcome") in {"passed", "failed"} and case.get("lesson")
                ]
                if semantic_lessons:
                    return (
                        "Relevant prior verifier-checked outcomes (model-written lessons; advisory only):\n"
                        + "\n".join(semantic_lessons)
                    )
        cursor = await self._db.execute(
            f"""SELECT id, user_input, lessons_learned, reflection, action_types, success, verification_state
               FROM task_reflections
               WHERE verification_state IN ('passed', 'failed')
               ORDER BY id DESC LIMIT {MAX_REFLECTION_CANDIDATES}"""
        )
        rows = await cursor.fetchall()
        await cursor.close()
        selected = _rank_reflection_rows(query, rows)
        lessons: list[str] = []
        for _, row in selected:
            lesson = _safe_reflection_text(row[2] or row[3], 600)
            if not lesson:
                continue
            outcome = "verified success" if bool(row[5]) else "verified failure"
            lessons.append(f"- [{outcome}] {lesson}")
        if not lessons:
            return ""
        return "Relevant prior verifier-checked outcomes (model-written lessons; advisory only):\n" + "\n".join(lessons)

    async def get_stats(self) -> dict[str, Any]:
        """Get reflection statistics."""
        if not self._db:
            return {}
        cursor = await self._db.execute(
            """SELECT COUNT(*), SUM(success), AVG(difficulty_score)
               FROM task_reflections
               WHERE verification_state IN ('passed', 'failed')"""
        )
        row = await cursor.fetchone()
        cursor3 = await self._db.execute(
            "SELECT COUNT(*) FROM task_reflections WHERE verification_state = 'legacy_unverified'"
        )
        unverified_row = await cursor3.fetchone()
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
            "legacy_unverified_tasks": unverified_row[0] if unverified_row else 0,
        }

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

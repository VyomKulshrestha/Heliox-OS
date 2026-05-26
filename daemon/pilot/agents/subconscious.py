"""Subconscious Agent — background long-term memory consolidation and personalization.

Periodically reviews the day's actions, extracts user preferences,
clusters behavioral patterns, and writes refined "rules" into a
persistent persona prompt that evolves over time.

Architecture:
  1. ReviewCycle:   Runs every N minutes (default: 30)
  2. Clustering:    Groups recent actions by category (file, code, web, media)
  3. Preference:    Extracts user habits ("always uses Python 3.11", "prefers dark mode")
  4. Rule Writing:  Generates system-prompt rules from patterns
   5. Persona:       Maintains a persona.md (in DATA_DIR) that is injected into planner context

Schema (in pilot.db):
  user_persona:
    - rule_id:      unique identifier
    - rule_text:    the personalized rule
    - confidence:   0.0-1.0 how confident the agent is in this rule
    - source_count: how many observations led to this rule
    - category:     preference | habit | constraint | style
    - created_at:   timestamp
    - updated_at:   last seen
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from pilot.config import PERSONA_FILE

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.subconscious")

PERSONA_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_persona (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id TEXT UNIQUE NOT NULL,
    rule_text TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    category TEXT DEFAULT 'preference',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_name TEXT NOT NULL,
    action_types TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    last_seen TEXT NOT NULL,
    sample_inputs TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_persona_category ON user_persona(category);
CREATE INDEX IF NOT EXISTS idx_persona_confidence ON user_persona(confidence);
CREATE INDEX IF NOT EXISTS idx_clusters_freq ON action_clusters(frequency);
"""

CONSOLIDATION_PROMPT = """\
You are the Subconscious Agent for Heliox OS. Your job is to analyze the user's
recent actions and extract personalized rules about their preferences and habits.

Here are the user's recent actions (last 24 hours):
{recent_actions}

Here are existing persona rules:
{existing_rules}

Here are recent TRIBE v2 Cognitive Saliency predictions (Brain Fingerprint):
{tribe_insights}

Analyze the actions and cognitive engagement to output ONLY valid JSON:
{{
  "new_rules": [
    {{
      "rule_id": "pref_python_version",
      "rule_text": "User prefers Python 3.11 for all projects",
      "confidence": 0.8,
      "category": "preference"
    }}
  ],
  "updated_rules": [
    {{
      "rule_id": "existing_rule_id",
      "confidence_delta": 0.1
    }}
  ],
  "clusters": [
    {{
      "cluster_name": "code_workflow",
      "action_types": ["file_write", "code_execute", "git_commit"]
    }}
  ]
}}

Rules should be:
- Actionable (the planner can use them to make better decisions)
- Specific (not vague like "user likes coding")
- Based on repeated patterns (not one-off actions)

Categories: preference, habit, constraint, style
"""


class SubconsciousAgent:
    """Background agent for long-term memory consolidation and personalization."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router
        self._db: aiosqlite.Connection | None = None
        self._task: asyncio.Task[None] | None = None
        self._interval_minutes: int = 30
        self._running = False

    async def initialize(self, db_path: str) -> None:
        """Initialize the subconscious agent's database tables."""
        self._db = await aiosqlite.connect(db_path)
        await self._db.executescript(PERSONA_SCHEMA)
        await self._db.commit()
        logger.info("SubconsciousAgent initialized")

    async def start(self, interval_minutes: int = 30) -> None:
        """Start the background consolidation loop."""
        self._interval_minutes = interval_minutes
        self._running = True
        self._task = asyncio.create_task(self._consolidation_loop())
        logger.info("Subconscious consolidation loop started (every %dm)", interval_minutes)

    async def stop(self) -> None:
        """Stop the consolidation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Subconscious consolidation loop stopped")

    async def _consolidation_loop(self) -> None:
        """Main loop: periodically review actions and update persona."""
        while self._running:
            try:
                await asyncio.sleep(self._interval_minutes * 60)
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Consolidation cycle error", exc_info=True)

    async def consolidate(self) -> dict[str, Any]:
        """Run a single consolidation cycle: analyze actions → update persona."""
        if not self._db:
            return {"error": "Not initialized"}

        recent = await self._get_recent_actions(hours=24)
        if not recent:
            return {"status": "no_actions"}

        existing = await self._get_persona_rules()

        # ── Feature 3: Subconscious Persona Brain Fingerprint ──
        # Try to gather cognitive insight history or use active state
        tribe_insight_text = "No neural engagement data available."
        try:
            from pilot.cognitive.tribe_engine import TribeEngine

            tribe = TribeEngine.get_instance()
            if tribe.is_loaded and hasattr(tribe, "_last_cognitive_load"):
                cog_load = tribe._last_cognitive_load
                tribe_insight_text = f"Latest session cognitive load average: {cog_load:.2f}. User neural engagement indicates high plasticity towards visual UI changes."
        except Exception:
            pass

        prompt = CONSOLIDATION_PROMPT.format(
            recent_actions=json.dumps(recent, indent=2),
            existing_rules=json.dumps(existing, indent=2),
            tribe_insights=tribe_insight_text,
        )

        try:
            response = await self._model.generate(prompt)
            data = json.loads(response)
        except Exception:
            logger.debug("LLM consolidation failed", exc_info=True)
            return {"status": "llm_error"}

        now = datetime.now(UTC).isoformat()
        results: dict[str, int] = {"new_rules": 0, "updated_rules": 0, "clusters": 0}

        # Process new rules
        for rule in data.get("new_rules", []):
            await self._upsert_rule(
                rule_id=rule["rule_id"],
                rule_text=rule["rule_text"],
                confidence=rule.get("confidence", 0.5),
                category=rule.get("category", "preference"),
                now=now,
            )
            results["new_rules"] += 1

        # Process confidence updates
        for update in data.get("updated_rules", []):
            await self._db.execute(
                """UPDATE user_persona
                   SET confidence = MIN(1.0, MAX(0.0, confidence + ?)),
                       source_count = source_count + 1,
                       updated_at = ?
                   WHERE rule_id = ?""",
                (update.get("confidence_delta", 0.1), now, update["rule_id"]),
            )
            results["updated_rules"] += 1

        # Process clusters
        for cluster in data.get("clusters", []):
            await self._upsert_cluster(cluster, now)
            results["clusters"] += 1

        await self._db.commit()

        # Regenerate the persona file
        await self._write_persona_file()

        logger.info(
            "Consolidation complete: %d new rules, %d updated, %d clusters",
            results["new_rules"],
            results["updated_rules"],
            results["clusters"],
        )
        return results

    async def _upsert_rule(self, rule_id: str, rule_text: str, confidence: float, category: str, now: str) -> None:
        """Insert or update a persona rule."""
        if not self._db:
            return
        await self._db.execute(
            """INSERT INTO user_persona (rule_id, rule_text, confidence, category, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(rule_id) DO UPDATE SET
                 rule_text = excluded.rule_text,
                 confidence = MAX(confidence, excluded.confidence),
                 source_count = source_count + 1,
                 updated_at = excluded.updated_at""",
            (rule_id, rule_text, confidence, category, now, now),
        )

    async def _upsert_cluster(self, cluster: dict[str, Any], now: str) -> None:
        """Insert or update an action cluster."""
        if not self._db:
            return
        name = cluster.get("cluster_name", "unknown")
        types = json.dumps(cluster.get("action_types", []))

        cursor = await self._db.execute("SELECT id FROM action_clusters WHERE cluster_name = ?", (name,))
        existing = await cursor.fetchone()
        if existing:
            await self._db.execute(
                """UPDATE action_clusters
                   SET frequency = frequency + 1, last_seen = ?
                   WHERE cluster_name = ?""",
                (now, name),
            )
        else:
            await self._db.execute(
                """INSERT INTO action_clusters (cluster_name, action_types, last_seen)
                   VALUES (?, ?, ?)""",
                (name, types, now),
            )

    async def _get_recent_actions(self, hours: int = 24) -> list[dict[str, Any]]:
        """Fetch recent actions from the action_history table."""
        if not self._db:
            return []
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        cursor = await self._db.execute(
            """SELECT user_input, plan_json, success, explanation
               FROM action_history
               WHERE timestamp > ?
               ORDER BY timestamp DESC
               LIMIT 50""",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "user_input": row[0],
                "plan_summary": row[1][:200] if row[1] else "",
                "success": bool(row[2]),
                "explanation": row[3] or "",
            }
            for row in rows
        ]

    async def _get_persona_rules(self) -> list[dict[str, Any]]:
        """Get all persona rules sorted by confidence."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            """SELECT rule_id, rule_text, confidence, category, source_count
               FROM user_persona
               ORDER BY confidence DESC"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "rule_id": row[0],
                "rule_text": row[1],
                "confidence": row[2],
                "category": row[3],
                "source_count": row[4],
            }
            for row in rows
        ]

    async def _write_persona_file(self) -> None:
        """Write high-confidence rules to the persona.md file for planner injection."""
        rules = await self._get_persona_rules()
        high_confidence = [r for r in rules if r["confidence"] >= 0.6]

        if not high_confidence:
            return

        PERSONA_FILE.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Heliox OS — User Persona",
            "",
            "These rules are automatically maintained by the Subconscious Agent.",
            "They inform the Planner about user preferences and habits.",
            "",
        ]

        # Group by category
        categories: dict[str, list[dict[str, Any]]] = {}
        for rule in high_confidence:
            cat = rule["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rule)

        for cat, cat_rules in sorted(categories.items()):
            lines.append(f"## {cat.capitalize()}")
            for r in cat_rules:
                conf = int(r["confidence"] * 100)
                lines.append(f"- {r['rule_text']} (confidence: {conf}%, seen: {r['source_count']}x)")
            lines.append("")

        PERSONA_FILE.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Persona file updated: %d rules written", len(high_confidence))

    async def get_persona_context(self) -> str:
        """Return persona rules formatted for planner system prompt injection."""
        rules = await self._get_persona_rules()
        high_confidence = [r for r in rules if r["confidence"] >= 0.5]
        if not high_confidence:
            return ""

        lines = ["User persona (learned preferences):"]
        for r in high_confidence[:15]:
            lines.append(f"  - [{r['category']}] {r['rule_text']}")
        return "\n".join(lines)

    async def add_manual_preference(self, key: str, value: str) -> None:
        """Allow user to manually set a preference rule."""
        if not self._db:
            return
        now = datetime.now(UTC).isoformat()
        rule_id = f"manual_{key.lower().replace(' ', '_')}"
        await self._upsert_rule(
            rule_id=rule_id,
            rule_text=f"User explicitly stated: {value}",
            confidence=1.0,
            category="preference",
            now=now,
        )
        await self._db.commit()
        await self._write_persona_file()

    async def get_stats(self) -> dict[str, Any]:
        """Return subconscious agent statistics."""
        if not self._db:
            return {}

        cursor = await self._db.execute("SELECT COUNT(*) FROM user_persona")
        total_rules = (await cursor.fetchone())[0]

        cursor = await self._db.execute("SELECT COUNT(*) FROM user_persona WHERE confidence >= 0.6")
        high_conf = (await cursor.fetchone())[0]

        cursor = await self._db.execute("SELECT COUNT(*) FROM action_clusters")
        clusters = (await cursor.fetchone())[0]

        return {
            "total_rules": total_rules,
            "high_confidence_rules": high_conf,
            "action_clusters": clusters,
            "persona_file": str(PERSONA_FILE),
            "consolidation_interval_min": self._interval_minutes,
            "running": self._running,
        }

    async def close(self) -> None:
        await self.stop()
        if self._db:
            await self._db.close()
            self._db = None

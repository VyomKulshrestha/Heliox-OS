"""Memory store — SQLite for action history, ChromaDB for semantic search.

Memory updates are asynchronous and never block the main execution pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiosqlite

from pilot.config import DATA_DIR, DB_FILE

if TYPE_CHECKING:
    from pilot.actions import ActionPlan, ActionResult

logger = logging.getLogger("pilot.memory.store")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS action_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_input TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    results_json TEXT,
    success INTEGER DEFAULT 1,
    explanation TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_timestamp ON action_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_history_success ON action_history(success);
CREATE INDEX IF NOT EXISTS idx_prefs_key ON user_preferences(key);
"""


class MemoryStore:
    """Persistent memory with action history and semantic preference learning."""

    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None
        self._chroma_collection: Any = None

    async def initialize(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(DB_FILE))
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB for semantic search (best-effort)."""
        try:
            import chromadb

            chroma_dir = DATA_DIR / "chroma"
            chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(chroma_dir))
            self._chroma_collection = client.get_or_create_collection(
                name="pilot_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB initialized at %s", chroma_dir)
        except ImportError:
            logger.warning("ChromaDB not available — semantic memory disabled")
        except Exception:
            logger.exception("ChromaDB initialization failed")

    async def record(
        self,
        user_input: str,
        plan: ActionPlan,
        results: list[ActionResult],
    ) -> None:
        """Record an executed plan and its results."""
        if not self._db:
            return

        now = datetime.now(UTC).isoformat()
        plan_json = plan.model_dump_json()
        results_json = json.dumps([r.model_dump() for r in results])
        success = all(r.success for r in results)

        await self._db.execute(
            """INSERT INTO action_history
               (timestamp, user_input, plan_json, results_json, success, explanation)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, user_input, plan_json, results_json, int(success), plan.explanation),
        )
        await self._db.commit()

        if self._chroma_collection is not None:
            try:
                self._chroma_collection.add(
                    documents=[user_input],
                    metadatas=[
                        {
                            "timestamp": now,
                            "success": str(success),
                            "explanation": plan.explanation[:500],
                        }
                    ],
                    ids=[f"history-{now}"],
                )
            except Exception:
                logger.debug("ChromaDB write failed", exc_info=True)

    async def get_context(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant context for a query using semantic search."""
        parts: list[str] = []

        if self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(query_texts=[query], n_results=n_results)
                if results["documents"] and results["documents"][0]:
                    parts.append("Related past requests:")
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=False):
                        parts.append(f'  - "{doc}" (result: {meta.get("explanation", "N/A")})')
            except Exception:
                logger.debug("ChromaDB query failed", exc_info=True)

        if self._db:
            prefs = await self._get_preferences()
            if prefs:
                parts.append("User preferences:")
                for k, v in prefs.items():
                    parts.append(f"  - {k}: {v}")

        return "\n".join(parts) if parts else ""

    async def get_history(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if not self._db:
            return []

        cursor = await self._db.execute(
            """SELECT id, timestamp, user_input, success, explanation
               FROM action_history ORDER BY id DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "user_input": r[2],
                "success": bool(r[3]),
                "explanation": r[4],
            }
            for r in rows
        ]

    async def set_preference(self, key: str, value: str) -> None:
        if not self._db:
            return
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """INSERT INTO user_preferences (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, now),
        )
        await self._db.commit()

    async def _get_preferences(self) -> dict[str, str]:
        if not self._db:
            return {}
        cursor = await self._db.execute("SELECT key, value FROM user_preferences")
        rows = await cursor.fetchall()
        return {r[0]: r[1] for r in rows}

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

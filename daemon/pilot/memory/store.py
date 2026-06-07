"""Memory store — SQLite for action history, ChromaDB for semantic search.

Memory updates are asynchronous and never block the main execution pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiofiles.os

from pilot.config import DATA_DIR, DB_FILE
from pilot.db.sqlite_pool import AsyncSqlitePool
from pilot.models.router import ModelRouter

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

    def __init__(
        self,
        checkpoint_interval_seconds: int = 300,
        pruning_interval_seconds: int = 3600,
        pruning_min_memories: int = 10,
    ) -> None:
        self._pool: AsyncSqlitePool | None = None
        self._chroma_collection: Any = None
        self._workspace_index = None

        self._checkpoint_task: asyncio.Task[None] | None = None
        self._checkpoint_interval_seconds = checkpoint_interval_seconds

        self._pruning_task: asyncio.Task[None] | None = None
        self._pruning_interval_seconds = pruning_interval_seconds
        self._pruning_min_memories = pruning_min_memories

    async def initialize(self, router: ModelRouter = None) -> None:
        await aiofiles.os.makedirs(DATA_DIR, exist_ok=True)

        self._pool = AsyncSqlitePool(DB_FILE)
        await self._pool.start()

        async with self._pool.write() as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

        await asyncio.to_thread(self._init_chroma)

        self._init_workspace_index()

        if self._checkpoint_interval_seconds > 0:
            self._checkpoint_task = asyncio.create_task(self._periodic_checkpoint_loop())
            logger.info(
                "Memory WAL checkpoint scheduler started (interval=%ss)",
                self._checkpoint_interval_seconds,
            )

        if router and self._pruning_interval_seconds > 0:
            self._pruning_task = asyncio.create_task(self._periodic_pruning_loop(router))
            logger.info("Semantic memory pruning scheduler started.")

    def _init_workspace_index(self) -> None:
        """Initialize the workspace RAG index."""
        from pilot.memory.workspace_index import WorkspaceIndex

        workspace_dir = DATA_DIR / "workspace_index"
        self._workspace_index = WorkspaceIndex(workspace_dir)

        logger.info("WorkspaceIndex initialized at %s", workspace_dir)

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

    async def checkpoint(self) -> dict[str, Any]:
        """Trigger a manual SQLite WAL checkpoint."""
        if not self._pool:
            return {"status": "error", "message": "Memory store is not initialized"}

        result = await self._pool.checkpoint()

        logger.info("Memory WAL checkpoint completed: %s", result)
        return {"status": "ok", **result}

    async def _periodic_checkpoint_loop(self) -> None:
        """Periodically checkpoint SQLite WAL data."""
        while True:
            await asyncio.sleep(self._checkpoint_interval_seconds)

            try:
                await self.checkpoint()

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("Periodic memory WAL checkpoint failed")

    async def _periodic_pruning_loop(self, router: ModelRouter) -> None:
        """Periodically cluster and prune semantic memory."""
        while True:
            await asyncio.sleep(self._pruning_interval_seconds)

            try:
                logger.info("Starting background semantic memory pruning...")
                await self._cluster_and_prune(router)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Background semantic memory pruning failed")

    async def _cluster_and_prune(self, router: ModelRouter) -> None:
        """Background task to cluster semantic memories and prune redundancies."""
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.warning("Optional dependencies 'numpy' or 'scikit-learn' missing. Semantic pruning disabled.")
            return

        """Identify semantic clusters in ChromaDB, summarize, and prune SQLite/Chroma."""
        if self._chroma_collection is None or not self._pool:
            return

        # 1. Fetch unsummarized memories from ChromaDB
        # We assume we add a metadata flag like "is_macro: false" to raw logs
        chroma_data = await asyncio.to_thread(
            self._chroma_collection.get,
            where={"is_macro": {"$ne": True}},  # Fetch granular logs only
            include=["embeddings", "documents", "metadatas"],
        )

        if not chroma_data["documents"] or len(chroma_data["documents"]) < self._pruning_min_memories:
            logger.debug("Not enough granular memories to justify pruning.")
            return

        embeddings = chroma_data["embeddings"]
        documents = chroma_data["documents"]
        ids = chroma_data["ids"]

        # 2. Apply Clustering (e.g., DBSCAN via scikit-learn)
        X = np.array(embeddings)
        # eps defines the semantic proximity threshold
        clustering = DBSCAN(eps=0.3, min_samples=3, metric="cosine").fit(X)

        # 3. Process each cluster
        clusters = set(clustering.labels_)
        for cluster_id in clusters:
            if cluster_id == -1:
                continue  # Skip noise/unclustered items

            # Gather the memories belonging to this cluster
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_docs = [documents[i] for i in cluster_indices]
            cluster_ids = [ids[i] for i in cluster_indices]

            # Extract SQLite IDs from the Chroma IDs (assuming format "history-{id}")
            # Ensure safe parsing based on how they format IDs

            # 4. Synthesize the Macro-Learning using the ModelRouter
            # We construct a prompt asking the LLM to summarize the patterns
            synthesis_prompt = (
                "Identify the core user preference or workflow pattern from "
                "these related historical actions:\n" + "\n".join(cluster_docs)
            )

            # Pass to the local LLM router
            macro_summary = await router.generate(prompt=synthesis_prompt)

            # 5. Commit Macro-Node & Prune Granular Logs
            await self._commit_and_prune(macro_summary, cluster_ids)

    async def _commit_and_prune(self, macro_summary: str, old_chroma_ids: list[str]) -> None:
        """Insert the new macro-learning and delete the granular logs."""
        now = datetime.now(UTC).isoformat()
        macro_id_str = f"macro-{now}"

        # A. Update SQLite
        async with self._pool.write() as db:
            # 1. Insert the new macro summary as a high-level plan
            await db.execute(
                """INSERT INTO action_history
                   (timestamp, user_input, plan_json, results_json, success, explanation)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (now, "MACRO_LEARNING", "{}", "[]", 1, macro_summary),
            )

            # 2. Prune old records (Extracting SQLite timestamps/IDs from old_chroma_ids)
            # You will need to parse the timestamp out of the 'history-{timestamp}' string
            for c_id in old_chroma_ids:
                ts = c_id.replace("history-", "")
                await db.execute("DELETE FROM action_history WHERE timestamp = ?", (ts,))

            await db.commit()

        # B. Update ChromaDB
        if self._chroma_collection is not None:
            # Delete the granular embeddings
            await asyncio.to_thread(self._chroma_collection.delete, ids=old_chroma_ids)

            # Add the new macro embedding
            await asyncio.to_thread(
                self._chroma_collection.add,
                documents=[macro_summary],
                metadatas=[{"timestamp": now, "is_macro": True}],
                ids=[macro_id_str],
            )

    async def record(
        self,
        user_input: str,
        plan: ActionPlan,
        results: list[ActionResult],
    ) -> None:
        """Record an executed plan and its results."""
        if not self._pool:
            return

        now = datetime.now(UTC).isoformat()
        plan_json = plan.model_dump_json()

        results_json = json.dumps([r.model_dump() for r in results])

        success = all(r.success for r in results)

        async with self._pool.write() as db:
            await db.execute(
                """INSERT INTO action_history
                   (timestamp, user_input, plan_json, results_json, success, explanation)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    user_input,
                    plan_json,
                    results_json,
                    int(success),
                    plan.explanation,
                ),
            )

            await db.commit()

        if self._chroma_collection is not None:
            try:
                await asyncio.to_thread(
                    self._chroma_collection.add,
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
                results = await asyncio.to_thread(
                    self._chroma_collection.query,
                    query_texts=[query],
                    n_results=n_results,
                )

                if results["documents"] and results["documents"][0]:
                    parts.append("Related past requests:")

                    for doc, meta in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        strict=False,
                    ):
                        parts.append(f'  - "{doc}" (result: {meta.get("explanation", "N/A")})')

            except Exception:
                logger.debug("ChromaDB query failed", exc_info=True)

        if self._pool:
            prefs = await self._get_preferences()

            if prefs:
                parts.append("User preferences:")

                for k, v in prefs.items():
                    parts.append(f"  - {k}: {v}")

        return "\n".join(parts) if parts else ""

    async def get_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []

        async with self._pool.read() as db:
            cursor = await db.execute(
                """SELECT id, timestamp, user_input, success, explanation
                   FROM action_history
                   ORDER BY id DESC
                   LIMIT ?
                   OFFSET ?""",
                (limit, offset),
            )

            rows = await cursor.fetchall()

            await cursor.close()

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
        if not self._pool:
            return

        now = datetime.now(UTC).isoformat()

        async with self._pool.write() as db:
            await db.execute(
                """INSERT INTO user_preferences (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key)
                   DO UPDATE SET
                       value=excluded.value,
                       updated_at=excluded.updated_at""",
                (key, value, now),
            )

            await db.commit()

    async def get_preference(self, key: str) -> str | None:
        """Return the stored value for *key*, or None if not found."""
        if not self._pool:
            return None

        async with self._pool.read() as db:
            cursor = await db.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
            row = await cursor.fetchone()
            await cursor.close()

        return row[0] if row else None

    async def _get_preferences(self) -> dict[str, str]:
        if not self._pool:
            return {}

        async with self._pool.read() as db:
            cursor = await db.execute("SELECT key, value FROM user_preferences")

            rows = await cursor.fetchall()

            await cursor.close()

        return {r[0]: r[1] for r in rows}

    async def index_workspace(self, folder_path: str) -> dict:
        """Index a workspace folder for semantic search."""
        if self._workspace_index is None:
            return {
                "success": False,
                "error": "Workspace index not initialized",
            }

        return await asyncio.to_thread(
            self._workspace_index.index_workspace,
            folder_path,
        )

    async def search_workspace(
        self,
        query: str,
        n_results: int = 5,
    ) -> list:
        """Search the workspace index semantically."""
        if self._workspace_index is None:
            return []

        return await asyncio.to_thread(
            self._workspace_index.search,
            query,
            n_results,
        )

    async def close(self) -> None:
        if self._checkpoint_task:
            self._checkpoint_task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await self._checkpoint_task

            self._checkpoint_task = None

        if self._pool:
            # Final checkpoint before shutdown
            await self.checkpoint()

            await self._pool.close()

            self._pool = None

        if self._pruning_task:
            self._pruning_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pruning_task
            self._pruning_task = None

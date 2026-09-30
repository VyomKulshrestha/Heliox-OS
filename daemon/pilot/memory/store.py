"""Memory store — SQLite for action history, ChromaDB for semantic search.

Memory updates are asynchronous and never block the main execution pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles.os

from pilot.config import DATA_DIR, DB_FILE
from pilot.db.sqlite_pool import AsyncSqlitePool
from pilot.intelligence.experience import (
    PrivacyClass,
    get_experience_context,
)
from pilot.memory.assembler import TemporalContextAssembler
from pilot.memory.sliding_window import get_token_count
from pilot.memory.temporal import (
    MemoryProvenance,
    MemoryScope,
    TemporalFact,
    TemporalMemoryStore,
)
from pilot.models.router import ModelRouter

if TYPE_CHECKING:
    from pilot.actions import ActionPlan, ActionResult

logger = logging.getLogger("pilot.memory.store")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS action_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT 'default',
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
        temporal_db_file: str | Path | None = None,
    ) -> None:
        self._pool: AsyncSqlitePool | None = None
        self._chroma_collection: Any = None
        self._workspace_index = None
        self._temporal: TemporalMemoryStore | None = None
        self._context_assembler: TemporalContextAssembler | None = None
        self._temporal_db_file = Path(temporal_db_file) if temporal_db_file else None

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
            cursor = await db.execute("PRAGMA table_info(action_history)")
            columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()
            if "session_id" not in columns:
                await db.execute("ALTER TABLE action_history ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON action_history(session_id, id)")
            await db.commit()

        temporal_path = self._temporal_db_file or Path(DB_FILE).with_name("temporal_memory.db")
        self._temporal = TemporalMemoryStore(temporal_path)
        await self._temporal.initialize()
        self._context_assembler = TemporalContextAssembler(self._temporal)

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
        *,
        session_id: str = "default",
        task_id: str = "",
        provenance_event_id: str = "",
        verification: Any | None = None,
        task_complete: bool = True,
    ) -> None:
        """Record execution results; temporal success requires task completion and a passing verifier."""
        if not self._pool:
            return

        now = datetime.now(UTC).isoformat()
        plan_json = plan.model_dump_json()

        results_json = json.dumps([r.model_dump() for r in results])

        success = all(r.success for r in results)

        async with self._pool.write() as db:
            await db.execute(
                """INSERT INTO action_history
                   (timestamp, session_id, user_input, plan_json, results_json, success, explanation)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    session_id,
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
                            "session_id": session_id,
                            "success": str(success),
                            "explanation": plan.explanation[:500],
                        }
                    ],
                    ids=[f"history-{now}"],
                )

            except Exception:
                logger.debug("ChromaDB write failed", exc_info=True)

        if self._temporal is not None and getattr(plan, "actions", None):
            context = get_experience_context()
            effective_session_id = session_id or context.session_id or "default"
            effective_task_id = task_id or context.task_id
            verified = (
                bool(results)
                and bool(getattr(plan, "actions", None))
                and success
                and task_complete
                and verification is not None
                and bool(getattr(verification, "passed", False))
            )
            verification_failed = verification is not None and not bool(getattr(verification, "passed", False))
            outcome = (
                "success"
                if verified
                else "failure"
                if not success or verification_failed
                else "in_progress"
                if task_complete is False and verification is not None
                else "unverified"
            )
            action_types = sorted(
                {
                    str(getattr(getattr(action, "action_type", ""), "value", ""))
                    for action in getattr(plan, "actions", [])
                    if getattr(getattr(action, "action_type", ""), "value", "")
                }
            )
            result_summaries = [
                {
                    "success": bool(getattr(result, "success", False)),
                    "error": str(getattr(result, "error", "") or "")[:300],
                }
                for result in results
            ]
            await self._temporal.record_episode(
                session_id=effective_session_id,
                task_id=effective_task_id,
                summary=(f"Request: {user_input[:500]}. Plan: {str(getattr(plan, 'explanation', ''))[:500]}"),
                outcome=outcome,
                tags=action_types,
                importance=0.65 if success else 0.8,
                provenance_event_id=provenance_event_id,
                payload={
                    "results": result_summaries,
                    "verification_passed": bool(getattr(verification, "passed", False)),
                },
                provenance=(
                    MemoryProvenance.VERIFIED_OUTCOME if verified else MemoryProvenance.INFERRED
                ),
            )

    async def record_task_completion(
        self,
        goal: str,
        *,
        session_id: str,
        task_id: str,
        action_types: list[str] | None = None,
        independently_verified: bool = False,
    ) -> None:
        """Store a completion claim, promoting it only when deterministic evidence verifies it."""
        if self._temporal is None:
            return
        context = get_experience_context()
        await self._temporal.record_episode(
            session_id=session_id or context.session_id or "default",
            task_id=task_id or context.task_id,
            summary=(
                f"Verified task completion: {goal[:500]}"
                if independently_verified
                else f"Completion claim: {goal[:500]}"
            ),
            outcome="success" if independently_verified else "unverified",
            tags=action_types or [],
            importance=0.75,
            payload={
                "verification_passed": independently_verified,
                "completion_basis": (
                    "exact_editable_text_match"
                    if independently_verified
                    else "model_claim_cites_fresh_observation"
                ),
            },
            provenance=(
                MemoryProvenance.VERIFIED_OUTCOME
                if independently_verified
                else MemoryProvenance.MODEL_SYNTHESIS
            ),
        )

    async def get_context(
        self,
        query: str,
        n_results: int = 5,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """Retrieve ranked memory context without exceeding the token budget."""
        parts: list[str] = []
        max_tokens = max(0, int(max_tokens))
        context = get_experience_context()
        effective_session_id = session_id or context.session_id or "default"
        effective_task_id = task_id if task_id is not None else context.task_id

        if self._context_assembler is not None:
            assembled = await self._context_assembler.assemble(
                query,
                session_id=effective_session_id,
                task_id=effective_task_id,
                max_tokens=max_tokens,
            )
            if assembled.text:
                parts.append(assembled.text)

        def _append_if_fits(line: str) -> bool:
            candidate = "\n".join([*parts, line])
            if get_token_count(candidate) > max_tokens:
                return False
            parts.append(line)
            return True

        if self._chroma_collection is not None:
            try:
                query_args: dict[str, Any] = {
                    "query_texts": [query],
                    "n_results": n_results,
                }
                if session_id is not None:
                    query_args["where"] = {"session_id": effective_session_id}
                results = await asyncio.to_thread(self._chroma_collection.query, **query_args)

                if results["documents"] and results["documents"][0]:
                    for doc, meta in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        strict=False,
                    ):
                        meta = meta if isinstance(meta, dict) else {}
                        if meta.get("source") == "verified_reflection":
                            continue
                        _append_if_fits(
                            "- [legacy episode; source=semantic history] "
                            f'"{doc}" (result: {meta.get("explanation", "N/A")})'
                        )

            except Exception:
                logger.debug("ChromaDB query failed", exc_info=True)

        if self._pool:
            prefs = await self._get_preferences()

            if prefs:
                for k, v in prefs.items():
                    _append_if_fits(f"- [legacy preference; source=stored setting] {k}: {v}")

        return "\n".join(parts) if parts else ""

    async def record_verified_reflection(
        self,
        lesson: str,
        *,
        outcome: str,
        session_id: str,
        occurred_at: str,
    ) -> bool:
        """Index a distilled verifier-backed lesson for semantic case retrieval."""
        if (
            self._chroma_collection is None
            or not isinstance(lesson, str)
            or not lesson.strip()
            or outcome not in {"passed", "failed"}
        ):
            return False
        memory_id = hashlib.sha256(
            f"{session_id}\0{occurred_at}\0{outcome}\0{lesson}".encode()
        ).hexdigest()
        await asyncio.to_thread(
            self._chroma_collection.upsert,
            documents=[lesson],
            metadatas=[{
                "source": "verified_reflection",
                "outcome": outcome,
                "success": str(outcome == "passed"),
                "session_id": session_id or "default",
                "timestamp": occurred_at,
            }],
            ids=[f"reflection-{memory_id}"],
        )
        return True

    async def search_verified_reflections(self, query: str, *, limit: int = 3) -> list[dict[str, str]]:
        """Search only distilled, verifier-backed cases in the semantic memory index."""
        if self._chroma_collection is None or not query.strip():
            return []
        bounded_limit = max(1, min(int(limit), 10))
        try:
            results = await asyncio.to_thread(
                self._chroma_collection.query,
                query_texts=[query],
                n_results=bounded_limit,
                where={"source": "verified_reflection"},
            )
        except Exception:
            logger.debug("Verified reflection search failed", exc_info=True)
            return []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        documents = documents[0] if documents and isinstance(documents[0], list) else []
        metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else []
        found: list[dict[str, str]] = []
        for document, metadata in zip(documents, metadatas, strict=False):
            if not isinstance(document, str) or not document.strip():
                continue
            if not isinstance(metadata, dict):
                continue
            if metadata.get("source") != "verified_reflection":
                continue
            outcome = metadata.get("outcome", "")
            if outcome not in {"passed", "failed"}:
                continue
            found.append({"lesson": document, "outcome": outcome})
        return found

    async def get_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []

        async with self._pool.read() as db:
            if session_id is None:
                cursor = await db.execute(
                    """SELECT id, timestamp, user_input, success, explanation
                       FROM action_history
                       ORDER BY id DESC
                       LIMIT ?
                       OFFSET ?""",
                    (limit, offset),
                )
            else:
                cursor = await db.execute(
                    """SELECT id, timestamp, user_input, success, explanation
                       FROM action_history
                       WHERE session_id = ?
                       ORDER BY id DESC
                       LIMIT ?
                       OFFSET ?""",
                    (session_id, limit, offset),
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

    async def set_preference(
        self,
        key: str,
        value: str,
        *,
        provenance: MemoryProvenance = MemoryProvenance.SYSTEM_OBSERVATION,
        confidence: float = 0.75,
        event_id: str = "",
    ) -> None:
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

        if self._temporal is not None:
            await self._temporal.remember_fact(
                subject="user",
                predicate=f"preference:{key}",
                value=value,
                scope=MemoryScope.USER,
                confidence=confidence,
                provenance=provenance,
                event_id=event_id,
                privacy_class=PrivacyClass.SENSITIVE,
            )

    async def remember_fact(self, **kwargs: Any) -> TemporalFact | None:
        """Store a provenance-labelled temporal fact when temporal memory is active."""
        if self._temporal is None:
            return None
        return await self._temporal.remember_fact(**kwargs)

    async def put_working(
        self,
        *,
        session_id: str,
        task_id: str,
        key: str,
        value: Any,
        priority: float = 0.5,
        ttl_seconds: float = 3600,
    ) -> None:
        """Upsert short-lived task state for bounded planner context."""
        if self._temporal is None:
            return
        await self._temporal.put_working(
            session_id=session_id,
            task_id=task_id,
            key=key,
            value=value,
            priority=priority,
            ttl_seconds=ttl_seconds,
        )

    async def clear_task_working(self, *, session_id: str, task_id: str) -> int:
        """Forget completed task state while keeping its verified episode."""
        if self._temporal is None:
            return 0
        return await self._temporal.clear_task_working(
            session_id=session_id,
            task_id=task_id,
        )

    async def temporal_status(self, *, limit: int = 50) -> dict[str, Any]:
        """Return reviewable facts and counts for the local memory controls."""
        if self._temporal is None:
            return {
                "available": False,
                "facts": [],
                "counts": {
                    "facts": {},
                    "episodes": 0,
                    "working_items": 0,
                },
            }
        facts = await self._temporal.list_facts(limit=limit)
        counts = await self._temporal.stats()
        return {
            "available": True,
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": fact.value,
                    "scope": fact.scope.value,
                    "status": fact.status.value,
                    "confidence": fact.confidence,
                    "provenance": fact.provenance.value,
                    "evidence_count": fact.evidence_count,
                    "valid_from": fact.valid_from,
                    "valid_until": fact.valid_until,
                    "updated_at": fact.updated_at,
                }
                for fact in facts
            ],
            "counts": counts,
        }

    async def retract_fact(self, fact_id: str, *, reason: str = "") -> TemporalFact:
        """Retract one learned fact at the user's request."""
        if self._temporal is None:
            raise RuntimeError("Temporal memory is not initialized")
        return await self._temporal.retract_fact(fact_id, reason=reason)

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

        if self._temporal is not None:
            await self._temporal.close()
            self._temporal = None
            self._context_assembler = None

        if self._pruning_task:
            self._pruning_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pruning_task
            self._pruning_task = None

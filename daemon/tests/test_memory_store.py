"""Unit tests for pilot/memory/store.py

Tests cover:
- MemoryStore initialization
- record() — storing action history
- get_history() — retrieving history with limit/offset
- get_context() — semantic search + preferences
- set_preference() / _get_preferences()
- ChromaDB mocked (no external dependency)
- aiosqlite used in-memory (no file system writes)
- close() cleanup
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from pilot.db.sqlite_pool import AsyncSqlitePool
from pilot.memory.assembler import TemporalContextAssembler
from pilot.memory.sliding_window import get_token_count
from pilot.memory.store import SCHEMA_SQL, MemoryStore
from pilot.memory.temporal import (
    FactStatus,
    MemoryProvenance,
    TemporalMemoryStore,
)

# ---------------------------------------------------------------------------
# Minimal stubs for ActionPlan and ActionResult
# ---------------------------------------------------------------------------


class _FakePlan:
    explanation = "open browser"

    def model_dump_json(self) -> str:
        return json.dumps({"explanation": self.explanation, "actions": []})


class _FakeResult:
    def __init__(self, success: bool = True) -> None:
        self.success = success

    def model_dump(self) -> dict:
        return {"success": self.success, "output": "ok"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_plan() -> _FakePlan:
    return _FakePlan()


@pytest.fixture
def fake_results() -> list[_FakeResult]:
    return [_FakeResult(success=True), _FakeResult(success=True)]


@pytest.fixture
def fake_results_failed() -> list[_FakeResult]:
    return [_FakeResult(success=False)]


@pytest_asyncio.fixture
async def store(tmp_path):
    """
    Return an initialised MemoryStore backed by a real in-memory SQLite DB.
    ChromaDB and WorkspaceIndex are fully mocked — no external services needed.

    Key fix: the real aiosqlite connection is created BEFORE any patching
    so the patch on aiosqlite.connect never intercepts our own :memory: call.
    """
    pool = AsyncSqlitePool(tmp_path / "memory.db")
    await pool.start()
    async with pool.write() as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()

    chroma_mock = MagicMock()
    chroma_mock.add = MagicMock()
    chroma_mock.query = MagicMock(return_value={"documents": [[]], "metadatas": [[]]})

    s = MemoryStore()
    s._pool = pool
    s._chroma_collection = chroma_mock
    s._workspace_index = None
    s._temporal = TemporalMemoryStore(tmp_path / "temporal.db")
    await s._temporal.initialize()
    s._context_assembler = TemporalContextAssembler(s._temporal)

    yield s

    if s._temporal:
        await s._temporal.close()
        s._temporal = None
    if s._pool:
        await s._pool.close()
        s._pool = None


async def _fetch_one(store, query: str, params: tuple = ()):
    async with store._pool.read() as db:
        cursor = await db.execute(query, params)
        row = await cursor.fetchone()
        await cursor.close()
        return row


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


class TestMemoryStoreInit:
    def test_initial_state_is_none(self):
        """MemoryStore attributes start as None before initialize() is called."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        assert s._pool is None
        assert s._chroma_collection is None
        assert s._workspace_index is None

    @pytest.mark.asyncio
    async def test_initialize_sets_up_pool(self, tmp_path):
        """After manually wiring a pool, _pool is not None."""
        pool = AsyncSqlitePool(tmp_path / "memory.db")
        await pool.start()
        async with pool.write() as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

        s = MemoryStore()
        s._pool = pool
        assert s._pool is not None
        await pool.close()

    @pytest.mark.asyncio
    async def test_initialize_migrates_existing_history_to_session_column(self, tmp_path):
        database = tmp_path / "legacy.db"
        pool = AsyncSqlitePool(database)
        await pool.start()
        async with pool.write() as db:
            await db.execute(
                """CREATE TABLE action_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    results_json TEXT,
                    success INTEGER DEFAULT 1,
                    explanation TEXT DEFAULT ''
                )"""
            )
            await db.commit()
        await pool.close()

        s = MemoryStore(checkpoint_interval_seconds=0, pruning_interval_seconds=0)
        with (
            patch("pilot.memory.store.DATA_DIR", tmp_path),
            patch("pilot.memory.store.DB_FILE", database),
            patch.object(s, "_init_chroma"),
            patch.object(s, "_init_workspace_index"),
        ):
            await s.initialize()

        async with s._pool.read() as db:
            cursor = await db.execute("PRAGMA table_info(action_history)")
            columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()

        assert "session_id" in columns
        await s.close()


class TestCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoint_returns_ok_status(self, store):
        """checkpoint() manually flushes the SQLite WAL and returns stats."""
        result = await store.checkpoint()

        assert result["status"] == "ok"
        assert result["mode"] == "TRUNCATE"
        assert "busy" in result
        assert "log_frames" in result
        assert "checkpointed_frames" in result


# ---------------------------------------------------------------------------
# Tests: record()
# ---------------------------------------------------------------------------


class TestRecord:
    @pytest.mark.asyncio
    async def test_record_stores_entry(self, store, fake_plan, fake_results):
        """record() inserts a row into action_history."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("open browser", fake_plan, fake_results)

        row = await _fetch_one(store, "SELECT COUNT(*) FROM action_history")
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_record_stores_correct_user_input(self, store, fake_plan, fake_results):
        """record() saves the exact user input string."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("close window", fake_plan, fake_results)

        row = await _fetch_one(store, "SELECT user_input FROM action_history")
        assert row[0] == "close window"

    @pytest.mark.asyncio
    async def test_record_stores_chat_session_id(self, store, fake_plan, fake_results):
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record(
                "session-specific task",
                fake_plan,
                fake_results,
                session_id="chat-123",
            )

        row = await _fetch_one(store, "SELECT session_id FROM action_history")
        assert row[0] == "chat-123"

    @pytest.mark.asyncio
    async def test_record_success_flag_true(self, store, fake_plan, fake_results):
        """record() sets success=1 when all results succeed."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("open terminal", fake_plan, fake_results)

        row = await _fetch_one(store, "SELECT success FROM action_history")
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_record_success_flag_false(self, store, fake_plan, fake_results_failed):
        """record() sets success=0 when any result fails."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("delete file", fake_plan, fake_results_failed)

        row = await _fetch_one(store, "SELECT success FROM action_history")
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_record_multiple_entries(self, store, fake_plan, fake_results):
        """record() can store multiple entries sequentially."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("task one", fake_plan, fake_results)
            await store.record("task two", fake_plan, fake_results)
            await store.record("task three", fake_plan, fake_results)

        row = await _fetch_one(store, "SELECT COUNT(*) FROM action_history")
        assert row[0] == 3

    @pytest.mark.asyncio
    async def test_record_skips_when_pool_none(self, fake_plan, fake_results):
        """record() returns silently when _pool is None (no crash)."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        s._pool = None
        await s.record("test input", fake_plan, fake_results)

    @pytest.mark.asyncio
    async def test_record_writes_to_chroma(self, store, fake_plan, fake_results):
        """record() calls asyncio.to_thread (ChromaDB write) when collection available."""
        called = []

        async def fake_to_thread(fn, *args, **kwargs):
            called.append(True)

        with patch("pilot.memory.store.asyncio.to_thread", side_effect=fake_to_thread):
            await store.record("search web", fake_plan, fake_results)

        assert len(called) >= 1

    @pytest.mark.asyncio
    async def test_action_success_is_not_verified_without_verifier(self, store, fake_plan, fake_results):
        from types import SimpleNamespace

        fake_plan.actions = [SimpleNamespace(action_type=SimpleNamespace(value="browser_navigate"))]
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record(
                "search the web for release notes",
                fake_plan,
                fake_results,
                session_id="chat-1",
                task_id="task-1",
            )

        episodes = await store._temporal.query_episodes(
            "release notes",
            session_id="chat-1",
        )

        assert len(episodes) == 1
        assert episodes[0].provenance == MemoryProvenance.INFERRED.value
        assert "Outcome: unverified" in episodes[0].text
        assert "search the web" in episodes[0].text

    @pytest.mark.asyncio
    async def test_temporal_episode_is_verified_only_after_goal_verification(
        self, store, fake_plan, fake_results
    ):
        from types import SimpleNamespace

        from pilot.actions import VerificationResult

        fake_plan.actions = [SimpleNamespace(action_type=SimpleNamespace(value="browser_navigate"))]
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record(
                "search the web for release notes",
                fake_plan,
                fake_results,
                session_id="chat-verified",
                task_id="task-verified",
                verification=VerificationResult(passed=True),
            )

        episodes = await store._temporal.query_episodes(
            "release notes",
            session_id="chat-verified",
        )

        assert len(episodes) == 1
        assert episodes[0].provenance == MemoryProvenance.VERIFIED_OUTCOME.value
        assert "Outcome: success" in episodes[0].text

    @pytest.mark.asyncio
    async def test_verification_failure_overrides_successful_action_results(
        self, store, fake_plan, fake_results
    ):
        from types import SimpleNamespace

        from pilot.actions import VerificationResult

        fake_plan.actions = [SimpleNamespace(action_type=SimpleNamespace(value="browser_navigate"))]
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record(
                "search the web for release notes",
                fake_plan,
                fake_results,
                session_id="chat-failed-verify",
                task_id="task-failed-verify",
                verification=VerificationResult(passed=False),
            )

        episodes = await store._temporal.query_episodes(
            "release notes",
            session_id="chat-failed-verify",
        )

        assert len(episodes) == 1
        assert episodes[0].provenance == MemoryProvenance.INFERRED.value
        assert "Outcome: failure" in episodes[0].text


# ---------------------------------------------------------------------------
# Tests: get_history()
# ---------------------------------------------------------------------------


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_get_history_empty(self, store):
        """get_history() returns empty list when no records exist."""
        result = await store.get_history()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_history_returns_entries(self, store, fake_plan, fake_results):
        """get_history() returns stored entries as dicts."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("open notes", fake_plan, fake_results)

        history = await store.get_history()
        assert len(history) == 1
        assert history[0]["user_input"] == "open notes"

    @pytest.mark.asyncio
    async def test_get_history_order_is_newest_first(self, store, fake_plan, fake_results):
        """get_history() returns entries newest first (DESC by id)."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("first task", fake_plan, fake_results)
            await store.record("second task", fake_plan, fake_results)

        history = await store.get_history()
        assert history[0]["user_input"] == "second task"
        assert history[1]["user_input"] == "first task"

    @pytest.mark.asyncio
    async def test_get_history_limit(self, store, fake_plan, fake_results):
        """get_history() respects the limit parameter."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            for i in range(5):
                await store.record(f"task {i}", fake_plan, fake_results)

        history = await store.get_history(limit=3)
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_history_offset(self, store, fake_plan, fake_results):
        """get_history() respects the offset parameter for pagination."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            for i in range(4):
                await store.record(f"task {i}", fake_plan, fake_results)

        page1 = await store.get_history(limit=2, offset=0)
        page2 = await store.get_history(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]["user_input"] != page2[0]["user_input"]

    @pytest.mark.asyncio
    async def test_get_history_filters_by_chat_session(self, store, fake_plan, fake_results):
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("first chat", fake_plan, fake_results, session_id="chat-a")
            await store.record("second chat", fake_plan, fake_results, session_id="chat-b")

        history = await store.get_history(session_id="chat-a")

        assert [entry["user_input"] for entry in history] == ["first chat"]

    @pytest.mark.asyncio
    async def test_get_history_entry_has_required_keys(self, store, fake_plan, fake_results):
        """Each history entry contains id, timestamp, user_input, success, explanation."""
        with patch("pilot.memory.store.asyncio.to_thread", new_callable=AsyncMock):
            await store.record("check email", fake_plan, fake_results)

        history = await store.get_history()
        entry = history[0]
        for key in ("id", "timestamp", "user_input", "success", "explanation"):
            assert key in entry

    @pytest.mark.asyncio
    async def test_get_history_returns_empty_when_pool_none(self):
        """get_history() returns [] when _pool is None."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        s._pool = None
        result = await s.get_history()
        assert result == []


# ---------------------------------------------------------------------------
# Tests: set_preference() and _get_preferences()
# ---------------------------------------------------------------------------


class TestPreferences:
    @pytest.mark.asyncio
    async def test_set_and_get_preference(self, store):
        """set_preference() stores a key-value pair retrievable by _get_preferences()."""
        await store.set_preference("theme", "dark")
        prefs = await store._get_preferences()
        assert prefs["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_set_preference_overwrites_existing(self, store):
        """set_preference() updates value on duplicate key (upsert)."""
        await store.set_preference("theme", "dark")
        await store.set_preference("theme", "light")
        prefs = await store._get_preferences()
        assert prefs["theme"] == "light"

    @pytest.mark.asyncio
    async def test_set_multiple_preferences(self, store):
        """Multiple preferences can be stored independently."""
        await store.set_preference("theme", "dark")
        await store.set_preference("language", "python")
        prefs = await store._get_preferences()
        assert prefs["theme"] == "dark"
        assert prefs["language"] == "python"

    @pytest.mark.asyncio
    async def test_set_preference_creates_provenance_labelled_fact(self, store):
        await store.set_preference("editor", "vscode")

        facts = await store._temporal.query_facts("editor", session_id="chat-1")

        assert len(facts) == 1
        fact = await store._temporal.get_fact(facts[0].memory_id)
        assert fact is not None
        assert fact.status == FactStatus.ACTIVE
        assert fact.provenance == MemoryProvenance.SYSTEM_OBSERVATION

    @pytest.mark.asyncio
    async def test_get_preferences_empty(self, store):
        """_get_preferences() returns empty dict when no prefs set."""
        prefs = await store._get_preferences()
        assert prefs == {}

    @pytest.mark.asyncio
    async def test_set_preference_skips_when_pool_none(self):
        """set_preference() returns silently when _pool is None."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        s._pool = None
        await s.set_preference("key", "value")  # should not raise

    @pytest.mark.asyncio
    async def test_get_preferences_returns_empty_when_pool_none(self):
        """_get_preferences() returns {} when _pool is None."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        s._pool = None
        result = await s._get_preferences()
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_context()
# ---------------------------------------------------------------------------


class TestGetContext:
    @pytest.mark.asyncio
    async def test_get_context_empty_store_returns_empty_string(self, store):
        """get_context() returns empty string when no history or preferences."""
        store._chroma_collection = None
        result = await store.get_context("open browser")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_context_includes_preferences(self, store):
        """get_context() includes user preferences in returned string."""
        store._chroma_collection = None
        await store.set_preference("editor", "vscode")
        result = await store.get_context("open editor")
        assert "editor" in result
        assert "vscode" in result

    @pytest.mark.asyncio
    async def test_task_phase_query_retrieves_matching_desktop_experience(self, store):
        await store._temporal.record_episode(
            session_id="chat-1",
            task_id="task-1",
            summary="Hermes desktop workflow opened its settings panel",
            outcome="success",
            importance=0.8,
        )
        await store._temporal.record_episode(
            session_id="chat-1",
            task_id="task-2",
            summary="Heliox desktop workflow opened its settings panel",
            outcome="success",
            importance=0.8,
        )

        for app in ("Hermes", "Heliox"):
            context = await store.get_context(
                f"Task goal: Continue the desktop workflow. Current app: {app}",
                session_id="chat-1",
            )
            episodes = [line for line in context.splitlines() if line.startswith("- [episode;")]
            assert episodes
            assert app.lower() in episodes[0].lower()

    @pytest.mark.asyncio
    async def test_get_context_with_chroma_results(self, store):
        """get_context() includes semantic search results from ChromaDB."""
        mock_results = {
            "documents": [["open browser last time"]],
            "metadatas": [[{"explanation": "launched chrome"}]],
        }

        async def fake_to_thread(fn, *args, **kwargs):
            return mock_results

        with patch("pilot.memory.store.asyncio.to_thread", side_effect=fake_to_thread):
            result = await store.get_context("open browser")

        assert "open browser last time" in result

    @pytest.mark.asyncio
    async def test_get_context_filters_semantic_memory_by_chat_session(self, store):
        await store.set_preference("interaction_style", "concise")

        async def fake_to_thread(fn, *args, **kwargs):
            assert kwargs["where"] == {"session_id": "chat-123"}
            return {"documents": [[]], "metadatas": [[]]}

        with patch("pilot.memory.store.asyncio.to_thread", side_effect=fake_to_thread):
            context = await store.get_context("open browser", session_id="chat-123")

        assert "interaction_style: concise" in context

    @pytest.mark.asyncio
    async def test_get_context_chroma_failure_does_not_crash(self, store):
        """get_context() handles ChromaDB query failure gracefully."""

        async def fail_to_thread(fn, *args, **kwargs):
            raise Exception("ChromaDB connection error")

        with patch("pilot.memory.store.asyncio.to_thread", side_effect=fail_to_thread):
            result = await store.get_context("test query")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_context_is_strictly_token_bounded(self, store):
        store._chroma_collection = None
        for index in range(20):
            await store.set_preference(f"preference_{index}", "very long preference value " * 10)

        result = await store.get_context(
            "preference",
            session_id="chat-1",
            max_tokens=80,
        )

        assert get_token_count(result) <= 80
        assert "source=" in result


# ---------------------------------------------------------------------------
# Tests: close()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_sets_pool_to_none(self, tmp_path):
        """close() sets _pool to None after closing the pool."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        s._pool = AsyncSqlitePool(tmp_path / "memory.db")
        await s._pool.start()
        await s.close()
        assert s._pool is None

    @pytest.mark.asyncio
    async def test_close_is_safe_when_already_none(self):
        """close() does not raise when _pool is already None."""
        from pilot.memory.store import MemoryStore

        s = MemoryStore()
        s._pool = None
        await s.close()  # should not raise

"""A checkpoint store and a durable task store written by the version before sources were recorded open, gain the
source and scope columns, and their rows resume under the narrowest scope that could have made them.

The old tables are built here with the schema of the earlier version (its CREATE TABLE statements, copied), one row
each, in a test folder."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pilot.actions import Action, ActionPlan, ActionType, FileParams
from pilot.security.gateway import InvocationSource
from pilot.security.source_scope import LEGACY_SOURCE, stored_plan_scope
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.durable_tasks import DurableTaskStore

OLD_CHECKPOINTS = """
CREATE TABLE workflow_checkpoints (
    plan_id TEXT PRIMARY KEY,
    user_input TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    completed_count INTEGER NOT NULL,
    last_output TEXT NOT NULL,
    snapshot_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approval TEXT NOT NULL DEFAULT 'unknown'
)
"""
OLD_TASKS = """
CREATE TABLE durable_tasks (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_input TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_id TEXT NOT NULL DEFAULT '',
    resume_token_hash TEXT NOT NULL UNIQUE,
    cancellation_requested INTEGER NOT NULL DEFAULT 0,
    terminal_response_json TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _columns(path: Path, table: str) -> set[str]:
    with sqlite3.connect(path) as db:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


@pytest.mark.asyncio
async def test_an_old_checkpoint_store_gains_the_columns_and_its_plan_resumes_under_the_legacy_scope(tmp_path) -> None:
    path = tmp_path / "checkpoints.db"
    plan = ActionPlan(
        actions=[
            Action(action_type=ActionType.FILE_READ, target="C:/w/a.txt", parameters=FileParams(path="C:/w/a.txt"))
        ]
    )
    with sqlite3.connect(path) as db:
        db.execute(OLD_CHECKPOINTS)
        db.execute(
            "INSERT INTO workflow_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "p-old",
                "read a.txt",
                plan.model_dump_json(),
                "[]",
                0,
                "",
                "[]",
                "running",
                "2026-11-20T10:00:00",
                "approved",
            ),
        )

    store = WorkflowCheckpointStore(path)
    await store.initialize()
    try:
        assert {"source", "scope_json"} <= _columns(path, "workflow_checkpoints")
        checkpoint = await store.get("p-old")
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()
    assert checkpoint is not None and checkpoint.source == "" and checkpoint.scope_override is None
    source, scope = stored_plan_scope(None, checkpoint.source, checkpoint.scope_override)
    assert source == LEGACY_SOURCE and source is not InvocationSource.INTERACTIVE
    assert scope is not None


@pytest.mark.asyncio
async def test_an_old_durable_task_store_gains_the_columns_and_keeps_its_rows(tmp_path) -> None:
    path = tmp_path / "durable_tasks.db"
    with sqlite3.connect(path) as db:
        db.execute(OLD_TASKS)
        db.execute(
            "INSERT INTO durable_tasks (task_id, session_id, user_id, user_input, status, resume_token_hash, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("t-old", "s", "local", "read a.txt", "executing", "hash-1", "2026-11-20T10:00:00", "2026-11-20T10:00:00"),
        )

    store = DurableTaskStore(path)
    await store.initialize()
    try:
        assert {"source", "scope_json"} <= _columns(path, "durable_tasks")
        task = await store.get("t-old")
        scope = await store.scope("t-old")
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()
    assert task is not None and scope == ("", None)
    source, _override = stored_plan_scope(None, *scope)
    assert source == LEGACY_SOURCE

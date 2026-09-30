from __future__ import annotations

import json
from types import SimpleNamespace

import aiosqlite
import pytest

import pilot.agents.reflector as reflector_module
from pilot.agents.reflector import Reflector


class _Model:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "reflection": "The verifier did not observe the expected result.",
                "lessons_learned": "Confirm the requested report exists at the destination before finishing.",
                "difficulty_score": 0.7,
                "discovered_skills": [],
                "related_task_patterns": [],
            }
        )


async def _open_reflector(tmp_path, monkeypatch, memory_store=None) -> tuple[Reflector, _Model]:
    monkeypatch.setattr(reflector_module, "DB_FILE", tmp_path / "reflector.db")
    model = _Model()
    reflector = Reflector(model, memory_store=memory_store)
    await reflector.initialize()
    return reflector, model


def _plan() -> SimpleNamespace:
    action = SimpleNamespace(action_type=SimpleNamespace(value="browser_download"))
    return SimpleNamespace(actions=[action], explanation="Download the requested report.")


@pytest.mark.asyncio
async def test_verified_failure_is_stored_and_retrievable_as_a_reusable_lesson(tmp_path, monkeypatch):
    reflector, model = await _open_reflector(tmp_path, monkeypatch)
    try:
        result = SimpleNamespace(success=True, error=None)
        verification = SimpleNamespace(
            passed=False,
            details=["Expected report was not present."],
            failed_actions=[0],
        )

        stored = await reflector.reflect(
            "Download the payroll report using Bearer abcdefghijklmnop",
            _plan(),
            [result],
            verification,
        )

        assert stored == {"stored": True, "verification_state": "failed"}
        assert "Verifier Outcome: failed" in model.prompts[0]
        assert "abcdefghijklmnop" not in model.prompts[0]
        cursor = await reflector._db.execute("SELECT success, verification_state, user_input FROM task_reflections")
        row = await cursor.fetchone()
        await cursor.close()
        assert row[0:2] == (0, "failed")
        assert "abcdefghijklmnop" not in row[2]

        context = await reflector.get_improvement_context("payroll report download")
        assert "[verified failure]" in context
        assert "Confirm the requested report exists at the destination" in context
        assert "Download the payroll report" not in context
        assert "abcdefghijklmnop" not in context
    finally:
        await reflector.close()


@pytest.mark.asyncio
async def test_reflector_uses_existing_semantic_case_search_when_available(tmp_path, monkeypatch):
    class _SemanticMemory:
        def __init__(self) -> None:
            self.recorded: list[dict] = []

        async def record_verified_reflection(self, lesson, **kwargs):
            self.recorded.append({"lesson": lesson, **kwargs})
            return True

        async def search_verified_reflections(self, query, *, limit=3):
            assert query == "archive this web page as a PDF"
            assert limit == 3
            return [{"lesson": "Check the saved artifact at its destination.", "outcome": "failed"}]

    memory = _SemanticMemory()
    reflector, _ = await _open_reflector(tmp_path, monkeypatch, memory)
    try:
        await reflector.reflect(
            "Download the payroll report",
            _plan(),
            [SimpleNamespace(success=True, error=None)],
            SimpleNamespace(passed=False, details=["Report not found"], failed_actions=[0]),
            session_id="session-a",
        )

        assert len(memory.recorded) == 1
        assert memory.recorded[0]["lesson"] == (
            "Confirm the requested report exists at the destination before finishing."
        )
        assert memory.recorded[0]["outcome"] == "failed"
        assert memory.recorded[0]["session_id"] == "session-a"
        assert memory.recorded[0]["occurred_at"]
        context = await reflector.get_improvement_context("archive this web page as a PDF")
        assert "[verified failure]" in context
        assert "saved artifact at its destination" in context
        assert "payroll" not in context
    finally:
        await reflector.close()


@pytest.mark.asyncio
async def test_reflector_skips_dry_runs_missing_verification_and_contradictory_results(tmp_path, monkeypatch):
    reflector, model = await _open_reflector(tmp_path, monkeypatch)
    try:
        action_result = SimpleNamespace(success=True, error=None)
        passed = SimpleNamespace(passed=True, details=["Verified"])
        failed_action = SimpleNamespace(success=False, error="Execution failed")

        dry_run = await reflector.reflect("download report", _plan(), [action_result], passed, dry_run=True)
        unverified = await reflector.reflect("download report", _plan(), [action_result], None)
        inconsistent = await reflector.reflect("download report", _plan(), [failed_action], passed)

        assert dry_run["reason"] == "dry_run"
        assert unverified["reason"] == "missing_verified_action_outcome"
        assert inconsistent["reason"] == "inconsistent_verifier_and_action_results"
        assert model.prompts == []
        cursor = await reflector._db.execute("SELECT COUNT(*) FROM task_reflections")
        assert (await cursor.fetchone())[0] == 0
        await cursor.close()
    finally:
        await reflector.close()


@pytest.mark.asyncio
async def test_only_independently_observed_successes_are_learned(tmp_path, monkeypatch):
    reflector, model = await _open_reflector(tmp_path, monkeypatch)
    try:
        result = SimpleNamespace(success=True, error=None)
        reported_only = await reflector.reflect(
            "open the report",
            _plan(),
            [result],
            SimpleNamespace(
                passed=True,
                details=["REPORTED — no independent postcondition"],
                failed_actions=[],
                independently_observed_actions=[],
            ),
        )
        unverifiable_failure = await reflector.reflect(
            "open the report",
            _plan(),
            [result],
            SimpleNamespace(passed=False, details=["No verification data"], failed_actions=[]),
        )
        observed = await reflector.reflect(
            "open the report",
            _plan(),
            [result],
            SimpleNamespace(
                passed=True,
                details=["Destination exists"],
                failed_actions=[],
                independently_observed_actions=[0],
            ),
        )

        assert reported_only["reason"] == "missing_independent_postcondition_evidence"
        assert unverifiable_failure["reason"] == "missing_verified_failure_evidence"
        assert observed == {"stored": True, "verification_state": "passed"}
        assert len(model.prompts) == 1
    finally:
        await reflector.close()


@pytest.mark.asyncio
async def test_context_retrieves_relevant_verified_lessons_without_past_task_text(tmp_path, monkeypatch):
    reflector, _ = await _open_reflector(tmp_path, monkeypatch)
    try:
        rows = [
            (
                "CONFIDENTIAL payroll download from browser",
                0,
                "Check that the exported report exists at the requested destination.",
                "failed",
            ),
            ("Create a calendar invitation", 1, "Check the attendee timezone before sending the invitation.", "passed"),
            ("browser payroll report export", 1, "This legacy lesson must remain untrusted.", "legacy_unverified"),
        ]
        for user_input, success, lesson, state in rows:
            await reflector._db.execute(
                """INSERT INTO task_reflections
                   (timestamp, user_input, success, reflection, lessons_learned,
                    difficulty_score, action_types, verification_state)
                   VALUES ('2026-09-30T00:00:00Z', ?, ?, '', ?, 0.5, '[]', ?)""",
                (user_input, success, lesson, state),
            )
        await reflector._db.commit()

        relevant = await reflector.get_improvement_context("browser payroll report export")
        unrelated = await reflector.get_improvement_context("calendar invitation timezone")

        assert "[verified failure]" in relevant
        assert "requested destination" in relevant
        assert "CONFIDENTIAL" not in relevant
        assert "timezone" not in relevant
        assert "legacy lesson" not in relevant
        assert "[verified success]" in unrelated
        assert "timezone before sending" in unrelated
    finally:
        await reflector.close()


@pytest.mark.asyncio
async def test_legacy_reflections_are_migrated_but_not_recalled_as_verified(tmp_path, monkeypatch):
    db_path = tmp_path / "reflector.db"
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """CREATE TABLE task_reflections (
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
            INSERT INTO task_reflections
                (timestamp, user_input, success, reflection, lessons_learned)
            VALUES ('2026-09-29T00:00:00Z', 'browser report', 1, '', 'Unverified old lesson');"""
        )
        await db.commit()

    monkeypatch.setattr(reflector_module, "DB_FILE", db_path)
    reflector = Reflector(_Model())
    await reflector.initialize()
    try:
        await reflector._db.execute(
            """INSERT INTO task_reflections
               (timestamp, user_input, success, reflection, lessons_learned, verification_state)
               VALUES ('2026-09-30T00:00:00Z', 'send calendar invite', 1, '',
                       'Check the attendee timezone.', 'passed')"""
        )
        await reflector._db.commit()
        cursor = await reflector._db.execute("SELECT verification_state FROM task_reflections")
        assert {row[0] for row in await cursor.fetchall()} == {"legacy_unverified", "passed"}
        await cursor.close()
        assert await reflector.get_improvement_context("browser report") == ""
        stats = await reflector.get_stats()
        assert stats["total_tasks"] == 1
        assert stats["success_rate"] == 100.0
        assert stats["legacy_unverified_tasks"] == 1
    finally:
        await reflector.close()

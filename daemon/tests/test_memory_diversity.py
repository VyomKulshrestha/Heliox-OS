"""Memory recall favours recent memories and does not repeat itself."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pilot.memory.assembler import TemporalContextAssembler, diversify
from pilot.memory.store import MemoryStore
from pilot.memory.temporal import MemoryProvenance, RankedMemory, TemporalMemoryStore


def _memory(memory_id: str, text: str, score: float) -> RankedMemory:
    return RankedMemory(kind="episode", text=text, score=score, confidence=1.0, provenance="t", memory_id=memory_id)


@pytest.fixture
async def temporal(tmp_path):
    store = TemporalMemoryStore(tmp_path / "temporal.db")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def test_diversify_drops_near_duplicates_and_promotes_different_memories():
    items = [
        _memory("a1", "Opened Notepad and saved notes.txt to Documents", 1.0),
        _memory("a2", "Opened Notepad and saved notes.txt to Documents", 0.99),  # exact repeat
        _memory("a3", "Opened Notepad and saved notes.txt to the Documents folder", 0.98),  # close variant
        _memory("b1", "Backed up the photos folder to drive D", 0.8),
    ]

    ordered, dropped = diversify(items)

    assert [item.memory_id for item in ordered] == ["a1", "b1", "a3"]
    assert dropped == 1


def test_diversify_keeps_order_when_everything_is_different():
    items = [_memory("x", "rename report", 0.9), _memory("y", "email Priya the invoice", 0.5)]

    assert diversify(items) == (items, 0)
    assert diversify([]) == ([], 0)


async def test_repeated_episodes_do_not_crowd_out_a_relevant_fact(temporal):
    for _ in range(6):
        await temporal.record_episode(
            session_id="s1",
            task_id="t",
            summary="Exported the quarterly report to PDF",
            outcome="success",
            importance=0.9,
        )
    await temporal.remember_fact(
        subject="user",
        predicate="report folder",
        value="D:/Reports/Quarterly",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )

    context = await TemporalContextAssembler(temporal).assemble("quarterly report", session_id="s1", max_tokens=400)

    assert "D:/Reports/Quarterly" in context.text
    assert context.text.count("Exported the quarterly report to PDF") == 1
    assert context.duplicate_count == 5
    assert context.omitted_count >= 5


async def test_each_memory_stays_on_one_line(temporal):
    await temporal.record_episode(
        session_id="s1",
        task_id="t",
        summary='Report free disk space\n\n(Scheduled run "Disk check".)',
        outcome="success",
    )

    context = await TemporalContextAssembler(temporal).assemble("disk space", session_id="s1", max_tokens=400)

    [line] = context.text.splitlines()[1:]
    assert 'Report free disk space (Scheduled run "Disk check".)' in line


async def test_older_episodes_rank_below_recent_ones(temporal):
    now = datetime.now(UTC)
    await temporal.record_episode(
        session_id="s1",
        task_id="old",
        summary="Cleaned the Downloads folder",
        outcome="success",
        occurred_at=(now - timedelta(days=60)).isoformat(),
    )
    await temporal.record_episode(
        session_id="s1",
        task_id="new",
        summary="Cleaned the Downloads folder by type",
        outcome="success",
        occurred_at=(now - timedelta(hours=2)).isoformat(),
    )

    ranked = await temporal.query_episodes("clean downloads folder", session_id="s1", now=now)

    assert ranked[0].text.startswith("Cleaned the Downloads folder by type")
    assert ranked[0].score > 2 * ranked[1].score


class _FakeChroma:
    def __init__(self, documents, metadatas, distances):
        self.result = {"documents": [documents], "metadatas": [metadatas], "distances": [distances]}
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


async def test_legacy_recall_weighs_age_and_skips_repeats():
    now = datetime.now(UTC)
    store = MemoryStore()
    store._chroma_collection = _FakeChroma(
        documents=[
            "open the budget spreadsheet",  # closest match, but a year old
            "open the budget spreadsheet",  # identical repeat
            "open the budget spreadsheet in Excel",  # slightly further, recorded today
            "rename the budget spreadsheet",
        ],
        metadatas=[
            {"timestamp": (now - timedelta(days=365)).isoformat(), "explanation": "old"},
            {"timestamp": (now - timedelta(days=360)).isoformat(), "explanation": "old repeat"},
            {"timestamp": now.isoformat(), "explanation": "today"},
            {"timestamp": now.isoformat(), "explanation": "renamed"},
        ],
        distances=[0.10, 0.10, 0.30, 0.60],
    )

    context = await store.get_context("open budget spreadsheet", n_results=3)

    lines = context.splitlines()
    assert lines[0].endswith('"open the budget spreadsheet in Excel" (result: today)')
    assert sum('"open the budget spreadsheet" (result' in line for line in lines) == 1
    assert store._chroma_collection.calls[0]["n_results"] == 9


def test_legacy_recency_falls_back_for_unreadable_timestamps():
    assert MemoryStore._legacy_recency("not a date") == 0.25
    fresh = MemoryStore._legacy_recency(datetime.now(UTC).isoformat())
    assert 0.99 < fresh <= 1.0

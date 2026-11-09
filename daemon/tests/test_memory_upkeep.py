"""Memory, part two: consolidation, links between people and memories, and a gated tuning loop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pilot.actions import ActionPlan
from pilot.memory import assembler, bench, store, tune
from pilot.memory.consolidate import consolidate
from pilot.memory.links import expand_query
from pilot.memory.store import MemoryStore
from pilot.memory.temporal import TemporalMemoryStore


# ── consolidation ──


@pytest.mark.asyncio
async def test_consolidation_merges_repeats_fades_old_answers_and_keeps_action_outcomes(tmp_path):
    temporal = TemporalMemoryStore(tmp_path / "t.db")
    await temporal.initialize()
    now = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
    old = (now - timedelta(days=120)).isoformat()
    try:
        for day in range(5):  # the same question, five days running
            await temporal.record_episode(
                session_id="s", task_id="", summary="Request: check disk space. Answer: 112 GB free on C.",
                outcome="answered", importance=0.45, occurred_at=(now - timedelta(days=day)).isoformat(),
            )  # fmt: skip
        await temporal.record_episode(
            session_id="s", task_id="", summary="Request: what's the weather. Answer: sunny.", outcome="answered",
            importance=0.45, occurred_at=old,
        )  # fmt: skip
        await temporal.record_episode(
            session_id="s", task_id="", summary="Request: install the printer driver. Plan: ran the installer.",
            outcome="success", importance=0.65, occurred_at=old,
        )  # fmt: skip
        report = await consolidate(temporal, now=now)
        assert report["merged"] == 4 and report["faded"] == 1
        left = {e["summary"]: e for e in await temporal.recent_episodes()}
        assert len(left) == 2 and any("install the printer driver" in summary for summary in left)  # outcome kept
        assert left["Request: check disk space. Answer: 112 GB free on C."]["importance"] == pytest.approx(0.65)
        assert await consolidate(temporal, now=now) == {"working_expired": 0, "merged": 0, "faded": 0}
    finally:
        await temporal.close()


@pytest.mark.asyncio
async def test_the_store_runs_upkeep_and_reports_it(tmp_path):
    memory = MemoryStore(checkpoint_interval_seconds=0, pruning_interval_seconds=0, db_file=tmp_path / "m.db",
                         semantic_search=False)  # fmt: skip
    await memory.initialize()
    try:
        for _ in range(3):
            plan = ActionPlan(actions=[], explanation="You have 112 GB free on C.")
            await memory.record("Check how much disk space is left", plan, [], session_id="s")
        report = await memory.consolidate()
        assert report["merged"] == 2 and memory.last_consolidation["merged"] == 2
    finally:
        await memory.close()


# ── links ──


def test_a_relation_in_the_request_brings_in_the_name_behind_it():
    people = {"manager": "Rahul", "sister": "Ana"}
    assert expand_query("what did my manager say about staff", people) == "what did my manager say about staff Rahul"
    assert expand_query("gifts for my sisters", people).endswith("Ana")
    assert expand_query("ask Rahul, my manager", people) == "ask Rahul, my manager"  # already named
    assert expand_query("plan the weekend", people) == "plan the weekend"


@pytest.mark.asyncio
async def test_under_pressure_the_link_is_what_finds_the_managers_decision(monkeypatch):
    linked = await bench.run(pressure=1, budget=1500)
    assert next(r for r in linked["rows"] if r["id"] == "via-manager")["pass"]
    monkeypatch.setattr(MemoryStore, "set_people_source", lambda self, source: None)
    unlinked = await bench.run(pressure=1, budget=1500)
    assert not next(r for r in unlinked["rows"] if r["id"] == "via-manager")["pass"]


@pytest.mark.asyncio
async def test_under_pressure_every_probe_still_passes():
    for budget in (1500, 4000):
        result = await bench.run(pressure=2, budget=budget)
        assert result["passed"] == result["probes"] and result["stale"] == 0, result["rows"]


# ── the gated tuning loop ──


def test_tuning_settings_are_applied_only_inside_the_trial_and_always_restored():
    before = tune.current()
    with tune.applied({"pilot.memory.store.PROFILE_TOKENS": 200, "pilot.memory.assembler.MMR_RELEVANCE_WEIGHT": 0.5}):
        assert store.PROFILE_TOKENS == 200 and assembler.MMR_RELEVANCE_WEIGHT == 0.5
    assert tune.current() == before
    train, held_out = tune.split(["a", "b", "c", "d", "e"])
    assert train == {"a", "c", "e"} and held_out == {"b", "d"} and not train & held_out
    assert tune.candidates(5) == tune.candidates(5)  # the same search every run


@pytest.mark.asyncio
async def test_the_gate_refuses_a_change_that_breaks_a_probe(monkeypatch):
    monkeypatch.setattr(tune, "BUDGETS", (1500,))
    monkeypatch.setattr(
        tune,
        "candidates",
        lambda count, seed=11: [{**tune.current(), "pilot.memory.store.PROFILE_TOKENS": 20}],  # starves the profile
    )
    report = await tune.tune(count=1, pressure=1)
    [tried] = report["tried"]
    assert tried["regressions"] and not tried["accepted"] and report["recommendation"] is None

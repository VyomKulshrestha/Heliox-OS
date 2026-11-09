"""Context and memory for long sessions: what reaches the model after weeks of use, measured."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import ActionPlan
from pilot.memory import bench
from pilot.memory.session_recall import earlier_turns
from pilot.memory.sliding_window import get_token_count
from pilot.memory.store import MemoryStore
from pilot.models.ollama import OllamaClient, context_size
from pilot.workflows.user_profile import UserProfile


@pytest.mark.asyncio
async def test_the_memory_bench_passes_every_probe_and_brings_back_nothing_stale():
    result = await bench.run()
    failed = [row for row in result["rows"] if not row["pass"]]
    assert not failed, failed
    assert result["probes"] >= 10 and result["stale"] == 0


@pytest.mark.asyncio
async def test_why_profile_facts_need_their_own_memory_labels(tmp_path):
    """One shared label made each new fact supersede the last: only the newest "about" fact stayed active."""
    from pilot.memory.temporal import FactStatus, MemoryProvenance, TemporalMemoryStore

    store = TemporalMemoryStore(tmp_path / "t.db")
    await store.initialize()
    facts = ("I work at a bakery", "I have two cats", "I live in Pune")

    async def remember(label: str, value: str):
        return await store.remember_fact(
            subject="user", predicate=label, value=value, confidence=1.0, provenance=MemoryProvenance.EXPLICIT_USER
        )

    try:
        shared = [await remember("profile:about", value) for value in facts]
        own = [await remember(f"profile:about:{i}", value) for i, value in enumerate(facts)]
        active = {fact.fact_id for fact in await store.list_facts(statuses=(FactStatus.ACTIVE,))}
        assert [f.fact_id in active for f in shared] == [False, False, True]
        assert all(f.fact_id in active for f in own)
    finally:
        await store.close()


# ── the profile: slots and the always-on block ──


@pytest.mark.asyncio
async def test_one_current_value_per_slot_and_everything_else_side_by_side(tmp_path):
    mirrored: dict[str, tuple[str, str]] = {}
    retracted: list[str] = []

    async def remember(predicate, value):
        mirrored[f"f{len(mirrored)}"] = (predicate, value)
        return f"f{len(mirrored) - 1}"

    async def forget(fact_id):
        retracted.append(fact_id)

    profile = UserProfile(tmp_path / "profile.json", remember=remember, forget=forget)
    for said in (
        "Remember that I live in Mumbai",
        "Remember that I work at a bakery",
        "Remember that I have two cats",
        "Remember that I live in Pune now",
        "My manager is Priya",
        "My new manager is Rahul",
        "My friend is Sam",
        "My friend is Alex",
    ):
        await profile.answer_async(said)
    lines = profile.context_lines()
    assert "About them: I live in Pune now" in lines and not any("Mumbai" in line for line in lines)
    assert "About them: I work at a bakery" in lines and "About them: I have two cats" in lines
    assert [line for line in lines if "manager" in line] == ["Person they know: Rahul (manager)"]
    assert sum("(friend)" in line for line in lines) == 2
    labels = [predicate for predicate, _ in mirrored.values()]
    assert len(set(labels)) == len(labels) - 2  # only home and manager were reused, by the replacements
    assert len(retracted) == 2


@pytest.mark.asyncio
async def test_the_profile_leads_the_context_within_its_budget(tmp_path):
    store = MemoryStore(checkpoint_interval_seconds=0, pruning_interval_seconds=0, db_file=tmp_path / "m.db",
                        semantic_search=False)  # fmt: skip
    await store.initialize()
    try:
        store.set_profile_source(lambda: ["About them: I have two cats", "They prefer: short answers"])
        context = await store.get_context("order food for my pets", max_tokens=600)
        assert context.startswith("What they have told Heliox about themselves:")
        assert "I have two cats" in context and "short answers" in context
        store.set_profile_source(lambda: [f"About them: fact number {i} " + "word " * 30 for i in range(50)])
        crowded = await store.get_context("x", max_tokens=600)
        # The block has a ceiling; over it, what matches the request comes first, then the most recent.
        assert "fact number 49" in crowded and "fact number 0 " not in crowded
        assert get_token_count(crowded.split("Relevant memory")[0]) <= 300
        assert "fact number 3 " in await store.get_context("what about fact number 3", max_tokens=600)

        # An answer is remembered as an episode, so it can be found later without the vector index.
        plan = ActionPlan(actions=[], explanation="Drafted the Q4 summary for Lisbon: revenue up 12 percent.")
        await store.record("Draft the Q4 summary for Lisbon", plan, [], session_id="s1")
        store.set_profile_source(None)
        assert "revenue up 12 percent" in await store.get_context("update the Lisbon summary", max_tokens=600)
    finally:
        await store.close()


# ── long sessions ──


def test_earlier_turns_keep_the_opening_and_what_matches_within_the_budget():
    older = [{"user_input": "Today's report is for the Porto warehouse", "explanation": "Got it."}]
    older += [{"user_input": f"Unrelated step {i}", "explanation": f"Done {i}."} for i in range(30)]
    older.insert(12, {"user_input": "Count the late deliveries", "explanation": "14 late deliveries of 210."})
    text = earlier_turns(older, "put the late deliveries count in the Porto summary", max_tokens=200)
    assert text.startswith("Earlier in this conversation:")
    assert "Porto warehouse" in text and "14 late deliveries" in text
    assert text.index("Porto") < text.index("14 late")  # in the order they happened
    assert "Unrelated step 29" not in text
    assert earlier_turns(older, "anything", max_tokens=0) == ""
    assert earlier_turns([], "anything", max_tokens=500) == ""


# ── the model's real context window ──


def test_the_ollama_window_fits_the_prompt_and_never_exceeds_the_model():
    assert context_size(100, 262144) == 4096
    assert context_size(4100, 262144) == 8192  # the planner prompt no longer risks a 4,096 window
    assert context_size(20000, 16384) == 16384  # never more than the model supports
    assert context_size(100000, None) == 32768


@pytest.mark.asyncio
async def test_the_model_context_length_is_asked_once():
    client = OllamaClient.__new__(OllamaClient)
    client._base_url = "http://127.0.0.1:11434"
    client._model_max = {}
    response = MagicMock(status_code=200)
    response.json.return_value = {"model_info": {"qwen35.context_length": 262144, "general.name": "x"}}
    client._client = MagicMock(post=AsyncMock(return_value=response))
    assert await client._max_context("qwen3.5:4b") == 262144
    assert await client._max_context("qwen3.5:4b") == 262144
    assert client._client.post.await_count == 1


def test_the_bench_script_is_well_formed():
    script = json.loads(bench.SCRIPT.read_text(encoding="utf-8"))
    sessions = {session["id"] for session in script["sessions"]}
    for probe in script["probes"]:
        assert probe["session"] == "new" or probe["session"] in sessions
        assert probe.get("expect") or probe.get("forbid")


# ── forgetting everything, from Your data ──


@pytest.mark.asyncio
async def test_conversation_history_and_long_term_memory_can_each_be_wiped(tmp_path):
    from pilot.memory.temporal import MemoryProvenance
    from pilot.workflows.privacy import STORES

    assert {"conversation_history", "long_term_memory"} <= set(STORES)
    store = MemoryStore(
        checkpoint_interval_seconds=0, pruning_interval_seconds=0, db_file=tmp_path / "m.db", semantic_search=False
    )
    await store.initialize()
    try:
        plan = ActionPlan(actions=[], explanation="Found three flour suppliers.")
        await store.record("Find a flour supplier", plan, [], session_id="s1")
        await store.remember_fact(
            subject="user", predicate="profile:about:x", value="I bake", provenance=MemoryProvenance.EXPLICIT_USER,
            confidence=1.0,
        )  # fmt: skip
        assert await store.counts() == {"history": 1, "long_term": 2}  # the turn's episode and the fact
        assert await store.clear_history() == 1
        assert (await store.counts())["history"] == 0 and (await store.counts())["long_term"] == 2
        assert await store.clear_long_term() == 2
        assert await store.counts() == {"history": 0, "long_term": 0}
        assert "flour" not in await store.get_context("flour supplier", max_tokens=600)
    finally:
        await store.close()
    # The append-only guard on evidence is back once the wipe is done.
    import sqlite3

    with sqlite3.connect(tmp_path / "temporal_memory.db") as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert "fact_evidence_no_delete" in names

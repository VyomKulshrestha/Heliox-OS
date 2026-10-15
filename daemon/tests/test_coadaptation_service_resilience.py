"""CoAdaptationService under partial failure, cancellation, real clocks and stuck clients (spec 10.7, 13.3, 14.3).

tmp_path, InMemoryKeyProvider and fake clocks only. Each test pins one property the happy-path tests in
test_coadaptation_service.py cannot see: erasure that survives an export file another program holds open or a locked
keyring, a retraction stamped at the instant its fallbacks were issued, a study call the RPC layer cancels half way, a
decision the store refuses, a session the UI lost, and notifications that never hold the service lock.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import pilot.coadaptation.service as service_module
import pilot.coadaptation.store as store_module
from pilot.coadaptation.keys import CoAdaptationKeyUnavailable
from tests.test_coadaptation_service import (
    ALL_PURPOSES,
    FakeClock,
    _veto_trials,
    begin,
    begin_params,
    block_context,
    build_trial,
    candidate_rects,
    decide_block,
    grant_params,
    make_harness,
    new_participant,
    pointer_activation,
    pref_params,
    run_block,
    run_session,
    sql,
)


@pytest.fixture
async def harness(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    yield h
    await h.svc.close()


class TickingClock(FakeClock):
    """Like a real clock: every reading is a little later than the one before."""

    def __call__(self) -> datetime:
        self.now += timedelta(microseconds=7)
        return self.now


@pytest.fixture
def locked_files(monkeypatch):
    """Paths whose unlink fails as on Windows when another program holds the file open (WinError 32)."""
    locked: set[str] = set()
    original = Path.unlink

    def unlink(self, missing_ok=False):
        if str(self) in locked:
            raise PermissionError(13, "The process cannot access the file because it is being used", str(self))
        return original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    return locked


def _pending(h) -> store_module.PendingErasure:
    return h.svc.store.pending_erasure()


def _trials_seen(h, participant_key: str) -> int:
    return h.svc._engines[participant_key].status().trials_seen


# ── retraction time = fallbacks issue time (finding: retraction stamped after its fallbacks) ─────────────────────


async def test_retraction_is_stamped_at_the_issue_time_of_its_fallbacks(tmp_path):
    h = make_harness(tmp_path)
    h.clock = TickingClock()
    h.svc = h.new_service()
    await h.svc.start()
    try:
        pk = await new_participant(h)
        live = await begin(h, pk)
        freeze = await h.svc.set_preference(
            pref_params(
                "freeze_adaptation", {"regime": "confirm_selection:hold_700"}, participant_key=pk, modality="mouse"
            )
        )
        assert freeze["status"] == "ok", freeze
        rec, table = await decide_block(live, 2)
        assert rec["variant"] == "confirm_selection:hold_700"
        pref_id = freeze["preference"]["pref_id"]
        retract = await h.svc.retract_preference({"pref_id": pref_id})
        assert retract["status"] == "ok"
        fallbacks = retract["fallbacks"]
        issued = datetime.fromisoformat(fallbacks["issued_at"])
        assert fallbacks["refresh"] == "assistance_off"
        # The file, the rule history and every reconstruction at issued_at agree that the rule is gone then.
        assert h.svc._prefs_store.get(pref_id).retracted_at == issued
        (stored_rule,) = [rule for rule in await h.svc.store.rules(pk) if rule.pref_id == pref_id]
        assert stored_rule.retracted_at == issued
        assert h.svc._engines[pk]._fallbacks_at("mouse", issued).refresh == fallbacks["refresh"]

        # The UI reverts the running trial to the reply's refresh regime at that issue time.
        task = next(b["task"] for b in live.blocks if b["index"] == 2)
        first = pointer_activation(live, candidate_rects(task), rec["spread"], 0, rec["variant"], index=0)
        trial = build_trial(
            live,
            2,
            task,
            rec["decision_id"],
            rec["variant"],
            rec["spread"],
            0,
            intended=0,
            table_ids=[table["decision_id"]] if table else [],
            activations=[first],
            revert={
                "reason": "refresh",
                "from_activation": 1,
                "regime": fallbacks["refresh"],
                "fallbacks_issued_at": fallbacks["issued_at"],
            },
        )
        reply = await h.svc.record_trial({"session_id": live.session_id, "trial": trial})
        assert reply["status"] == "recorded", reply
        assert reply["preference_violation"] is False
        assert reply["verification"] == "ok"
    finally:
        await h.svc.close()


# ── erasure that survives a locked export file (finding: a failed export delete aborts revoke/expiry/erase) ────


async def test_revoke_with_a_locked_export_finishes_and_the_file_goes_later(harness, locked_files):
    keep = await new_participant(harness)
    pk = await new_participant(harness, ("selection_study", "research_export"))
    await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    export = await harness.svc.export({"participant_key": pk})
    locked_files.add(export["path"])

    reply = await harness.svc.revoke_consent({"participant_key": pk})
    assert reply["status"] == "ok", reply
    erased = reply["erased"]
    assert erased["rows"] > 0 and erased["key_destroyed"] is True and erased["rules_deleted"] == 1
    assert erased["export_files"] == 0 and erased["export_files_left"] == 1 and erased["pending"] is True
    assert pk not in harness.keys
    assert pk not in harness.preferences_path.read_text(encoding="utf-8")
    assert harness.broadcasts.of("revoked")[-1]["participant_key"] == pk
    assert await harness.svc.store.participants() == [keep]
    assert Path(export["path"]).exists()
    assert _pending(harness).participants == (pk,)

    # A repeated withdrawal completes the work instead of answering no_consent.
    again = await harness.svc.revoke_consent({"participant_key": pk})
    assert again["status"] == "ok" and again["erased"]["pending"] is True

    locked_files.clear()
    assert (await harness.svc.status({}))["status"] == "ok"  # status runs the prune, which retries
    assert not Path(export["path"]).exists()
    assert _pending(harness).empty and not harness.svc.store.pending_path.exists()
    assert await harness.svc.revoke_consent({"participant_key": pk}) == {"status": "ignored", "reason": "no_consent"}


async def test_pending_erasure_survives_a_restart(harness, locked_files):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    export = await harness.svc.export({"participant_key": pk})
    locked_files.add(export["path"])
    assert (await harness.svc.revoke_consent({"participant_key": pk}))["status"] == "ok"
    await harness.svc.close()
    assert not harness.db_path.exists()  # the last participant took the database with it
    assert json.loads(harness.svc.store.pending_path.read_text(encoding="utf-8"))["participants"] == [pk]
    locked_files.clear()
    harness.svc = harness.new_service()
    await harness.svc.start()
    assert not Path(export["path"]).exists()
    assert _pending(harness).empty


async def test_erase_today_with_a_locked_export_deletes_it_on_the_retry(harness, locked_files):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    live = await begin(harness, pk)
    await run_block(live, 2, trials=1)
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    export = await harness.svc.export({"participant_key": pk})
    locked_files.add(export["path"])
    first = await harness.svc.erase({"participant_key": pk, "scope": "today"})
    assert first["status"] == "ok", first
    assert first["erased"]["rows"] > 0 and first["erased"]["export_files_left"] == 1
    assert _pending(harness).files == (Path(export["path"]).name,)
    locked_files.clear()
    # Today has no rows left; the export holding the erased day is still deleted.
    again = await harness.svc.erase({"participant_key": pk, "scope": "today"})
    assert again["status"] == "ok" and again["erased"]["rows"] == 0
    assert not Path(export["path"]).exists()
    assert _pending(harness).empty


async def test_removing_research_export_with_a_locked_export_deletes_it_on_the_retry(harness, locked_files):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    export = await harness.svc.export({"participant_key": pk})
    locked_files.add(export["path"])
    reply = await harness.svc.grant_consent(grant_params(("selection_study",), participant_key=pk))
    assert reply["status"] == "ok", reply
    assert reply["removed"]["purposes"] == ["research_export"]
    assert reply["removed"]["export_files"] == 0 and reply["removed"]["export_files_left"] == 1
    locked_files.clear()
    await harness.svc.status({})
    assert not Path(export["path"]).exists()


async def test_expiry_with_a_locked_export_finishes_the_rest_and_retries_the_file(harness, locked_files):
    pk = await new_participant(harness, ("selection_study", "research_export"), retention_days=1)
    await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    export = await harness.svc.export({"participant_key": pk})
    locked_files.add(export["path"])
    harness.clock.advance(days=1, seconds=1)
    status = await harness.svc.status({})
    assert status["participants"] == [] and status["erasure_pending"] is True
    assert pk not in harness.keys
    assert pk not in harness.preferences_path.read_text(encoding="utf-8")
    assert Path(export["path"]).exists() and _pending(harness).participants == (pk,)
    locked_files.clear()
    assert (await harness.svc.status({}))["erasure_pending"] is False
    assert not Path(export["path"]).exists() and _pending(harness).empty


async def test_unreadable_rules_file_at_revoke_is_retried(harness):
    pk = await new_participant(harness)
    await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    good = harness.preferences_path.read_text(encoding="utf-8")
    harness.preferences_path.write_text("{not json", encoding="utf-8")
    reply = await harness.svc.revoke_consent({"participant_key": pk})
    assert reply["status"] == "ok" and reply["erased"]["pending"] is True
    assert _pending(harness).rules == (pk,)
    harness.preferences_path.write_text(good, encoding="utf-8")
    await harness.svc.status({})
    assert pk not in harness.preferences_path.read_text(encoding="utf-8")
    assert _pending(harness).empty


async def test_a_tampered_pending_list_deletes_only_export_names(harness, tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x", encoding="utf-8")
    harness.svc.store.pending_path.write_text(
        json.dumps({"schema_version": 1, "participants": ["not-a-key"], "rules": [], "files": [str(victim), "../x"]}),
        encoding="utf-8",
    )
    assert _pending(harness).empty
    await harness.svc.status({})
    assert victim.exists()


# ── a keyring key that could not be deleted (finding: key_destroyed=false is never retried) ─────────────────────


async def test_key_left_by_a_degraded_revoke_is_deleted_once_the_keyring_returns(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    keep = await new_participant(h)
    pk = await new_participant(h)
    await h.svc.close()
    h.keys.set_available(False)
    h.svc = h.new_service()
    await h.svc.start()
    try:
        reply = await h.svc.revoke_consent({"participant_key": pk})
        assert reply["status"] == "ok"
        assert reply["erased"]["key_destroyed"] is False and reply["erased"]["pending"] is True
        assert pk in h.keys
        await h.svc.status({})  # still locked: the key stays listed
        assert pk in h.keys and _pending(h).participants == (pk,)
        h.keys.set_available(True)
        await h.svc.status({})
        assert pk not in h.keys and keep in h.keys
        assert _pending(h).empty
    finally:
        await h.svc.close()


async def test_key_left_by_an_expiry_while_locked_is_deleted_at_a_later_start(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    pk = await new_participant(h, retention_days=1)
    await h.svc.close()
    h.keys.set_available(False)
    h.clock.advance(days=2)
    h.svc = h.new_service()
    await h.svc.start()  # prunes the expired participant without the keyring
    await h.svc.close()
    assert pk in h.keys and not h.db_path.exists()
    h.keys.set_available(True)
    h.svc = h.new_service()
    await h.svc.start()
    try:
        assert pk not in h.keys and _pending(h).empty
    finally:
        await h.svc.close()


# ── erase-today in degraded mode (finding: the snapshot learned from erased rows stays) ─────────────────────────


async def test_degraded_erase_today_deletes_the_snapshot_with_the_rows(tmp_path):
    h = make_harness(tmp_path, learn_across_sessions=True)
    await h.svc.start()
    pk = await new_participant(h, ("selection_study", "cross_session_learning"))
    await run_session(h, pk, blocks=1, trials=2)
    h.clock.advance(days=1)
    await run_session(h, pk, blocks=1, trials=2)
    await h.svc.close()
    assert await _model_state_rows(h, pk) == 1
    h.keys.set_available(False)
    h.svc = h.new_service()
    await h.svc.start()
    try:
        reply = await h.svc.erase({"participant_key": pk, "scope": "today"})
        assert reply["status"] == "ok" and reply["erased"]["rows"] > 0
        assert await sql(h, "SELECT COUNT(*) FROM model_state WHERE participant_key = ?", (pk,)) == [(0,)]
    finally:
        await h.svc.close()


async def _model_state_rows(h, participant_key: str) -> int:
    store = store_module.CoAdaptationStore(h.db_path, h.exports_dir, h.keys)
    await store.open(create=False)
    try:
        rows = await store._fetchall("SELECT COUNT(*) FROM model_state WHERE participant_key = ?", (participant_key,))
        return int(rows[0][0])
    finally:
        await store.close()


# ── a cancelled or partly failed record_trial (finding: learned twice or session killed) ────────────────────────


async def test_a_cancelled_record_trial_finishes_once_and_a_resend_is_not_learned_again(harness, monkeypatch):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    trial = build_trial(
        live, 2, "pointing", rec["decision_id"], rec["variant"], rec["spread"], 0,
        table_ids=[table["decision_id"]] if table else [],
    )  # fmt: skip
    before = _trials_seen(harness, pk)
    store = harness.svc.store
    original = store._key
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_key(participant_key):
        entered.set()
        await release.wait()
        return await original(participant_key)

    monkeypatch.setattr(store, "_key", slow_key)
    call = asyncio.create_task(harness.svc.record_trial({"session_id": live.session_id, "trial": trial}))
    await entered.wait()  # inside put_trial's write transaction
    call.cancel()  # the UI socket closed: the server cancels its request tasks
    with pytest.raises(asyncio.CancelledError):
        await call
    release.set()
    again = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert again["status"] == "recorded", again
    assert _trials_seen(harness, pk) == before + 1
    assert await sql(harness, "SELECT COUNT(*) FROM trials") == [(1,)]
    assert live.session_id in harness.svc._sessions


async def test_a_stored_trial_without_a_cached_reply_is_answered_from_its_row(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    trial = build_trial(
        live, 2, "pointing", rec["decision_id"], rec["variant"], rec["spread"], 0,
        table_ids=[table["decision_id"]] if table else [],
    )  # fmt: skip
    first = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    seen = _trials_seen(harness, pk)
    harness.svc._sessions[live.session_id].trial_replies.clear()
    resend = dict(trial, interrupted=True, interruption_reason="refused")
    again = await harness.svc.record_trial({"session_id": live.session_id, "trial": resend})
    assert again == first
    assert _trials_seen(harness, pk) == seen
    assert live.session_id in harness.svc._sessions
    other = await begin(harness, await new_participant(harness))
    clash = await harness.svc.record_trial({"session_id": other.session_id, "trial": trial})
    assert clash == {"status": "error", "message": "Duplicate trial_id"}


async def test_a_trial_is_never_stored_without_the_relabel_of_its_fallback(harness, monkeypatch):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, _table = await decide_block(live, 2)
    trial = build_trial(
        live, 2, "pointing", rec["decision_id"], live.fallbacks["l1"], live.begin_spread(), 0,
        l1_fallback_reason="late",
    )  # fmt: skip
    original = store_module.DecisionRecord.from_dict

    def failing(payload):
        if payload.get("fallback_reason") == "late":
            raise ValueError("relabel failed")
        return original(payload)

    monkeypatch.setattr(store_module.DecisionRecord, "from_dict", staticmethod(failing))
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["status"] == "error"
    assert await sql(harness, "SELECT COUNT(*) FROM trials") == [(0,)]
    assert await sql(harness, "SELECT behavior FROM decisions WHERE decision_id = ?", (rec["decision_id"],)) == [
        ("policy",)
    ]
    # The engine learned a trial that is not stored, so the session is dropped and rebuilt from rows.
    assert live.session_id not in harness.svc._sessions and pk not in harness.svc._engines


# ── a decision the store refuses (finding: the C5 veto consumed by a decision never delivered) ──────────────────


async def test_a_failed_decision_write_never_leaves_the_decision_applied(tmp_path):
    h = make_harness(tmp_path, adaptive_enabled=True)
    await h.svc.start()
    try:
        pk = await new_participant(h, ("selection_study", "adaptive_assistance"))
        await run_session(h, pk, blocks=2, trials=2)
        h.clock.advance(days=1)
        live = await begin(h, pk)
        b_blocks = [b["index"] for b in live.blocks if b["phase"] == "B"]
        engine = h.svc._engines[pk]
        engine._session.constraint.record_veto("block_regime", "act")
        original = h.keys.get
        state = {"fail": True}

        def failing(participant_key):
            if state["fail"]:
                state["fail"] = False
                raise CoAdaptationKeyUnavailable("transient")
            return original(participant_key)

        h.keys.get = failing
        task = next(b["task"] for b in live.blocks if b["index"] == b_blocks[0])
        reply = await h.svc.recommend(
            {
                "session_id": live.session_id,
                "decision_id": str(uuid.uuid4()),
                "level": "block_regime",
                "context": block_context(b_blocks[0], task),
            }
        )
        assert reply == {
            "status": "refused",
            "reason": "keyring_unavailable",
            "message": service_module.KEYRING_MESSAGE,
        }
        # The engine that consumed the veto for an undelivered decision is gone; nothing keeps that state.
        assert live.session_id not in h.svc._sessions and pk not in h.svc._engines
        assert (await h.svc.resume_session({"session_id": live.session_id}))["reason"] == "session_closed"
        assert await sql(h, "SELECT end_reason FROM sessions WHERE session_id = ?", (live.session_id,)) == [
            ("interrupted",)
        ]
    finally:
        await h.svc.close()


async def test_a_decision_id_stored_by_another_session_is_refused_before_the_engine_decides(harness):
    pk = await new_participant(harness)
    first, runs = await run_session(harness, pk, blocks=1, trials=1)
    used = runs[0].rec["decision_id"]
    harness.clock.advance(minutes=5)
    live = await begin(harness, pk)
    engine = harness.svc._engines[pk]
    engine._session.constraint.record_veto("block_regime", "act")
    reply = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": used,
            "level": "block_regime",
            "context": block_context(2, "pointing"),
        }
    )
    assert reply == {"status": "error", "message": "Duplicate decision_id"}
    assert used not in engine._session.decisions
    assert engine._session.constraint.vetoed_at("block_regime") == frozenset({"act"})
    assert live.session_id in harness.svc._sessions


# ── proposals and session 1 without decrypting the whole history (finding: full decrypt on the event loop) ─────


async def test_proposals_read_only_the_window_once_and_off_the_event_loop(harness, monkeypatch):
    harness.settings.learn_across_sessions = True
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    await run_session(harness, pk, blocks=2, trials=2)
    harness.clock.advance(days=8)
    live = await begin(harness, pk)
    store = harness.svc.store
    calls: list[datetime] = []
    threads: set[str] = set()
    decoded = {"n": 0}
    original_since = store.trials_since
    original_from_dict = store_module.StoredTrial.from_dict

    async def results_rows(*_args, **_kwargs):
        raise AssertionError("proposals must not read the whole history")

    async def trials_since(participant_key, since):
        calls.append(since)
        return await original_since(participant_key, since)

    def from_dict(payload):
        threads.add(threading.current_thread().name)
        decoded["n"] += 1
        return original_from_dict(payload)

    monkeypatch.setattr(store, "results_rows", results_rows)
    monkeypatch.setattr(store, "trials_since", trials_since)
    monkeypatch.setattr(store_module.StoredTrial, "from_dict", staticmethod(from_dict))
    variant = await _veto_trials(live, 3)
    status = await harness.svc.status({})
    assert any(p["variant"] == variant for p in status["proposals"])
    await harness.svc.status({})
    # Read once, for the 7-day window only: the 8-day-old session is never decrypted, later vetoes are appended to
    # the cache and later status calls use it.
    assert len(calls) == 1
    assert decoded["n"] <= 3
    assert threads and threading.main_thread().name not in threads


async def test_session_one_is_read_alone_at_a_later_session_begin(tmp_path, monkeypatch):
    h = make_harness(tmp_path, adaptive_enabled=True)
    await h.svc.start()
    try:
        pk = await new_participant(h, ("selection_study", "adaptive_assistance"))
        first, _runs = await run_session(h, pk, blocks=1, trials=2)
        h.clock.advance(days=1)
        await run_session(h, pk, blocks=1, trials=2)
        h.clock.advance(days=1)
        reads: list[tuple[tuple[str, ...], bool]] = []
        original = h.svc.store._rows_of_sessions

        async def spy(participant_key, key, session_ids, *, every_session, with_decisions=True):
            reads.append((tuple(session_ids), every_session))
            return await original(
                participant_key, key, session_ids, every_session=every_session, with_decisions=with_decisions
            )

        monkeypatch.setattr(h.svc.store, "_rows_of_sessions", spy)
        rows = await h.svc.store.session_rows(pk, 1)
        assert rows is not None and rows.session.session_id == first.session_id
        assert reads == [((first.session_id,), False)]
    finally:
        await h.svc.close()


# ── a session the UI lost (finding: session_begin blocked until restart) ────────────────────────────────────────


async def test_a_lost_session_is_ended_as_abandoned_once_it_is_quiet(harness):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    lost = await begin(harness, pk)
    await decide_block(lost, 2)
    # The window reloaded: the UI no longer knows lost.session_id. A begin right away is still refused.
    assert (await harness.svc.begin_session(begin_params(pk)))["reason"] == "session_active"
    refused = await harness.svc.grant_consent(grant_params(("selection_study",), participant_key=pk))
    assert refused["reason"] == "session_active"
    harness.clock.advance(seconds=service_module.STALE_SESSION_AFTER.total_seconds())
    fresh = await harness.svc.begin_session(begin_params(pk))
    assert fresh["status"] == "ok", fresh
    assert await sql(harness, "SELECT end_reason FROM sessions WHERE session_id = ?", (lost.session_id,)) == [
        ("abandoned",)
    ]
    assert (await harness.svc.resume_session({"session_id": lost.session_id}))["reason"] == "session_closed"
    harness.clock.advance(seconds=service_module.STALE_SESSION_AFTER.total_seconds())
    updated = await harness.svc.grant_consent(grant_params(("selection_study",), participant_key=pk))
    assert updated["status"] == "ok" and updated["removed"]["purposes"] == ["research_export"]


async def test_a_session_in_use_is_never_taken_over(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    for _ in range(3):
        harness.clock.advance(seconds=service_module.STALE_SESSION_AFTER.total_seconds() / 2)
        assert (await harness.svc.resume_session({"session_id": live.session_id}))["status"] == "ok"
        assert (await harness.svc.begin_session(begin_params(pk)))["reason"] == "session_active"
    assert live.session_id in harness.svc._sessions


# ── notifications never hold the lock (finding: a stuck client stalls every study call) ─────────────────────────


async def test_a_stuck_notification_never_holds_the_service_lock(tmp_path):
    h = make_harness(tmp_path)
    stuck = asyncio.Event()
    sent: list[dict] = []

    async def broadcast(method: str, params: dict) -> None:
        sent.append(params)
        await stuck.wait()  # a UI client that stopped reading its socket

    h.svc = h.new_service()
    h.svc._broadcast_fn = broadcast
    await h.svc.start()
    try:
        grant = asyncio.create_task(h.svc.grant_consent(grant_params(ALL_PURPOSES)))
        while not sent:
            await asyncio.sleep(0)
        status = await asyncio.wait_for(h.svc.status({}), timeout=5)
        assert status["status"] == "ok" and len(status["participants"]) == 1
        stuck.set()
        assert (await grant)["status"] == "ok"
        assert [params["reason"] for params in sent] == ["consent"]
    finally:
        stuck.set()
        await h.svc.close()

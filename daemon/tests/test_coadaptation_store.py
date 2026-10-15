"""Encrypted co-adaptation store (spec 10.6-10.9, tests of 22.2). tmp_path and in-memory keys only."""

from __future__ import annotations

import base64
import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import pilot.coadaptation.store as store_module
from pilot.coadaptation.consent import CoAdaptationConsentError
from pilot.coadaptation.core.consent import ConsentPeriod
from pilot.coadaptation.core.engine import EngineEvent, TrialObservation
from pilot.coadaptation.core.preferences import Preference, PreferenceKind, PreferenceScope, preference_to_dict
from pilot.coadaptation.core.records import BurdenReport, DecisionRecord, SessionRow, StoredTrial
from pilot.coadaptation.keys import InMemoryKeyProvider
from pilot.coadaptation.store import CoAdaptationStore, _decrypt_json
from tests.test_coadaptation_service import (
    ALL_PURPOSES,
    START,
    begin,
    decide_block,
    grant_params,
    inject_alarm,
    make_harness,
    new_participant,
    pref_params,
    run_block,
    run_session,
    sql,
    stored_session,
)

LABEL = "Zelda-Marker-Label"


class _SpyKeys(InMemoryKeyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.gets = 0
        self.creates = 0

    def get(self, participant_key: str) -> bytes:
        self.gets += 1
        return super().get(participant_key)

    def get_or_create(self, participant_key: str) -> bytes:
        self.creates += 1
        return super().get_or_create(participant_key)


@pytest.fixture
async def harness(tmp_path):
    h = make_harness(tmp_path, gaze_prior_enabled=True, hand_pointer_enabled=True, learn_across_sessions=True)
    h.keys = _SpyKeys()
    h.svc = h.new_service()
    await h.svc.start()
    yield h
    await h.svc.close()


async def _populate(h, monkeypatch, *, label: str | None = LABEL):
    """One participant with a closed gaze session holding a drift event, a burden report and an export."""
    pk = await new_participant(h, ALL_PURPOSES, label=label)
    live = await begin(h, pk, gaze_check=0.9)
    inject_alarm(monkeypatch, "drift", head="l1", outcome="ua")
    run = await run_block(
        live,
        2,
        trials=3,
        gaze=("center", 0.0),
        gaze_summary={"agreed_with_intended": True},
    )
    await h.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    return pk, live, run


def _decrypt_export(path: Path, key: bytes) -> dict:
    header_line, body = path.read_bytes().split(b"\n")[:2]
    header = json.loads(header_line)
    assert header["cipher"] == "AES-256-GCM" and header["schema_version"] == 1
    plaintext = AESGCM(key).decrypt(base64.b64decode(header["nonce"]), base64.b64decode(body), header_line)
    return json.loads(plaintext)


def _pref(pref_id: str, created: datetime, participant_key: str | None = None) -> Preference:
    return Preference(
        pref_id=pref_id,
        kind=PreferenceKind.FORBID_VARIANT,
        scope=PreferenceScope(context="study", participant_key=participant_key),
        value="ask_clarification:3",
        origin="explicit_user",
        created_at=created,
    )


async def test_no_database_before_grant(tmp_path):
    store = CoAdaptationStore(tmp_path / "co.db", tmp_path / "exports", InMemoryKeyProvider())
    await store.open(create=False)
    assert not store.exists and not store.is_open
    assert await store.consents() == [] and await store.open_sessions() == []
    assert (await store.counts())["exists"] is False
    with pytest.raises(CoAdaptationConsentError):
        await store.put_rule("p-00000000", _pref("r1", START))
    assert not (tmp_path / "co.db").exists()


async def test_every_write_without_consent_raises_and_writes_nothing(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    store = harness.svc.store
    rows = await store.results_rows(pk, day_index=None)
    session = rows.sessions[0]
    decision = rows.decisions[0]
    stored = rows.trials[0]
    report = rows.burden[0]
    counts_before = {
        table: (await sql(harness, f"SELECT COUNT(*) FROM {table}"))[0][0]
        for table in ("sessions", "decisions", "trials", "burden", "events", "rules", "model_state", "meta")
    }
    payloads_before = await sql(harness, "SELECT decision_id, behavior, superseded_by FROM decisions")
    harness.clock.advance(days=91)
    gets = harness.keys.gets
    new_session = dataclasses.replace(session, session_id="11111111-2222-4333-8444-555555555555", ended_at=None)
    new_trial = dataclasses.replace(stored, trial=dataclasses.replace(stored.trial, trial_id="t-new"))
    writes = [
        store.put_session(new_session),
        store.close_session(session.session_id, harness.clock.now, "stopped"),
        store.put_rule(pk, _pref("r9", harness.clock.now, pk)),
        store.put_decision(dataclasses.replace(decision, decision_id="d-new"), participant_key=pk),
        store.supersede_decision(decision.decision_id, "d-other"),
        store.relabel_decision(decision.decision_id, applied=None, fallback_reason="late"),
        store.put_trial(
            TrialObservation(stored=new_trial, learned=True, refresh_required=False, events=()), participant_key=pk
        ),
        store.put_burden(dataclasses.replace(report, report_id="b-new"), participant_key=pk),
        store.put_event(pk, "learning_reset", {}, harness.clock.now),
        store.save_snapshot(pk, b"state", "hash", harness.clock.now),
        store.mark_learning_reset(pk, harness.clock.now),
        store.set_participant_label(pk, "Other"),
    ]
    for write in writes:
        with pytest.raises(CoAdaptationConsentError) as caught:
            await write
        assert caught.value.code == "consent_expired"
    assert harness.keys.gets == gets
    counts_after = {
        table: (await sql(harness, f"SELECT COUNT(*) FROM {table}"))[0][0]
        for table in ("sessions", "decisions", "trials", "burden", "events", "rules", "model_state", "meta")
    }
    assert counts_after == counts_before
    assert await sql(harness, "SELECT decision_id, behavior, superseded_by FROM decisions") == payloads_before


async def test_purpose_gate_on_export(harness):
    pk = await new_participant(harness)
    with pytest.raises(CoAdaptationConsentError) as caught:
        await harness.svc.store.export(pk, harness.clock.now)
    assert caught.value.code == "purpose_missing"
    assert not harness.exports_dir.exists()


async def test_after_erase_every_write_raises_and_no_key_is_created(harness, monkeypatch):
    keep = await new_participant(harness)
    pk, live, run = await _populate(harness, monkeypatch)
    store = harness.svc.store
    rows = await store.results_rows(pk, day_index=None)
    creates = harness.keys.creates
    result = await store.erase_participant(pk)
    assert result.rows > 0 and result.key_destroyed is True
    for write in (
        store.put_session(rows.sessions[0]),
        store.put_rule(pk, _pref("r1", START, pk)),
        store.put_decision(rows.decisions[0], participant_key=pk),
        store.put_trial(
            TrialObservation(stored=rows.trials[0], learned=True, refresh_required=False, events=()),
            participant_key=pk,
        ),
        store.put_burden(rows.burden[0], participant_key=pk),
        store.put_event(pk, "prune", {}, harness.clock.now),
        store.save_snapshot(pk, b"x", "h", harness.clock.now),
        store.mark_learning_reset(pk, harness.clock.now),
        store.set_participant_label(pk, None),
    ):
        with pytest.raises(CoAdaptationConsentError):
            await write
    assert harness.keys.creates == creates
    assert pk not in harness.keys
    assert keep in harness.keys


async def test_rule_history_upserts_and_goes_with_the_participant(harness):
    pk = await new_participant(harness)
    store = harness.svc.store
    first = _pref("r-first", START + timedelta(minutes=1), pk)
    second = _pref("r-second", START + timedelta(minutes=2))
    await store.put_rule(pk, second)
    await store.put_rule(pk, first)
    retracted = dataclasses.replace(first, retracted_at=START + timedelta(minutes=5))
    await store.put_rule(pk, retracted)
    rules = await store.rules(pk)
    assert [rule.pref_id for rule in rules] == ["r-first", "r-second"]
    assert rules[0].retracted_at == START + timedelta(minutes=5)
    assert await sql(harness, "SELECT COUNT(*) FROM rules") == [(2,)]
    assert [preference_to_dict(rule) for rule in rules] == [preference_to_dict(retracted), preference_to_dict(second)]
    await harness.svc.revoke_consent({"participant_key": pk})
    assert not harness.db_path.exists()


async def test_open_sessions_needs_no_key(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    harness.keys.set_available(False)
    gets = harness.keys.gets
    assert await harness.svc.store.open_sessions() == [(pk, live.session_id)]
    assert harness.keys.gets == gets
    harness.keys.set_available(True)


async def test_put_trial_writes_the_row_and_its_events_in_one_transaction(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    store = harness.svc.store
    rows = await store.results_rows(pk, day_index=None)
    stored = rows.trials[0]
    session = await stored_session(harness, live.session_id)
    reopened = dataclasses.replace(session, session_id="22222222-2222-4333-8444-555555555555", ended_at=None)
    await store.put_session(reopened)
    trial = dataclasses.replace(stored.trial, trial_id="t-atomic", session_id=reopened.session_id)
    event = EngineEvent.create(
        kind="calibration_suspended",
        session_id=reopened.session_id,
        trial_id="t-atomic",
        created_at=trial.recorded_at,
    )
    observation = TrialObservation(
        stored=dataclasses.replace(stored, trial=trial), learned=True, refresh_required=True, events=(event,)
    )
    original = store_module._encrypt

    def failing(key, aad, plaintext):
        if aad.startswith("events:"):
            raise OSError("disk full")
        return original(key, aad, plaintext)

    monkeypatch.setattr(store_module, "_encrypt", failing)
    with pytest.raises(OSError):
        await store.put_trial(observation, participant_key=pk)
    assert await sql(harness, "SELECT COUNT(*) FROM trials WHERE trial_id = 't-atomic'") == [(0,)]
    assert await sql(harness, "SELECT COUNT(*) FROM events WHERE trial_id = 't-atomic'") == [(0,)]
    monkeypatch.setattr(store_module, "_encrypt", original)
    await store.put_trial(observation, participant_key=pk)
    assert await sql(harness, "SELECT COUNT(*) FROM trials WHERE trial_id = 't-atomic'") == [(1,)]
    assert await sql(harness, "SELECT kind FROM events WHERE trial_id = 't-atomic'") == [("calibration_suspended",)]


async def test_payloads_are_exactly_the_to_dict_outputs(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    key = harness.keys.get(pk)
    store = harness.svc.store
    rows = await store.results_rows(pk, day_index=None)

    async def payload(table: str, id_column: str, row_id: str) -> dict:
        ((blob,),) = await sql(harness, f"SELECT payload_enc FROM {table} WHERE {id_column} = ?", (row_id,))
        return _decrypt_json(key, f"{table}:{row_id}", blob)

    for decision in rows.decisions:
        assert await payload("decisions", "decision_id", decision.decision_id) == decision.to_dict()
        assert DecisionRecord.from_dict(decision.to_dict()) == decision
    for stored in rows.trials:
        assert await payload("trials", "trial_id", stored.trial.trial_id) == stored.to_dict()
    for report in rows.burden:
        assert await payload("burden", "report_id", report.report_id) == report.to_dict()
    session_payload = await payload("sessions", "session_id", live.session_id)
    assert session_payload == dataclasses.replace(rows.sessions[0], ended_at=None, end_reason=None).to_dict()
    ((event_id,),) = await sql(harness, "SELECT event_id FROM events WHERE kind = 'drift'")
    event = EngineEvent.from_dict(await payload("events", "event_id", event_id))
    assert event.kind == "drift" and event.head == "l1" and event.outcome == "ua"
    for rule in await store.rules(pk):
        assert preference_to_dict(rule)["pref_id"] == rule.pref_id


async def test_encrypted_columns_hold_no_plaintext_and_the_label_only_in_label_enc(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    export = await harness.svc.export({"participant_key": pk})
    salt = (await stored_session(harness, live.session_id)).seed_salt
    await harness.svc.close()
    raw = b""
    for suffix in ("", "-wal"):
        path = Path(f"{harness.db_path}{suffix}")
        if path.exists():
            raw += path.read_bytes()
    for marker in (LABEL, "intended_index", "agreed_with_intended", "seed_salt", salt, "ask_clarification:3"):
        assert marker.encode() not in raw, marker
    assert list(harness.keys._keys) == [pk]
    assert LABEL not in export["path"]
    assert all(LABEL not in path.name for path in harness.exports_dir.iterdir())
    assert LABEL not in harness.preferences_path.read_text(encoding="utf-8")
    harness.svc = harness.new_service()
    await harness.svc.start()
    for table, columns in (
        ("participants", ("participant_key", "created_at")),
        ("consents", ("consent_id", "record_json", "granted_at", "expires_at")),
        ("sessions", ("session_id", "participant_key", "end_reason", "protocol_id")),
        ("decisions", ("decision_id", "level", "mode", "behavior")),
        ("trials", ("trial_id", "task", "verification")),
        ("events", ("event_id", "kind")),
        ("meta", ("key", "value")),
    ):
        for row in await sql(harness, f"SELECT {', '.join(columns)} FROM {table}"):
            assert all(LABEL not in str(value) for value in row)
    assert (await harness.svc.store.participant_labels())[pk] == LABEL


async def test_aad_binds_each_blob_to_its_row(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    trials = await sql(harness, "SELECT trial_id, payload_enc FROM trials ORDER BY seq")
    (first_id, first_blob), (second_id, second_blob) = trials[0], trials[1]
    async with harness.svc.store._pool.write() as conn:
        await conn.execute("UPDATE trials SET payload_enc = ? WHERE trial_id = ?", (second_blob, first_id))
        await conn.execute("UPDATE trials SET payload_enc = ? WHERE trial_id = ?", (first_blob, second_id))
        await conn.commit()
    with pytest.raises(InvalidTag):
        await harness.svc.store.results_rows(pk, day_index=None)


async def test_secure_delete_is_on(harness):
    await new_participant(harness)
    async with harness.svc.store._pool.write() as conn, conn.execute("PRAGMA secure_delete") as cursor:
        assert (await cursor.fetchone())[0] == 1


async def test_revoke_deletes_rows_exports_and_key_then_the_file(harness, monkeypatch):
    other = await new_participant(harness, ("selection_study", "research_export"))
    await harness.svc.export({"participant_key": other})
    pk, live, run = await _populate(harness, monkeypatch)
    await harness.svc.export({"participant_key": pk})
    reply = await harness.svc.revoke_consent({"participant_key": pk})
    assert reply["erased"]["export_files"] == 1 and reply["erased"]["key_destroyed"] is True
    assert harness.db_path.exists()
    for table in (
        "participants",
        "consents",
        "sessions",
        "decisions",
        "trials",
        "burden",
        "events",
        "rules",
        "model_state",
    ):
        assert await sql(harness, f"SELECT COUNT(*) FROM {table} WHERE participant_key = ?", (pk,)) == [(0,)]
    assert [path.name.split("_")[1] for path in harness.exports_dir.iterdir()] == [other]
    assert pk not in harness.keys
    await harness.svc.revoke_consent({"participant_key": other})
    for suffix in ("", "-wal", "-shm"):
        assert not Path(f"{harness.db_path}{suffix}").exists()
    assert list(harness.exports_dir.iterdir()) == []


async def test_erase_day_deletes_the_day_and_its_exports(harness, monkeypatch):
    other = await new_participant(harness, ("selection_study", "research_export"))
    pk = await new_participant(harness, ALL_PURPOSES)
    yesterday, _runs = await run_session(harness, pk, blocks=1, trials=1)
    harness.clock.advance(minutes=1)
    old_export = await harness.svc.export({"participant_key": pk})
    harness.clock.advance(days=1)
    await harness.svc.export({"participant_key": other})
    live = await begin(harness, pk)
    inject_alarm(monkeypatch)
    await run_block(live, 2, trials=2)
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    harness.clock.advance(minutes=1)
    new_export = await harness.svc.export({"participant_key": pk})
    day_index = live.reply["day_index"]
    expected_rows = 0
    for table in ("sessions", "decisions", "trials", "burden", "events"):
        counted = await sql(harness, f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (live.session_id,))
        expected_rows += counted[0][0]
    assert expected_rows >= 1 + 1 + 2 + 1 + 1
    result = await harness.svc.store.erase_day(pk, day_index)
    assert (result.rows, result.export_files, result.export_files_left) == (expected_rows, 1, 0)
    assert not Path(new_export["path"]).exists()
    assert Path(old_export["path"]).exists()
    assert any(path.name.startswith(f"coadapt_{other}_") for path in harness.exports_dir.iterdir())
    assert await sql(harness, "SELECT session_id FROM sessions") == [(yesterday.session_id,)]
    assert await sql(harness, "SELECT COUNT(*) FROM events WHERE session_id = ?", (live.session_id,)) == [(0,)]
    assert await sql(harness, "SELECT COUNT(*) FROM decisions WHERE session_id = ?", (live.session_id,)) == [(0,)]
    assert await harness.svc.store.erase_day(pk, day_index) == store_module.EraseDayResult(rows=0, export_files=0)


async def test_consent_update_removes_what_each_purpose_stored(harness, monkeypatch):
    harness.settings.hand_pointer_enabled = True
    pk, live, run = await _populate(harness, monkeypatch)
    hand_live = await begin(harness, pk, modality="hand_pointer")
    await run_block(hand_live, 2, trials=2, hand={"quality_mean": 0.8, "quality_min": 0.5, "tracking_losses": 1})
    await harness.svc.end_session({"session_id": hand_live.session_id, "reason": "completed"})
    await harness.svc.export({"participant_key": pk})
    store = harness.svc.store
    before = await store.results_rows(pk, day_index=None)
    reply = await harness.svc.grant_consent(
        grant_params(("selection_study", "adaptive_assistance", "cross_session_learning"), participant_key=pk)
    )
    assert reply["status"] == "ok"
    assert reply["removed"]["purposes"] == ["gaze_prior", "hand_pointer", "research_export"]
    assert reply["removed"]["export_files"] == 1
    assert reply["removed"]["sessions_rewritten"] == 1
    assert reply["removed"]["trials_rewritten"] == 5
    after = await store.results_rows(pk, day_index=None)
    assert all(session.gaze_check_accuracy is None for session in after.sessions)
    for old, new in zip(before.trials, after.trials):
        assert new.trial.gaze is None and new.trial.hand is None
        assert all(activation.gaze is None for activation in new.trial.activations)
        assert new.verification == old.verification and new.labels == old.labels
        assert [a.a_gate for a in new.trial.activations] == [a.a_gate for a in old.trial.activations]
    records = await store.consents()
    assert records[0][2] is not None and records[1][2] is None
    assert not any(harness.exports_dir.iterdir())


@pytest.mark.parametrize("dropped", ["gaze_prior", "hand_pointer"])
async def test_unticking_gaze_or_hand_deletes_the_exports_that_hold_it(harness, monkeypatch, dropped):
    harness.settings.hand_pointer_enabled = True
    pk, live, run = await _populate(harness, monkeypatch)
    hand_live = await begin(harness, pk, modality="hand_pointer")
    await run_block(hand_live, 2, trials=1, hand={"quality_mean": 0.8, "quality_min": 0.5, "tracking_losses": 1})
    await harness.svc.end_session({"session_id": hand_live.session_id, "reason": "completed"})
    export = await harness.svc.export({"participant_key": pk})
    bundle = _decrypt_export(Path(export["path"]), harness.keys.get(pk))
    if dropped == "gaze_prior":
        assert any(trial["trial"]["gaze"] is not None for trial in bundle["trials"])
    else:
        assert any(trial["trial"]["hand"] is not None for trial in bundle["trials"])
    kept = [purpose for purpose in ALL_PURPOSES if purpose != dropped]
    assert "research_export" in kept
    reply = await harness.svc.grant_consent(grant_params(kept, participant_key=pk))
    assert reply["status"] == "ok", reply
    assert reply["removed"]["purposes"] == [dropped]
    # Every bundle written earlier holds what the unticked purpose alone stored, so it goes too.
    assert reply["removed"]["export_files"] == 1 and reply["removed"]["export_files_left"] == 0
    assert not Path(export["path"]).exists()
    assert (await harness.svc.export({"participant_key": pk}))["status"] == "ok"


async def test_consent_update_is_one_transaction(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    store = harness.svc.store
    before_trials = await sql(harness, "SELECT trial_id, payload_enc FROM trials ORDER BY trial_id")
    before_consents = await store.consents()
    original = store_module._encrypt

    def failing(key, aad, plaintext):
        if aad.startswith("trials:"):
            raise OSError("disk full")
        return original(key, aad, plaintext)

    monkeypatch.setattr(store_module, "_encrypt", failing)
    reply = await harness.svc.grant_consent(grant_params(("selection_study",), participant_key=pk))
    assert reply["status"] == "error"
    monkeypatch.setattr(store_module, "_encrypt", original)
    assert await store.consents() == before_consents
    assert await sql(harness, "SELECT trial_id, payload_enc FROM trials ORDER BY trial_id") == before_trials
    assert (await store.results_rows(pk, day_index=None)).sessions[0].gaze_check_accuracy == 0.9


async def test_consent_history_returns_every_period_oldest_first(harness):
    pk = await new_participant(harness)
    harness.clock.advance(days=1)
    await harness.svc.grant_consent(grant_params(("selection_study", "research_export"), participant_key=pk))
    harness.clock.advance(days=1)
    await harness.svc.grant_consent(grant_params(("selection_study", "gaze_prior"), participant_key=pk))
    history = await harness.svc.store.consent_history(pk)
    assert all(isinstance(period, ConsentPeriod) for period in history)
    assert [period.record.granted_at for period in history] == [
        START,
        START + timedelta(days=1),
        START + timedelta(days=2),
    ]
    assert [period.replaced_at for period in history] == [START + timedelta(days=1), START + timedelta(days=2), None]
    assert all(period.record.expires_at == START + timedelta(days=90) for period in history)


async def test_prune_erases_expired_consents(harness):
    short = await new_participant(harness, retention_days=1)
    keep = await new_participant(harness)
    await begin(harness, short)
    erased = await harness.svc.store.prune(harness.clock.now + timedelta(days=1, seconds=1))
    assert erased == [short]
    assert await harness.svc.store.participants() == [keep]
    assert short not in harness.keys


async def test_decision_cap_deletes_the_oldest_closed_sessions_with_their_events(harness, monkeypatch):
    pk = await new_participant(harness)
    first, _runs = await run_session(harness, pk, blocks=1, trials=1)
    live = await begin(harness, pk)
    inject_alarm(monkeypatch)
    await run_block(live, 2, trials=1)
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    third = await begin(harness, pk)
    await decide_block(third, 1)
    store = harness.svc.store
    deleted = await store.apply_decision_cap(pk, keep=1)
    assert deleted == 2
    assert await sql(harness, "SELECT session_id FROM sessions") == [(third.session_id,)]
    assert await sql(harness, "SELECT COUNT(*) FROM events") == [(0,)]
    assert (await store.counts())["pruned_sessions"] == 2
    assert await store.apply_decision_cap(pk, keep=0) == 0


async def test_replay_rows_order(harness, monkeypatch):
    pk = await new_participant(harness)
    first, _runs = await run_session(harness, pk, blocks=2, trials=2)
    harness.clock.advance(hours=2)
    live = await begin(harness, pk)
    inject_alarm(monkeypatch)
    await run_block(live, 2, trials=2)
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    items = [item async for item in harness.svc.store.replay_rows(pk)]
    kinds = [type(item).__name__ for item in items]
    assert kinds[0] == "SessionRow"
    sessions = [item for item in items if isinstance(item, SessionRow)]
    assert [s.session_id for s in sessions] == [first.session_id, live.session_id]
    assert all(s.ended_at is not None for s in sessions)
    event_index = kinds.index("EngineEvent")
    assert isinstance(items[event_index - 1], StoredTrial)
    assert items[event_index - 1].trial.trial_id == items[event_index].trial_id
    current: list = []
    for item in items:
        if isinstance(item, SessionRow):
            seqs = [x.trial.seq if isinstance(x, StoredTrial) else x.seq for x in current]
            assert seqs == sorted(seqs)
            current = []
        elif isinstance(item, StoredTrial | BurdenReport):
            current.append(item)


async def test_replay_skips_rows_before_the_learning_reset(harness):
    pk = await new_participant(harness)
    await run_session(harness, pk, blocks=1, trials=1)
    harness.clock.advance(minutes=1)
    await harness.svc.reset_learning({"participant_key": pk})
    assert [item async for item in harness.svc.store.replay_rows(pk)] == []
    harness.clock.advance(minutes=1)
    live, _runs = await run_session(harness, pk, blocks=1, trials=1)
    sessions = [item for item in [x async for x in harness.svc.store.replay_rows(pk)] if isinstance(item, SessionRow)]
    assert [s.session_id for s in sessions] == [live.session_id]
    assert len((await harness.svc.store.results_rows(pk, day_index=None)).sessions) == 2


async def test_session_rows_reads_one_session_or_none(harness):
    pk = await new_participant(harness)
    first, _runs = await run_session(harness, pk, blocks=2, trials=2)
    rows = await harness.svc.store.session_rows(pk, 1)
    assert rows.session.session_id == first.session_id
    assert len(rows.trials) == 4 and len(rows.burden) == 2 and len(rows.decisions) >= 2
    assert all(isinstance(record, DecisionRecord) for record in rows.decisions)
    assert await harness.svc.store.session_rows(pk, 2) is None
    await harness.svc.erase({"participant_key": pk, "scope": "today"})
    assert await harness.svc.store.session_rows(pk, 1) is None


async def test_export_round_trip(harness, monkeypatch):
    pk, live, run = await _populate(harness, monkeypatch)
    await harness.svc.set_preference(pref_params("forbid_variant", "offer_modality", participant_key=pk))
    harness.clock.advance(minutes=1)
    await harness.svc.reset_learning({"participant_key": pk})
    harness.clock.advance(minutes=1)
    await harness.svc.store.put_event(pk, "prune", {"sessions": 0}, harness.clock.now)
    reply = await harness.svc.export({"participant_key": pk})
    path = Path(reply["path"])
    assert path.name == f"coadapt_{pk}_{harness.clock.now.astimezone(UTC):%Y%m%dT%H%M%SZ}.enc"
    content = path.read_bytes()
    assert len(content) == reply["bytes"]
    bundle = _decrypt_export(path, harness.keys.get(pk))
    assert LABEL not in json.dumps(bundle)
    assert bundle["bundle_schema"] == 1 and bundle["record_schema"] == "coadapt-records-v1"
    assert set(bundle["versions"]) == {"strategy", "feature_schema", "model", "prior", "policy", "sdk"}
    store = harness.svc.store
    rows = await store.results_rows(pk, day_index=None)
    assert [ConsentPeriod.from_dict(d).to_dict() for d in bundle["consents"]] == [
        period.to_dict() for period in await store.consent_history(pk)
    ]
    assert [SessionRow.from_dict(d) for d in bundle["sessions"]] == list(rows.sessions)
    assert [DecisionRecord.from_dict(d) for d in bundle["decisions"]] == list(rows.decisions)
    assert [StoredTrial.from_dict(d) for d in bundle["trials"]] == list(rows.trials)
    assert [BurdenReport.from_dict(d) for d in bundle["burden"]] == list(rows.burden)
    events = [EngineEvent.from_dict(d) for d in bundle["events"]]
    assert [event.kind for event in events] == ["drift"]
    stored_kinds = sorted(kind for (kind,) in await sql(harness, "SELECT kind FROM events"))
    assert stored_kinds == ["drift", "learning_reset", "prune"]
    assert bundle["preferences"] == [preference_to_dict(rule) for rule in await store.rules(pk)]
    assert any(rule["value"] == "offer_modality" for rule in bundle["preferences"])
    marker = await store.learning_reset_at(pk)
    assert marker is not None
    assert datetime.fromisoformat(bundle["learning_reset_at"]) == marker
    # Floats are written unrounded: every propensity and draw reads back bit-identical.
    for exported, stored in zip(bundle["decisions"], rows.decisions):
        assert exported["payload"] == stored.payload

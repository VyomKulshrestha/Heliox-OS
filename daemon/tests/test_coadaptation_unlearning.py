"""Unlearning by recomputation (spec 10.8, tests of 22.2): the stored snapshot always equals rebuild(retained rows)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot.coadaptation.core.engine import CoAdaptationEngine, EngineSettings
from pilot.coadaptation.core.preferences import PreferenceSet
from tests.test_coadaptation_service import (
    ALL_PURPOSES,
    begin,
    build_trial,
    decide_block,
    grant_params,
    inject_alarm,
    make_harness,
    new_participant,
    run_block,
    run_session,
    same_state,
    sql,
)

DAEMON_DIR = Path(__file__).resolve().parents[1]
PRACTICE_ARRAYS = {"model.practice_total", "model.practice_variant", "model.item_eff", "model.item_time"}


@pytest.fixture
async def harness(tmp_path):
    h = make_harness(tmp_path, learn_across_sessions=True, hand_pointer_enabled=True, gaze_prior_enabled=True)
    await h.svc.start()
    yield h
    await h.svc.close()


async def _stored_snapshot(h, participant_key: str) -> bytes:
    return (await h.svc.store.load_snapshot(participant_key))[0]


async def _assert_snapshot_is_rebuild(h, participant_key: str) -> bytes:
    stored = await _stored_snapshot(h, participant_key)
    rebuilt = await h.svc.rebuild(participant_key)
    assert same_state(stored, rebuilt.snapshot())
    return stored


def _changed_arrays(a: bytes, b: bytes) -> set[str]:
    left = np.load(io.BytesIO(a), allow_pickle=False)
    right = np.load(io.BytesIO(b), allow_pickle=False)
    changed = set()
    for name in left.files:
        x, y = left[name], right[name]
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            changed.add(name)
    return changed


async def test_snapshot_equals_rebuild_after_prune(harness):
    short = await new_participant(harness, ("selection_study", "cross_session_learning"), retention_days=1)
    keep = await new_participant(harness, ("selection_study", "cross_session_learning"))
    await run_session(harness, short, blocks=2, trials=2)
    await run_session(harness, keep, blocks=2, trials=2)
    before = await _stored_snapshot(harness, keep)
    harness.clock.advance(days=1, seconds=1)
    await harness.svc.status({})
    assert await harness.svc.store.participants() == [keep]
    after = await _assert_snapshot_is_rebuild(harness, keep)
    assert same_state(before, after)


async def test_snapshot_equals_rebuild_after_erase_today(harness):
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    await run_session(harness, pk, blocks=2, trials=3)
    harness.clock.advance(days=1)
    await run_session(harness, pk, blocks=2, trials=3)
    before = await _stored_snapshot(harness, pk)
    reply = await harness.svc.erase({"participant_key": pk, "scope": "today"})
    assert reply["erased"]["rows"] > 0
    after = await _assert_snapshot_is_rebuild(harness, pk)
    assert not same_state(before, after)
    assert await sql(harness, "SELECT COUNT(*) FROM sessions") == [(1,)]


async def test_snapshot_equals_rebuild_after_removing_hand_pointer(harness):
    pk = await new_participant(harness, ALL_PURPOSES)
    live = await begin(harness, pk, modality="hand_pointer")
    await run_block(live, 2, trials=3, hand={"quality_mean": 0.6, "quality_min": 0.2, "tracking_losses": 3})
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    reply = await harness.svc.grant_consent(
        grant_params([p for p in ALL_PURPOSES if p != "hand_pointer"], participant_key=pk)
    )
    assert reply["removed"]["purposes"] == ["hand_pointer"] and reply["removed"]["trials_rewritten"] == 3
    await _assert_snapshot_is_rebuild(harness, pk)


async def test_removing_gaze_prior_leaves_the_snapshot_unchanged(harness):
    pk = await new_participant(harness, ALL_PURPOSES)
    live = await begin(harness, pk, gaze_check=0.9)
    await run_block(live, 2, trials=3, gaze=("left", 0.0), gaze_summary={"agreed_with_intended": False})
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    before = await _stored_snapshot(harness, pk)
    reply = await harness.svc.grant_consent(
        grant_params([p for p in ALL_PURPOSES if p != "gaze_prior"], participant_key=pk)
    )
    assert reply["removed"]["purposes"] == ["gaze_prior"] and reply["removed"]["sessions_rewritten"] == 1
    after = await _assert_snapshot_is_rebuild(harness, pk)
    assert before == after


async def test_learning_reset_marker_is_respected(harness):
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    await run_session(harness, pk, blocks=2, trials=3)
    harness.clock.advance(minutes=1)
    await harness.svc.reset_learning({"participant_key": pk})
    fresh = CoAdaptationEngine(participant_key=pk, settings=EngineSettings(), preferences=PreferenceSet(()))
    assert same_state(await _stored_snapshot(harness, pk), fresh.snapshot())
    harness.clock.advance(minutes=1)
    await run_session(harness, pk, blocks=1, trials=2)
    await _assert_snapshot_is_rebuild(harness, pk)
    status = (await harness.svc.status({}))["participants"][0]["model"]
    assert status["trials_seen"] == 2 and status["sessions_seen"] == 1


async def test_rebuild_with_and_without_the_snapshot_cache_agree(harness):
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    await run_session(harness, pk, blocks=3, trials=3)
    cached = await _stored_snapshot(harness, pk)
    await harness.svc.close()
    harness.svc = harness.new_service()
    await harness.svc.start()
    async with harness.svc.store._pool.write() as conn:
        await conn.execute("DELETE FROM model_state")
        await conn.commit()
    await harness.svc.status({})
    assert same_state(await _stored_snapshot(harness, pk), cached)


async def test_mismatch_rows_are_not_learned_and_fallback_rows_only_advance_practice(harness):
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    live = await begin(harness, pk)
    run = await run_block(live, 2, trials=2)
    engine = harness.svc._engines[pk]
    table_ids = [run.table["decision_id"]] if run.table else []

    before = engine.snapshot()
    tampered = build_trial(
        live, 2, run.task, run.rec["decision_id"], run.variant, run.rec["spread"], 3, table_ids=table_ids
    )
    tampered["activations"][0]["a_gate"] = min(1.0, tampered["activations"][0]["a_gate"] + 0.2)
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": tampered})
    assert reply["verification"] == "posterior_mismatch"
    assert engine.snapshot() == before

    other = "confirm_selection:hold_700" if run.variant != "confirm_selection:hold_700" else "act"
    mismatch = build_trial(
        live,
        2,
        run.task,
        run.rec["decision_id"],
        run.variant,
        run.rec["spread"],
        4,
        table_ids=table_ids,
        applied=other,
    )
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": mismatch})
    assert reply["verification"] == "draw_mismatch"
    assert engine.snapshot() == before

    rec, _table = await decide_block(live, 3)
    fallback_regime = live.fallbacks["l1"]
    fallback = build_trial(
        live, 3, "menu", rec["decision_id"], fallback_regime, live.begin_spread(), 0, l1_fallback_reason="rpc_error"
    )
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": fallback})
    assert reply["verification"] == "fallback"
    after = engine.snapshot()
    assert after != before
    assert _changed_arrays(before, after) <= PRACTICE_ARRAYS

    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    await _assert_snapshot_is_rebuild(harness, pk)


async def test_calibration_suspension_and_drift_survive_a_restart(harness, monkeypatch):
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    table_ids = [table["decision_id"]] if table else []
    inject_alarm(monkeypatch, "drift", head="l1", outcome="complete")
    first = build_trial(live, 2, "pointing", rec["decision_id"], rec["variant"], rec["spread"], 0, table_ids=table_ids)
    assert (await harness.svc.record_trial({"session_id": live.session_id, "trial": first}))["status"] == "recorded"
    inject_alarm(monkeypatch, "calibration_suspended")
    second = build_trial(live, 2, "pointing", rec["decision_id"], rec["variant"], rec["spread"], 1, table_ids=table_ids)
    assert (await harness.svc.record_trial({"session_id": live.session_id, "trial": second}))["status"] == "recorded"
    monkeypatch.undo()
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    stored = await _stored_snapshot(harness, pk)
    await harness.svc.close()
    harness.svc = harness.new_service()
    await harness.svc.start()
    async with harness.svc.store._pool.write() as conn:
        await conn.execute("DELETE FROM model_state")
        await conn.commit()
    model = (await harness.svc.status({}))["participants"][0]["model"]
    assert model["calibration_suspended"] is True
    assert model["drift_flags"] == ["l1:complete"]
    assert same_state(await _stored_snapshot(harness, pk), stored)


_TZ_SCRIPT = r"""
import asyncio, json, sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from pilot.coadaptation.keys import InMemoryKeyProvider
from pilot.coadaptation.service import CoAdaptationService

args = json.loads(sys.argv[1])
settings = SimpleNamespace(**args["settings"])
now = datetime.fromisoformat(args["now"])


async def main():
    service = CoAdaptationService(
        settings_provider=lambda: settings,
        gaze_tracking_enabled=lambda: False,
        db_path=Path(args["db"]),
        preferences_path=Path(args["prefs"]),
        exports_dir=Path(args["exports"]),
        key_provider=InMemoryKeyProvider(keys={args["pk"]: bytes.fromhex(args["key"])}),
        clock=lambda: now,
    )
    await service.start()
    engine = await service.rebuild(args["pk"])
    Path(args["out"]).write_bytes(engine.snapshot())
    await service.close()
    print(datetime.now().astimezone().utcoffset().total_seconds())


asyncio.run(main())
"""


async def test_rebuild_does_not_depend_on_the_process_time_zone(harness, tmp_path):
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    await run_session(harness, pk, blocks=2, trials=3)
    harness.clock.advance(days=1, hours=7)
    await run_session(harness, pk, blocks=2, trials=3)
    expected = await _stored_snapshot(harness, pk)
    await harness.svc.close()
    args = {
        "settings": {
            name: getattr(harness.settings, name)
            for name in harness.settings.__dataclass_fields__  # type: ignore[attr-defined]
        },
        "now": harness.clock.now.isoformat(),
        "db": str(harness.db_path),
        "prefs": str(harness.preferences_path),
        "exports": str(harness.exports_dir),
        "pk": pk,
        "key": harness.keys.get(pk).hex(),
    }
    offsets = {}
    for tz in ("UTC0", "EST5EDT", "JST-9"):
        out = tmp_path / f"snapshot-{tz}.npz"
        env = {**os.environ, "TZ": tz}
        result = subprocess.run(
            [sys.executable, "-c", _TZ_SCRIPT, json.dumps({**args, "out": str(out)})],
            cwd=DAEMON_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        offsets[tz] = float(result.stdout.strip().splitlines()[-1])
        assert same_state(out.read_bytes(), expected), tz
    assert offsets["UTC0"] == 0.0 and offsets["JST-9"] == 9 * 3600.0
    harness.svc = harness.new_service()
    await harness.svc.start()

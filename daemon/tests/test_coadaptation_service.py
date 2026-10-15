"""CoAdaptationService (spec 13.3, 14.2, 14.3, tests of 22.2).

Every test runs on tmp_path with InMemoryKeyProvider and a fake clock; nothing touches DATA_DIR or the OS keyring.
The helpers at the top build synthetic but verifiable study traffic: simple layouts, and activations whose posteriors
are computed with ``pilot.coadaptation.core.target_posterior`` so the engine's verification returns ``ok``. The store
and unlearning tests import them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pilot.coadaptation.core.constraints import fixed_l2_rule
from pilot.coadaptation.core.engine import CoAdaptationEngine, EngineEvent, NoAllowedRegime, TrialObservation
from pilot.coadaptation.core.policy import select_from_cell
from pilot.coadaptation.core.records import SessionRow
from pilot.coadaptation.core.state import Rect, day_index_for
from pilot.coadaptation.core.study_plan import STUDY_PROTOCOL_ID, rank_fixed_regimes, study_protocol_sha256
from pilot.coadaptation.core.target_posterior import GazeInput, cell_id, compute_posterior, keyboard_endpoint
from pilot.coadaptation.keys import InMemoryKeyProvider
from pilot.coadaptation.schemas import FallbacksV1
from pilot.coadaptation.service import CoAdaptationService
from pilot.coadaptation.store import _decrypt_json

CONSENT_SHA = "c" * 64
START = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
SURFACE = (700.0, 500.0)
POINT_CENTRES = [(230.0 + 80.0 * col, 150.0 + 80.0 * row) for row in range(3) for col in range(4)]
POINT_SIZE = (44.0, 44.0)
POINT_ORIGIN = (350.0, 470.0)
MENU_CENTRES = [(350.0, 80.0 + 48.0 * i) for i in range(6)]
MENU_SIZE = (220.0, 24.0)
MENU_ORIGIN = (350.0, 40.0)
ACT_GUARDED = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})
RESULT_OF = {
    "assistance_off": ("committed", "direct"),
    "act": ("committed", "direct"),
    "enlarge_target:1.5": ("committed", "direct"),
    "enlarge_target:2.0": ("committed", "direct"),
    "confirm_selection:second_click": ("confirming", "confirm"),
    "confirm_selection:hold_400": ("held", "hold"),
    "confirm_selection:hold_700": ("held", "hold"),
}
ALL_PURPOSES = (
    "selection_study",
    "adaptive_assistance",
    "cross_session_learning",
    "gaze_prior",
    "hand_pointer",
    "research_export",
)


# ── harness ──────────────────────────────────────────────────────────────────


@dataclass
class SettingsView:
    """Satisfies CoAdaptationSettingsView structurally."""

    enabled: bool = True
    adaptive_enabled: bool = False
    learn_across_sessions: bool = False
    gaze_prior_enabled: bool = False
    hand_pointer_enabled: bool = False
    ambiguity_threshold: float = 0.25
    unintended_activation_tolerance: float = 0.05
    noninferiority_margin: float = 0.01
    upper_bound_credibility: float = 0.95
    exploration_floor: float = 0.1
    switching_penalty: float = 0.1
    burden_prompt_every_blocks: int = 1
    day_boundary_hour: int = 4


class FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class Broadcasts(list):
    async def __call__(self, method: str, params: dict) -> None:
        self.append((method, dict(params)))

    def reasons(self) -> list[str]:
        return [params["reason"] for _method, params in self]

    def of(self, reason: str) -> list[dict]:
        return [params for _method, params in self if params["reason"] == reason]


@dataclass
class Harness:
    tmp: Path
    settings: SettingsView
    clock: FakeClock
    keys: InMemoryKeyProvider
    broadcasts: Broadcasts
    gaze_tracking: list[bool]
    svc: CoAdaptationService = field(init=False)

    def __post_init__(self) -> None:
        self.svc = self.new_service()

    @property
    def db_path(self) -> Path:
        return self.tmp / "co_adaptation.db"

    @property
    def preferences_path(self) -> Path:
        return self.tmp / "co_adaptation_preferences.json"

    @property
    def exports_dir(self) -> Path:
        return self.tmp / "co_adaptation_exports"

    def new_service(self) -> CoAdaptationService:
        return CoAdaptationService(
            settings_provider=lambda: self.settings,
            gaze_tracking_enabled=lambda: self.gaze_tracking[0],
            db_path=self.db_path,
            preferences_path=self.preferences_path,
            exports_dir=self.exports_dir,
            key_provider=self.keys,
            broadcast_fn=self.broadcasts,
            clock=self.clock,
        )

    async def restart(self) -> CoAdaptationService:
        await self.svc.close()
        self.svc = self.new_service()
        await self.svc.start()
        return self.svc


def make_harness(tmp_path: Path, **settings: Any) -> Harness:
    return Harness(
        tmp=tmp_path,
        settings=SettingsView(**settings),
        clock=FakeClock(),
        keys=InMemoryKeyProvider(),
        broadcasts=Broadcasts(),
        gaze_tracking=[True],
    )


@pytest.fixture
async def harness(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    yield h
    await h.svc.close()


def grant_params(
    purposes: Iterable[str] = ("selection_study",),
    *,
    participant_key: str | None = None,
    label: str | None = None,
    retention_days: int = 90,
    consent_sha: str = CONSENT_SHA,
) -> dict:
    return {
        "participant_key": participant_key,
        "label": label,
        "purposes": list(purposes),
        "retention_days": retention_days,
        "consent_text_version": "coadapt-consent-v1",
        "consent_text_sha256": consent_sha,
        "protocol_id": STUDY_PROTOCOL_ID,
        "protocol_sha256": study_protocol_sha256(),
        "authorized": True,
    }


async def new_participant(h: Harness, purposes: Iterable[str] = ("selection_study",), **kwargs: Any) -> str:
    reply = await h.svc.grant_consent(grant_params(purposes, **kwargs))
    assert reply["status"] == "ok", reply
    return reply["participant_key"]


def device(modality: str = "mouse", *, gaze: bool = False) -> dict:
    return {
        "modality": modality,
        "pointer_types": [] if modality in ("keyboard", "hand_pointer") else ["mouse"],
        "dpr": 1.0,
        "surface_w": SURFACE[0],
        "surface_h": SURFACE[1],
        "gaze": gaze,
        "hand_pointer": modality == "hand_pointer",
    }


def begin_params(
    participant_key: str,
    *,
    modality: str = "mouse",
    gaze_check: float | None = None,
    consent_sha: str = CONSENT_SHA,
    protocol_id: str = STUDY_PROTOCOL_ID,
) -> dict:
    return {
        "participant_key": participant_key,
        "context": "study",
        "consent_text_sha256": consent_sha,
        "device": device(modality, gaze=gaze_check is not None),
        "protocol_id": protocol_id,
        "gaze_check": None if gaze_check is None else {"accuracy": gaze_check},
    }


@dataclass
class Live:
    """One open study session as the UI would hold it."""

    h: Harness
    participant_key: str
    reply: dict
    modality: str
    gaze_accuracy: float | None
    trial_counter: int = 0

    @property
    def session_id(self) -> str:
        return self.reply["session_id"]

    @property
    def fallbacks(self) -> dict:
        return self.reply["fallbacks"]

    @property
    def tau(self) -> float:
        return self.reply["thresholds"]["ambiguity"]

    @property
    def blocks(self) -> list[dict]:
        return self.reply["plan"]["blocks"]

    def begin_spread(self) -> dict | None:
        if self.modality == "keyboard":
            return None
        return {"r": self.reply["spread"][self.modality], "floor": self.reply["floors"][self.modality]}


async def begin(h: Harness, participant_key: str, **kwargs: Any) -> Live:
    reply = await h.svc.begin_session(begin_params(participant_key, **kwargs))
    assert reply["status"] == "ok", reply
    return Live(
        h=h,
        participant_key=participant_key,
        reply=reply,
        modality=kwargs.get("modality", "mouse"),
        gaze_accuracy=kwargs.get("gaze_check"),
    )


def centres_of(task: str) -> list[tuple[float, float]]:
    return MENU_CENTRES if task == "menu" else POINT_CENTRES


def size_of(task: str) -> tuple[float, float]:
    return MENU_SIZE if task == "menu" else POINT_SIZE


def candidate_rects(task: str) -> list[dict]:
    w, h = size_of(task)
    return [{"cx": cx, "cy": cy, "w": w, "h": h} for cx, cy in centres_of(task)]


def layout_summary(task: str) -> dict:
    w, h = size_of(task)
    origin = MENU_ORIGIN if task == "menu" else POINT_ORIGIN
    centres = centres_of(task)
    id_bar = sum(math.log2(math.hypot(cx - origin[0], cy - origin[1]) / w + 1.0) for cx, cy in centres) / len(centres)
    return {
        "task": task,
        "n_candidates": len(centres),
        "w_px": w,
        "h_px": h,
        "gap_px": 24.0 if task == "menu" else 36.0,
        "id_bar": id_bar,
        "menu_n": len(centres) if task == "menu" else 0,
        "origin_x": origin[0],
        "origin_y": origin[1],
    }


def block_context(block_index: int, task: str, **extra: Any) -> dict:
    context = {
        "block_index": block_index,
        "layouts": [layout_summary(task)] * 8,
        "previous_regime": None,
        "session_minute": 1.0,
        "trials_today": 0,
        "hand_quality_recent": None,
    }
    context.update(extra)
    return context


def core_rects(candidates: list[dict]) -> list[Rect]:
    return [Rect(c["cx"], c["cy"], c["w"], c["h"]) for c in candidates]


def pointer_activation(
    live: Live,
    candidates: list[dict],
    spread: dict | None,
    intended: int,
    regime: str,
    *,
    index: int = 0,
    keyboard: bool = False,
    gaze: tuple[str, float] | None = None,
    endpoint: tuple[int, int] | None = None,
) -> dict:
    """One activation on ``intended`` with the posterior fields the engine recomputes."""
    rect = candidates[intended]
    result, _via = RESULT_OF[regime]
    if keyboard or live.modality == "keyboard":
        x, y = keyboard_endpoint(Rect(rect["cx"], rect["cy"], rect["w"], rect["h"]))
        return {
            "index": index,
            "t_ms": 700.0 + index,
            "modality": "keyboard",
            "x": x,
            "y": y,
            "hit_index": intended,
            "press_index": intended,
            "a_gate": 0.0,
            "n_plausible": 1,
            "expanded_index": None,
            "hit_rect": dict(rect),
            "gaze": None,
            "l2": None,
            "irreversible_rule": False,
            "result": result,
            "outcome": "committed",
            "committed_index": intended,
        }
    x, y = endpoint if endpoint is not None else (int(math.floor(rect["cx"] + 0.5)), int(math.floor(rect["cy"] + 0.5)))
    gaze_input = None
    if gaze is not None:
        gaze_input = GazeInput(region=gaze[0], conf=gaze[1], acc=float(live.gaze_accuracy or 0.0))
    posterior = compute_posterior(
        core_rects(candidates),
        (x, y),
        modality=live.modality,
        r=spread["r"],
        floor=spread["floor"],
        surface=SURFACE,
        gaze=gaze_input,
        press_index=intended,
    )
    assert posterior.hit_index == intended
    return {
        "index": index,
        "t_ms": 700.0 + index,
        "modality": live.modality,
        "x": x,
        "y": y,
        "hit_index": intended,
        "press_index": intended,
        "a_gate": posterior.ambiguity_gate,
        "n_plausible": posterior.n_plausible,
        "expanded_index": None,
        "hit_rect": dict(rect),
        "gaze": None if gaze is None else {"region": gaze[0], "conf": gaze[1]},
        "l2": None,
        "irreversible_rule": False,
        "result": result,
        "outcome": "committed",
        "committed_index": intended,
    }


def build_trial(
    live: Live,
    block_index: int,
    task: str,
    regime_decision_id: str,
    regime: str,
    spread: dict | None,
    trial_index: int,
    *,
    intended: int | None = None,
    table_ids: Iterable[str] = (),
    applied: str | None = None,
    l1_fallback_reason: str | None = None,
    activations: list[dict] | None = None,
    candidates: list[dict] | None = None,
    keyboard: bool = False,
    gaze: tuple[str, float] | None = None,
    gaze_summary: dict | None = None,
    hand: dict | None = None,
    veto: dict | None = None,
    revert: dict | None = None,
    offers: Iterable[dict] = (),
    interrupted: bool = False,
    issued_at: str | None = None,
    committed_index: Any = "intended",
    commit_via: str | None = None,
) -> dict:
    candidates = candidates if candidates is not None else candidate_rects(task)
    if intended is None:
        intended = trial_index % len(candidates)
    applied = applied or regime
    if activations is None:
        activations = [pointer_activation(live, candidates, spread, intended, applied, keyboard=keyboard, gaze=gaze)]
    committed = intended if committed_index == "intended" else committed_index
    live.h.clock.advance(seconds=3)
    cue_at = live.h.clock.now - timedelta(seconds=2)
    return {
        "trial_id": str(uuid.uuid4()),
        "block_index": block_index,
        "trial_index": trial_index,
        "task": task,
        "regime": regime,
        "regime_decision_id": regime_decision_id,
        "table_decision_ids": list(table_ids),
        "applied_regime": applied,
        "l1_fallback_reason": l1_fallback_reason,
        "fallbacks_issued_at": issued_at or live.fallbacks["issued_at"],
        "spread": spread,
        "layout": layout_summary(task),
        "candidates": candidates,
        "surface_w": SURFACE[0],
        "surface_h": SURFACE[1],
        "dpr": 1.0,
        "reversible": [True] * len(candidates),
        "cue_at": cue_at.isoformat(),
        "intended_index": intended,
        "modality": live.modality,
        "activations": activations,
        "revert": revert,
        "committed_index": committed,
        "commit_t_ms": 900.0 if committed is not None else None,
        "commit_via": commit_via or (RESULT_OF[applied][1] if committed is not None else "none"),
        "corrections": 0,
        "time_ms": 900.0 + trial_index,
        "limit_ms": 15000 if task == "menu" else 10000,
        "duration_ms": 900.0 + trial_index,
        "movement": {
            "rt_ms": 250.0,
            "mt_ms": 600.0 + 10 * trial_index,
            "path_ratio": 1.2,
            "submovements": 1,
            "peak_speed_px_s": 1500.0,
            "reentries": 0,
            "endpoint_dx_norm": 0.02 * ((trial_index % 3) - 1),
            "endpoint_dy_norm": 0.01 * ((trial_index % 2) - 0.5),
        }
        if task == "pointing" and not keyboard and live.modality != "keyboard"
        else None,
        "veto": veto,
        "offers": list(offers),
        "hand": hand,
        "gaze": gaze_summary,
        "interrupted": interrupted,
        "interruption_reason": "paused" if interrupted else None,
        "explanation_views": 0,
    }


@dataclass
class BlockRun:
    index: int
    task: str
    rec: dict
    table: dict | None
    replies: list[dict]

    @property
    def variant(self) -> str:
        return self.rec["variant"]


async def decide_block(live: Live, block_index: int, *, table: bool = True) -> tuple[dict, dict | None]:
    task = next(block["task"] for block in live.blocks if block["index"] == block_index)
    context = block_context(block_index, task)
    reply = await live.h.svc.recommend(
        {"session_id": live.session_id, "decision_id": str(uuid.uuid4()), "level": "block_regime", "context": context}
    )
    assert reply["status"] == "ok", reply
    rec = reply["recommendation"]
    table_rec = None
    if table and rec["variant"] in ACT_GUARDED:
        table_reply = await live.h.svc.recommend(
            {
                "session_id": live.session_id,
                "decision_id": str(uuid.uuid4()),
                "level": "ambiguity_table",
                "context": {
                    **context,
                    "block_decision_id": rec["decision_id"],
                    "start_trial_index": 0,
                    "trials_in_block": 8,
                },
            }
        )
        assert table_reply["status"] == "ok", table_reply
        table_rec = table_reply["recommendation"]
    return rec, table_rec


async def run_block(
    live: Live, block_index: int, *, trials: int = 4, burden: bool = True, **trial_kwargs: Any
) -> BlockRun:
    task = next(block["task"] for block in live.blocks if block["index"] == block_index)
    rec, table = await decide_block(live, block_index)
    table_ids = [table["decision_id"]] if table else []
    replies = []
    for trial_index in range(trials):
        trial = build_trial(
            live,
            block_index,
            task,
            rec["decision_id"],
            rec["variant"],
            rec["spread"],
            trial_index,
            table_ids=table_ids,
            **trial_kwargs,
        )
        reply = await live.h.svc.record_trial({"session_id": live.session_id, "trial": trial})
        assert reply["status"] == "recorded", reply
        replies.append(reply)
    if burden:
        report = {"block_index": block_index, "scope": "block", "effort": 3, "control": 5}
        if rec["variant"] != "assistance_off":
            report["intrusiveness"] = 2
        assert (await live.h.svc.report_burden({"session_id": live.session_id, "report": report}))["status"] == (
            "recorded"
        )
    return BlockRun(block_index, task, rec, table, replies)


async def run_session(
    h: Harness,
    participant_key: str,
    *,
    blocks: int | None = None,
    trials: int = 4,
    end: bool = True,
    **begin_kwargs: Any,
) -> tuple[Live, list[BlockRun]]:
    live = await begin(h, participant_key, **begin_kwargs)
    runs = []
    main_blocks = [block for block in live.blocks if block["phase"] != "practice"]
    for block in main_blocks[: blocks if blocks is not None else len(main_blocks)]:
        runs.append(await run_block(live, block["index"], trials=trials))
    if end:
        reply = await h.svc.end_session({"session_id": live.session_id, "reason": "completed"})
        assert reply["status"] == "ok", reply
    return live, runs


async def stored_session(h: Harness, session_id: str) -> SessionRow:
    store = h.svc.store
    async with (
        store._read() as conn,
        conn.execute("SELECT participant_key, payload_enc FROM sessions WHERE session_id = ?", (session_id,)) as cursor,
    ):
        participant_key, blob = await cursor.fetchone()
    payload = _decrypt_json(h.keys.get(participant_key), f"sessions:{session_id}", blob)
    return SessionRow.from_dict(payload)


async def sql(h: Harness, query: str, args: Iterable[Any] = ()) -> list[tuple]:
    return await h.svc.store._fetchall(query, args)


def same_state(a: bytes, b: bytes, *, tol: float = 1e-12) -> bool:
    """Engine snapshots hold the same learned arrays (within ``tol``)."""
    left = np.load(io.BytesIO(a), allow_pickle=False)
    right = np.load(io.BytesIO(b), allow_pickle=False)
    if sorted(left.files) != sorted(right.files):
        return False
    for name in left.files:
        x, y = left[name], right[name]
        if x.shape != y.shape:
            return False
        if x.dtype.kind in "fc":
            if not np.allclose(x, y, rtol=0.0, atol=tol, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def assert_fallbacks(fallbacks: dict) -> FallbacksV1:
    return FallbacksV1.parse(fallbacks)


def inject_alarm(
    monkeypatch, kind: str = "calibration_suspended", *, head: str | None = None, outcome: str | None = None
):
    """Make the next observed trial fire a real alarm event through the engine's own apply_event and mark_stale."""
    original = CoAdaptationEngine.observe_trial
    state = {"armed": True}

    def observe(self, trial):
        observation = original(self, trial)
        if not state["armed"]:
            return observation
        state["armed"] = False
        event = EngineEvent.create(
            kind=kind,
            session_id=trial.session_id,
            trial_id=trial.trial_id,
            created_at=trial.recorded_at,
            head=head,
            outcome=outcome,
        )
        self.apply_event(event)
        marked = self.mark_stale(kind)
        return TrialObservation(
            stored=observation.stored,
            learned=observation.learned,
            refresh_required=observation.refresh_required or marked,
            events=(*observation.events, event),
        )

    monkeypatch.setattr(CoAdaptationEngine, "observe_trial", observe)
    return state


def pref_params(kind: str, value: Any, *, participant_key: str | None = None, modality: str | None = None) -> dict:
    return {
        "kind": kind,
        "scope": {"context": "study", "modality": modality, "participant_key": participant_key},
        "value": value,
    }


# ── status, consent ──────────────────────────────────────────────────────────


async def test_status_without_a_store_creates_nothing(harness):
    reply = await harness.svc.status({})
    assert reply["status"] == "ok"
    assert reply["available"] is True
    assert reply["authority"] == "presentation_only" and reply["execution_authority"] is False
    assert reply["participants"] == [] and reply["proposals"] == [] and reply["preferences"] == []
    assert reply["store"]["exists"] is False
    assert reply["protocol"] == {"protocol_id": STUDY_PROTOCOL_ID, "sha256": study_protocol_sha256()}
    assert reply["versions"]["consent_text"] == "coadapt-consent-v1"
    assert set(reply["config"]) == {f.name for f in dataclasses.fields(SettingsView)}
    assert not harness.db_path.exists()
    assert not harness.exports_dir.exists()
    assert len(harness.keys) == 0


@pytest.mark.parametrize(
    ("level", "context", "expected"),
    [
        ("block_regime", block_context(1, "menu"), {"variant": "assistance_off"}),
        (
            "ambiguity_table",
            block_context(1, "menu", block_decision_id=str(uuid.uuid4()), start_trial_index=0, trials_in_block=8),
            {"level": "ambiguity_table", "cells": []},
        ),
        (
            "offer",
            {
                "block_index": 1,
                "trial_index": 2,
                "task": "menu",
                "struggle_kind": "corrections",
                "keyboard_available": True,
            },
            {"variant": "no_offer"},
        ),
    ],
)
async def test_recommend_without_consent_returns_baseline_and_creates_no_db(harness, level, context, expected):
    reply = await harness.svc.recommend(
        {"session_id": str(uuid.uuid4()), "decision_id": str(uuid.uuid4()), "level": level, "context": context}
    )
    assert reply["status"] == "ignored"
    assert reply["logged"] is False
    assert reply["reason"] == "no_consent"
    assert reply["recommendation"] == {**expected, "authority": "presentation_only", "execution_authority": False}
    assert not harness.db_path.exists()


async def test_recommend_when_disabled_returns_baseline(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    harness.settings.enabled = False
    reply = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "block_regime",
            "context": block_context(1, "menu"),
        }
    )
    assert reply["status"] == "ignored" and reply["reason"] == "disabled" and reply["logged"] is False
    assert reply["recommendation"]["variant"] == "assistance_off"


async def test_grant_with_null_creates_a_daemon_key(harness):
    reply = await harness.svc.grant_consent(grant_params(label="Lab A"))
    assert reply["status"] == "ok"
    pk = reply["participant_key"]
    assert len(pk) == 10 and pk.startswith("p-")
    assert pk in harness.keys
    assert harness.db_path.exists()
    assert reply["consent"]["active"] is True
    assert reply["removed"] == {
        "purposes": [],
        "export_files": 0,
        "trials_rewritten": 0,
        "sessions_rewritten": 0,
        "export_files_left": 0,
    }
    assert harness.broadcasts.of("consent") == [{"reason": "consent", "participant_key": pk, "refresh_seq": None}]
    assert harness.svc.has_active_consent() is True
    status = await harness.svc.status({})
    (participant,) = status["participants"]
    assert participant["label"] == "Lab A"
    labels = await sql(harness, "SELECT label_enc FROM participants")
    assert labels[0][0] is not None and b"Lab A" not in bytes(labels[0][0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"authorized": False},
        {"purposes": ["gaze_prior"]},
        {"consent_text_sha256": "not-a-hash"},
        {"retention_days": 0},
        {"retention_days": 366},
        {"participant_key": "Lab A"},
    ],
)
async def test_grant_validation_errors(harness, overrides):
    reply = await harness.svc.grant_consent({**grant_params(), **overrides})
    assert reply["status"] == "error"
    assert not harness.db_path.exists()


async def test_grant_for_an_unknown_key_fails(harness):
    reply = await harness.svc.grant_consent(grant_params(participant_key="p-00000000"))
    assert reply["status"] == "error" and reply["reason"] == "unknown_participant"
    pk = await new_participant(harness)
    await harness.svc.revoke_consent({"participant_key": pk})
    again = await harness.svc.grant_consent(grant_params(participant_key=pk))
    assert again["status"] == "error" and again["reason"] == "unknown_participant"


async def test_consent_update_keeps_the_old_record_and_never_extends_expiry(harness):
    pk = await new_participant(harness)
    harness.clock.advance(days=80)
    same = await harness.svc.grant_consent(grant_params(("selection_study", "research_export"), participant_key=pk))
    assert same["status"] == "ok"
    assert datetime.fromisoformat(same["consent"]["expires_at"]) == START + timedelta(days=90)
    shorter = await harness.svc.grant_consent(
        grant_params(("selection_study", "research_export"), participant_key=pk, retention_days=5)
    )
    assert datetime.fromisoformat(shorter["consent"]["expires_at"]) == START + timedelta(days=85)
    records = await harness.svc.store.consents()
    assert len(records) == 3
    assert [replaced is not None for _record, _revoked, replaced in records] == [True, True, False]
    history = await harness.svc.store.consent_history(pk)
    assert [period.replaced_at is not None for period in history] == [True, True, False]
    assert history[1].record.replaces_consent_id == history[0].record.consent_id


async def test_update_removing_a_purpose_during_an_open_session_is_refused(harness):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    await begin(harness, pk)
    before = await harness.svc.store.consents()
    reply = await harness.svc.grant_consent(grant_params(participant_key=pk))
    assert reply == {
        "status": "refused",
        "reason": "session_active",
        "message": "Stop the running session before removing a permission.",
    }
    assert await harness.svc.store.consents() == before
    added = await harness.svc.grant_consent(grant_params(ALL_PURPOSES, participant_key=pk))
    assert added["status"] == "ok"


async def test_update_reports_what_a_removed_purpose_deleted(harness):
    harness.settings.gaze_prior_enabled = True
    pk = await new_participant(harness, ("selection_study", "gaze_prior", "research_export"))
    live = await begin(harness, pk, gaze_check=0.9)
    assert live.reply["gaze_prior"] is True
    await run_block(live, 2, trials=2, gaze=("center", 0.0), gaze_summary={"agreed_with_intended": True})
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    assert (await harness.svc.export({"participant_key": pk}))["status"] == "ok"
    reply = await harness.svc.grant_consent(grant_params(("selection_study",), participant_key=pk))
    assert reply["removed"] == {
        "purposes": ["gaze_prior", "research_export"],
        "export_files": 1,
        "trials_rewritten": 2,
        "sessions_rewritten": 1,
        "export_files_left": 0,
    }
    assert list(harness.exports_dir.iterdir()) == []
    row = await stored_session(harness, live.session_id)
    assert row.gaze_check_accuracy is None


# ── session begin ────────────────────────────────────────────────────────────


async def test_session_begin_refusals(harness):
    harness.settings.enabled = False
    pk = await new_participant(harness)
    assert (await harness.svc.begin_session(begin_params(pk)))["reason"] == "disabled"
    harness.settings.enabled = True
    assert (await harness.svc.begin_session(begin_params("p-00000000")))["reason"] == "no_consent"
    changed = await harness.svc.begin_session(begin_params(pk, consent_sha="d" * 64))
    assert changed == {"status": "refused", "reason": "consent_text_changed"}
    protocol = await harness.svc.begin_session(begin_params(pk, protocol_id="coadapt-study-v2"))
    assert protocol == {"status": "refused", "reason": "protocol_changed"}
    assert (await harness.svc.begin_session(begin_params(pk, modality="hand_pointer")))["reason"] == (
        "hand_pointer_not_allowed"
    )
    await begin(harness, pk)
    assert (await harness.svc.begin_session(begin_params(pk)))["reason"] == "session_active"


async def test_protocol_change_is_refused(harness, monkeypatch):
    pk = await new_participant(harness)
    import pilot.coadaptation.service as service_module

    monkeypatch.setattr(service_module, "study_protocol_sha256", lambda: "e" * 64)
    assert (await harness.svc.begin_session(begin_params(pk)))["reason"] == "protocol_changed"


async def test_session_begin_reply(harness):
    pk = await new_participant(harness)
    harness.clock.advance(minutes=3)
    started = harness.clock.now
    live = await begin(harness, pk)
    reply = live.reply
    assert set(reply) == {
        "status",
        "session_id",
        "participant_key",
        "day_index",
        "session_number",
        "plan",
        "layout_seed",
        "spread",
        "floors",
        "thresholds",
        "gaze_prior",
        "hand_pointer",
        "burden_prompt_every_blocks",
        "evidence_kind",
        "fallbacks",
        "refresh_seq",
    }
    assert reply["session_number"] == 1
    assert reply["refresh_seq"] == 0
    assert reply["day_index"] == day_index_for(started.astimezone(), 4)
    assert set(reply["spread"]) == {"mouse"} and reply["floors"] == {"mouse": 1.0}
    assert reply["thresholds"] == {"ambiguity": 0.25, "bins": {"b1": [0.25, 0.4], "b2": [0.4, 0.6], "b3": [0.6, 1.0]}}
    fallbacks = assert_fallbacks(reply["fallbacks"])
    assert fallbacks.issued_at == started
    assert [block["phase"] for block in reply["plan"]["blocks"]][:2] == ["practice", "A"]
    row = await stored_session(harness, live.session_id)
    assert row.day_index == reply["day_index"]
    assert row.started_at == started
    assert row.seed_salt not in json.dumps(reply)


async def test_keyboard_session_has_no_spread(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk, modality="keyboard")
    assert live.reply["spread"] == {} and live.reply["floors"] == {}
    assert len([block for block in live.blocks if block["phase"] == "A"]) == 12
    run = await run_block(live, 2, trials=3)
    assert run.rec["spread"] is None
    assert [reply["verification"] for reply in run.replies] == ["ok", "ok", "ok"]
    bad = build_trial(live, 2, "pointing", run.rec["decision_id"], run.variant, {"r": 1.0, "floor": 1.0}, 5)
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": bad})
    assert reply["status"] == "error"


async def test_pointer_session_rejects_a_trial_without_spread(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, _table = await decide_block(live, 2)
    trial = build_trial(live, 2, "pointing", rec["decision_id"], rec["variant"], rec["spread"], 0)
    trial["spread"] = None
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["status"] == "error"


async def test_menu_trial_with_a_keyboard_activation_in_a_mouse_session(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 1)
    trial = build_trial(
        live,
        1,
        "menu",
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        keyboard=True,
        table_ids=[table["decision_id"]] if table else [],
    )
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["verification"] == "ok", reply


@pytest.mark.parametrize(
    ("purposes", "config", "accuracy", "stored", "used"),
    [
        (ALL_PURPOSES, True, 0.8, 0.8, True),
        (ALL_PURPOSES, True, 0.5, 0.5, False),
        (("selection_study",), True, 0.8, None, False),
        (ALL_PURPOSES, False, 0.8, None, False),
    ],
)
async def test_gaze_check_accuracy_is_stored_only_with_the_purpose_effective(
    harness, purposes, config, accuracy, stored, used
):
    harness.settings.gaze_prior_enabled = config
    pk = await new_participant(harness, purposes)
    live = await begin(harness, pk, gaze_check=accuracy)
    assert live.reply["gaze_prior"] is used
    row = await stored_session(harness, live.session_id)
    assert row.gaze_check_accuracy == stored
    assert row.device.gaze_available is used


async def test_gaze_and_hand_fields_are_dropped_when_not_effective(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    trial = build_trial(
        live,
        2,
        "pointing",
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        table_ids=[table["decision_id"]] if table else [],
        gaze_summary={"agreed_with_intended": True},
        hand={"quality_mean": 0.9, "quality_min": 0.4, "tracking_losses": 1},
    )
    trial["activations"][0]["gaze"] = {"region": "center", "conf": 0.0}
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["status"] == "recorded"
    rows = await harness.svc.store.results_rows(pk, day_index=None)
    stored = rows.trials[0].trial
    assert stored.gaze is None and stored.hand is None and stored.activations[0].gaze is None


async def test_hand_pointer_session(harness):
    harness.settings.hand_pointer_enabled = True
    pk = await new_participant(harness, ("selection_study", "hand_pointer"))
    live = await begin(harness, pk, modality="hand_pointer")
    assert live.reply["hand_pointer"] is True
    assert live.reply["floors"] == {"hand_pointer": 6.0}
    run = await run_block(live, 2, trials=2, hand={"quality_mean": 0.9, "quality_min": 0.6, "tracking_losses": 0})
    assert all(reply["verification"] == "ok" for reply in run.replies), run.replies
    rows = await harness.svc.store.results_rows(pk, day_index=None)
    assert rows.trials[0].trial.hand is not None


# ── trials, decisions, sessions ──────────────────────────────────────────────


async def test_record_trial_sets_recorded_at_and_seq_and_is_idempotent(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    trial = build_trial(
        live,
        2,
        "pointing",
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        table_ids=[table["decision_id"]] if table else [],
    )
    recorded_at = harness.clock.now
    first = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert first["status"] == "recorded" and first["verification"] == "ok"
    assert set(first) == {
        "status",
        "verification",
        "labels",
        "preference_violation",
        "refresh_required",
        "refresh_seq",
        "fallbacks",
    }
    assert first["labels"] == {"complete": True, "ua": False, "correction": False, "veto": False, "time_ms": 900.0}
    harness.clock.advance(seconds=30)
    again = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert again == first
    rows = await sql(harness, "SELECT recorded_at, seq FROM trials")
    assert len(rows) == 1
    assert datetime.fromisoformat(rows[0][0]) == recorded_at
    decision_seqs = [row[0] for row in await sql(harness, "SELECT seq FROM decisions ORDER BY seq")]
    assert rows[0][1] == max(decision_seqs) + 1
    assert sorted(decision_seqs) == list(range(1, len(decision_seqs) + 1))


async def test_duplicate_decision_id_is_an_error(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    params = {
        "session_id": live.session_id,
        "decision_id": str(uuid.uuid4()),
        "level": "block_regime",
        "context": block_context(1, "menu"),
    }
    assert (await harness.svc.recommend(params))["status"] == "ok"
    assert await harness.svc.recommend(params) == {"status": "error", "message": "Duplicate decision_id"}


async def test_every_l1_decision_carries_spread_and_priorities(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, _table = await decide_block(live, 1)
    assert rec["authority"] == "presentation_only" and rec["execution_authority"] is False
    assert set(rec["priorities"]) == {"fewer_mistakes", "faster", "fewer_interruptions"}
    assert set(rec["spread"]) == {"r", "floor"}


async def test_a_new_table_supersedes_the_old_one(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    for block in [b["index"] for b in live.blocks if b["phase"] == "A"]:
        rec, table = await decide_block(live, block)
        if table is not None:
            break
    else:
        pytest.fail("no act-guarded block in Phase A")
    second = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "ambiguity_table",
            "context": {
                **block_context(block, next(b["task"] for b in live.blocks if b["index"] == block)),
                "block_decision_id": rec["decision_id"],
                "start_trial_index": 3,
                "trials_in_block": 5,
            },
        }
    )
    assert second["status"] == "ok"
    assert second["recommendation"]["start_trial_index"] == 3
    assert len(second["recommendation"]["draws"]) == 5
    rows = dict(await sql(harness, "SELECT decision_id, superseded_by FROM decisions"))
    assert rows[table["decision_id"]] == second["recommendation"]["decision_id"]


async def test_unknown_block_decision_and_block_index_are_errors(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    unknown = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "ambiguity_table",
            "context": {
                **block_context(2, "pointing"),
                "block_decision_id": str(uuid.uuid4()),
                "start_trial_index": 0,
                "trials_in_block": 8,
            },
        }
    )
    assert unknown["status"] == "error"
    outside = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "block_regime",
            "context": block_context(40, "pointing"),
        }
    )
    assert outside["status"] == "error"


async def test_relabel_only_the_level_that_fell_back(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    fallbacks = FallbacksV1.parse(live.fallbacks).to_core()
    # L1 fallback: the regime did not arrive, the block ran fallbacks.l1 with the session_begin spread.
    rec_l1, table_l1 = await decide_block(live, 2)
    l1_trial = build_trial(
        live,
        2,
        "pointing",
        rec_l1["decision_id"],
        fallbacks.l1,
        live.begin_spread(),
        0,
        l1_fallback_reason="late",
    )
    l1_reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": l1_trial})
    assert l1_reply["verification"] == "fallback", l1_reply

    # L2-only fallback: an act-guarded block whose table arrived late; a press on a gap is ambiguous.
    for block in [b["index"] for b in live.blocks if b["phase"] == "A" and b["index"] > 2]:
        rec_l2, table_l2 = await decide_block(live, block)
        if table_l2 is not None:
            break
    else:
        pytest.fail("no act-guarded block in Phase A")
    task = next(b["task"] for b in live.blocks if b["index"] == block)
    candidates = candidate_rects(task)
    gap = (
        (
            int((candidates[0]["cx"] + candidates[1]["cx"]) / 2),
            int(candidates[0]["cy"]),
        )
        if task == "pointing"
        else (int(candidates[0]["cx"]), int((candidates[0]["cy"] + candidates[1]["cy"]) / 2))
    )
    posterior = compute_posterior(
        core_rects(candidates),
        gap,
        modality="mouse",
        r=rec_l2["spread"]["r"],
        floor=rec_l2["spread"]["floor"],
        surface=SURFACE,
        press_index=-1,
    )
    cell = cell_id("b3", posterior.n_plausible, False)
    response = fixed_l2_rule(fallbacks, cell)
    ambiguous = {
        "index": 0,
        "t_ms": 500.0,
        "modality": "mouse",
        "x": gap[0],
        "y": gap[1],
        "hit_index": None,
        "press_index": -1,
        "a_gate": posterior.ambiguity_gate,
        "n_plausible": posterior.n_plausible,
        "expanded_index": None,
        "hit_rect": None,
        "gaze": None,
        "l2": {
            "decision_id": table_l2["decision_id"],
            "cell_id": cell,
            "draw_index": 0,
            "variant": response,
            "source": "fallback",
            "fallback_reason": "late",
        },
        "irreversible_rule": False,
        "result": "asked" if response.startswith("ask") else "none",
        "outcome": "cancelled" if response.startswith("ask") else "none",
        "committed_index": None,
    }
    second = pointer_activation(live, candidates, rec_l2["spread"], 1, rec_l2["variant"], index=1)
    l2_trial = build_trial(
        live,
        block,
        task,
        rec_l2["decision_id"],
        rec_l2["variant"],
        rec_l2["spread"],
        0,
        intended=1,
        table_ids=[table_l2["decision_id"]],
        activations=[ambiguous, second],
    )
    l2_trial["corrections"] = 1 if response.startswith("ask") else 0
    l2_reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": l2_trial})
    assert l2_reply["verification"] == "ok", l2_reply

    # Late offer: no offer was shown, only the offer decision is relabelled.
    offer = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "offer",
            "context": {
                "block_index": block,
                "trial_index": 1,
                "task": task,
                "struggle_kind": "corrections",
                "keyboard_available": True,
            },
        }
    )
    assert offer["status"] == "ok"
    offer_id = offer["recommendation"]["decision_id"]
    offer_trial = build_trial(
        live,
        block,
        task,
        rec_l2["decision_id"],
        rec_l2["variant"],
        rec_l2["spread"],
        1,
        table_ids=[table_l2["decision_id"]],
        offers=[{"decision_id": offer_id, "variant": "no_offer", "response": "ignored", "fallback_reason": "late"}],
    )
    offer_reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": offer_trial})
    assert offer_reply["verification"] == "ok", offer_reply

    behavior = dict(await sql(harness, "SELECT decision_id, behavior FROM decisions"))
    baseline = {decision_id for decision_id, value in behavior.items() if value == "baseline"}
    assert baseline == {rec_l1["decision_id"], table_l2["decision_id"], offer_id}
    records = {
        record.decision_id: record for record in (await harness.svc.store.results_rows(pk, day_index=None)).decisions
    }
    assert records[rec_l1["decision_id"]].applied == fallbacks.l1
    assert records[rec_l1["decision_id"]].fallback_reason == "late"
    assert records[table_l2["decision_id"]].applied is None
    assert records[offer_id].applied == "no_offer"
    assert records[rec_l2["decision_id"]].behavior == "policy"


async def test_a_stale_table_the_trial_then_applied_is_not_relabelled(harness):
    # The table requested after a veto had not arrived at the first ambiguous press (stale_table names it); it arrived
    # and the running trial adopted it, so its draws were applied and it stays a policy decision.
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    fallbacks = FallbacksV1.parse(live.fallbacks).to_core()
    for block in [b["index"] for b in live.blocks if b["phase"] == "A" and b["index"] > 1]:
        rec, table = await decide_block(live, block)
        if table is not None:
            break
    else:
        pytest.fail("no act-guarded block in Phase A")
    task = next(b["task"] for b in live.blocks if b["index"] == block)
    candidates = candidate_rects(task)
    if task == "pointing":
        gap = (int((candidates[0]["cx"] + candidates[1]["cx"]) / 2), int(candidates[0]["cy"]))
    else:
        gap = (int(candidates[0]["cx"]), int((candidates[0]["cy"] + candidates[1]["cy"]) / 2))
    posterior = compute_posterior(
        core_rects(candidates),
        gap,
        modality="mouse",
        r=rec["spread"]["r"],
        floor=rec["spread"]["floor"],
        surface=SURFACE,
        press_index=-1,
    )
    cell = cell_id("b3", posterior.n_plausible, False)
    entry = next(c for c in table["cells"] if c["cell_id"] == cell)
    responses = [
        ("fallback", fixed_l2_rule(fallbacks, cell), "stale_table"),
        ("table", select_from_cell(entry["order"], entry["propensities"], float(table["draws"][0][0])), None),
    ]
    activations = []
    for index, (source, response, reason) in enumerate(responses):
        activations.append(
            {
                "index": index,
                "t_ms": 500.0 + 300.0 * index,
                "modality": "mouse",
                "x": gap[0],
                "y": gap[1],
                "hit_index": None,
                "press_index": -1,
                "a_gate": posterior.ambiguity_gate,
                "n_plausible": posterior.n_plausible,
                "expanded_index": None,
                "hit_rect": None,
                "gaze": None,
                "l2": {
                    "decision_id": table["decision_id"],
                    "cell_id": cell,
                    "draw_index": 0,
                    "variant": response,
                    "source": source,
                    "fallback_reason": reason,
                },
                "irreversible_rule": False,
                "result": "asked" if response.startswith("ask") else "none",
                "outcome": "cancelled" if response.startswith("ask") else "none",
                "committed_index": None,
            }
        )
    activations.append(pointer_activation(live, candidates, rec["spread"], 1, rec["variant"], index=2))
    trial = build_trial(
        live,
        block,
        task,
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        intended=1,
        table_ids=[table["decision_id"]],
        activations=activations,
    )
    trial["corrections"] = sum(response.startswith("ask") for _, response, _ in responses)
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["status"] == "recorded" and reply["verification"] == "ok", reply
    behavior = dict(await sql(harness, "SELECT decision_id, behavior FROM decisions"))
    assert behavior[table["decision_id"]] == "policy"
    assert behavior[rec["decision_id"]] == "policy"


async def test_end_session_and_closed_session_calls(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=2)
    ended = await harness.svc.end_session({"session_id": live.session_id, "reason": "stopped"})
    assert ended["status"] == "ok"
    assert {"trials", "completion", "ua_rate", "predictions"} <= set(ended["summary"])
    json.dumps(ended)
    closed = await harness.svc.end_session({"session_id": live.session_id, "reason": "stopped"})
    assert closed == {"status": "ignored", "reason": "session_closed"}
    resume = await harness.svc.resume_session({"session_id": live.session_id})
    assert resume == {"status": "ignored", "reason": "session_closed"}
    unknown = await harness.svc.resume_session({"session_id": str(uuid.uuid4())})
    assert unknown == {"status": "ignored", "reason": "unknown_session"}
    end_reason = await sql(harness, "SELECT end_reason FROM sessions")
    assert end_reason == [("stopped",)]


async def test_report_burden_sets_recorded_at_and_seq(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=1, burden=False)
    reply = await harness.svc.report_burden(
        {"session_id": live.session_id, "report": {"block_index": 2, "scope": "block", "effort": 4, "control": 6}}
    )
    assert reply == {"status": "recorded"}
    rows = await sql(harness, "SELECT recorded_at, seq FROM burden")
    assert datetime.fromisoformat(rows[0][0]) == harness.clock.now
    assert rows[0][1] == 1 + max(row[0] for row in await sql(harness, "SELECT seq FROM trials"))


async def test_error_paths_never_raise(harness):
    assert (await harness.svc.recommend({"session_id": "x"}))["status"] == "error"
    assert (await harness.svc.record_trial({"session_id": str(uuid.uuid4()), "trial": {}}))["status"] == "error"
    assert (await harness.svc.erase({"participant_key": "p-00000000", "scope": "all"}))["status"] == "error"
    assert (await harness.svc.results({"participant_key": "p-00000000", "scope": "week"}))["status"] == "error"
    assert (await harness.svc.end_session({"session_id": str(uuid.uuid4()), "reason": "bored"}))["status"] == "error"
    assert (await harness.svc.status({"x": 1}))["status"] == "error"
    big = await harness.svc.set_preference({"kind": "pin_act", "scope": {"context": "study"}, "value": "x" * 9000})
    assert big["status"] == "error" and big["reason"] == "payload_too_large"
    assert (await harness.svc.retract_preference({"pref_id": "missing"})) == {
        "status": "error",
        "message": "Unknown rule",
    }
    assert (await harness.svc.status("not a dict"))["status"] == "ok"  # type: ignore[arg-type]


async def test_ignored_paths(harness):
    for method, params in (
        (harness.svc.reset_learning, {"participant_key": "p-00000000"}),
        (harness.svc.erase, {"participant_key": "p-00000000", "scope": "today"}),
        (harness.svc.results, {"participant_key": "p-00000000", "scope": "all"}),
        (harness.svc.export, {"participant_key": "p-00000000"}),
        (harness.svc.revoke_consent, {"participant_key": "p-00000000"}),
        (harness.svc.record_trial, {"session_id": str(uuid.uuid4()), "trial": build_trial_stub()}),
        (harness.svc.report_burden, {"session_id": str(uuid.uuid4()), "report": {"scope": "session"}}),
        (harness.svc.resume_session, {"session_id": str(uuid.uuid4())}),
    ):
        reply = await method(params)
        assert reply["status"] == "ignored", (method, reply)
        assert reply["reason"] == "no_consent"


def build_trial_stub() -> dict:
    return {
        "trial_id": str(uuid.uuid4()),
        "block_index": 1,
        "trial_index": 0,
        "task": "pointing",
        "regime": "act",
        "regime_decision_id": str(uuid.uuid4()),
        "table_decision_ids": [],
        "applied_regime": "act",
        "fallbacks_issued_at": START.isoformat(),
        "spread": {"r": 1.0, "floor": 1.0},
        "layout": layout_summary("pointing"),
        "candidates": candidate_rects("pointing"),
        "surface_w": 700.0,
        "surface_h": 500.0,
        "dpr": 1.0,
        "reversible": [True] * 12,
        "cue_at": START.isoformat(),
        "intended_index": 0,
        "modality": "mouse",
        "activations": [],
        "committed_index": None,
        "commit_via": "none",
        "corrections": 0,
        "time_ms": 10000.0,
        "limit_ms": 10000,
        "duration_ms": 10000.0,
        "interrupted": False,
        "explanation_views": 0,
    }


async def test_export_requires_research_export(harness):
    pk = await new_participant(harness)
    assert await harness.svc.export({"participant_key": pk}) == {"status": "refused", "reason": "purpose_missing"}
    other = await new_participant(harness, ("selection_study", "research_export"))
    reply = await harness.svc.export({"participant_key": other})
    assert reply["status"] == "ok"
    path = Path(reply["path"])
    assert path.parent == harness.exports_dir and path.name.startswith(f"coadapt_{other}_")
    assert reply["bytes"] == path.stat().st_size


# ── settings ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "config_value", "effective", "deviates"),
    [
        ("exploration_floor", 0.2, 0.1, True),
        ("exploration_floor", 0.1, 0.1, False),
        ("switching_penalty", 0.5, 0.1, True),
        ("ambiguity_threshold", 0.15, 0.15, True),
        ("ambiguity_threshold", 0.25, 0.25, False),
        ("unintended_activation_tolerance", 0.1, 0.05, True),
        ("unintended_activation_tolerance", 0.02, 0.02, True),
        ("noninferiority_margin", 0.03, 0.01, True),
        ("noninferiority_margin", 0.0, 0.0, True),
        ("upper_bound_credibility", 0.99, 0.99, True),
        ("upper_bound_credibility", 0.9, 0.95, True),
    ],
)
async def test_protocol_pinned_and_stricter_settings(harness, name, config_value, effective, deviates):
    setattr(harness.settings, name, config_value)
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    row = await stored_session(harness, live.session_id)
    assert row.settings[name] == pytest.approx(effective)
    assert (name in row.protocol_deviations) is deviates
    assert row.settings["thompson_draws"] == 256


async def test_settings_are_fixed_per_session(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    previous = dataclasses.replace(harness.settings)
    harness.settings.ambiguity_threshold = 0.1
    await harness.svc.on_config_changed(previous)
    # An act-guarded block of this session still uses the session's tau of 0.25.
    for block in [b["index"] for b in live.blocks if b["phase"] == "A"]:
        rec, table = await decide_block(live, block)
        if table is not None:
            break
    else:
        pytest.fail("no act-guarded block in Phase A")
    assert table["ambiguity_threshold"] == 0.25
    assert [b["lo"] for b in table["bins"]] == [0.25, 0.4, 0.6]
    task = next(b["task"] for b in live.blocks if b["index"] == block)
    tight = [
        {"cx": 300.0, "cy": 200.0, "w": 28.0, "h": 28.0},
        {"cx": 330.0, "cy": 200.0, "w": 28.0, "h": 28.0},
        {"cx": 360.0, "cy": 200.0, "w": 28.0, "h": 28.0},
    ]
    endpoint = None
    for dx in range(0, 15):
        posterior = compute_posterior(
            core_rects(tight),
            (300 + dx, 200),
            modality="mouse",
            r=rec["spread"]["r"],
            floor=rec["spread"]["floor"],
            surface=SURFACE,
            press_index=0,
        )
        if posterior.hit_index == 0 and 0.12 < posterior.ambiguity_gate <= 0.25:
            endpoint = (300 + dx, 200)
            break
    assert endpoint is not None, "no endpoint between the old and the new tau"
    activation = pointer_activation(live, tight, rec["spread"], 0, rec["variant"], endpoint=endpoint)
    trial = build_trial(
        live,
        block,
        task,
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        intended=0,
        candidates=tight,
        activations=[activation],
        table_ids=[table["decision_id"]],
    )
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["verification"] == "ok", reply
    row = await stored_session(harness, live.session_id)
    assert row.settings["ambiguity_threshold"] == 0.25
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    stored_snapshot = (await harness.svc.store.load_snapshot(pk))[0]
    nxt = await begin(harness, pk)
    assert nxt.reply["thresholds"]["ambiguity"] == 0.1
    assert (await stored_session(harness, nxt.session_id)).settings["ambiguity_threshold"] == 0.1
    await harness.svc.end_session({"session_id": nxt.session_id, "reason": "completed"})
    live_snapshot = (await harness.svc.store.load_snapshot(pk))[0]
    rebuilt = await harness.svc.rebuild(pk)
    assert same_state(rebuilt.snapshot(), live_snapshot)
    assert stored_snapshot != live_snapshot


async def test_paused_session_keeps_its_tau(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    previous = dataclasses.replace(harness.settings)
    harness.settings.ambiguity_threshold = 0.1
    await harness.svc.on_config_changed(previous)
    resume = await harness.svc.resume_session({"session_id": live.session_id})
    assert resume["status"] == "ok"
    row = await stored_session(harness, live.session_id)
    assert row.settings["ambiguity_threshold"] == 0.25


async def test_disabling_ends_open_sessions(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    previous = dataclasses.replace(harness.settings)
    harness.settings.enabled = False
    await harness.svc.on_config_changed(previous)
    assert await sql(harness, "SELECT end_reason FROM sessions") == [("disabled",)]
    harness.settings.enabled = True
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": build_trial_stub()})
    assert reply == {"status": "ignored", "reason": "session_closed"}


# ── restart and start-up ─────────────────────────────────────────────────────


async def test_restart_closes_open_sessions(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=1)
    harness.clock.advance(minutes=5)
    await harness.restart()
    assert await sql(harness, "SELECT end_reason FROM sessions") == [("daemon_restart",)]
    assert (await harness.svc.resume_session({"session_id": live.session_id}))["status"] == "ignored"
    status = await harness.svc.status({})
    assert status["participants"][0]["model"]["sessions_seen"] == 1


async def test_restart_after_expiry_with_an_open_session(harness):
    short = await new_participant(harness, ("selection_study", "research_export"), retention_days=1)
    keep = await new_participant(harness)
    await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=short))
    assert (await harness.svc.export({"participant_key": short}))["status"] == "ok"
    await begin(harness, short)
    other = await begin(harness, keep)
    await harness.svc.close()
    harness.clock.advance(days=2)
    harness.svc = harness.new_service()
    await harness.svc.start()
    assert short not in harness.keys
    assert list(harness.exports_dir.iterdir()) == []
    assert short not in harness.preferences_path.read_text(encoding="utf-8")
    assert await sql(harness, "SELECT COUNT(*) FROM sessions WHERE participant_key = ?", (short,)) == [(0,)]
    assert await sql(harness, "SELECT COUNT(*) FROM rules WHERE participant_key = ?", (short,)) == [(0,)]
    assert await sql(harness, "SELECT end_reason FROM sessions WHERE session_id = ?", (other.session_id,)) == [
        ("daemon_restart",)
    ]
    status = await harness.svc.status({})
    assert status["status"] == "ok"
    assert [p["participant_key"] for p in status["participants"]] == [keep]
    assert harness.broadcasts.of("expired")[-1]["participant_key"] == short


async def test_start_survives_a_failing_participant(harness, monkeypatch):
    first = await new_participant(harness)
    second = await new_participant(harness)
    a = await begin(harness, first)
    b = await begin(harness, second)
    await harness.svc.close()
    service = harness.new_service()
    original = service.store.close_session

    async def failing(session_id, ended_at, reason):
        if session_id == a.session_id:
            raise RuntimeError("disk error")
        return await original(session_id, ended_at, reason)

    monkeypatch.setattr(service.store, "close_session", failing)
    harness.svc = service
    await service.start()
    rows = dict(await sql(harness, "SELECT session_id, end_reason FROM sessions"))
    assert rows[a.session_id] is None
    assert rows[b.session_id] == "daemon_restart"
    assert (await service.status({}))["status"] == "ok"


async def test_degraded_start_without_keyring(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    pk = await new_participant(h, ("selection_study", "research_export"))
    other = await new_participant(h)
    live, _runs = await run_session(h, pk, blocks=1, trials=1)
    await h.svc.close()
    h.keys.set_available(False)
    h.svc = h.new_service()
    await h.svc.start()
    try:
        status = await h.svc.status({})
        assert status["status"] == "ok" and status["keyring_available"] is False
        assert [p["label"] for p in status["participants"]] == [None, None]
        for method, params in (
            (h.svc.begin_session, begin_params(other)),
            (h.svc.set_preference, pref_params("pin_act", True)),
            (h.svc.retract_preference, {"pref_id": "x"}),
            (h.svc.reset_learning, {"participant_key": other}),
            (h.svc.results, {"participant_key": other, "scope": "all"}),
            (h.svc.export, {"participant_key": pk}),
            (
                h.svc.recommend,
                {
                    "session_id": live.session_id,
                    "decision_id": str(uuid.uuid4()),
                    "level": "block_regime",
                    "context": block_context(1, "menu"),
                },
            ),
            (h.svc.resume_session, {"session_id": live.session_id}),
        ):
            reply = await method(params)
            assert reply["status"] == "refused" and reply["reason"] == "keyring_unavailable", (method, reply)
        grant = await h.svc.grant_consent(grant_params())
        assert grant["status"] == "error"
        assert grant["message"] == "Secure OS key storage is unavailable; co-adaptation stays off."
        erased = await h.svc.erase({"participant_key": pk, "scope": "today"})
        assert erased["status"] == "ok" and erased["erased"]["rows"] > 0
        revoked = await h.svc.revoke_consent({"participant_key": other})
        assert revoked["status"] == "ok" and revoked["erased"]["key_destroyed"] is False
    finally:
        await h.svc.close()


# ── status and notifications ─────────────────────────────────────────────────


async def test_status_shape_with_labels_and_fallbacks(harness):
    pk = await new_participant(harness, label="Ana")
    unlabelled = await new_participant(harness)
    await run_session(harness, pk, blocks=1, trials=1)
    status = await harness.svc.status({})
    participants = {p["participant_key"]: p for p in status["participants"]}
    assert participants[pk]["label"] == "Ana" and participants[unlabelled]["label"] is None
    entry = participants[pk]
    assert set(entry) == {"participant_key", "label", "consent", "model", "fallbacks", "sessions", "last_session_at"}
    assert entry["sessions"] == 1
    assert set(entry["model"]) == {
        "trials_seen",
        "sessions_seen",
        "last_day_index",
        "drift_flags",
        "calibration_suspended",
        "learn_across_sessions_effective",
    }
    assert set(entry["fallbacks"]) == {"mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer"}
    for fallbacks in entry["fallbacks"].values():
        parsed = assert_fallbacks(fallbacks)
        assert parsed.freeze_conflict is None
    assert status["store"]["participants"] == 2 and status["store"]["sessions"] == 1
    json.dumps(status)


async def test_lifecycle_notifications(harness):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    await harness.svc.reset_learning({"participant_key": pk})
    await harness.svc.erase({"participant_key": pk, "scope": "today"})
    await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3"))
    await harness.svc.revoke_consent({"participant_key": pk})
    short = await new_participant(harness, retention_days=1)
    harness.clock.advance(days=1, seconds=1)
    await harness.svc.status({})
    assert harness.broadcasts.reasons() == [
        "consent",
        "reset",
        "erased",
        "preferences",
        "revoked",
        "consent",
        "expired",
    ]
    assert all(method == "co_adaptation_updated" for method, _params in harness.broadcasts)
    assert harness.broadcasts.of("expired") == [{"reason": "expired", "participant_key": short, "refresh_seq": None}]


async def test_concurrent_calls_are_serialised(harness, monkeypatch):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    inside = {"now": 0, "max": 0}
    original = harness.svc.store.put_decision

    async def tracked(record, *, participant_key, **kwargs):
        inside["now"] += 1
        inside["max"] = max(inside["max"], inside["now"])
        await asyncio.sleep(0.01)
        try:
            return await original(record, participant_key=participant_key, **kwargs)
        finally:
            inside["now"] -= 1

    monkeypatch.setattr(harness.svc.store, "put_decision", tracked)
    replies = await asyncio.gather(
        *[
            harness.svc.recommend(
                {
                    "session_id": live.session_id,
                    "decision_id": str(uuid.uuid4()),
                    "level": "block_regime",
                    "context": block_context(index, "menu" if index % 2 else "pointing"),
                }
            )
            for index in range(1, 7)
        ],
        harness.svc.status({}),
    )
    assert all(reply["status"] == "ok" for reply in replies)
    assert inside["max"] == 1
    seqs = [row[0] for row in await sql(harness, "SELECT seq FROM decisions ORDER BY seq")]
    assert seqs == list(range(1, 7))


async def test_seed_salt_never_leaves_the_daemon(harness):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    live, runs = await run_session(harness, pk, blocks=3, trials=2)
    salt = (await stored_session(harness, live.session_id)).seed_salt
    blob = json.dumps(
        [live.reply, [run.rec for run in runs], [run.table for run in runs], [run.replies for run in runs]]
    )
    blob += json.dumps(await harness.svc.status({}))
    blob += json.dumps(await harness.svc.results({"participant_key": pk, "scope": "all"}))
    blob += json.dumps(list(harness.broadcasts))
    assert salt not in blob


# ── eviction ─────────────────────────────────────────────────────────────────


async def test_revoke_evicts_and_close_writes_nothing(harness):
    keep = await new_participant(harness)
    pk = await new_participant(harness, ("selection_study", "research_export"))
    live = await begin(harness, pk)
    run = await run_block(live, 2, trials=2)
    assert pk in harness.svc._engines
    reply = await harness.svc.revoke_consent({"participant_key": pk})
    assert reply["status"] == "ok"
    assert reply["erased"]["key_destroyed"] is True and reply["erased"]["rows"] > 0
    assert pk not in harness.svc._engines
    assert live.session_id not in harness.svc._sessions
    later = build_trial(live, 2, run.task, run.rec["decision_id"], run.variant, run.rec["spread"], 5)
    assert await harness.svc.record_trial({"session_id": live.session_id, "trial": later}) == {
        "status": "ignored",
        "reason": "no_consent",
    }
    baseline = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "block_regime",
            "context": block_context(3, "menu"),
        }
    )
    assert baseline["status"] == "ignored" and baseline["logged"] is False
    assert (await harness.svc.begin_session(begin_params(pk)))["reason"] == "no_consent"
    await harness.svc.close()
    assert pk not in harness.keys
    assert await harness.svc.store._fetchall("SELECT 1") == []
    harness.svc = harness.new_service()
    await harness.svc.start()
    for table in ("participants", "sessions", "decisions", "trials", "model_state", "events", "rules", "consents"):
        assert await sql(harness, f"SELECT COUNT(*) FROM {table} WHERE participant_key = ?", (pk,)) == [(0,)]
    assert keep in await harness.svc.store.participants()


async def test_expiry_evicts_through_the_lazy_prune(harness):
    pk = await new_participant(harness, retention_days=1)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=1)
    harness.clock.advance(days=1, seconds=1)
    status = await harness.svc.status({})
    assert status["participants"] == []
    assert pk not in harness.svc._engines and live.session_id not in harness.svc._sessions
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": build_trial_stub()})
    assert reply == {"status": "ignored", "reason": "no_consent"}
    await harness.svc.close()
    assert pk not in harness.keys
    assert not harness.db_path.exists()


async def test_erase_today_evicts_and_rebuilds(harness):
    pk = await new_participant(harness, ("selection_study", "research_export"))
    live = await begin(harness, pk)
    run = await run_block(live, 2, trials=2)
    await harness.svc.export({"participant_key": pk})
    reply = await harness.svc.erase({"participant_key": pk, "scope": "today"})
    assert reply["status"] == "ok" and reply["session_ended"] is True
    assert reply["erased"]["export_files"] == 1 and reply["erased"]["rows"] > 0
    assert live.session_id not in harness.svc._sessions
    later = build_trial(live, 2, run.task, run.rec["decision_id"], run.variant, run.rec["spread"], 5)
    assert (await harness.svc.record_trial({"session_id": live.session_id, "trial": later}))["status"] == "ignored"
    assert await sql(harness, "SELECT COUNT(*) FROM trials") == [(0,)]
    status = (await harness.svc.status({}))["participants"][0]
    assert status["model"]["trials_seen"] == 0


async def test_reset_learning_ends_the_open_session_first(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=2)
    harness.clock.advance(minutes=1)
    reply = await harness.svc.reset_learning({"participant_key": pk})
    assert reply["status"] == "ok" and reply["session_ended"] is True
    assert reply["model"]["trials_seen"] == 0
    assert await sql(harness, "SELECT end_reason FROM sessions") == [("learning_reset",)]
    assert await sql(harness, "SELECT kind FROM events") == [("learning_reset",)]
    assert harness.broadcasts.of("reset")


async def test_no_rebuild_or_restore_while_a_session_is_open(harness, monkeypatch):
    pk = await new_participant(harness)
    await run_session(harness, pk, blocks=1, trials=1)
    await harness.restart()
    calls: list[tuple[str, bool]] = []
    service = harness.svc
    original_load, original_rebuild = service._load_engine, service._rebuild_unlocked

    async def load(participant_key, consent, now):
        calls.append(("load", service._open_session_of(participant_key) is not None))
        return await original_load(participant_key, consent, now)

    async def rebuild(participant_key, consent, now):
        calls.append(("rebuild", service._open_session_of(participant_key) is not None))
        return await original_rebuild(participant_key, consent, now)

    monkeypatch.setattr(service, "_load_engine", load)
    monkeypatch.setattr(service, "_rebuild_unlocked", rebuild)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=2)
    await service.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    await service.status({})
    await service.results({"participant_key": pk, "scope": "all"})
    await service.resume_session({"session_id": live.session_id})
    assert calls and all(not open_session for _name, open_session in calls)


# ── preferences, staleness and refresh_seq ───────────────────────────────────


async def test_preference_change_marks_stale_with_one_refresh_seq(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    assert live.reply["refresh_seq"] == 0
    rec, table = await decide_block(live, 2)
    reply = await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    assert reply["status"] == "ok"
    assert reply["refresh_required"] is True and reply["refresh_seq"] == 1
    assert_fallbacks(reply["fallbacks"])
    assert harness.broadcasts.of("preferences") == [{"reason": "preferences", "participant_key": pk, "refresh_seq": 1}]
    trial = build_trial(
        live,
        2,
        "pointing",
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        table_ids=[table["decision_id"]] if table else [],
    )
    recorded = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert recorded["verification"] == "ok"
    assert recorded["refresh_required"] is True and recorded["refresh_seq"] == 1
    assert_fallbacks(recorded["fallbacks"])
    resumed = await harness.svc.resume_session({"session_id": live.session_id})
    assert resumed["refresh_required"] is True and resumed["refresh_seq"] == 1
    # A second, different change in the same block carries the next value.
    retracted = await harness.svc.retract_preference({"pref_id": reply["preference"]["pref_id"]})
    assert retracted["status"] == "ok"
    assert retracted["refresh_required"] is True and retracted["refresh_seq"] == 2
    # Fresh decisions clear the stale state; a change before any new decision of the session would carry null.
    await decide_block(live, 3)
    resumed = await harness.svc.resume_session({"session_id": live.session_id})
    assert resumed["refresh_required"] is False and resumed["refresh_seq"] == 2
    third = await harness.svc.set_preference(pref_params("forbid_variant", "enlarge_target", participant_key=pk))
    assert third["refresh_seq"] == 3 and third["refresh_required"] is True
    assert [b["refresh_seq"] for b in harness.broadcasts.of("preferences")] == [1, 2, 3]


async def test_device_wide_rule_refreshes_every_open_session(harness):
    first = await new_participant(harness)
    second = await new_participant(harness)
    idle = await new_participant(harness)
    a = await begin(harness, first)
    b = await begin(harness, second)
    c = await begin(harness, idle)
    # Give the two sessions different refresh_seq values: only the first one holds an open decision here.
    await decide_block(a, 2)
    own = await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=first))
    assert own["refresh_seq"] == 1
    assert harness.broadcasts.of("preferences") == [
        {"reason": "preferences", "participant_key": first, "refresh_seq": 1}
    ]
    await decide_block(a, 3)
    await decide_block(b, 2)
    reply = await harness.svc.set_preference(pref_params("max_interruptions_per_block", 8))
    assert reply["refresh_required"] is False and reply["refresh_seq"] is None and reply["fallbacks"] is None
    device_wide = {item["participant_key"]: item["refresh_seq"] for item in harness.broadcasts.of("preferences")[1:]}
    # Every open session refreshes, the idle one too: its fallbacks are stale even before its first decision (9.7).
    assert device_wide == {second: 1, first: 2, idle: 1}
    # With no open session, the change only asks every window to reload status.
    for live in (a, b, c):
        await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    nobody = await harness.svc.set_preference(pref_params("forbid_variant", "offer_modality"))
    assert nobody["status"] == "ok"
    assert harness.broadcasts.of("preferences")[-1] == {
        "reason": "preferences",
        "participant_key": None,
        "refresh_seq": None,
    }


@pytest.mark.parametrize(("kind", "head", "outcome"), [("calibration_suspended", None, None), ("drift", "l1", "ua")])
async def test_an_alarm_raises_refresh_seq_once(harness, monkeypatch, kind, head, outcome):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    inject_alarm(monkeypatch, kind, head=head, outcome=outcome)
    trial = build_trial(
        live,
        2,
        "pointing",
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        table_ids=[table["decision_id"]] if table else [],
    )
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["refresh_required"] is True and reply["refresh_seq"] == 1
    fallbacks = assert_fallbacks(reply["fallbacks"])
    assert harness.broadcasts.of(kind) == [{"reason": kind, "participant_key": pk, "refresh_seq": 1}]
    assert await sql(harness, "SELECT kind, session_id, trial_id FROM events") == [
        (kind, live.session_id, trial["trial_id"])
    ]
    again = await harness.svc.record_trial(
        {
            "session_id": live.session_id,
            "trial": build_trial(
                live,
                2,
                "pointing",
                rec["decision_id"],
                rec["variant"],
                rec["spread"],
                1,
                table_ids=[table["decision_id"]] if table else [],
            ),
        }
    )
    assert again["refresh_required"] is True and again["refresh_seq"] == 1
    if kind != "calibration_suspended":
        assert (await harness.svc.status({}))["participants"][0]["model"]["drift_flags"] == [f"{head}:{outcome}"]
        return
    for name in ("plain", "refresh", "practice", "l1"):
        assert getattr(fallbacks, name) not in ACT_GUARDED | {"assistance_off"}
    status = await harness.svc.status({})
    for value in status["participants"][0]["fallbacks"].values():
        parsed = assert_fallbacks(value)
        assert parsed.plain != "assistance_off" and parsed.practice != "assistance_off"


async def test_refresh_seq_never_decreases_across_a_rebuild(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    await decide_block(live, 2)
    first = await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3", participant_key=pk))
    assert first["refresh_seq"] == 1
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    await harness.svc.reset_learning({"participant_key": pk})
    await harness.svc.rebuild(pk)
    again = await begin(harness, pk)
    assert again.reply["refresh_seq"] == 1


async def test_no_allowed_regime_at_session_begin_under_c7(harness, monkeypatch):
    harness.settings.learn_across_sessions = True
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    live = await begin(harness, pk)
    rec, table = await decide_block(live, 2)
    inject_alarm(monkeypatch)
    trial = build_trial(
        live,
        2,
        "pointing",
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        table_ids=[table["decision_id"]] if table else [],
    )
    await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    # The suspension carries over within the day: the next session's fallbacks are never acting.
    harness.clock.advance(minutes=5)
    suspended = await begin(harness, pk)
    fallbacks = assert_fallbacks(suspended.fallbacks)
    for name in ("plain", "refresh", "practice", "l1"):
        assert getattr(fallbacks, name) not in ACT_GUARDED | {"assistance_off"}
    await harness.svc.end_session({"session_id": suspended.session_id, "reason": "completed"})
    forbid = await harness.svc.set_preference(pref_params("forbid_variant", "confirm_selection", participant_key=pk))
    assert forbid["status"] == "ok", forbid
    reply = await harness.svc.begin_session(begin_params(pk))
    assert reply == {"status": "refused", "reason": "no_allowed_regime"}


async def test_no_allowed_regime_at_recommend(harness, monkeypatch):
    pk = await new_participant(harness)
    live = await begin(harness, pk)

    def refuse(self, **kwargs):
        raise NoAllowedRegime()

    monkeypatch.setattr(CoAdaptationEngine, "decide_block", refuse)
    reply = await harness.svc.recommend(
        {
            "session_id": live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "block_regime",
            "context": block_context(1, "menu"),
        }
    )
    assert reply == {"status": "refused", "reason": "no_allowed_regime"}
    assert await sql(harness, "SELECT COUNT(*) FROM decisions") == [(0,)]


async def test_freeze_fallbacks_follow_the_frozen_regime(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    await decide_block(live, 2)
    reply = await harness.svc.set_preference(
        pref_params("freeze_adaptation", {"regime": "confirm_selection:hold_700"}, participant_key=pk, modality="mouse")
    )
    assert reply["status"] == "ok"
    fallbacks = assert_fallbacks(reply["fallbacks"])
    assert fallbacks.l1 == fallbacks.refresh == fallbacks.practice == "confirm_selection:hold_700"
    assert fallbacks.plain == "assistance_off"
    rec, _table = await decide_block(live, 3)
    assert rec["mode"] == "pinned" and rec["variant"] == "confirm_selection:hold_700"


async def test_conflicting_rule_writes_nothing_and_marks_nothing_stale(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    first = await harness.svc.set_preference(
        pref_params("min_caution", "confirm_selection:second_click", participant_key=pk)
    )
    assert first["status"] == "ok"
    await decide_block(live, 2)
    file_before = harness.preferences_path.read_bytes()
    rules_before = await sql(harness, "SELECT pref_id FROM rules")
    broadcasts_before = len(harness.broadcasts)
    reply = await harness.svc.set_preference(
        pref_params("freeze_adaptation", {"regime": "act"}, participant_key=pk, modality="mouse")
    )
    assert reply["status"] == "error" and reply["reason"] == "preference_conflict"
    assert reply["message"]
    assert [item["pref_id"] for item in reply["conflicting"]] == [first["preference"]["pref_id"]]
    assert reply["conflicting"][0]["summary"]
    assert harness.preferences_path.read_bytes() == file_before
    assert await sql(harness, "SELECT pref_id FROM rules") == rules_before
    assert len(harness.broadcasts) == broadcasts_before
    resumed = await harness.svc.resume_session({"session_id": live.session_id})
    # The rejected rule raised nothing: refresh_seq is still the value the accepted rule left.
    assert resumed["refresh_required"] is False and resumed["refresh_seq"] == first["refresh_seq"]


async def test_same_kind_conflicts_and_the_pause_replacement_sequence(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    pin = await harness.svc.set_preference(pref_params("pin_regime", "act", participant_key=pk))
    second_pin = await harness.svc.set_preference(pref_params("pin_regime", "enlarge_target:1.5", participant_key=pk))
    assert second_pin["reason"] == "preference_conflict"
    assert [item["pref_id"] for item in second_pin["conflicting"]] == [pin["preference"]["pref_id"]]
    await harness.svc.retract_preference({"pref_id": pin["preference"]["pref_id"]})
    freeze = await harness.svc.set_preference(
        pref_params("freeze_adaptation", {"regime": "act"}, participant_key=pk, modality="mouse")
    )
    assert freeze["status"] == "ok"
    other = await harness.svc.set_preference(
        pref_params("freeze_adaptation", {"regime": "enlarge_target:2.0"}, participant_key=pk, modality="mouse")
    )
    assert other["reason"] == "preference_conflict"
    assert [item["pref_id"] for item in other["conflicting"]] == [freeze["preference"]["pref_id"]]
    # The study's replacement during a block: retract the earlier pause, then send the new one.
    await decide_block(live, 2)
    before = (await harness.svc.resume_session({"session_id": live.session_id}))["refresh_seq"]
    retract = await harness.svc.retract_preference({"pref_id": freeze["preference"]["pref_id"]})
    replacement = await harness.svc.set_preference(
        pref_params("freeze_adaptation", {"regime": "enlarge_target:2.0"}, participant_key=pk, modality="mouse")
    )
    assert replacement["status"] == "ok"
    assert retract["refresh_required"] is True and replacement["refresh_required"] is True
    assert before < retract["refresh_seq"] < replacement["refresh_seq"]
    assert [b["refresh_seq"] for b in harness.broadcasts.of("preferences")][-2:] == [
        retract["refresh_seq"],
        replacement["refresh_seq"],
    ]
    active = [
        p
        for p in (await harness.svc.status({}))["preferences"]
        if p["kind"] == "freeze_adaptation" and p["retracted_at"] is None
    ]
    assert [p["value"] for p in active] == [{"regime": "enlarge_target:2.0"}]


async def test_smallest_cap_and_freeze_conflict(harness):
    pk = await new_participant(harness)
    await harness.svc.set_preference(pref_params("max_interruptions_per_block", 4))
    cap = await harness.svc.set_preference(pref_params("max_interruptions_per_block", 2, participant_key=pk))
    assert cap["status"] == "ok"
    status = await harness.svc.status({})
    mouse = assert_fallbacks(status["participants"][0]["fallbacks"]["mouse"])
    assert mouse.interruption_cap == 2
    assert mouse.freeze_conflict is None
    minimum = await harness.svc.set_preference(
        pref_params("min_caution", "confirm_selection:second_click", participant_key=pk)
    )
    assert minimum["status"] == "ok" and minimum["notes"]
    status = await harness.svc.status({})
    mouse = assert_fallbacks(status["participants"][0]["fallbacks"]["mouse"])
    assert mouse.freezable == []
    assert mouse.freeze_conflict is not None
    assert cap["preference"]["pref_id"] in mouse.freeze_conflict.conflicting


async def test_freeze_conflict_uses_the_open_sessions_latest_regime(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    rec, _table = await decide_block(live, 2)
    status = await harness.svc.status({})
    mouse = assert_fallbacks(status["participants"][0]["fallbacks"]["mouse"])
    assert rec["variant"] in mouse.freezable
    assert mouse.freeze_conflict is None


async def test_unknown_participant_in_a_rule_scope(harness):
    reply = await harness.svc.set_preference(pref_params("pin_act", True, participant_key="p-00000000"))
    assert reply["status"] == "error" and reply["reason"] == "unknown_participant"
    assert not harness.preferences_path.exists()


async def test_preferences_work_without_consent(harness):
    reply = await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3"))
    assert reply["status"] == "ok" and reply["refresh_seq"] is None
    assert not harness.db_path.exists()


# ── proposals ────────────────────────────────────────────────────────────────


async def _veto_trials(live: Live, count: int) -> str:
    for block in [b["index"] for b in live.blocks if b["phase"] == "A"]:
        rec, table = await decide_block(live, block)
        if rec["variant"] != "assistance_off":
            break
    task = next(b["task"] for b in live.blocks if b["index"] == block)
    fallbacks = live.fallbacks
    for trial_index in range(count):
        candidates = candidate_rects(task)
        first = pointer_activation(live, candidates, rec["spread"], 0, rec["variant"], index=0)
        trial = build_trial(
            live,
            block,
            task,
            rec["decision_id"],
            rec["variant"],
            rec["spread"],
            trial_index,
            intended=0,
            table_ids=[table["decision_id"]] if table else [],
            activations=[first],
            veto={"decision_id": rec["decision_id"], "level": "block_regime"},
            revert={
                "reason": "veto",
                "from_activation": 1,
                "regime": fallbacks["plain"],
                "fallbacks_issued_at": fallbacks["issued_at"],
            },
        )
        reply = await live.h.svc.record_trial({"session_id": live.session_id, "trial": trial})
        assert reply["status"] == "recorded", reply
    return rec["variant"]


async def test_three_vetoes_make_a_session_proposal(harness):
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    variant = await _veto_trials(live, 3)
    status = await harness.svc.status({})
    assert {"participant_key": pk, "kind": "forbid_variant", "variant": variant, "reason": "three_vetoes"} in status[
        "proposals"
    ]
    assert harness.broadcasts.of("proposal")
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    assert (await harness.svc.status({}))["proposals"] == []


async def test_a_veto_after_a_refresh_revert_counts_the_revert_regime(harness):
    # The refresh reverted the trial before the veto, so the chip showed the refresh regime: the service counts the
    # variant C5 excludes (records.vetoed_variant), never the cue regime the refresh had replaced.
    pk = await new_participant(harness)
    live = await begin(harness, pk)
    for block in [b["index"] for b in live.blocks if b["phase"] == "A"]:
        rec, table = await decide_block(live, block)
        if rec["variant"] not in ("assistance_off", live.fallbacks["refresh"]):
            break
    else:
        pytest.fail("no block whose regime differs from the refresh regime")
    task = next(b["task"] for b in live.blocks if b["index"] == block)
    candidates = candidate_rects(task)
    trial = build_trial(
        live,
        block,
        task,
        rec["decision_id"],
        rec["variant"],
        rec["spread"],
        0,
        intended=0,
        table_ids=[table["decision_id"]] if table else [],
        activations=[pointer_activation(live, candidates, rec["spread"], 0, rec["variant"], index=0)],
        veto={"decision_id": rec["decision_id"], "level": "block_regime"},
        revert={
            "reason": "refresh",
            "from_activation": 1,
            "regime": live.fallbacks["refresh"],
            "fallbacks_issued_at": live.fallbacks["issued_at"],
        },
    )
    reply = await harness.svc.record_trial({"session_id": live.session_id, "trial": trial})
    assert reply["status"] == "recorded", reply
    assert dict(harness.svc._sessions[live.session_id].veto_counts) == {live.fallbacks["refresh"]: 1}


async def test_cross_session_vetoes_count_only_with_the_purpose(harness):
    harness.settings.learn_across_sessions = True
    pk = await new_participant(harness, ("selection_study", "cross_session_learning"))
    live = await begin(harness, pk)
    variant = await _veto_trials(live, 2)
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    second = await begin(harness, pk)
    for _attempt in range(30):
        if await _veto_trials(second, 1) == variant:
            break
    else:
        pytest.fail("the vetoed variant never came up again")
    status = await harness.svc.status({})
    assert any(p["variant"] == variant for p in status["proposals"])
    harness.clock.advance(days=8)
    assert (await harness.svc.status({}))["proposals"] == []


# ── comparator, cap, results ─────────────────────────────────────────────────


async def test_comparator_ranking_comes_from_session_one(harness):
    harness.settings.adaptive_enabled = True
    pk = await new_participant(harness, ("selection_study", "adaptive_assistance"))
    await run_session(harness, pk, trials=4)
    harness.clock.advance(days=1)
    live = await begin(harness, pk)
    assert live.reply["session_number"] == 2
    row = await stored_session(harness, live.session_id)
    first = await harness.svc.store.session_rows(pk, 1)
    expected = rank_fixed_regimes(first.trials, first.burden, first.decisions, modality="mouse")
    assert row.comparator_ranking == tuple(expected)
    assert expected
    phases = [block["phase"] for block in live.blocks]
    assert phases.count("comparator") == 2 and phases.count("probe") == 2 and phases.count("A_lite") == 6
    b_block = next(block["index"] for block in live.blocks if block["phase"] == "B")
    rec, _table = await decide_block(live, b_block)
    assert rec["mode"] == "adaptive"
    assert rec["priorities"] is not None


async def test_comparator_is_empty_when_session_one_was_erased(harness, monkeypatch):
    import pilot.coadaptation.store as store_module

    harness.settings.adaptive_enabled = True
    pk = await new_participant(harness, ("selection_study", "adaptive_assistance"))
    await run_session(harness, pk, blocks=2, trials=2)
    harness.clock.advance(days=1)
    second, _runs = await run_session(harness, pk, blocks=1, trials=1)
    second_decisions = (
        await sql(harness, "SELECT COUNT(*) FROM decisions WHERE session_id = ?", (second.session_id,))
    )[0][0]
    monkeypatch.setattr(store_module, "DECISION_CAP_KEEP", second_decisions)
    harness.clock.advance(days=1)
    live = await begin(harness, pk)
    assert live.reply["session_number"] == 2
    row = await stored_session(harness, live.session_id)
    assert row.comparator_ranking == ()
    assert "comparator" not in [block["phase"] for block in live.blocks]
    assert (await harness.svc.status({}))["store"]["pruned_sessions"] == 1
    assert await harness.svc.store.session_rows(pk, 1) is None


async def test_decision_cap_runs_only_at_session_begin(harness, monkeypatch):
    import pilot.coadaptation.store as store_module

    pk = await new_participant(harness)
    monkeypatch.setattr(store_module, "DECISION_CAP_KEEP", 1)
    first, _runs = await run_session(harness, pk, blocks=2, trials=1)
    assert await sql(harness, "SELECT COUNT(*) FROM sessions") == [(1,)]
    harness.clock.advance(hours=1)
    second = await begin(harness, pk)
    sessions = [row[0] for row in await sql(harness, "SELECT session_id FROM sessions")]
    assert sessions == [second.session_id]
    for table in ("decisions", "trials", "burden"):
        assert await sql(harness, f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (first.session_id,)) == [(0,)]
    assert await sql(harness, "SELECT kind FROM events") == [("prune",)]
    await decide_block(second, 1)
    await decide_block(second, 2)
    assert await sql(harness, "SELECT COUNT(*) FROM sessions") == [(1,)]


async def test_results_shape_and_temporary_calibration(harness):
    pk = await new_participant(harness)
    await run_session(harness, pk, blocks=3, trials=3)
    live = await begin(harness, pk)
    await run_block(live, 2, trials=2)
    engine = harness.svc._engines[pk]
    before = engine.snapshot()
    for scope in ("today", "all"):
        reply = await harness.svc.results({"participant_key": pk, "scope": scope})
        assert reply["status"] == "ok"
        assert set(reply) == {
            "status",
            "evidence_kind",
            "claim",
            "per_regime",
            "calibration",
            "probe_trend",
            "drift_flags",
            "allowed_now",
            "predictions_now",
        }
        assert reply["evidence_kind"] == "live_human"
        assert set(reply["calibration"]) == {"ua", "complete", "log_time"}
        assert reply["calibration"]["ua"]["n"] > 0
        for entry in reply["per_regime"]:
            assert set(entry["verification"]) == {
                "ok",
                "fallback",
                "draw_mismatch",
                "posterior_mismatch",
                "interrupted",
            }
            assert set(entry["burden"]) == {"effort", "intrusiveness", "control"}
        assert set(reply["allowed_now"]) == {"eligible", "excluded"}
        assert reply["predictions_now"]
        json.dumps(reply)
    assert harness.svc._engines[pk] is engine
    assert engine.snapshot() == before


# ── consent history, rule history and the audit (needs WP4's core.audit) ─────


async def test_consent_history_and_audit_a6(harness):
    from pilot.coadaptation.core.audit import agency_audit

    harness.settings.adaptive_enabled = True
    pk = await new_participant(harness, ("selection_study", "research_export"))
    await run_session(harness, pk, blocks=2, trials=2)
    harness.clock.advance(days=1)
    update = await harness.svc.grant_consent(
        grant_params(("selection_study", "research_export", "adaptive_assistance"), participant_key=pk)
    )
    assert update["status"] == "ok"
    harness.clock.advance(minutes=5)
    live = await begin(harness, pk)
    b_block = next(block["index"] for block in live.blocks if block["phase"] == "B")
    await run_block(live, b_block, trials=2)
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    history = await harness.svc.store.consent_history(pk)
    assert len(history) == 2 and history[0].replaced_at is not None and history[1].replaced_at is None
    rows = await harness.svc.store.results_rows(pk, day_index=None)
    rules = await harness.svc.store.rules(pk)
    report = agency_audit(rows.decisions, rows.trials, rows.sessions, rows.burden, rules, history)
    assert report.counts["A6"] == 0, report.findings
    granted = history[1].record.granted_at
    adaptive = next(record for record in rows.decisions if record.mode.value == "adaptive")
    moved = dataclasses.replace(adaptive, created_at=granted - timedelta(minutes=1))
    tampered = [moved if record.decision_id == adaptive.decision_id else record for record in rows.decisions]
    failing = agency_audit(tampered, rows.trials, rows.sessions, rows.burden, rules, history)
    assert failing.counts["A6"] > 0


async def test_rule_history(harness):
    from pilot.coadaptation.core.audit import agency_audit

    device_rule = await harness.svc.set_preference(pref_params("forbid_variant", "ask_clarification:3"))
    pk = await new_participant(harness, ("selection_study", "research_export"))
    history = await harness.svc.store.rules(pk)
    assert [rule.pref_id for rule in history] == [device_rule["preference"]["pref_id"]]
    live = await begin(harness, pk)
    pin = await harness.svc.set_preference(pref_params("pin_act", True, participant_key=pk))
    assert pin["status"] == "ok"
    collapse = None
    for block in [b["index"] for b in live.blocks if b["phase"] == "A"]:
        rec, table = await decide_block(live, block)
        if table is not None:
            collapse = table
            break
    assert collapse is not None and collapse["mode"] == "pinned"
    assert any(cell["propensities"]["act"] == 1.0 for cell in collapse["cells"])
    harness.clock.advance(minutes=2)
    retracted_at = harness.clock.now
    await harness.svc.retract_preference({"pref_id": pin["preference"]["pref_id"]})
    await harness.svc.end_session({"session_id": live.session_id, "reason": "completed"})
    rules = {rule.pref_id: rule for rule in await harness.svc.store.rules(pk)}
    stored_pin = rules[pin["preference"]["pref_id"]]
    assert stored_pin.created_at < stored_pin.retracted_at == retracted_at
    harness.clock.advance(days=31)
    await harness.svc.set_preference(pref_params("forbid_variant", "offer_modality"))
    assert pin["preference"]["pref_id"] not in harness.preferences_path.read_text(encoding="utf-8")
    exported = await harness.svc.export({"participant_key": pk})
    assert exported["status"] == "ok"
    rows = await harness.svc.store.results_rows(pk, day_index=None)
    rule_history = await harness.svc.store.rules(pk)
    assert pin["preference"]["pref_id"] in {rule.pref_id for rule in rule_history}
    report = agency_audit(
        rows.decisions,
        rows.trials,
        rows.sessions,
        rows.burden,
        rule_history,
        await harness.svc.store.consent_history(pk),
    )
    assert report.counts["A2"] == 0 and report.counts["A5"] == 0, report.findings

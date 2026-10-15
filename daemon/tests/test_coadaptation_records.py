"""Stored records: labels, schemas and exact serialization (spec 13.1.2, 22.1).

The builders at the top are shared with the constraints, policy, engine, utility and study-plan tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from pilot.coadaptation.core.consent import CoAdaptationPurpose, ConsentPeriod, ConsentRecord
from pilot.coadaptation.core.engine import EngineEvent, EngineSettings
from pilot.coadaptation.core.evidence import EvidenceKind
from pilot.coadaptation.core.policy import PolicyMode
from pilot.coadaptation.core.records import (
    RECORD_SCHEMA_VERSION,
    RECORD_SCHEMAS,
    ActivationGaze,
    ActivationRecord,
    BurdenReport,
    DecisionRecord,
    EpisodeLabels,
    GazeSummary,
    HandSummary,
    L2Applied,
    MovementSummary,
    OfferResponse,
    RevertRecord,
    SessionRow,
    SpreadUsed,
    StoredTrial,
    TrialLabels,
    TrialRecord,
    VetoRecord,
    derive_episodes,
    derive_labels,
)
from pilot.coadaptation.core.state import (
    ActivationContext,
    BlockContext,
    ClockContext,
    DeviceContext,
    LayoutSummary,
    Rect,
)
from pilot.coadaptation.core.study_plan import build_session_plan
from pilot.coadaptation.core.target_posterior import compute_posterior, keyboard_endpoint, rendered_rects
from pilot.coadaptation.core.vocabulary import DecisionLevel

# ----------------------------------------------------------------------------------------------- shared builders

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
PK = "p-0000abcd"
SALT = "0123456789abcdef0123456789abcdef"
SURFACE = (700.0, 500.0)


def grid(
    w: float = 16.0, gap: float = 2.0, cols: int = 4, rows: int = 3, centre: tuple[float, float] = (350.0, 200.0)
) -> tuple[Rect, ...]:
    pitch = w + gap
    x0 = centre[0] - (cols - 1) * pitch / 2.0
    y0 = centre[1] - (rows - 1) * pitch / 2.0
    return tuple(Rect(cx=x0 + c * pitch, cy=y0 + r * pitch, w=w, h=w) for r in range(rows) for c in range(cols))


def menu(n: int = 6, origin: tuple[float, float] = (300.0, 60.0)) -> tuple[Rect, ...]:
    top = origin[1] + 16.0
    return tuple(Rect(cx=origin[0], cy=top + 12.0 + 24.0 * i, w=220.0, h=24.0) for i in range(n))


def layout_for(
    cands: tuple[Rect, ...], *, task: str = "pointing", origin: tuple[float, float] = (350.0, 460.0), gap: float = 2.0
) -> LayoutSummary:
    ids = [math.log2(math.hypot(c.cx - origin[0], c.cy - origin[1]) / min(c.w, c.h) + 1.0) for c in cands]
    return LayoutSummary(
        task=task,  # type: ignore[arg-type]
        n_candidates=len(cands),
        w_px=cands[0].w,
        h_px=cands[0].h,
        gap_px=gap,
        id_bar=sum(ids) / len(ids),
        menu_n=len(cands) if task == "menu" else 0,
        origin_x=origin[0],
        origin_y=origin[1],
    )


def device(modality: str = "mouse") -> DeviceContext:
    return DeviceContext(
        modality=modality,  # type: ignore[arg-type]
        pointer_types=() if modality == "keyboard" else ("mouse",),
        dpr=1.0,
        surface_w=SURFACE[0],
        surface_h=SURFACE[1],
        gaze_available=False,
        hand_pointer=modality == "hand_pointer",
    )


def engine_settings(**overrides: Any) -> EngineSettings:
    values: dict[str, Any] = {"thompson_draws": 32, "learn_across_sessions": True}
    values.update(overrides)
    return EngineSettings(**values)


def session_row(
    session_id: str = "s1",
    *,
    day_index: int = 740000,
    started_at: datetime = T0,
    modality: str = "mouse",
    settings: EngineSettings | None = None,
    session_number: int = 1,
    gaze_accuracy: float | None = None,
    comparator: tuple[str, ...] = (),
    participant_key: str = PK,
) -> SessionRow:
    settings = settings or engine_settings()
    return SessionRow(
        session_id=session_id,
        participant_key=participant_key,
        context="study",
        day_index=day_index,
        session_number=session_number,
        protocol_id=None,
        protocol_sha256=None,
        evidence_kind=EvidenceKind.SIMULATED,
        started_at=started_at,
        ended_at=None,
        end_reason=None,
        device=device(modality),
        plan=build_session_plan(
            session_number=session_number,
            adaptive_allowed=settings.adaptive_allowed and session_number > 1,
            comparator_available=bool(comparator),
            plan_seed=7,
            burden_every_blocks=1,
            n_arms=4 if modality == "keyboard" else 6,
        ),
        layout_seed=11,
        seed_salt=SALT,
        gaze_check_accuracy=gaze_accuracy,
        comparator_ranking=comparator,
        settings=settings.as_dict(),
        protocol_deviations=(),
        versions={"model": "coadapt-owm-v1"},
    )


def block_context(
    row: SessionRow,
    block_index: int,
    phase: str,
    layout: LayoutSummary,
    *,
    minute: float = 1.0,
    previous: str | None = None,
) -> BlockContext:
    return BlockContext(
        session_id=row.session_id,
        block_index=block_index,
        phase=phase,  # type: ignore[arg-type]
        task=layout.task,
        layouts=(layout,) * 8,
        device=row.device,
        clock=ClockContext(
            day_index=row.day_index, session_minute=minute, trials_today=0, days_since_last_session=None
        ),
        previous_regime=previous,
        hand_quality_recent=None,
    )


def pointer_activation(
    cands: tuple[Rect, ...],
    endpoint: tuple[int, int],
    *,
    index: int = 0,
    spread: SpreadUsed,
    press_index: int,
    modality: str = "mouse",
    expanded_index: int | None = None,
    scale: float = 1.0,
    result: str = "committed",
    outcome: str | None = None,
    l2: L2Applied | None = None,
    committed_index: int | None = -2,
    gaze: ActivationGaze | None = None,
    gaze_accuracy: float | None = None,
    irreversible_rule: bool = False,
    t_ms: float = 600.0,
) -> ActivationRecord:
    """An activation whose recorded posterior fields are exactly what the daemon recomputes."""
    from pilot.coadaptation.core.target_posterior import GazeInput

    rendered = rendered_rects(cands, expanded_index, scale)
    gaze_in = None if gaze is None else GazeInput(region=gaze.region, conf=gaze.conf, acc=float(gaze_accuracy or 0.0))
    post = compute_posterior(
        rendered,
        endpoint,
        modality=modality,
        r=spread.r,
        floor=spread.floor,
        surface=SURFACE,
        gaze=gaze_in,
        press_index=press_index,
    )
    hit = post.hit_index
    if committed_index == -2:
        committed_index = hit if result == "committed" else None
    return ActivationRecord(
        index=index,
        t_ms=t_ms,
        modality=modality,
        x=int(endpoint[0]),
        y=int(endpoint[1]),
        hit_index=hit,
        a_gate=post.ambiguity_gate,
        n_plausible=post.n_plausible,
        press_index=press_index,
        expanded_index=expanded_index,
        hit_rect=None if hit is None else rendered[hit],
        gaze=gaze,
        l2=l2,
        irreversible_rule=irreversible_rule,
        result=result,  # type: ignore[arg-type]
        outcome=(outcome or ("none" if result == "none" else "committed")),  # type: ignore[arg-type]
        committed_index=committed_index,
    )


def keyboard_activation(
    cands: tuple[Rect, ...], focused: int, *, index: int = 0, result: str = "committed", outcome: str | None = None
) -> ActivationRecord:
    x, y = keyboard_endpoint(cands[focused])
    return ActivationRecord(
        index=index,
        t_ms=800.0,
        modality="keyboard",
        x=x,
        y=y,
        hit_index=focused,
        a_gate=0.0,
        n_plausible=1,
        press_index=focused,
        expanded_index=None,
        hit_rect=cands[focused],
        gaze=None,
        l2=None,
        irreversible_rule=False,
        result=result,  # type: ignore[arg-type]
        outcome=(outcome or ("none" if result == "none" else "committed")),  # type: ignore[arg-type]
        committed_index=focused if result == "committed" else None,
    )


def trial_record(
    *,
    activations: tuple[ActivationRecord, ...],
    cands: tuple[Rect, ...],
    layout: LayoutSummary,
    regime: str = "assistance_off",
    decision_id: str = "d1",
    session_id: str = "s1",
    block_index: int = 1,
    trial_index: int = 0,
    intended: int | None = 5,
    issued_at: datetime = T0,
    recorded_at: datetime | None = None,
    spread: SpreadUsed | None = None,
    modality: str = "mouse",
    seq: int = 0,
    **overrides: Any,
) -> TrialRecord:
    committed = [a.committed_index for a in activations if a.outcome == "committed" and a.committed_index is not None]
    final = committed[-1] if committed else None
    task = "idle" if intended is None else layout.task
    values: dict[str, Any] = {
        "trial_id": f"{session_id}-{block_index}-{trial_index}-{seq}",
        "session_id": session_id,
        "block_index": block_index,
        "trial_index": trial_index,
        "task": task,
        "regime": regime,
        "regime_decision_id": decision_id,
        "table_decision_ids": (),
        "applied_regime": regime,
        "l1_fallback_reason": None,
        "fallbacks_issued_at": issued_at,
        "spread": spread,
        "layout": layout,
        "candidates": cands,
        "surface_w": SURFACE[0],
        "surface_h": SURFACE[1],
        "dpr": 1.0,
        "reversible": (True,) * len(cands),
        "cue_at": issued_at + timedelta(seconds=1),
        "intended_index": intended,
        "modality": modality,
        "activations": activations,
        "revert": None,
        "committed_index": final,
        "commit_t_ms": None if final is None else 1200.0,
        "commit_via": "direct" if final is not None else "none",
        "corrections": 0,
        "time_ms": 1200.0,
        "limit_ms": 30000.0 if intended is None else (15000.0 if layout.task == "menu" else 10000.0),
        "duration_ms": 1200.0,
        "movement": None,
        "veto": None,
        "offers": (),
        "hand": None,
        "gaze": None,
        "interrupted": False,
        "interruption_reason": None,
        "explanation_views": 0,
        "recorded_at": recorded_at or issued_at + timedelta(seconds=5),
        "seq": seq,
    }
    values.update(overrides)
    return TrialRecord(**values)


# ------------------------------------------------------------------------------------------------------- labels

CANDS = grid()
LAYOUT = layout_for(CANDS)
SPREAD = SpreadUsed(r=1.0, floor=1.0)


def _commit(index: int, *, committed: int, i: int = 0) -> ActivationRecord:
    target = CANDS[committed]
    return pointer_activation(
        CANDS, (int(target.cx), int(target.cy)), index=i, spread=SPREAD, press_index=committed, t_ms=500.0 + 100 * i
    )


def test_derive_labels_truth_table() -> None:
    correct = trial_record(activations=(_commit(0, committed=5),), cands=CANDS, layout=LAYOUT, spread=SPREAD)
    labels = derive_labels(correct)
    assert (labels.complete, labels.ua, labels.correction, labels.veto) == (True, False, False, False)
    assert labels.time_ms == 1200.0 and not labels.censored and labels.limit_ms == 10000.0

    wrong = trial_record(activations=(_commit(0, committed=6),), cands=CANDS, layout=LAYOUT, spread=SPREAD)
    labels = derive_labels(wrong)
    assert (labels.complete, labels.ua) == (False, True)
    assert labels.censored and labels.time_ms == 10000.0  # an uncompleted trial is censored at its limit

    undone = trial_record(
        activations=(_commit(0, committed=6), _commit(1, committed=5, i=1)),
        cands=CANDS,
        layout=LAYOUT,
        spread=SPREAD,
        corrections=1,
    )
    labels = derive_labels(undone)
    assert (labels.complete, labels.ua, labels.correction) == (True, True, True)

    target = CANDS[5]
    cancelled = pointer_activation(
        CANDS,
        (int(target.cx), int(target.cy)),
        spread=SPREAD,
        press_index=5,
        result="confirming",
        outcome="cancelled",
    )
    labels = derive_labels(trial_record(activations=(cancelled,), cands=CANDS, layout=LAYOUT, spread=SPREAD))
    assert (labels.complete, labels.ua, labels.correction) == (False, False, True)

    aborted = replace(cancelled, outcome="aborted")
    labels = derive_labels(trial_record(activations=(aborted,), cands=CANDS, layout=LAYOUT, spread=SPREAD))
    assert labels.correction is False  # an aborted hold is not a correction

    late = trial_record(
        activations=(_commit(0, committed=5),), cands=CANDS, layout=LAYOUT, spread=SPREAD, commit_t_ms=10500.0
    )
    assert derive_labels(late).complete is False  # committed after the limit

    vetoed = trial_record(
        activations=(_commit(0, committed=5),),
        cands=CANDS,
        layout=LAYOUT,
        spread=SPREAD,
        veto=VetoRecord(decision_id="d1", level=DecisionLevel.BLOCK_REGIME),
    )
    assert derive_labels(vetoed).veto is True


def test_derive_labels_idle_epochs_count_every_commit() -> None:
    idle_layout = replace(LAYOUT, task="idle")
    quiet = trial_record(activations=(), cands=CANDS, layout=idle_layout, intended=None, duration_ms=30000.0)
    labels = derive_labels(quiet)
    assert labels.idle and labels.idle_commits == 0 and not labels.ua and labels.idle_minutes == 0.5
    slip = trial_record(
        activations=(_commit(0, committed=2),), cands=CANDS, layout=idle_layout, intended=None, duration_ms=30000.0
    )
    labels = derive_labels(slip)
    assert labels.ua and labels.idle_commits == 1 and not labels.complete


def test_episodes_follow_each_l2_response() -> None:
    target = CANDS[5]
    l2 = L2Applied("t1", "b2:p2:h1", 0, "confirm_selection:second_click", "table", None)
    first = pointer_activation(
        CANDS,
        (int(target.cx + 8), int(target.cy + 5)),
        spread=SPREAD,
        press_index=5,
        result="confirming",
        outcome="cancelled",
        l2=l2,
        t_ms=400.0,
    )
    second = _commit(5, committed=5, i=1)
    trial = trial_record(activations=(first, second), cands=CANDS, layout=LAYOUT, spread=SPREAD, corrections=1)
    ctx = ActivationContext(0.5, "b2", 2, True, 0.5, 0.5, "mouse", LAYOUT)
    episodes = derive_episodes(trial, {0: ctx})
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.variant == "confirm_selection:second_click" and episode.cell_id == "b2:p2:h1"
    assert episode.complete and episode.correction and not episode.ua and not episode.veto
    assert episode.time_ms == pytest.approx(1200.0 - 400.0)


# ------------------------------------------------------------------------------------------------------ schemas


RECORD_CLASSES = (
    SpreadUsed,
    ActivationGaze,
    L2Applied,
    ActivationRecord,
    MovementSummary,
    OfferResponse,
    RevertRecord,
    VetoRecord,
    HandSummary,
    GazeSummary,
    TrialRecord,
    TrialLabels,
    EpisodeLabels,
    StoredTrial,
    BurdenReport,
    SessionRow,
    DecisionRecord,
)


def test_record_schemas_match_dataclass_fields() -> None:
    assert RECORD_SCHEMA_VERSION == "coadapt-records-v1"
    for cls in RECORD_CLASSES:
        assert list(RECORD_SCHEMAS[cls.__name__]) == [f.name for f in fields(cls)], cls.__name__
    for payload in ("RecommendationV1", "AmbiguityTableV1", "PredictionV1"):
        assert RECORD_SCHEMAS[payload]


def test_activation_contract() -> None:
    base = _commit(0, committed=5)
    assert isinstance(base.press_index, int)
    with pytest.raises(ValueError):
        replace(base, press_index=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replace(base, result="cancelled")  # type: ignore[arg-type]
    for result in ("committed", "confirming", "held", "asked"):
        for outcome in ("committed", "cancelled", "aborted", "open_at_end"):
            replace(base, result=result, outcome=outcome)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            replace(base, result=result, outcome="none")  # type: ignore[arg-type]
    replace(base, result="none", outcome="none", committed_index=None)
    with pytest.raises(ValueError):
        replace(base, result="none", outcome="committed")
    with pytest.raises(ValueError):
        replace(base, outcome="cancelled_late")  # type: ignore[arg-type]
    assert "plain_regime" not in RECORD_SCHEMAS["RevertRecord"] and "regime" in RECORD_SCHEMAS["RevertRecord"]
    no_spread = trial_record(activations=(), cands=CANDS, layout=LAYOUT, spread=None, modality="keyboard")
    assert no_spread.spread is None
    assert TrialRecord.from_dict(json.loads(json.dumps(no_spread.to_dict()))) == no_spread


# ------------------------------------------------------------------------------------------ seeded round trips


def _f(rng: np.random.Generator, scale: float = 1000.0) -> float:
    return float(rng.random() * scale - scale / 3)


def _dt(rng: np.random.Generator) -> datetime:
    tz = timezone(timedelta(minutes=int(rng.integers(-12 * 60, 14 * 60))))
    return (T0 + timedelta(microseconds=int(rng.integers(0, 10**12)))).astimezone(tz)


def _rect(rng: np.random.Generator) -> Rect:
    return Rect(_f(rng), _f(rng), float(rng.random() * 50 + 1), float(rng.random() * 50 + 1))


def random_records(rng: np.random.Generator) -> list[Any]:
    spread = SpreadUsed(r=float(rng.random() * 3 + 0.1), floor=float(rng.random() * 6))
    gaze = ActivationGaze(region=str(rng.choice(["center", "left", "right", "up", "down"])), conf=float(rng.random()))
    l2 = L2Applied("t", "b3:p1:h0", int(rng.integers(0, 5)), "wait_for_evidence", "fallback", "stale_table")
    cap = L2Applied("t", "b1:p2:h1", 0, "wait_for_evidence", "cap", None)
    hit = _rect(rng)
    activation = ActivationRecord(
        index=int(rng.integers(0, 31)),
        t_ms=_f(rng, 60000) % 60000,
        modality="pen",
        x=int(rng.integers(-100, 1000)),
        y=int(rng.integers(-100, 1000)),
        hit_index=3,
        a_gate=float(rng.random()),
        n_plausible=int(rng.integers(1, 4)),
        press_index=int(rng.integers(-1, 30)),
        expanded_index=None,
        hit_rect=hit,
        gaze=gaze,
        l2=l2 if rng.random() < 0.5 else cap,
        irreversible_rule=bool(rng.random() < 0.5),
        result="asked",
        outcome="open_at_end",
        committed_index=None,
    )
    movement = MovementSummary(_f(rng), _f(rng), 1.0 + rng.random(), 3, _f(rng), 1, _f(rng, 1), _f(rng, 1))
    offer = OfferResponse("o1", "offer_modality:keyboard", "dismissed", None)
    late_offer = OfferResponse("o2", "no_offer", "ignored", "late")
    revert = RevertRecord("refresh", 1, "confirm_selection:hold_700", _dt(rng))
    veto = VetoRecord("d", DecisionLevel.AMBIGUITY_RESPONSE)
    layout = replace(LAYOUT, id_bar=float(rng.random() * 5), origin_x=_f(rng), origin_y=_f(rng))
    trial = trial_record(
        activations=(activation,),
        cands=(hit, _rect(rng), _rect(rng), _rect(rng)),
        layout=layout,
        spread=spread,
        intended=1,
        issued_at=_dt(rng),
        recorded_at=_dt(rng),
        movement=movement,
        offers=(offer, late_offer),
        revert=revert,
        veto=veto,
        hand=HandSummary(float(rng.random()), float(rng.random()), 2),
        gaze=GazeSummary(agreed_with_intended=None),
        l1_fallback_reason="late",
        table_decision_ids=("a", "b"),
        modality="pen",
        corrections=2,
        time_ms=_f(rng, 60000) % 60000,
    )
    labels = derive_labels(trial)
    context = ActivationContext(
        float(rng.random()), "b2", 2, True, float(rng.random()), float(rng.random()), "pen", layout
    )
    episode = EpisodeLabels(
        0, "t", "ask_clarification:2", "b2:p2:h1", "table", context, True, False, True, False, _f(rng), True
    )
    stored = StoredTrial(trial, "ok", 740001, float(rng.random()), labels, (episode,), ("l1:act:forbidden",))
    burden = BurdenReport(
        "r1",
        "s1",
        4,
        2,
        "block",
        3,
        None,
        7,
        {"mental": 40, "physical": None, "temporal": 10, "performance": 0, "effort": 100, "frustration": 5},
        6,
        "act",
        ("intrusiveness",),
        _dt(rng),
    )
    row = replace(
        session_row(),
        started_at=_dt(rng),
        ended_at=_dt(rng),
        end_reason="completed",
        gaze_check_accuracy=float(rng.random()),
    )
    decision = DecisionRecord(
        decision_id="d",
        session_id="s1",
        participant_key=PK,
        seq=3,
        level=DecisionLevel.BLOCK_REGIME,
        mode=PolicyMode.ADAPTIVE,
        behavior="baseline",
        created_at=_dt(rng),
        superseded_by=None,
        payload={key: {"value": _f(rng), "nested": [1, None, "x", True]} for key in RECORD_SCHEMAS["RecommendationV1"]},
        applied="act",
        fallback_reason="late",
        versions={"policy": "coadapt-policy-v1"},
    )
    event = EngineEvent.create(
        kind="drift", session_id="s1", trial_id="t1", created_at=_dt(rng), head="l1", outcome="ua"
    )
    granted = _dt(rng)
    consent = ConsentRecord(
        participant_key=PK,
        purposes=frozenset({CoAdaptationPurpose.SELECTION_STUDY, CoAdaptationPurpose.HAND_POINTER}),
        consent_text_version="coadapt-consent-v1",
        consent_text_sha256="b" * 64,
        granted_at=granted,
        expires_at=granted + timedelta(days=30),
        retention_days=30,
        authorized=True,
    )
    period = ConsentPeriod(consent, replaced_at=granted + timedelta(days=2))
    return [
        spread,
        gaze,
        l2,
        cap,
        activation,
        movement,
        offer,
        late_offer,
        revert,
        veto,
        trial,
        labels,
        episode,
        stored,
        burden,
        row,
        decision,
        event,
        consent,
        period,
    ]


def test_seeded_records_round_trip_bit_identically() -> None:
    for seed in range(25):
        for record in random_records(np.random.default_rng(seed)):
            text = json.dumps(record.to_dict())
            again = type(record).from_dict(json.loads(text))
            assert again == record, type(record).__name__
            assert json.dumps(again.to_dict()) == text, type(record).__name__


def test_from_dict_rejects_unknown_missing_and_wrong_keys_without_echoing() -> None:
    secret = "secret-value-98765"
    for record in random_records(np.random.default_rng(99)):
        cls = type(record)
        good = record.to_dict()
        with pytest.raises(ValueError) as error:
            cls.from_dict({**good, "unexpected": secret})
        assert secret not in str(error.value)
        required = "participant_key" if cls is ConsentRecord else next(iter(good))  # schema_version has a default
        with pytest.raises(ValueError):
            cls.from_dict({key: value for key, value in good.items() if key != required})
        for key, value in good.items():
            if isinstance(value, str):
                with pytest.raises(ValueError) as error:
                    cls.from_dict({**good, key: [secret]})
                assert secret not in str(error.value)
                break


def test_a_layouts_mean_presses_survive_storage_and_older_rows_still_read() -> None:
    """A pointing layout's mean keyboard presses (the decision-time prior, 7.3) stay with the stored trial, so a block
    rebuilt from its trials predicts as the live decision did; a row without them encodes and reads as before."""
    keyboard = replace(LAYOUT, presses_mean=3.5)
    trial = trial_record(activations=(), cands=CANDS, layout=keyboard, spread=None, modality="keyboard")
    encoded = trial.to_dict()
    assert encoded["layout"]["presses_mean"] == 3.5
    again = TrialRecord.from_dict(json.loads(json.dumps(encoded)))
    assert again == trial and again.layout.presses_mean == 3.5
    context = ActivationContext(0.3, "b2", 2, True, 0.6, 0.6, "keyboard", keyboard)
    episode = EpisodeLabels(0, "t", "act", "b2:p2:h1", "table", context, True, False, False, False, 900.0, False)
    assert EpisodeLabels.from_dict(json.loads(json.dumps(episode.to_dict()))).context.layout == keyboard
    plain = trial_record(activations=(), cands=CANDS, layout=LAYOUT, spread=None, modality="keyboard")
    assert LAYOUT.presses_mean is None and "presses_mean" not in plain.to_dict()["layout"]
    assert TrialRecord.from_dict(plain.to_dict()) == plain
    for bad in ("3", True, None, 0.0):
        layout = {**plain.to_dict()["layout"], "presses_mean": bad}
        if bad is None:
            assert TrialRecord.from_dict({**plain.to_dict(), "layout": layout}) == plain
            continue
        with pytest.raises(ValueError):
            TrialRecord.from_dict({**plain.to_dict(), "layout": layout})


def test_wrongly_typed_nested_values_are_rejected() -> None:
    trial = trial_record(activations=(_commit(0, committed=5),), cands=CANDS, layout=LAYOUT, spread=SPREAD)
    good = trial.to_dict()
    for key, bad in (
        ("block_index", "1"),
        ("interrupted", 0),
        ("candidates", [{"cx": 1, "cy": 2, "w": 3}]),
        ("fallbacks_issued_at", "2026-10-15T09:00:00"),  # no UTC offset
        ("spread", {"r": True, "floor": 1.0}),
        ("time_ms", float("inf")),
    ):
        with pytest.raises(ValueError):
            TrialRecord.from_dict({**good, key: bad})
    activation = dict(good["activations"][0])
    activation["press_index"] = None
    with pytest.raises(ValueError):
        TrialRecord.from_dict({**good, "activations": [activation]})


def test_decision_record_payload_needs_the_rpc_keys() -> None:
    with pytest.raises(ValueError):
        DecisionRecord(
            decision_id="d",
            session_id="s",
            participant_key=PK,
            seq=0,
            level=DecisionLevel.AMBIGUITY_RESPONSE,
            mode=PolicyMode.RANDOMIZED,
            behavior="policy",
            created_at=T0,
            superseded_by=None,
            payload={"decision_id": "d"},
            applied=None,
            fallback_reason=None,
            versions={},
        )


def test_block_context_builder_is_valid() -> None:
    row = session_row()
    block = block_context(row, 1, "A", LAYOUT)
    assert isinstance(block, BlockContext) and len(block.layouts) == 8

"""Session plans, the comparator ranking and the protocol hash (spec 12.3, 22.1)."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from pilot.coadaptation.core import study_plan
from pilot.coadaptation.core.policy import PolicyMode
from pilot.coadaptation.core.priors import menu_sdp_time_s
from pilot.coadaptation.core.records import (
    RECORD_SCHEMAS,
    BurdenReport,
    DecisionRecord,
    SpreadUsed,
    StoredTrial,
    derive_labels,
)
from pilot.coadaptation.core.study_plan import (
    LAYOUT_GRID,
    PLAN_RULES_VERSION,
    PROTOCOL_HYPERPARAMETERS,
    STUDY_PROTOCOL_ID,
    SessionPlan,
    build_session_plan,
    phase_b_positions,
    plan_seed,
    rank_fixed_regimes,
    study_protocol_sha256,
)
from pilot.coadaptation.core.vocabulary import DecisionLevel
from tests.test_coadaptation_records import CANDS, LAYOUT, PK, SPREAD, T0, pointer_activation, trial_record


def _later(seed: int, *, comparator: bool = True, n_arms: int = 6, every: int = 1) -> SessionPlan:
    return build_session_plan(
        session_number=2,
        adaptive_allowed=True,
        comparator_available=comparator,
        plan_seed=seed,
        burden_every_blocks=every,
        n_arms=n_arms,
    )


def test_session_one_plan() -> None:
    plan = build_session_plan(
        session_number=1, adaptive_allowed=True, comparator_available=True, plan_seed=1, burden_every_blocks=1
    )
    assert plan.blocks[0].phase == "practice" and plan.blocks[0].index == 0 and not plan.blocks[0].burden_after
    main = plan.blocks[1:]
    assert len(main) == 12 and all(b.phase == "A" for b in main)
    assert [b.task for b in main] == ["menu", "pointing"] * 6  # even index pointing, odd index menu
    assert [b.index for b in main if b.idle_after] == [4, 10]
    assert all(b.burden_after for b in main) and all(b.trials == 8 for b in plan.blocks)
    assert plan.active_trials == 104 and plan.end_questionnaire
    no_adaptive = build_session_plan(
        session_number=3, adaptive_allowed=False, comparator_available=True, plan_seed=1, burden_every_blocks=1
    )
    assert [b.phase for b in no_adaptive.blocks[1:]] == ["A"] * 12


def test_later_session_plan_structure() -> None:
    for seed in range(200):
        for n_arms in (6, 4):
            plan = _later(seed, n_arms=n_arms)
            phases = [b.phase for b in plan.blocks]
            assert phases[0] == "practice" and phases[1 : 1 + n_arms] == ["A_lite"] * n_arms
            b_blocks = plan.blocks[1 + n_arms :]
            assert len(b_blocks) == 10
            for half in (b_blocks[:5], b_blocks[5:]):
                counts = Counter(b.phase for b in half)
                assert counts == {"B": 3, "probe": 1, "comparator": 1}
                assert half[0].phase == "B"  # B1 and B6 are adaptive
            probes = {b.task for b in b_blocks if b.phase == "probe"}
            comparators = {b.task for b in b_blocks if b.phase == "comparator"}
            assert probes == comparators == {"pointing", "menu"}
            assert [b.index for b in plan.blocks if b.idle_after] == [3, n_arms + 6]
            assert plan.active_trials == 8 * (1 + n_arms + 10)
    assert _later(5) == _later(5)
    without = _later(5, comparator=False)
    assert Counter(b.phase for b in without.blocks) == {"practice": 1, "A_lite": 6, "B": 8, "probe": 2}
    assert [b.phase == "probe" for b in without.blocks] == [b.phase == "probe" for b in _later(5).blocks]


def test_every_allowed_position_occurs_over_a_thousand_seeds() -> None:
    options = phase_b_positions(n_arms=6)
    seen = Counter()
    for seed in range(1000):
        plan = _later(seed)
        b_blocks = plan.blocks[7:]
        probe = tuple(i % 5 + 1 for i, b in enumerate(b_blocks) if b.phase == "probe")
        comparator = tuple(i % 5 + 1 for i, b in enumerate(b_blocks) if b.phase == "comparator")
        seen[(probe[0], comparator[0], probe[1], comparator[1])] += 1
        for position in probe + comparator:
            assert 2 <= position <= 5
    assert set(seen) == set(options)


def test_burden_schedule() -> None:
    plan = _later(3, every=3)
    assert [b.index for b in plan.blocks if b.burden_after] == [3, 6, 9, 12, 15]
    with pytest.raises(ValueError):
        _later(3, every=0)


def test_plan_seed_and_serialization() -> None:
    import hashlib

    assert plan_seed(PK, 2) == int.from_bytes(
        hashlib.sha256(f"{PK}|2|{PLAN_RULES_VERSION}".encode()).digest()[:8], "big"
    )
    assert plan_seed(PK, 2) != plan_seed(PK, 3)
    plan = _later(9)
    assert SessionPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    assert plan.to_dict()["blocks"][0] == {
        "index": 0,
        "phase": "practice",
        "task": "pointing",
        "trials": 8,
        "idle_after": False,
        "burden_after": False,
    }
    with pytest.raises(ValueError):
        SessionPlan.from_dict({**plan.to_dict(), "gaze": True})


def test_layout_grid_and_protocol_hash() -> None:
    menu = LAYOUT_GRID["menu"]
    assert menu["n_items"] == [6, 12, 16] and menu["item_h_px"] == 24 and menu["button_h_px"] == 32
    assert max(menu["n_items"]) * menu["item_h_px"] + menu["button_h_px"] <= LAYOUT_GRID["fit_px"][1]
    assert LAYOUT_GRID["pointing"] == {
        "w_px": [16, 28, 44],
        "gap_px": [2, 8],
        "d_px": [120, 240, 360],
        "grids": ["4x3", "6x2"],
    }
    assert STUDY_PROTOCOL_ID == "coadapt-study-v1" and PROTOCOL_HYPERPARAMETERS["thompson_draws"] == 256
    digest = study_protocol_sha256()
    assert len(digest) == 64 and digest == study_protocol_sha256()
    original = json.loads(json.dumps(LAYOUT_GRID))
    try:
        study_plan.LAYOUT_GRID["menu"]["n_items"] = [6, 12, 24]
        assert study_protocol_sha256() != digest
    finally:
        study_plan.LAYOUT_GRID["menu"]["n_items"] = original["menu"]["n_items"]
    assert study_protocol_sha256() == digest


# -------------------------------------------------------------------------------------------- comparator ranking


def _decision(decision_id: str, priorities: dict[str, float] | None) -> DecisionRecord:
    payload: dict[str, Any] = {key: None for key in RECORD_SCHEMAS["RecommendationV1"]}
    payload["priorities"] = priorities
    return DecisionRecord(
        decision_id=decision_id,
        session_id="s1",
        participant_key=PK,
        seq=0,
        level=DecisionLevel.BLOCK_REGIME,
        mode=PolicyMode.RANDOMIZED,
        behavior="policy",
        created_at=T0,
        superseded_by=None,
        payload=payload,
        applied=None,
        fallback_reason=None,
        versions={},
    )


def _rows(
    regime: str,
    block: int,
    n: int,
    *,
    slips: int = 0,
    time_ms: float = 1200.0,
    task: str = "pointing",
    modality: str = "mouse",
    interrupted: int = 0,
    corrections: int = 0,
) -> list[StoredTrial]:
    rows = []
    for t in range(n):
        target = CANDS[5 if t >= slips else 6]
        activation = pointer_activation(
            CANDS, (int(target.cx), int(target.cy)), spread=SPREAD, press_index=5 if t >= slips else 6
        )
        trial = trial_record(
            activations=(activation,),
            cands=CANDS,
            layout=replace(LAYOUT, task=task),
            spread=SPREAD,
            regime=regime,
            decision_id=f"d-{block}",
            block_index=block,
            trial_index=t,
            modality=modality,
            commit_t_ms=time_ms,
            interrupted=t < interrupted,
            seq=block * 100 + t,
            corrections=corrections,
        )
        rows.append(StoredTrial(trial, "ok", 740000, None, derive_labels(trial), (), ()))
    return rows


def test_rank_fixed_regimes_orders_by_realized_utility_under_logged_priorities() -> None:
    careful = {"fewer_mistakes": 1.0, "faster": 0.0, "fewer_interruptions": 0.5}
    hasty = {"fewer_mistakes": 0.0, "faster": 1.0, "fewer_interruptions": 0.5}
    rows = (
        _rows("act", 1, 8, time_ms=600.0, corrections=1)  # faster, a correction in every trial
        + _rows("confirm_selection:hold_700", 2, 8, time_ms=1800.0)  # slower, no correction
        + _rows("enlarge_target:1.5", 3, 7)  # too few trials
        + _rows("assistance_off", 4, 8)  # never ranked
        + _rows("enlarge_target:2.0", 5, 9, interrupted=2)  # only 7 non-interrupted trials
    )
    decisions = [_decision(f"d-{b}", careful) for b in range(1, 6)]
    ranking = rank_fixed_regimes(rows, [], decisions, modality="mouse")
    assert ranking == ("confirm_selection:hold_700", "act")
    decisions = [_decision(f"d-{b}", hasty) for b in range(1, 6)]
    assert rank_fixed_regimes(rows, [], decisions, modality="mouse") == ("act", "confirm_selection:hold_700")
    assert rank_fixed_regimes(rows, [], decisions, modality="keyboard") == ()
    assert rank_fixed_regimes([], [], [], modality="mouse") == ()


def test_rank_fixed_regimes_breaks_ties_toward_caution_and_reads_burden() -> None:
    rows = _rows("act", 1, 8) + _rows("enlarge_target:2.0", 2, 8)
    decisions = [_decision("d-1", None), _decision("d-2", None)]
    assert rank_fixed_regimes(rows, [], decisions, modality="mouse") == ("enlarge_target:2.0", "act")
    burden = [
        BurdenReport("r1", "s1", 50, 2, "block", 7, 7, 2, None, None, None, (), T0 + timedelta(minutes=5)),
        BurdenReport("r2", "s1", 51, 1, "block", 2, 1, 6, None, None, None, (), T0 + timedelta(minutes=6)),
    ]
    assert rank_fixed_regimes(rows, burden, decisions, modality="mouse") == ("act", "enlarge_target:2.0")


def test_the_physics_t_ref_fallback_uses_the_constraining_side_of_a_menu_item() -> None:
    """Section 21.2's fallback T_ref is the physics median for assistance_off (7.3), whose target width is the
    constraining dimension min(w, h): a 220 x 24 menu item is as hard to reach along the menu as a 24 px square."""
    from tests.test_coadaptation_records import layout_for, menu

    items = menu(12)
    origin = (300.0, 60.0)
    layout = layout_for(items, task="menu", origin=origin, gap=0.0)
    trial = trial_record(activations=(), cands=items, layout=layout, intended=11)
    squares = tuple(replace(item, w=24.0) for item in items)
    square_trial = trial_record(activations=(), cands=squares, layout=layout, intended=11)
    distance = 292.0  # from the menu button to the centre of the twelfth item
    novice = menu_sdp_time_s(12, 0.0) * 1000.0
    expected = 1.3 * (250.0 + 100.0 + 230.0 * math.log2(distance / 24.0 + 1.0) + novice) / 1000.0
    for item_trial in (trial, square_trial):
        assert study_plan.physics_reference_time_s(item_trial, "mouse") == pytest.approx(expected, rel=1e-12)
    # Without assistance_off menu trials the fallback is what reference_times_s reports for menus.
    stored = StoredTrial(trial, "ok", 740000, None, derive_labels(trial), (), ())
    assert study_plan.reference_times_s([stored], modality="mouse") == {"menu": pytest.approx(expected)}

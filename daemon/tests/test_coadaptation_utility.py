"""Utilities of the policy, realized utility and pointing throughput (spec 9.1, 21.2, 12.7, 22.1)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pilot.coadaptation.core.records import MovementSummary, TrialLabels
from pilot.coadaptation.core.utility import (
    PROTOCOL_WEIGHTS,
    UtilityWeights,
    block_burden_term,
    l1_utility,
    l2_utility,
    l3_utility,
    pointing_throughput,
    realized_utility,
)
from tests.test_coadaptation_records import CANDS, LAYOUT, SPREAD, pointer_activation, trial_record


def test_weight_relations_hold_on_the_whole_slider_grid() -> None:
    grid = np.linspace(0.0, 1.0, 21)
    for m in grid:
        for f in grid:
            for i in grid:
                w = UtilityWeights.from_priorities(float(m), float(f), float(i))
                assert w.w_ua >= 2.0 * w.w_t and w.w_ua >= 1.0 and w.w_c == 1.0
    with pytest.raises(ValueError):
        UtilityWeights.from_priorities(1.2, 0.5, 0.5)


def test_time_for_slips_trade_off_at_the_corners_and_defaults() -> None:
    assert UtilityWeights.from_priorities(1.0, 0.0, 0.5).speed_for_slips() == pytest.approx(0.052, abs=5e-4)
    assert PROTOCOL_WEIGHTS.speed_for_slips() == pytest.approx(0.104, abs=5e-4)
    assert UtilityWeights.from_priorities(0.0, 1.0, 0.5).speed_for_slips() == pytest.approx(0.208, abs=5e-4)


def _u1(variant: str = "act", **overrides: Any) -> float:
    values = {
        "p_complete": 0.95,
        "p_ua": 0.05,
        "p_correction": 0.1,
        "p_veto": 0.02,
        "log_time_s": math.log(1.2),
        "effort": 3.0,
        "intrusiveness": 2.0,
        "weights": PROTOCOL_WEIGHTS,
        "t_ref_s": 1.0,
        "previous_regime": variant,
        "switching_penalty": 0.1,
    }
    values.update(overrides)
    return float(l1_utility(variant, **values))


def test_utilities_are_monotone_in_each_outcome() -> None:
    base = _u1()
    assert _u1(p_complete=0.99) > base
    assert _u1(p_ua=0.10) < base
    assert _u1(p_correction=0.2) < base
    assert _u1(p_veto=0.1) < base
    assert _u1(log_time_s=math.log(2.0)) < base
    assert _u1(effort=5.0) < base and _u1(intrusiveness=5.0) < base
    assert _u1(previous_regime="enlarge_target:1.5") == pytest.approx(base - 0.1)
    l2_base = l2_utility(
        "ask_clarification:2",
        p_complete=0.9,
        p_ua=0.01,
        p_correction=0.1,
        p_veto=0.02,
        log_time_s=0.5,
        weights=PROTOCOL_WEIGHTS,
    )
    assert l2_utility(
        "wait_for_evidence",
        p_complete=0.9,
        p_ua=0.01,
        p_correction=0.1,
        p_veto=0.02,
        log_time_s=0.5,
        weights=PROTOCOL_WEIGHTS,
    ) == pytest.approx(l2_base + 0.3 * PROTOCOL_WEIGHTS.w_b)
    assert l3_utility("no_offer", p_accept=0.9, p_dismiss=0.1, weights=PROTOCOL_WEIGHTS) == 0.0
    assert l3_utility(
        "offer_modality:keyboard", p_accept=0.5, p_dismiss=0.2, weights=PROTOCOL_WEIGHTS
    ) == pytest.approx(0.3 * 0.5 - 0.5 * PROTOCOL_WEIGHTS.w_b * 0.2)
    # arrays of Thompson draws broadcast
    draws = l1_utility(
        "act",
        p_complete=np.array([0.9, 0.95]),
        p_ua=np.array([0.1, 0.05]),
        p_correction=0.1,
        p_veto=0.02,
        log_time_s=0.2,
        effort=np.array([3.0, 3.0]),
        intrusiveness=2.0,
        weights=PROTOCOL_WEIGHTS,
        t_ref_s=1.0,
        previous_regime=None,
        switching_penalty=0.1,
    )
    assert draws.shape == (2,) and draws[1] > draws[0]


def test_intrusiveness_is_one_without_help() -> None:
    off = _u1("assistance_off", intrusiveness=6.0)
    assert off == pytest.approx(_u1("assistance_off", intrusiveness=1.0))
    assert block_burden_term(4.0, None, regime="assistance_off") == pytest.approx(-1.0)
    assert block_burden_term(4.0, 7.0, regime="assistance_off") == pytest.approx(-1.0)


def _labels(complete: bool, **overrides: Any) -> TrialLabels:
    values = {
        "complete": complete,
        "ua": False,
        "correction": False,
        "veto": False,
        "time_ms": 2000.0 if complete else 10000.0,
        "censored": not complete,
        "limit_ms": 10000.0,
        "idle": False,
        "idle_commits": 0,
        "idle_minutes": 0.0,
    }
    values.update(overrides)
    return TrialLabels(**values)


def test_realized_utility_reads_observed_outcomes_only() -> None:
    w = PROTOCOL_WEIGHTS
    done = realized_utility(_labels(True), time_s=None, t_ref_s=1.0, burden_term=0.0, weights=w)
    assert done == pytest.approx(1.0 - w.w_t * math.log(2.0))
    # an uncompleted trial counts at its limit, whatever time is passed
    missed = realized_utility(_labels(False), time_s=0.5, t_ref_s=1.0, burden_term=0.0, weights=w)
    assert missed == pytest.approx(-w.w_t * math.log(10.0))
    mapping = {"complete": False, "ua": True, "correction": True, "veto": True}
    with_limit = realized_utility(mapping, time_s=15.0, t_ref_s=1.0, burden_term=0.5, weights=w)
    assert with_limit == pytest.approx(-w.w_ua - w.w_k - w.w_t * math.log(15.0) - w.w_b * 0.5 - w.w_v)
    # no switching term: the same labels score the same whatever regime came before
    assert "previous" not in realized_utility.__code__.co_varnames
    with pytest.raises(ValueError):
        realized_utility(
            {"complete": False, "ua": False, "correction": False, "veto": False},
            time_s=None,
            t_ref_s=1.0,
            burden_term=0.0,
            weights=w,
        )


def test_block_burden_term_imputation_order() -> None:
    assert block_burden_term(5.0, 3.0, regime="act") == pytest.approx((5 - 4) / 3 + (3 - 4) / 3)
    session = {"effort": 6.0, "intrusiveness": 2.0}
    overall = {"effort": 1.0, "intrusiveness": 7.0}
    assert block_burden_term(None, None, regime="act", session_means=session, all_means=overall) == pytest.approx(
        (6 - 4) / 3 + (2 - 4) / 3
    )
    assert block_burden_term(None, None, regime="act", session_means={"effort": None}, all_means=overall) == (
        pytest.approx((1 - 4) / 3 + (7 - 4) / 3)
    )
    assert block_burden_term(None, None, regime="act") == pytest.approx(0.0)  # the scale midpoint 4


# --------------------------------------------------------------------------------------------------- throughput


def _fake_trial(w: float, d: float, offset_norm: float, mt_s: float) -> SimpleNamespace:
    """A minimal pointing trial: origin (0, 0), target at distance d along x, W-normalised endpoint offset."""
    target = SimpleNamespace(cx=d, cy=0.0, w=w, h=w)
    return SimpleNamespace(
        task="pointing",
        interrupted=False,
        intended_index=0,
        activations=(SimpleNamespace(modality="mouse"),),
        movement=SimpleNamespace(mt_ms=mt_s * 1000.0, endpoint_dx_norm=offset_norm, endpoint_dy_norm=0.0),
        candidates=(target,),
        layout=SimpleNamespace(origin_x=0.0, origin_y=0.0),
    )


def _iso_per_condition(trials: list[SimpleNamespace]) -> float:
    by_condition: dict[tuple[float, float], list[SimpleNamespace]] = {}
    for trial in trials:
        target = trial.candidates[0]
        by_condition.setdefault((target.w, target.cx), []).append(trial)
    values = []
    for (w, d), items in by_condition.items():
        offsets_px = np.array([t.movement.endpoint_dx_norm * w for t in items])
        we = 4.133 * float(np.std(offsets_px, ddof=1))
        values.extend(math.log2(d / we + 1.0) / (t.movement.mt_ms / 1000.0) for t in items)
    return float(np.mean(values))


def test_throughput_equals_the_iso_computation_for_one_condition() -> None:
    rng = np.random.default_rng(1)
    trials = [_fake_trial(28.0, 240.0, float(rng.normal(0, 0.25)), float(rng.uniform(0.5, 1.0))) for _ in range(40)]
    assert pointing_throughput(trials) == pytest.approx(_iso_per_condition(trials), rel=1e-12)


def test_throughput_pools_mixed_widths_without_inflating_the_effective_width() -> None:
    rng = np.random.default_rng(2)
    trials = []
    for w in (16.0, 28.0, 44.0):
        for d in (120.0, 240.0, 360.0):
            for _ in range(1000):
                trials.append(_fake_trial(w, d, float(rng.normal(0, 0.24)), float(rng.uniform(0.4, 1.2))))
    pooled = pointing_throughput(trials)
    per_condition = _iso_per_condition(trials)
    assert pooled == pytest.approx(per_condition, rel=0.03)
    raw = np.array([t.movement.endpoint_dx_norm * t.candidates[0].w for t in trials])
    we_raw = 4.133 * float(np.std(raw, ddof=1))
    naive = float(np.mean([math.log2(t.candidates[0].cx / we_raw + 1) / (t.movement.mt_ms / 1000) for t in trials]))
    assert abs(naive - per_condition) / per_condition > 0.03  # one SD over raw offsets of mixed W is biased


def test_throughput_skips_keyboard_interrupted_and_menu_trials() -> None:
    target = CANDS[5]
    activation = pointer_activation(CANDS, (int(target.cx), int(target.cy)), spread=SPREAD, press_index=5)
    movement = MovementSummary(200.0, 600.0, 1.1, 1, 900.0, 0, 0.1, -0.05)
    real = trial_record(activations=(activation,), cands=CANDS, layout=LAYOUT, spread=SPREAD, movement=movement)
    other = trial_record(
        activations=(activation,),
        cands=CANDS,
        layout=LAYOUT,
        spread=SPREAD,
        movement=MovementSummary(200.0, 700.0, 1.1, 1, 900.0, 0, -0.12, 0.02),
    )
    assert pointing_throughput([real]) is None  # one trial has no scatter estimate
    assert pointing_throughput([real, other]) is not None
    stored = SimpleNamespace(trial=real), SimpleNamespace(trial=other)
    assert pointing_throughput(stored) == pointing_throughput([real, other])
    keyboard = _fake_trial(28.0, 240.0, 0.1, 0.5)
    keyboard.activations = (SimpleNamespace(modality="keyboard"),)
    interrupted = _fake_trial(28.0, 240.0, 0.1, 0.5)
    interrupted.interrupted = True
    menu_trial = _fake_trial(28.0, 240.0, 0.1, 0.5)
    menu_trial.task = "menu"
    assert pointing_throughput([keyboard, interrupted, menu_trial]) is None

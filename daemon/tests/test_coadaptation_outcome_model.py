"""The co-adaptation world model (spec sections 7 and 8): priors, learning, dynamics, variants and snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from pilot.coadaptation.core.dynamics import Q_SLOW
from pilot.coadaptation.core.outcome_model import (
    FILTER_KEYS,
    MODEL_VERSION,
    PRIOR_VERSION,
    CalibrationMonitor,
    CoAdaptationWorldModel,
    EpisodeOutcome,
    ModelVariant,
    PopulationPrior,
    TrialContext,
    TrialOutcome,
    beta_indices,
    prior_covariance,
    write_npz,
)
from pilot.coadaptation.core.priors import (
    l1_physics_prediction,
    l1_physics_update,
    menu_sdp_time_s,
    practice_multiplier,
)
from pilot.coadaptation.core.special import norm_ppf
from pilot.coadaptation.core.state import (
    ALLOWED_CONTEXT_KEYS,
    BURDEN_DIM,
    BURDEN_FEATURE_NAMES,
    DENIED_FEATURE_SUBSTRINGS,
    L1_DIM,
    L1_FEATURE_NAMES,
    L2_DIM,
    L2_FEATURE_NAMES,
    ActivationContext,
    BlockContext,
    ClockContext,
    DeviceContext,
    LayoutSummary,
    OfferContext,
    Rect,
    build_burden_features,
    build_l1_features,
    build_l2_features,
    day_index_for,
)
from pilot.coadaptation.core.vocabulary import L2_OPTIONS, VARIANTS, DecisionLevel, l1_arms

DAEMON = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
DAY = 740_000
MOUSE = DeviceContext("mouse", ("mouse",), 1.0, 700.0, 500.0, False, False)
HAND = DeviceContext("hand_pointer", (), 1.0, 700.0, 500.0, False, True)
POINTING = LayoutSummary("pointing", 12, 28.0, 28.0, 8.0, 3.2, 0, 350.0, 470.0)
MENU = LayoutSummary("menu", 12, 220.0, 24.0, 0.0, 2.6, 12, 350.0, 40.0)
GRID = tuple(Rect(296.0 + 36 * (i % 4), 164.0 + 36 * (i // 4), 28.0, 28.0) for i in range(12))
MENU_ITEMS = tuple(Rect(350.0, 68.0 + 24 * i, 220.0, 24.0) for i in range(12))
ARMS = l1_arms("mouse")


def _model(**kw) -> CoAdaptationWorldModel:
    model = CoAdaptationWorldModel(**kw)
    model.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    return model


def _block(layouts=(POINTING,), previous: str | None = "assistance_off", device: DeviceContext = MOUSE) -> BlockContext:
    return BlockContext(
        "s1", 1, "A", "pointing", tuple(layouts), device, ClockContext(DAY, 5.0, 10, None), previous, None
    )


def _ctx(i: int, variant: str, at: datetime, *, menu: bool = False, device: DeviceContext = MOUSE) -> TrialContext:
    layout, cands = (MENU, MENU_ITEMS) if menu else (POINTING, GRID)
    return TrialContext(at, layout, cands, i % 12, variant, device, float(i % 60), None, 0.25)


def _outcome(rng: np.random.Generator) -> TrialOutcome:
    complete = bool(rng.random() < 0.95)
    return TrialOutcome(
        complete=complete,
        ua=bool(rng.random() < 0.05),
        correction=bool(rng.random() < 0.1),
        veto=bool(rng.random() < 0.02),
        log_time=math.log(rng.uniform(0.8, 2.5)) if complete else math.log(10.0),
        censored=not complete,
    )


def _act_ctx(modality: str = "mouse") -> ActivationContext:
    return ActivationContext(0.5, "b2", 2, True, 0.5, 0.5, modality, POINTING)


def _feed(model: CoAdaptationWorldModel, start: datetime, n: int, seed: int = 0) -> None:
    rng = np.random.Generator(np.random.PCG64(seed))
    for i in range(n):
        at = start + timedelta(seconds=20 * i)
        variant = ARMS[(i // 8) % len(ARMS)]
        ctx = _ctx(i, variant, at, menu=bool(i % 3 == 0))
        model.observe_l1_trial(ctx, _outcome(rng), regime_ucb_unintended=0.04)
        if variant == "act" and i % 4 == 0:
            model.observe_l2_episode(
                at, _act_ctx(), "wait_for_evidence", EpisodeOutcome(True, False, False, False, 0.2, False)
            )
        target = ctx.candidates[ctx.target_index]
        if ctx.layout.task != "menu" and not variant.startswith("enlarge"):
            model.observe_spread("mouse", target.cx + rng.normal(0, 6), target.cy + rng.normal(0, 6), target)
        if i % 8 == 7:
            model.observe_burden(at, variant, "mouse", ipt=0.2, ratings={"effort": 3, "intrusiveness": 2, "control": 5})


# --- state and features (spec 5) ----------------------------------------------------------------------------------


def test_feature_vectors_are_ordered_finite_and_bounded() -> None:
    assert (len(L1_FEATURE_NAMES), len(L2_FEATURE_NAMES), len(BURDEN_FEATURE_NAMES)) == (L1_DIM, L2_DIM, BURDEN_DIM)
    assert (L1_DIM, L2_DIM, BURDEN_DIM) == (18, 12, 10)
    names = L1_FEATURE_NAMES + L2_FEATURE_NAMES + BURDEN_FEATURE_NAMES
    keys = set().union(*ALLOWED_CONTEXT_KEYS.values())
    assert not [n for n in names + tuple(keys) if any(bad in n for bad in DENIED_FEATURE_SUBSTRINGS)]
    model = _model()
    _feed(model, T0, 300)
    practice = model.practice()
    rng = np.random.Generator(np.random.PCG64(12))
    for _ in range(300):
        layout = LayoutSummary(
            str(rng.choice(["pointing", "menu"])),
            12,
            28.0,
            28.0,
            float(rng.uniform(0, 20)),
            float(rng.uniform(0, 12)),
            12,
            350.0,
            470.0,
        )
        device = HAND if rng.random() < 0.5 else MOUSE
        clock = ClockContext(DAY, float(rng.uniform(0, 300)), 0, None)
        for variant in l1_arms(device.modality):
            age = None if rng.random() < 0.3 else int(rng.integers(0, 40))
            f = build_l1_features(layout, variant, device, clock, practice, age, float(rng.uniform(0, 1)))
            assert f.shape == (18,) and np.all(np.isfinite(f)) and np.all(np.abs(f) <= 2.0)
            assert f[0] == f[15] == 1.0 and f[17] == f[3]
            fb = build_burden_features(variant, float(rng.uniform(0, 10)), practice, device.modality)
            assert fb.shape == (10,) and np.all((fb >= 0.0) & (fb <= 1.5))
        ctx = ActivationContext(float(rng.uniform(0, 1)), "b3", 2, True, 0.4, 0.5, device.modality, layout)
        for variant in L2_OPTIONS:
            f2 = build_l2_features(ctx, variant, practice)
            assert f2.shape == (12,) and np.all(np.isfinite(f2)) and np.all(np.abs(f2) <= 1.5)
            assert f2[11] == (0.0 if VARIANTS[variant].acting else 1.0)


def test_allowed_context_keys() -> None:
    block = {"block_index", "layouts", "previous_regime", "session_minute", "trials_today", "hand_quality_recent"}
    assert ALLOWED_CONTEXT_KEYS[DecisionLevel.BLOCK_REGIME] == block
    assert ALLOWED_CONTEXT_KEYS[DecisionLevel.AMBIGUITY_RESPONSE] == block | {
        "block_decision_id",
        "start_trial_index",
        "trials_in_block",
    }
    assert ALLOWED_CONTEXT_KEYS[DecisionLevel.OFFER] == {
        "block_index",
        "trial_index",
        "task",
        "struggle_kind",
        "keyboard_available",
    }
    for keys in ALLOWED_CONTEXT_KEYS.values():
        assert not keys & {"intended_index", "target", "cue", "is_idle", "nogo"}


def test_state_types_round_trip_and_reject_bad_input() -> None:
    block = _block((POINTING, MENU))
    offer = OfferContext("s1", 2, 3, "hand_pointer", "menu", "tracking_loss", True)
    items = (GRID[0], POINTING, MOUSE, block.clock, block, _act_ctx(), offer)
    for item in items:
        assert type(item).from_dict(json.loads(json.dumps(item.to_dict()))) == item
    good = POINTING.to_dict()
    for broken in (
        {**good, "intended_index": 3},
        {k: v for k, v in good.items() if k != "id_bar"},
        {**good, "n_candidates": 12.0},
        {**good, "task": "approval"},
        {**good, "w_px": True},
    ):
        with pytest.raises(ValueError) as err:
            LayoutSummary.from_dict(broken)
        assert "approval" not in str(err.value) and "12.0" not in str(err.value)


# --- priors and predictions -----------------------------------------------------------------------------------------


def test_prior_only_predictions_equal_physics() -> None:
    model = _model()
    for layout in (POINTING, MENU):
        block = _block((layout,))
        preds = model.predict_l1_block(block, ARMS, now=T0, tau=0.25, credibility=0.95)
        for variant in ARMS:
            phys = l1_physics_prediction(
                layout, variant, modality="mouse", n_item_eff_mean=0.0, n_variant=0, r=1.0, tau=0.25
            )
            pred = preds[variant]
            assert pred.time_ms.mean == pytest.approx(phys.time_ms, rel=1e-12)
            # The 90% bounds are symmetric around the physics latent on the probit scale.
            for outcome, estimate in (("ua", pred.p_unintended), ("complete", pred.p_complete), ("veto", pred.p_veto)):
                assert norm_ppf(estimate.lo90) + norm_ppf(estimate.hi90) == pytest.approx(2 * phys.h(outcome), abs=1e-9)
            assert pred.evidence.source == "literature_prior" and pred.evidence.trials_person == 0
            assert pred.authority == "presentation_only"
        assert preds["assistance_off"].burden["intrusiveness"] == 1.0
        assert preds["act"].burden["intrusiveness"] != 1.0


def test_prediction_dict_is_prediction_v1() -> None:
    model = _model()
    pred = model.predict_l1_block(_block(), ["act"], now=T0, tau=0.25, credibility=0.95)["act"]
    d = pred.to_dict()
    assert list(d) == [
        "variant",
        "p_complete",
        "p_unintended",
        "ucb_unintended",
        "p_correction",
        "p_veto",
        "time_ms",
        "burden",
        "evidence",
        "authority",
    ]
    assert d["authority"] == "presentation_only" and set(d["burden"]) == {"effort", "intrusiveness", "control"}
    assert set(d["evidence"]) == {"trials_person", "trials_today", "source"}
    assert type(pred).from_dict(json.loads(json.dumps(d))) == pred
    l2 = model.predict_l2(_act_ctx(), list(L2_OPTIONS[1:]), now=T0, credibility=0.95)
    assert all(p.burden is None for p in l2.values())


def test_a_time_contrast_carries_the_uncertainty_of_the_medians_not_trial_noise() -> None:
    """An explanation's delta_ci90 for time is a range for the difference of two predicted medians (section 9.8): the
    parameter uncertainty alone, as for slips and corrections. The (lo90, hi90) of a time estimate is a single-trial
    predictive range that also holds sigma_T; built from it the range came out about four times too wide."""
    from pilot.coadaptation.core.explain import _contrast, build_explanation
    from pilot.coadaptation.core.outcome_model import OutcomeEstimate, _time_estimate

    z90 = 1.6448536269514722
    one = _time_estimate([(0.0, 0.01)])
    assert (one.lo90, one.hi90) == pytest.approx((549.5, 1819.8), abs=0.1)  # PredictionV1 keeps this range
    assert one.credible90 == pytest.approx((1000.0 * math.exp(-z90 * 0.1), 1000.0 * math.exp(z90 * 0.1)))
    assert list(one.to_dict()) == ["mean", "lo90", "hi90"] and OutcomeEstimate.from_dict(one.to_dict()) == one
    lo, hi = _contrast("time", "act", "assistance_off", one, one).delta_ci90
    rng = np.random.Generator(np.random.PCG64(20261015))
    medians = 1000.0 * (np.exp(rng.normal(0.0, 0.1, 400_000)) - np.exp(rng.normal(0.0, 0.1, 400_000)))
    q05, q95 = np.quantile(medians, [0.05, 0.95])
    assert lo == pytest.approx(q05, rel=0.05) and hi == pytest.approx(q95, rel=0.05)
    assert hi < 300.0  # about 898 ms from the predictive ranges

    # The same holds for the explanation of a live decision built from the model's predictions.
    model = _model()
    hold = "confirm_selection:hold_700"
    preds = model.predict_l1_block(_block(), [hold, "assistance_off"], now=T0, tau=0.25, credibility=0.95)
    explanation = build_explanation(
        mode="adaptive", chosen=hold, predictions=preds, excluded={}, baseline="assistance_off"
    )
    time = next(contrast for contrast in explanation.contrasts if contrast.metric == "time")

    def sd(bounds: tuple[float, float]) -> float:
        return (bounds[1] - bounds[0]) / (2.0 * z90)

    a, b = preds[hold].time_ms, preds["assistance_off"].time_ms
    assert a.credible90 is not None and b.credible90 is not None
    half = z90 * math.hypot(sd(a.credible90), sd(b.credible90))
    assert (time.delta_ci90[1] - time.delta_ci90[0]) / 2.0 == pytest.approx(half, rel=1e-12)
    assert half < z90 * math.hypot(sd((a.lo90, a.hi90)), sd((b.lo90, b.hi90)))


def test_ucb_uses_the_credibility() -> None:
    model = _model()
    low = model.predict_l1_block(_block(), ["act"], now=T0, tau=0.25, credibility=0.8)["act"]
    high = model.predict_l1_block(_block(), ["act"], now=T0, tau=0.25, credibility=0.99)["act"]
    assert low.ucb_unintended < high.ucb_unintended
    assert low.p_unintended == high.p_unintended


def test_strategy_effects_are_recovered_with_calibrated_intervals() -> None:
    """True per-person effects drawn from the prior; the 90% intervals of the learned effects cover the truth."""
    hits = {("l1", "log_time"): 0, ("l1", "ua"): 0}
    effects = {("l1", "log_time"): 3, ("l1", "ua"): 2}  # g_conf on time, g_enl on slips
    seeds = 200
    for seed in range(seeds):
        rng = np.random.Generator(np.random.PCG64(seed))
        model = _model()
        truth = {k: rng.multivariate_normal(np.zeros(18), prior_covariance(*k)) for k in hits}
        for i in range(48):
            variant = ARMS[i % len(ARMS)]
            ctx = _ctx(i, variant, T0)
            h, f = model._l1_update_terms(ctx)
            log_time = h["log_time"] + f @ truth[("l1", "log_time")] + rng.normal(0.0, 0.35)
            ua = rng.random() < 0.5 * math.erfc(-(h["ua"] + f @ truth[("l1", "ua")]) / math.sqrt(2))
            model.observe_l1_trial(ctx, TrialOutcome(True, bool(ua), False, False, float(log_time), False))
        for key, idx in effects.items():
            flt = model.filters[key]
            half = 1.6448536269514722 * math.sqrt(flt.P[idx, idx])
            hits[key] += int(abs(flt.m[idx] - truth[key][idx]) <= half)
    for key, count in hits.items():
        assert 0.85 <= count / seeds <= 0.95, key


# --- dynamics -----------------------------------------------------------------------------------------------------


def _gamma_block_at_prior(model: CoAdaptationWorldModel) -> bool:
    for (head, outcome), flt in model.filters.items():
        g = flt.gamma_idx
        p0 = prior_covariance(head, outcome)
        if not (np.array_equal(flt.P[np.ix_(g, g)], p0[np.ix_(g, g)]) and np.all(flt.m[g] == 0.0)):
            return False
    return True


def test_day_boundary_resets_gamma_and_inflates_beta() -> None:
    for days, inflation in ((3, 3), (45, 30)):
        model = _model()
        _feed(model, T0, 40)
        assert not _gamma_block_at_prior(model)
        before = {k: model.filters[k].P.copy() for k in FILTER_KEYS}
        model.begin_session(day_index=DAY + days, device=MOUSE, learn_across_sessions=True)
        assert _gamma_block_at_prior(model)
        for key, flt in model.filters.items():
            b = beta_indices(key[0])
            expected = np.diag(before[key])[b] + Q_SLOW[flt.kind] * inflation
            assert np.allclose(np.diag(flt.P)[b], expected)


def test_same_day_sessions_share_gamma_and_trials_never_cross_days() -> None:
    model = _model()
    _feed(model, T0, 20)
    # Trials stamped two days later inside the same session do not start a new day.
    _feed(model, T0 + timedelta(days=2), 5, seed=1)
    gamma = model.filters[("l1", "ua")].m[15:].copy()
    assert model.last_day_index == DAY
    model.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    assert np.array_equal(model.filters[("l1", "ua")].m[15:], gamma)


def test_within_day_diffusion_is_capped_and_applied_to_copies() -> None:
    model = _model()
    _feed(model, T0, 40)
    key = ("l1", "log_time")
    flt = model.filters[key]
    p_before = flt.P.copy()
    last = T0 + timedelta(seconds=20 * 39)
    snap = model.snapshot()
    f = np.zeros(18)
    f[15] = 1.0
    _, s2_soon = flt.predict(f, minutes=model._minutes(key, _minutes(last + timedelta(minutes=10))))
    _, s2_late = flt.predict(f, minutes=model._minutes(key, _minutes(last + timedelta(days=0.9))))
    assert s2_soon == pytest.approx(p_before[15, 15] + 0.01**2 * 10)
    assert s2_late == pytest.approx(prior_covariance(*key)[15, 15])
    assert np.array_equal(flt.P, p_before) and model.snapshot() == snap


def _minutes(at: datetime) -> float:
    return (at - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() / 60.0


def test_device_change_resets_gamma_only() -> None:
    model = _model()
    _feed(model, T0, 40)
    before = {k: model.filters[k].P.copy() for k in FILTER_KEYS}
    means = {k: model.filters[k].m.copy() for k in FILTER_KEYS}
    model.begin_session(day_index=DAY, device=HAND, learn_across_sessions=True)
    assert _gamma_block_at_prior(model)
    for key, flt in model.filters.items():
        b = beta_indices(key[0])
        assert np.array_equal(flt.P[np.ix_(b, b)], before[key][np.ix_(b, b)])
        assert np.array_equal(flt.m[b], means[key][b])
    resized = DeviceContext("hand_pointer", (), 1.0, 900.0, 500.0, False, True)
    _feed(model, T0 + timedelta(hours=1), 10)
    model.begin_session(day_index=DAY, device=resized, learn_across_sessions=True)  # 700 -> 900 px: new class
    assert _gamma_block_at_prior(model)


def test_drift_alarm_widens_only_its_slow_block_up_to_the_prior() -> None:
    model = _model()
    _feed(model, T0, 60)
    before = {k: model.filters[k].P.copy() for k in FILTER_KEYS}
    for _ in range(5):
        model.apply_drift("l1", "ua")
    for key, flt in model.filters.items():
        if key != ("l1", "ua"):
            assert np.array_equal(flt.P, before[key])
    widened = model.filters[("l1", "ua")]
    p0 = prior_covariance("l1", "ua")
    b = beta_indices("l1")
    assert np.all(np.diag(widened.P)[b] <= np.diag(p0)[b] + 1e-15)
    assert np.all(np.diag(widened.P)[b] >= np.diag(before[("l1", "ua")])[b])
    assert np.array_equal(widened.P[15:, 15:], before[("l1", "ua")][15:, 15:])
    assert model.drift_flags == {"l1:ua"}
    model.begin_session(day_index=DAY + 1, device=MOUSE, learn_across_sessions=True)
    assert model.drift_flags == set()


def test_calibration_suspension_triggers_on_excess_slips_and_clears_at_a_day_boundary() -> None:
    model = _model()
    for i in range(70):
        ctx = _ctx(i, "act", T0 + timedelta(seconds=10 * i))
        slip = i % 5 == 0  # four times the predicted UCB
        model.observe_l1_trial(
            ctx, TrialOutcome(not slip, slip, slip, False, math.log(1.5), False), regime_ucb_unintended=0.05
        )
        if i < 49:
            assert not model.calibration_test()  # fewer than 50 buffered trials
    assert model.calibration_test()
    model.apply_calibration_suspension()
    assert model.calibration_suspended and not model.calibration_test()
    model.begin_session(day_index=DAY + 1, device=MOUSE, learn_across_sessions=True)
    assert not model.calibration_suspended
    # The boundary also drops the evidence that fired it: the new day's first row (here one that adds nothing to the
    # buffer) does not suspend acting regimes again, and the test waits for 50 new acting trials.
    later = T0 + timedelta(days=1)
    model.observe_l1_trial(
        _ctx(0, "assistance_off", later), TrialOutcome(True, False, False, False, math.log(1.5), False)
    )
    assert not model.calibration_test()
    for i in range(49):
        model.observe_l1_trial(
            _ctx(i, "act", later + timedelta(seconds=10 * (i + 1))),
            TrialOutcome(False, True, True, False, math.log(1.5), False),
            regime_ucb_unintended=0.05,
        )
    assert not model.calibration_test()  # 49 new trials, all slips: still fewer than 50
    calm = _model()
    for i in range(70):
        ctx = _ctx(i, "act", T0 + timedelta(seconds=10 * i))
        calm.observe_l1_trial(
            ctx, TrialOutcome(True, i % 25 == 0, False, False, math.log(1.5), False), regime_ucb_unintended=0.05
        )
    assert not calm.calibration_test()
    # Non-acting regimes never enter the C7 buffer.
    off = _model()
    for i in range(70):
        off.observe_l1_trial(
            _ctx(i, "confirm_selection:hold_700", T0),
            TrialOutcome(False, True, True, False, 2.3, True),
            regime_ucb_unintended=0.01,
        )
    assert not off.calibration_test()


def test_missing_labels_do_not_update() -> None:
    model = _model()
    _feed(model, T0, 16)
    snap = model.snapshot()
    model.observe_burden(T0, "act", "mouse", ipt=0.2, ratings={"effort": None, "intrusiveness": None, "control": None})
    model.observe_burden(T0, "assistance_off", "mouse", ipt=0.0, ratings={"intrusiveness": 5})
    model.observe_offer("hand_pointer", "pointing", "offer_modality:keyboard", "ignored")
    assert model.snapshot() == snap
    model.observe_offer("hand_pointer", "pointing", "offer_modality:keyboard", "accepted")
    assert model.offer_means("hand_pointer", "pointing", "offer_modality:keyboard")["accepted"] == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        model.observe_spread("keyboard", 1.0, 1.0, GRID[0])


def test_fallback_rows_advance_practice_only() -> None:
    model = _model()
    learned = {k: model.filters[k].m.copy() for k in FILTER_KEYS}
    for i in range(10):
        model.observe_practice_only(_ctx(i, "act", T0 + timedelta(minutes=i), menu=True))
    assert all(np.array_equal(model.filters[k].m, learned[k]) for k in FILTER_KEYS)
    practice = model.practice()
    assert practice.n_total == 10 and practice.n_variant["act"] == 10
    assert model.n_item_eff(12, 0, T0 + timedelta(minutes=10)) > 0.9


def test_menu_familiarity_has_a_seven_day_half_life() -> None:
    model = _model()
    model.observe_practice_only(_ctx(0, "act", T0, menu=True))
    assert model.n_item_eff(12, 0, T0) == pytest.approx(1.0)
    assert model.n_item_eff(12, 0, T0 + timedelta(days=7)) == pytest.approx(0.5)
    model.observe_practice_only(_ctx(0, "act", T0 + timedelta(days=7), menu=True))
    assert model.n_item_eff(12, 0, T0 + timedelta(days=7)) == pytest.approx(1.5)


# --- snapshots and determinism ------------------------------------------------------------------------------------


def test_snapshot_round_trip_and_strict_checks() -> None:
    model = _model()
    _feed(model, T0, 30)
    model.apply_calibration_suspension()
    model.apply_drift("l2", "log_time")
    snap = model.snapshot()
    restored = CoAdaptationWorldModel.restore(snap)
    assert restored.snapshot() == snap
    assert restored.calibration_suspended and restored.drift_flags == {"l2:log_time"}
    with np.load(io.BytesIO(snap), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        arrays = {name: data[name] for name in data.files}
    assert meta == {
        "model_version": MODEL_VERSION,
        "prior_version": PRIOR_VERSION,
        "feature_schema": "coadapt-features-v1",
        "variant": "full",
    }

    def rebuilt(**changes) -> bytes:
        out = dict(arrays)
        out.update(changes)
        return write_npz(out)

    bad_meta = dict(meta, model_version="coadapt-owm-v0")
    with pytest.raises(ValueError):
        CoAdaptationWorldModel.restore(rebuilt(metadata=np.array(json.dumps(bad_meta, sort_keys=True))))
    with pytest.raises(ValueError):
        CoAdaptationWorldModel.restore(rebuilt(**{"filter.l1.ua.m": np.zeros(17)}))
    with pytest.raises(ValueError):
        CoAdaptationWorldModel.restore(snap, variant=ModelVariant.SESSION_ONLY)
    pickled = io.BytesIO()
    np.savez(pickled, metadata=np.array(json.dumps(meta, sort_keys=True)), evil=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        CoAdaptationWorldModel.restore(pickled.getvalue())
    with pytest.raises(ValueError):
        CoAdaptationWorldModel.restore(b"not an archive")


def test_predictions_never_change_the_snapshot() -> None:
    model = _model()
    _feed(model, T0, 40)
    snap = model.snapshot()
    block = _block((POINTING, MENU) * 4, previous="act")
    later = T0 + timedelta(hours=2)
    for i in range(5):
        model.predict_l1_block(block, ARMS, now=later, tau=0.25, credibility=0.95)
        model.sample_l1(block, ARMS, np.random.Generator(np.random.PCG64(i)), 64, now=later, tau=0.25)
        model.predict_l2(_act_ctx(), L2_OPTIONS, now=later, credibility=0.95)
        model.sample_l2(_act_ctx(), L2_OPTIONS, np.random.Generator(np.random.PCG64(i)), 64, now=later)
        model.l1_trial_prediction(_ctx(i, "act", later))
        model.burden_latent("act", "mouse", ipt=0.3, now=later)
        model.predicted_ipt(block, "act", now=later, tau=0.25)
        model.sample_offers(
            "mouse", "menu", ["no_offer", "offer_modality:keyboard"], np.random.Generator(np.random.PCG64(i)), 32
        )
        model.beta_means()
        model.spread()
        model.practice()
        model.evidence("l1", "act")
    assert model.snapshot() == snap


def test_thompson_draws_are_reproducible_and_shaped() -> None:
    model = _model()
    _feed(model, T0, 24)
    block = _block((POINTING, MENU))
    a = model.sample_l1(block, ARMS, np.random.Generator(np.random.PCG64(9)), 32, now=T0, tau=0.25)
    b = model.sample_l1(block, ARMS, np.random.Generator(np.random.PCG64(9)), 32, now=T0, tau=0.25)
    assert a.variants == ARMS
    assert set(a.values) == {"complete", "ua", "correction", "veto", "log_time", "effort", "intrusiveness"}
    for key, arr in a.values.items():
        assert arr.shape == (32, len(ARMS)) and np.array_equal(arr, b.values[key])
    assert np.all(a.values["intrusiveness"][:, 0] == 1.0)
    by = a.by_variant()
    assert np.array_equal(by["act"]["ua"], a.values["ua"][:, 1])
    l2 = model.sample_l2(_act_ctx(), L2_OPTIONS, np.random.Generator(np.random.PCG64(9)), 16, now=T0)
    assert all(v.shape == (16, 5) for v in l2.values.values())


def test_l2_draws_from_shared_parameters_and_memos_equal_the_direct_ones() -> None:
    model = _model()
    _feed(model, T0, 24)
    thetas = model.sample_l2_parameters(np.random.Generator(np.random.PCG64(9)), 64, now=T0)
    direct = model.sample_l2(_act_ctx(), L2_OPTIONS, np.random.Generator(np.random.PCG64(9)), 64, now=T0)
    shared = model.l2_outcome_draws(_act_ctx(), L2_OPTIONS, thetas)
    assert all(np.array_equal(direct.values[key], shared.values[key]) for key in direct.values)
    # cells of one bin share the features; only the physics prior of some responses depends on n_plausible
    contexts = [
        _act_ctx(),
        ActivationContext(0.5, "b2", 3, True, 0.5, 0.5, "mouse", POINTING),
        ActivationContext(0.9, "b3", 2, True, 0.1, 0.5, "mouse", POINTING),
    ]
    draw_memo: dict = {}
    prediction_memo: dict = {}
    for ctx in contexts:
        plain = model.l2_outcome_draws(ctx, L2_OPTIONS, thetas)
        memoised = model.l2_outcome_draws(ctx, L2_OPTIONS, thetas, memo=draw_memo)
        assert all(np.array_equal(plain.values[key], memoised.values[key]) for key in plain.values)
        predictions = model.predict_l2(ctx, L2_OPTIONS, now=T0, credibility=0.95)
        assert model.predict_l2(ctx, L2_OPTIONS, now=T0, credibility=0.95, memo=prediction_memo) == predictions
    assert len(draw_memo) < len(contexts) * len(L2_OPTIONS) * len(direct.values)


def test_replay_reproduces_the_live_snapshot() -> None:
    def run() -> bytes:
        model = CoAdaptationWorldModel()
        _scenario(model)
        return model.snapshot()

    assert run() == run()


# --- model variants (spec 13.1.1) ---------------------------------------------------------------------------------


def _population() -> PopulationPrior:
    rng = np.random.Generator(np.random.PCG64(5))
    means = {k: rng.normal(0.0, 0.1, size=beta_indices(k[0]).size) for k in FILTER_KEYS}
    return PopulationPrior(beta_means=means, spread_r0={"mouse": 1.2, "hand_pointer": 1.4}, source_participants=15)


def test_population_variant_stays_at_its_prior() -> None:
    prior = _population()
    model = CoAdaptationWorldModel(variant=ModelVariant.POPULATION, population_prior=prior)
    model.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    _feed(model, T0, 60)
    model.begin_session(day_index=DAY + 2, device=MOUSE, learn_across_sessions=True)
    _feed(model, T0 + timedelta(days=2), 30)
    for key, flt in model.filters.items():
        b = beta_indices(key[0])
        assert np.array_equal(flt.m[b], prior.beta_means[key])
        assert np.array_equal(flt.P, prior_covariance(*key))
    assert model.spread()["mouse"] == pytest.approx(1.2)
    assert model.practice().n_total == 90
    assert model.drift_alarms() == []
    with pytest.raises(ValueError):
        CoAdaptationWorldModel(variant=ModelVariant.POPULATION)


def test_population_prior_changes_means_only() -> None:
    prior = _population()
    full = CoAdaptationWorldModel(population_prior=prior)
    plain = CoAdaptationWorldModel()
    for key in FILTER_KEYS:
        assert np.array_equal(full.filters[key].P, plain.filters[key].P)
        assert np.array_equal(full.filters[key].prior_cov, plain.filters[key].prior_cov)
        assert np.array_equal(full.filters[key].m[beta_indices(key[0])], prior.beta_means[key])
        assert np.all(full.filters[key].m[full.filters[key].gamma_idx] == 0.0)
    assert full.spread_prior("mouse") == 1.2 and full.spread_prior("pen") == 1.1


def test_session_only_restarts_learning_but_keeps_exposure() -> None:
    full = CoAdaptationWorldModel()
    session_only = CoAdaptationWorldModel(variant=ModelVariant.SESSION_ONLY)
    fresh = CoAdaptationWorldModel()
    for day in (DAY, DAY + 2):
        for model in (full, session_only):
            model.begin_session(day_index=day, device=MOUSE, learn_across_sessions=True)
        for key in FILTER_KEYS:
            assert np.array_equal(session_only.filters[key].m, fresh.filters[key].m)
            assert np.array_equal(session_only.filters[key].P, fresh.filters[key].P)
        assert session_only.spread() == fresh.spread()
        assert session_only.idle_rate("mouse", "act") == fresh.idle_rate("mouse", "act")
        assert all(d.chart.statistic == 0.0 for d in session_only.detectors.values())
        start = T0 + timedelta(days=day - DAY)
        _feed(full, start, 48)
        _feed(session_only, start, 48)
        session_only.observe_idle("mouse", "act", 2, 0.5)
        full.observe_idle("mouse", "act", 2, 0.5)
    assert session_only.practice() == full.practice()
    at = T0 + timedelta(days=3)
    for item in range(12):
        assert session_only.n_item_eff(12, item, at) == full.n_item_eff(12, item, at)
    n = session_only.practice().n_variant["act"]
    assert practice_multiplier(n) < practice_multiplier(0)
    assert menu_sdp_time_s(12, session_only.n_item_eff(12, 0, at)) == menu_sdp_time_s(12, full.n_item_eff(12, 0, at))
    assert not np.array_equal(full.filters[("l1", "ua")].m, session_only.filters[("l1", "ua")].m)


def test_learning_across_sessions_off_restarts_learning_at_the_prior() -> None:
    model = _model()
    _feed(model, T0, 40)
    model.apply_drift("l1", "ua")
    model.apply_calibration_suspension()
    model.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=False)
    fresh = CoAdaptationWorldModel()
    fresh.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=False)
    agency = {"c7_buffer", "c7_state", "drift_flags"}
    ours, theirs = model.to_arrays(), fresh.to_arrays()
    for name in ours:
        if name not in agency:
            assert np.array_equal(ours[name], theirs[name], equal_nan=True), name
    assert model.practice().n_total == 0 and model.spread()["mouse"] == 1.0
    # The C7 suspension and the day's drift flags last until the next day boundary, also across same-day sessions.
    assert model.calibration_suspended and model.drift_flags == {"l1:ua"}
    model.begin_session(day_index=DAY + 3, device=MOUSE, learn_across_sessions=False)
    assert not model.calibration_suspended and model.drift_flags == set()
    # After the day dynamics the learned state still starts exactly at the prior (no beta inflation).
    later = CoAdaptationWorldModel()
    later.begin_session(day_index=DAY + 3, device=MOUSE, learn_across_sessions=False)
    for key in FILTER_KEYS:
        assert np.array_equal(model.filters[key].P, later.filters[key].P)
    model.reset()
    assert not model.calibration_suspended and model.to_arrays()["c7_state"][0] == 0.0


def test_other_variants() -> None:
    physics = CoAdaptationWorldModel(variant=ModelVariant.PHYSICS_ONLY)
    physics.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    _feed(physics, T0, 40)
    physics.begin_session(day_index=DAY + 3, device=MOUSE, learn_across_sessions=True)
    for key in FILTER_KEYS:
        assert np.array_equal(physics.filters[key].P, prior_covariance(*key))
        assert np.all(physics.filters[key].m == 0.0)
    assert physics.spread()["mouse"] == 1.0 and physics.practice().n_total == 40

    no_today = CoAdaptationWorldModel(variant=ModelVariant.NO_TODAY)
    no_today.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    _feed(no_today, T0, 40)
    assert all(np.all(no_today.filters[k].m[no_today.filters[k].gamma_idx] == 0.0) for k in FILTER_KEYS)

    static = CoAdaptationWorldModel(variant=ModelVariant.PERSON_STATIC)
    static.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    _feed(static, T0, 40)
    before = static.filters[("l1", "ua")].P.copy()
    static.begin_session(day_index=DAY + 3, device=MOUSE, learn_across_sessions=True)
    b = beta_indices("l1")
    assert np.array_equal(static.filters[("l1", "ua")].P[np.ix_(b, b)], before[np.ix_(b, b)])
    assert static.drift_alarms() == [] and all(d.chart.statistic == 0.0 for d in static.detectors.values())

    no_practice = CoAdaptationWorldModel(variant=ModelVariant.NO_PRACTICE_SWITCH)
    no_practice.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    _feed(no_practice, T0, 40)
    _, f = no_practice._l1_update_terms(_ctx(41, "act", T0))
    assert np.all(f[[9, 10, 11]] == 0.0)
    ctx = _ctx(41, "act", T0)
    h, _ = no_practice._l1_update_terms(ctx)
    held = l1_physics_update(
        POINTING,
        GRID,
        ctx.target_index,
        "act",
        modality="mouse",
        n_item_eff=0.0,
        n_variant=0,
        r=no_practice.spread()["mouse"],
        tau=0.25,
    )
    assert no_practice.practice().n_variant["act"] > 0
    assert h["log_time"] == pytest.approx(held.h("log_time"))  # M_p held at its first-trial value


# --- calibration monitor ------------------------------------------------------------------------------------------


def test_calibration_monitor_report() -> None:
    monitor = CalibrationMonitor()
    empty = monitor.report()
    assert empty["ua"]["n"] == 0 and empty["log_time"] == {"coverage90": None, "n": 0}
    for p, y in ((0.05, False), (0.08, True), (0.92, True), (0.95, True)):
        monitor.add_binary("ua", p, y)
        monitor.add_binary("complete", 1 - p, not y)
    monitor.add_time(0.0, 0.5, 0.2, censored=False)
    monitor.add_time(0.0, 0.5, 2.0, censored=False)
    monitor.add_time(0.0, 0.5, 2.0, censored=True)  # above the interval: a miss
    monitor.add_time(0.0, 0.5, 0.1, censored=True)  # unknown: left out
    report = monitor.report()
    assert set(report) == {"ua", "complete", "log_time"}
    ua = report["ua"]
    assert set(ua) == {"brier", "ece", "log_score", "n", "bins"}
    assert ua["n"] == 4 and ua["brier"] == pytest.approx(np.mean([0.05**2, 0.92**2, 0.08**2, 0.05**2]))
    assert ua["bins"] == [
        {"p_mean": pytest.approx(0.065), "observed": 0.5, "n": 2},
        {"p_mean": pytest.approx(0.935), "observed": 1.0, "n": 2},
    ]
    assert ua["ece"] == pytest.approx(0.5 * abs(0.065 - 0.5) + 0.5 * abs(0.935 - 1.0))
    assert report["log_time"] == {"coverage90": pytest.approx(1 / 3), "n": 3}
    with pytest.raises(ValueError):
        monitor.add_binary("veto", 0.1, False)


# --- time zones ---------------------------------------------------------------------------------------------------


class _NoAstimezone(datetime):
    def astimezone(self, tz=None):  # pragma: no cover - fails the test if called
        raise AssertionError("day_index_for must not convert time zones")


def test_day_index_for_reads_the_offset_on_the_datetime() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    at = _NoAstimezone(2026, 10, 15, 3, 59, tzinfo=ist)
    assert day_index_for(at, 4) == datetime(2026, 10, 14).toordinal()
    assert day_index_for(_NoAstimezone(2026, 10, 15, 4, 0, tzinfo=ist), 4) == datetime(2026, 10, 15).toordinal()
    assert day_index_for(_NoAstimezone(2026, 10, 15, 0, 30, tzinfo=UTC), 0) == datetime(2026, 10, 15).toordinal()
    with pytest.raises(ValueError):
        day_index_for(datetime(2026, 10, 15, 12, 0), 4)
    with pytest.raises(ValueError):
        day_index_for(datetime(2026, 10, 15, 12, 0, tzinfo=UTC), 24)


def _scenario(model: CoAdaptationWorldModel) -> None:
    """A fixed multi-session history: two days, a device change, alarms from stored events."""
    model.begin_session(day_index=DAY, device=MOUSE, learn_across_sessions=True)
    _feed(model, T0, 48)
    model.apply_drift("l1", "log_time")
    model.observe_idle("mouse", "act", 1, 0.5)
    model.observe_offer("mouse", "menu", "offer_modality:keyboard", "dismissed")
    model.end_session()
    model.begin_session(day_index=DAY + 1, device=MOUSE, learn_across_sessions=True)
    _feed(model, T0 + timedelta(days=1), 48, seed=2)
    model.apply_calibration_suspension()
    model.begin_session(day_index=DAY + 1, device=HAND, learn_across_sessions=True)
    for i in range(16):
        ctx = _ctx(i, "enlarge_target:1.5", T0 + timedelta(days=1, hours=2, seconds=30 * i), device=HAND)
        model.observe_l1_trial(
            ctx, TrialOutcome(True, False, False, False, math.log(2.2), False), regime_ucb_unintended=0.1
        )


def replay_digest() -> str:
    """Snapshot digest of the scenario plus the study day of a fixed local time (run under different TZ values)."""
    model = CoAdaptationWorldModel()
    _scenario(model)
    ist = timezone(timedelta(hours=5, minutes=30))
    day = day_index_for(datetime(2026, 7, 15, 2, 30, tzinfo=ist), 4)
    return f"{hashlib.sha256(model.snapshot()).hexdigest()} {day}"


def test_results_do_not_depend_on_the_process_time_zone() -> None:
    code = (
        "import time, datetime;"
        "from tests.test_coadaptation_outcome_model import replay_digest;"
        "t = datetime.datetime(2026, 7, 15, 12, 0, tzinfo=datetime.timezone.utc).timestamp();"
        "print(time.localtime(t).tm_gmtoff, replay_digest())"
    )
    outputs = {}
    for tz in ("UTC0", "EST5EDT", "JST-9"):
        env = dict(os.environ, TZ=tz)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=DAEMON, env=env, check=True, capture_output=True, text=True
        )
        offset, digest, day = result.stdout.split()
        outputs[tz] = (int(offset), digest, int(day))
    assert [outputs[tz][0] for tz in ("UTC0", "EST5EDT", "JST-9")] == [0, -4 * 3600, 9 * 3600]
    digest, day = replay_digest().split()
    assert {(d, n) for _, d, n in outputs.values()} == {(digest, int(day))}

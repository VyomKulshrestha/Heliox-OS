"""Hard constraints C1-C8, fallbacks, the cap rule and preference violations (spec 9.2, 9.3, 9.7, 22.1)."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from pilot.coadaptation.core.constraints import (
    ConstraintState,
    ExclusionReason,
    FreezeConflict,
    apply_hard_constraints,
    cap_applies,
    control_floor_excludes,
    control_upper_bound,
    expected_rating,
    fallbacks,
    fixed_l2_rule,
    pin_act_collapse,
    predictive_mean_rating,
    preference_violations,
)
from pilot.coadaptation.core.policy import PolicyMode
from pilot.coadaptation.core.preferences import Preference, PreferenceKind, PreferenceScope, PreferenceSet
from pilot.coadaptation.core.records import RevertRecord, VetoRecord
from pilot.coadaptation.core.vocabulary import CAUTION_ORDER, L2_OPTIONS, L3_OPTIONS, VARIANTS, DecisionLevel, l1_arms
from tests.test_coadaptation_engine import AH2, INTENDED, Run, U, l2, pref
from tests.test_coadaptation_records import CANDS, LAYOUT, PK, T0, trial_record

MODALITIES = ("mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer")
_ids = itertools.count()


def apply(level: DecisionLevel, candidates, prefs: PreferenceSet, state: ConstraintState | None = None, **kw: Any):
    return apply_hard_constraints(
        level,
        list(candidates),
        preferences=prefs,
        state=state or ConstraintState(),
        modality=kw.pop("modality", "mouse"),
        participant_key=PK,
        adaptive=kw.pop("adaptive", True),
        **kw,
    )


# ------------------------------------------------------------------------------------------------------ C1-C8


def test_exclusion_reasons_are_the_sixteen_of_the_spec() -> None:
    assert [r.value for r in ExclusionReason] == [
        "forbidden_by_you",
        "below_your_min_caution",
        "pinned_other",
        "direct_selection_pinned",
        "not_applicable",
        "ambiguous_activation",
        "slip_risk_above_limit",
        "recent_veto",
        "low_control_rating",
        "calibration_suspended",
        "interruption_limit",
        "offer_cooldown",
        "used_this_round",
        "probe_schedule",
        "comparator_schedule",
        "purpose_missing",
    ]


def test_each_hard_constraint_excludes_with_its_reason() -> None:
    arms = l1_arms("mouse")
    # C1 forbid, minimum caution, pin, freeze, pin_act, cap
    _, excluded = apply(
        DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "enlarge_target")])
    )
    assert excluded["enlarge_target:1.5"] is excluded["enlarge_target:2.0"] is ExclusionReason.FORBIDDEN_BY_YOU
    _, excluded = apply(
        DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([pref(PreferenceKind.MIN_CAUTION, "enlarge_target:2.0")])
    )
    assert excluded["act"] is ExclusionReason.BELOW_YOUR_MIN_CAUTION
    allowed, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([pref(PreferenceKind.PIN_REGIME, "act")]))
    assert allowed == ["act"] and excluded["confirm_selection:hold_700"] is ExclusionReason.PINNED_OTHER
    allowed, _ = apply(
        DecisionLevel.BLOCK_REGIME,
        arms,
        PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "assistance_off"})]),
        adaptive=False,
    )
    assert allowed == ["assistance_off"]
    allowed, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([pref(PreferenceKind.PIN_ACT, True)]))
    assert excluded["confirm_selection:second_click"] is ExclusionReason.DIRECT_SELECTION_PINNED
    assert not any(v.startswith("confirm_selection") for v in allowed)
    allowed, excluded = apply(
        DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 3)])
    )
    assert excluded["confirm_selection:hold_700"] is ExclusionReason.INTERRUPTION_LIMIT
    # C2 at L1: the plain interface is never in the adaptive set
    _, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]))
    assert excluded["assistance_off"] is ExclusionReason.PROBE_SCHEDULE
    allowed, _ = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]), adaptive=False)
    assert "assistance_off" in allowed
    # C5 veto cooldown at the next decision of the same level
    state = ConstraintState()
    state.record_veto(DecisionLevel.BLOCK_REGIME, "act")
    state.record_veto(DecisionLevel.AMBIGUITY_RESPONSE, "wait_for_evidence")  # has no veto
    _, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]), state)
    assert excluded["act"] is ExclusionReason.RECENT_VETO
    assert state.vetoed_at(DecisionLevel.AMBIGUITY_RESPONSE) == frozenset()
    state.consume_vetoes(DecisionLevel.BLOCK_REGIME)
    _, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]), state)
    assert "act" not in excluded
    # C6 control floor
    state.add_control_floor("enlarge_target:2.0")
    _, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]), state)
    assert excluded["enlarge_target:2.0"] is ExclusionReason.LOW_CONTROL_RATING
    # C7 removes acting regimes in adaptive mode only
    suspended = ConstraintState(calibration_suspended=True)
    allowed, excluded = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]), suspended)
    assert excluded["act"] is excluded["enlarge_target:1.5"] is ExclusionReason.CALIBRATION_SUSPENDED
    assert allowed == ["confirm_selection:second_click", "confirm_selection:hold_700"]
    allowed, _ = apply(DecisionLevel.BLOCK_REGIME, arms, PreferenceSet([]), suspended, adaptive=False)
    assert "act" in allowed
    # C8 offer cooldown for 10 trials after a dismissed offer
    offers = ConstraintState()
    offers.dismiss_offer()
    _, excluded = apply(DecisionLevel.OFFER, L3_OPTIONS, PreferenceSet([]), offers)
    assert excluded["offer_modality:keyboard"] is ExclusionReason.OFFER_COOLDOWN and "no_offer" not in excluded
    offers.trials_recorded = 10
    _, excluded = apply(DecisionLevel.OFFER, L3_OPTIONS, PreferenceSet([]), offers)
    assert excluded == {}


def test_pin_act_collapses_l2_cells_and_skips_the_safety_filter() -> None:
    assert pin_act_collapse(["act", "confirm_selection:second_click", "wait_for_evidence", "ask_clarification:2"]) == (
        ["act"],
        {
            "confirm_selection:second_click": ExclusionReason.PINNED_OTHER,
            "wait_for_evidence": ExclusionReason.PINNED_OTHER,
            "ask_clarification:2": ExclusionReason.PINNED_OTHER,
        },
    )
    assert pin_act_collapse(["wait_for_evidence", "ask_clarification:2"])[0] == ["wait_for_evidence"]
    run = Run(PreferenceSet([pref(PreferenceKind.PIN_ACT, True)]))
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    table = run.engine.decide_ambiguity_table(
        decision_id="tbl",
        block_decision_id=decision.decision_id,
        block=block,
        start_trial_index=0,
        trials_in_block=8,
        now=run.tick(),
    )
    assert table.mode is PolicyMode.PINNED and table.pin_act
    for cell in table.cells:
        expected = "act" if cell.has_hit else "wait_for_evidence"
        assert cell.propensities[expected] == 1.0 and cell.pinned
        assert ExclusionReason.SLIP_RISK_ABOVE_LIMIT not in cell.excluded.values()


def test_a_veto_in_pinned_mode_adds_no_exclusion() -> None:
    # a pinned block's fixed-rule L2 response
    run = Run(PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"})]))
    run.begin()
    block = run.block(1)
    decision = run.decide(block)
    table = run.table(decision, block)
    assert table.mode is PolicyMode.PINNED
    fb = run.fallbacks()
    cell = "b1:p2:h1"
    rule = fixed_l2_rule(fb, cell)
    assert rule == "ask_clarification:2" and table.variant_for(cell, 0, 0) == rule
    ask = run.activation(
        AH2, l2=l2(table.decision_id, cell, rule), result="asked", outcome="cancelled", committed_index=None
    )
    after = run.activation(U, index=1)
    trial = run.trial(
        decision,
        (ask, after),
        trial_index=0,
        fallbacks=fb,
        veto=VetoRecord(table.decision_id, DecisionLevel.AMBIGUITY_RESPONSE),
        revert=RevertRecord("veto", 1, fb.plain, fb.issued_at),
    )
    assert run.observe(trial).stored.verification == "ok"
    constraint = run.engine._session.constraint  # noqa: SLF001
    assert constraint.vetoed_at(DecisionLevel.AMBIGUITY_RESPONSE) == frozenset()
    # a pin_act collapse
    run = Run(PreferenceSet([pref(PreferenceKind.PIN_ACT, True)]))
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    table = run.table(decision, block)
    collapse = run.activation(AH2, l2=l2(table.decision_id, cell, "act"), result="committed")
    trial = run.trial(
        decision,
        (collapse, run.activation(U, index=1)),
        trial_index=0,
        veto=VetoRecord(table.decision_id, DecisionLevel.AMBIGUITY_RESPONSE),
        revert=RevertRecord("veto", 1, run.fallbacks().plain, run.fallbacks().issued_at),
    )
    run.observe(trial)
    assert run.engine._session.constraint.vetoed_at(DecisionLevel.AMBIGUITY_RESPONSE) == frozenset()  # noqa: SLF001


# ---------------------------------------------------------------------------------------------------- fallbacks


def _random_prefs(rng: np.random.Generator) -> PreferenceSet:
    choices = [
        lambda: pref(
            PreferenceKind.FORBID_VARIANT,
            str(
                rng.choice(
                    [
                        "act",
                        "enlarge_target",
                        "assistance_off",
                        "confirm_selection",
                        "ask_clarification:3",
                        "confirm_selection:hold_700",
                    ]
                )
            ),
        ),
        lambda: pref(PreferenceKind.MIN_CAUTION, str(rng.choice(CAUTION_ORDER[:7]))),
        lambda: pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, int(rng.integers(0, 9))),
        lambda: pref(PreferenceKind.PIN_ACT, True),
        lambda: pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": str(rng.choice(l1_arms("mouse")))}),
        lambda: pref(PreferenceKind.PIN_REGIME, str(rng.choice(l1_arms("hand_pointer")[1:]))),
    ]
    prefs = PreferenceSet([])
    for _ in range(int(rng.integers(0, 4))):
        candidate = choices[int(rng.integers(0, len(choices)))]()
        try:
            prefs.validate_new(candidate)
        except ValueError:
            continue
        prefs = PreferenceSet([*prefs.all, candidate])
    return prefs


def test_fallbacks_over_a_seeded_grid_of_modalities_and_rules() -> None:
    rng = np.random.default_rng(20261015)
    for _ in range(300):
        prefs = _random_prefs(rng)
        for modality, suspended in itertools.product(MODALITIES, (False, True)):
            now = T0 + timedelta(hours=1)
            fb = fallbacks(prefs, modality, participant_key=PK, calibration_suspended=suspended, now=now)  # type: ignore[arg-type]
            allowed, _ = prefs.apply(
                DecisionLevel.BLOCK_REGIME,
                l1_arms(modality),
                context="study",
                modality=modality,  # type: ignore[arg-type]
                participant_key=PK,
                interruptions_used=0,
                hold=False,
            )
            held = prefs.held_regime(modality=modality, participant_key=PK)  # type: ignore[arg-type]
            pool = [v for v in allowed if not (suspended and VARIANTS[v].acting)]
            cautious = max(pool, key=lambda v: VARIANTS[v].caution) if pool else None
            if held is not None:
                assert fb.l1 == fb.refresh == fb.practice == held
            else:
                assert fb.l1 == cautious
                assert fb.refresh == fb.practice == fb.plain
                if fb.l1 is not None:
                    assert not any(p.forbids(fb.l1) for p in prefs.all)
                    assert not (suspended and VARIANTS[fb.l1].acting)
            if "assistance_off" in allowed and not suspended:
                assert fb.plain == "assistance_off"
            else:
                assert fb.plain == cautious
                if suspended and fb.plain is not None:
                    assert not VARIANTS[fb.plain].acting
            assert fb.freezable == prefs.freezable(modality=modality, participant_key=PK)  # type: ignore[arg-type]
            assert fb.l2[-1] == "wait_for_evidence"
            assert not any(p.forbids(v) for p in prefs.all for v in fb.l2)
            assert fb.issued_at == now
            caps = [p.value for p in prefs.all if p.kind is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK]
            assert fb.interruption_cap == (min(caps) if caps else None)
            target = fb.l1 or max(l1_arms(modality), key=lambda v: VARIANTS[v].caution)
            conflict = prefs.freeze_conflict(target, modality=modality, participant_key=PK)  # type: ignore[arg-type]
            assert (fb.freeze_conflict is None) == (conflict is None)
            assert (fb.freeze_conflict is None) == (target in fb.freezable)


def test_fallbacks_under_c7_and_under_freezes_and_pins() -> None:
    now = T0 + timedelta(hours=1)
    none = PreferenceSet([])
    fb = fallbacks(none, "mouse", participant_key=PK, calibration_suspended=False, now=now)
    assert (fb.l1, fb.plain, fb.refresh, fb.practice) == (
        "confirm_selection:hold_700",
        "assistance_off",
        "assistance_off",
        "assistance_off",
    )
    assert fb.l2 == ("ask_clarification:3", "ask_clarification:2", "wait_for_evidence")
    fb = fallbacks(none, "mouse", participant_key=PK, calibration_suspended=True, now=now)
    assert fb.plain == fb.refresh == fb.practice == "confirm_selection:hold_700"
    no_confirm = PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "confirm_selection")])
    fb = fallbacks(no_confirm, "mouse", participant_key=PK, calibration_suspended=True, now=now)
    assert fb.l1 is fb.plain is fb.refresh is fb.practice is None
    for regime in ("confirm_selection:hold_700", "assistance_off", "act"):
        frozen = PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": regime})])
        for suspended in (False, True):
            fb = fallbacks(frozen, "mouse", participant_key=PK, calibration_suspended=suspended, now=now)
            assert fb.l1 == fb.refresh == fb.practice == regime
    pinned = PreferenceSet([pref(PreferenceKind.PIN_REGIME, "enlarge_target:1.5")])
    fb = fallbacks(pinned, "mouse", participant_key=PK, calibration_suspended=True, now=now)
    assert fb.l1 == fb.refresh == fb.practice == "enlarge_target:1.5"
    assert fb.plain == "confirm_selection:hold_700"
    # a pin naming a regime the modality lacks does not apply there
    fb = fallbacks(pinned, "keyboard", participant_key=PK, calibration_suspended=False, now=now)
    assert fb.l1 == "confirm_selection:hold_700" and fb.refresh == "assistance_off"
    pin_act = PreferenceSet([pref(PreferenceKind.PIN_ACT, True)])
    assert fallbacks(pin_act, "mouse", participant_key=PK, calibration_suspended=False, now=now).l2 == (
        "act",
        "wait_for_evidence",
    )


def test_freeze_conflict_names_the_blocking_cap() -> None:
    cap = pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2)
    prefs = PreferenceSet([pref(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click"), cap])
    fb = fallbacks(prefs, "mouse", participant_key=PK, calibration_suspended=False, now=T0)
    assert fb.freezable == ()
    assert fb.l1 == "confirm_selection:hold_700"
    assert fb.freeze_conflict is not None and fb.freeze_conflict.conflicting == (cap.pref_id,)
    assert cap.summary() in fb.freeze_conflict.message
    assert fb.to_dict()["freeze_conflict"] == {"message": fb.freeze_conflict.message, "conflicting": [cap.pref_id]}
    current = fallbacks(
        prefs,
        "mouse",
        participant_key=PK,
        calibration_suspended=False,
        now=T0,
        current_regime="confirm_selection:second_click",
    )
    assert current.freeze_conflict is not None and current.freeze_conflict.conflicting == (cap.pref_id,)
    free = fallbacks(
        PreferenceSet([]), "mouse", participant_key=PK, calibration_suspended=False, now=T0, current_regime="act"
    )
    assert free.freeze_conflict is None and free.to_dict()["freeze_conflict"] is None
    assert FreezeConflict("m", ("a",)).to_dict() == {"message": "m", "conflicting": ["a"]}


def test_fixed_rule_cap_and_cap_applies() -> None:
    fb = fallbacks(PreferenceSet([]), "mouse", participant_key=PK, calibration_suspended=False, now=T0)
    assert fixed_l2_rule(fb, "b3:p3:h1") == "ask_clarification:3"
    assert fixed_l2_rule(fb, "b1:p2:h1") == "ask_clarification:2"
    assert fixed_l2_rule(fb, "b3:p1:h0") == "wait_for_evidence"
    pin_act = fallbacks(
        PreferenceSet([pref(PreferenceKind.PIN_ACT, True)]),
        "mouse",
        participant_key=PK,
        calibration_suspended=False,
        now=T0,
    )
    assert fixed_l2_rule(pin_act, "b1:p1:h1") == "act" and fixed_l2_rule(pin_act, "b3:p2:h0") == "wait_for_evidence"
    assert cap_applies(2, 2, pin_act=False) and cap_applies(0, 0, pin_act=False)
    assert not cap_applies(2, 1, pin_act=False)
    assert not cap_applies(None, 50, pin_act=False)
    assert not cap_applies(2, 5, pin_act=True)
    # interruption cap at L1: a cap below the block's 8 trials removes the confirm regimes
    allowed, _ = apply(
        DecisionLevel.BLOCK_REGIME,
        l1_arms("mouse"),
        PreferenceSet([pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 7)]),
        adaptive=False,
    )
    assert not any(v.startswith("confirm_selection") for v in allowed)
    allowed, _ = apply(
        DecisionLevel.BLOCK_REGIME,
        l1_arms("mouse"),
        PreferenceSet([pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 8)]),
        adaptive=False,
    )
    assert "confirm_selection:hold_700" in allowed


# --------------------------------------------------------------------------------------------- preference violations


def _trial(**overrides: Any):
    from tests.test_coadaptation_records import SPREAD, pointer_activation

    target = CANDS[INTENDED]
    activation = pointer_activation(CANDS, (int(target.cx), int(target.cy)), spread=SPREAD, press_index=INTENDED)
    overrides.setdefault("regime", "act")
    return trial_record(activations=(activation,), cands=CANDS, layout=LAYOUT, spread=SPREAD, **overrides)


def test_preference_violations_follow_the_binding_times() -> None:
    forbid_act = pref(PreferenceKind.FORBID_VARIANT, "act", at=T0 + timedelta(minutes=10))
    decided = {"d1": T0 + timedelta(minutes=5)}
    assert preference_violations(_trial(), [forbid_act], decision_times=decided) == ()
    decided_after = {"d1": T0 + timedelta(minutes=11)}
    assert preference_violations(_trial(), [forbid_act], decision_times=decided_after) == ("l1:act:forbidden",)
    # an L1 fallback is judged at the trial's fallbacks_issued_at
    fallback = _trial(l1_fallback_reason="late", issued_at=T0 + timedelta(minutes=12))
    assert preference_violations(fallback, [forbid_act], decision_times={}) == ("l1:act:forbidden",)


def test_refresh_reverts_must_keep_the_frozen_regime_and_veto_reverts_are_exempt() -> None:
    freeze = pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, at=T0 + timedelta(minutes=10))
    decided = {"d1": T0 + timedelta(minutes=1)}
    late = T0 + timedelta(minutes=20)
    refresh_off = _trial(revert=RevertRecord("refresh", 0, "assistance_off", late))
    assert preference_violations(refresh_off, [freeze], decision_times=decided) == (
        "revert:assistance_off:not_held_regime",
    )
    refresh_kept = _trial(revert=RevertRecord("refresh", 0, "act", late))
    assert preference_violations(refresh_kept, [freeze], decision_times=decided) == ()
    veto_plain = _trial(revert=RevertRecord("veto", 0, "assistance_off", late))
    assert preference_violations(veto_plain, [freeze], decision_times=decided) == ()
    # the decision's own regime is judged at the decision's time, before the freeze
    assert preference_violations(_trial(regime="enlarge_target:2.0"), [freeze], decision_times=decided) == ()


def test_an_l2_response_is_judged_at_its_source() -> None:
    forbid = pref(PreferenceKind.FORBID_VARIANT, "ask_clarification:2", at=T0 + timedelta(minutes=10))
    from tests.test_coadaptation_records import SPREAD, pointer_activation

    applied = l2("t1", "b1:p2:h1", "ask_clarification:2")
    ask = pointer_activation(
        CANDS,
        AH2,
        spread=SPREAD,
        press_index=INTENDED,
        l2=applied,
        result="asked",
        outcome="committed",
        committed_index=INTENDED,
    )
    trial = trial_record(activations=(ask,), cands=CANDS, layout=LAYOUT, spread=SPREAD, regime="act")
    assert preference_violations(trial, [forbid], decision_times={"d1": T0, "t1": T0}) == ()
    assert preference_violations(trial, [forbid], decision_times={"d1": T0, "t1": T0 + timedelta(minutes=11)}) == (
        "activation[0]:ask_clarification:2:forbidden",
    )
    fixed = replace(ask, l2=replace(applied, source="fixed_rule"))
    later_cue = trial_record(
        activations=(fixed,),
        cands=CANDS,
        layout=LAYOUT,
        spread=SPREAD,
        regime="act",
        issued_at=T0 + timedelta(minutes=11),
    )
    assert preference_violations(later_cue, [forbid], decision_times={"d1": T0}) == (
        "activation[0]:ask_clarification:2:forbidden",
    )


# ------------------------------------------------------------------------------------------------ C6 closed form


def test_control_floor_bound_is_the_closed_form_90th_percentile() -> None:
    from scipy.special import ndtr

    rng = np.random.default_rng(7)
    inner = (-1.5, -0.9, -0.3, 0.3, 0.9, 1.5)
    for mu, s in ((-1.2, 0.5), (0.0, 0.3), (0.8, 1.1), (-2.0, 0.05)):
        eta = rng.normal(mu, s, 100_000)
        ratings = 1.0 + sum(1.0 - ndtr(c - eta) for c in inner)  # E[y | eta] for each sampled latent
        assert expected_rating(float(eta[0])) == pytest.approx(float(ratings[0]), abs=1e-12)
        assert control_upper_bound(mu, s) == pytest.approx(float(np.quantile(ratings, 0.9)), abs=0.01)
    # deterministic: no random numbers are used
    state = np.random.get_state()
    assert control_upper_bound(-1.0, 0.4) == control_upper_bound(-1.0, 0.4)
    assert np.random.get_state()[1].tolist() == state[1].tolist()
    assert expected_rating(-10.0) == pytest.approx(1.0) and expected_rating(10.0) == pytest.approx(7.0)
    assert predictive_mean_rating(-1.5, 0.2) < 3.0
    assert control_floor_excludes(-1.6, 0.2) and not control_floor_excludes(0.5, 0.2)
    assert not control_floor_excludes(-0.9, 2.0)  # a wide posterior keeps the upper bound above 3.5
    assert math.isclose(expected_rating(0.0), 4.0, abs_tol=1e-12)


def test_l2_options_and_levels() -> None:
    assert L2_OPTIONS == (
        "act",
        "confirm_selection:second_click",
        "wait_for_evidence",
        "ask_clarification:2",
        "ask_clarification:3",
    )
    assert _ids is not None and Preference is not None and PreferenceScope is not None

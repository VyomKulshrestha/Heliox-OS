"""Simulated users: determinism, the layout port, family forms, behaviour, the shared preference layer and the
counterfactual truth used by the OPE validation (spec section 18.1)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from datetime import timedelta

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from benchmarks.coadaptation_suite import (
    ope_refusal_at_n,
    power_curve,
    probe_reliance_detection,
    select_confirmatory_n,
)
from pilot.coadaptation.core import ope
from pilot.coadaptation.core.audit import agency_audit
from pilot.coadaptation.core.constraints import fallbacks as compute_fallbacks
from pilot.coadaptation.core.engine import PolicyProbabilities
from pilot.coadaptation.core.preferences import (
    Preference,
    PreferenceKind,
    PreferenceScope,
    PreferenceSet,
)
from pilot.coadaptation.core.priors import PRIOR_TABLE_V1, keyboard_presses
from pilot.coadaptation.core.simulator import (
    ENLARGE_RIGHT,
    FAMILY_FORMS,
    FAMILY_SPLITS,
    MODEL_FORMS,
    SIM_EPOCH,
    SURFACE,
    BaselinePolicy,
    EnginePolicy,
    SimulatedPerson,
    SimulatedStudyRunner,
    first_aim_width,
    generate_layout,
    make_population,
    verify_disjoint,
)
from pilot.coadaptation.core.study_plan import LAYOUT_GRID
from pilot.coadaptation.core.utility import pointing_throughput

SEED = 20261015


def _person(family: str = "F1", index: int = 0, **changes):
    params = make_population([family], index + 1, SEED)[index]
    return replace(params, **changes) if changes else params


def _run(params, policy=None, *, sessions: int = 2, seed: int = 7):
    runner = SimulatedStudyRunner(
        SimulatedPerson(params),
        policy or (lambda: EnginePolicy()),
        seed=seed,
    )
    return runner, runner.run(sessions=sessions)


def test_runs_are_deterministic_by_seed() -> None:
    params = _person("F3")
    _, first = _run(params)
    _, second = _run(params)
    assert json.dumps(first.bundle(), sort_keys=True) == json.dumps(second.bundle(), sort_keys=True)
    assert make_population(["F2", "F4"], 3, SEED) == make_population(["F2", "F4"], 3, SEED)
    _, other = _run(params, seed=8)
    assert json.dumps(other.bundle(), sort_keys=True) != json.dumps(first.bundle(), sort_keys=True)


def test_layout_port_draws_only_from_the_layout_grid() -> None:
    seen: dict[str, set] = {
        "w_px": set(),
        "gap_px": set(),
        "d_px": set(),
        "grid": set(),
        "n_items": set(),
    }
    close = 0
    pointing = 0
    fit_w, fit_h = LAYOUT_GRID["fit_px"]
    for block in range(1, 13):
        for trial in range(8):
            for task in ("pointing", "menu"):
                plan = generate_layout(task, 4242, block, trial)
                drawn = dict(plan.drawn)
                for key in seen:
                    if key in drawn:
                        seen[key].add(drawn[key])
                for rect in plan.candidates:
                    assert rect.cx - rect.w / 2 >= 0 and rect.cx + rect.w / 2 <= fit_w
                    assert rect.cy - rect.h / 2 >= 0 and rect.cy + rect.h / 2 <= fit_h
                if task == "menu":
                    menu = LAYOUT_GRID["menu"]
                    assert plan.summary.menu_n in menu["n_items"]
                    item = (menu["item_w_px"], menu["item_h_px"])
                    assert all((rect.w, rect.h) == item for rect in plan.candidates)
                    # The button sits above the menu.
                    assert plan.summary.origin_y == menu["button_h_px"] / 2
                    assert menu["button_h_px"] + plan.summary.menu_n * menu["item_h_px"] <= fit_h
                else:
                    pointing += 1
                    assert drawn["w_px"] in LAYOUT_GRID["pointing"]["w_px"]
                    assert drawn["gap_px"] in LAYOUT_GRID["pointing"]["gap_px"]
                    assert drawn["d_px"] in LAYOUT_GRID["pointing"]["d_px"]
                    # The surface's mean keyboard presses travel with the grid, so keyboard decisions use them.
                    expected = keyboard_presses(plan.summary, None, candidates=plan.candidates)
                    assert plan.summary.presses_mean == expected
                    close += abs(plan.distance - drawn["d_px"]) <= 1.0
    assert seen["w_px"] == set(LAYOUT_GRID["pointing"]["w_px"])
    assert seen["gap_px"] == set(LAYOUT_GRID["pointing"]["gap_px"])
    assert seen["d_px"] == set(LAYOUT_GRID["pointing"]["d_px"])
    assert seen["grid"] == set(LAYOUT_GRID["pointing"]["grids"])
    assert seen["n_items"] == set(LAYOUT_GRID["menu"]["n_items"])
    # The cued target sits at the drawn D unless the grid cannot fit there.
    assert close >= 0.85 * pointing
    assert SURFACE == (640.0, 420.0)


def test_family_splits_are_disjoint_and_leakage_raises() -> None:
    verify_disjoint(FAMILY_SPLITS)
    with pytest.raises(ValueError):
        verify_disjoint({"tune": ("F1", "F3"), "holdout": ("F3",)})


def test_families_use_their_forms_and_holdouts_avoid_the_model_forms() -> None:
    for family in ("F1", "F3", "F5"):
        assert FAMILY_FORMS[family].day_form == "offset_fatigue"
        assert FAMILY_FORMS[family].switch_form == "exp3"
    for family in ("F4", "F6"):
        forms = FAMILY_FORMS[family]
        for field in ("switch_form", "endpoint_form", "aiming_form", "burden_form", "day_form"):
            assert getattr(forms, field) != getattr(MODEL_FORMS, field), (family, field)
    assert FAMILY_FORMS["F2"].day_form == "step"
    assert FAMILY_FORMS["F4"].day_form == "random_walk"
    assert FAMILY_FORMS["F6"].day_form == "none"
    for family in ("F2", "F4", "F6"):
        for params in make_population([family], 3, SEED):
            assert params.forms == FAMILY_FORMS[family]
            # No constant day offset and no fatigue slope: the model's gamma has nothing to fit.
            assert params.day_sd_log_spread == params.day_sd_log_time == 0
            assert params.fatigue_per_min == 0


def test_day_structures_behave_as_their_forms() -> None:
    rng = Generator(PCG64(1))
    step = SimulatedPerson(_person("F2"))
    step.begin_day(0, rng)
    step.begin_session(100, rng)
    before = step.spread("act")
    step.state.session_trials = int(step.state.step_trial)
    after = step.spread("act")
    assert 30 <= step.state.step_trial <= 70
    assert after / before == pytest.approx(math.exp(0.25))
    walk = SimulatedPerson(_person("F4"))
    walk.begin_day(0, rng)
    walk.begin_session(100, rng)
    start = walk.spread("act")
    plan = generate_layout("pointing", 1, 1, 0)
    for _ in range(40):
        walk.after_trial(plan, "act", 0.05, rng)
    assert walk.spread("act") != start
    walk.begin_session(100, rng)
    assert walk.state.walk_log_spread == 0.0  # the walk restarts at every session
    flat = SimulatedPerson(_person("F6"))
    flat.begin_day(0, rng)
    first_day = flat.spread("assistance_off")
    flat.begin_day(3, rng)
    flat.state.session_minute = 45.0
    assert flat.spread("assistance_off") == first_day  # no day effect and no fatigue


def test_learning_curves_decrease() -> None:
    for family in ("F1", "F2"):
        person = SimulatedPerson(_person(family))
        values = []
        for n in range(0, 60, 5):
            person.state.practice["act"] = n
            values.append(person.practice_multiplier("act"))
        assert all(a > b for a, b in zip(values, values[1:])), family


@pytest.mark.parametrize(
    ("form", "check"),
    [
        ("exp3", lambda c, k: c * math.exp(-k / 3.0)),
        ("hyperbolic", lambda c, k: c / (1.0 + k)),
        ("delayed", lambda c, k: c * (k / 2.0) * math.exp(1.0 - k / 2.0)),
    ],
)
def test_switch_transients_follow_their_form(form, check) -> None:
    params = _person("F1")
    params = replace(params, forms=replace(params.forms, switch_form=form), switch_cost_ms=300.0)
    person = SimulatedPerson(params)
    person.set_regime("act")
    assert person.switch_transient() == 0.0  # no change yet this session
    person.set_regime("confirm_selection:hold_700")
    values = []
    for k in range(8):
        person.state.since_switch = k
        values.append(person.switch_transient())
        assert values[-1] == pytest.approx(check(300.0, k))
    assert all(v >= 0 for v in values) and max(values) > 0
    if form == "delayed":
        assert values[0] == 0.0 and int(np.argmax(values)) == 2


def test_idle_activations_are_poisson() -> None:
    params = _person("F3", idle_activation_rate_per_min=2.0)
    runner = SimulatedStudyRunner(SimulatedPerson(params), lambda: EnginePolicy(), seed=3)
    _, result = _run(params, sessions=1)
    plan = generate_layout("idle", 9, 1, 900)
    fb = compute_fallbacks(
        PreferenceSet(()),
        "hand_pointer",
        participant_key=result.participant_key,
        calibration_suspended=False,
        now=result.sessions[0].started_at,
    )
    counts = []
    rng = Generator(PCG64(11))
    for _ in range(1500):
        record, _ = runner.simulate_trial(
            plan,
            session_id=result.sessions[0].session_id,
            block_index=4,
            trial_index=8,
            regime="assistance_off",
            regime_decision_id=result.decisions[0].decision_id,
            table=None,
            fallbacks=fb,
            spread=(1.6, 6.0),
            tau=0.25,
            modality="hand_pointer",
            k_start=0,
            cue_at=result.sessions[0].started_at,
            rng=rng,
            idle=True,
        )
        counts.append(len(record.activations))
    mean = float(np.mean(counts))
    assert mean == pytest.approx(1.0, abs=0.08)  # 2 per minute over a 30 s epoch
    assert float(np.var(counts)) == pytest.approx(mean, rel=0.15)


def test_enlarge_keeps_the_scatter_planned_for_the_cue_time_width_and_shortens_the_aim() -> None:
    """Spec 7.3, the model's own form that the tuning families share (18.1): under enlarge_target the endpoint
    scatter stays planned for the cue-time W while the literal hit uses the rendered rects, and the first aim's Fitts
    time uses W_eff, the p_x mixture of k W and W_bad."""
    assert {
        "pointer": PRIOR_TABLE_V1.enlarge_right_pointer.value,
        "hand_pointer": PRIOR_TABLE_V1.enlarge_right_hand.value,
    } == ENLARGE_RIGHT
    plan = next(
        layout
        for layout in (generate_layout("pointing", seed, 1, 1) for seed in range(1, 400))
        if dict(layout.drawn)["w_px"] == 16.0 and dict(layout.drawn)["gap_px"] == 2.0
    )
    w = plan.candidates[plan.intended].w
    assert first_aim_width(plan, "act", "mouse") == w
    assert first_aim_width(plan, "enlarge_target:2.0", "mouse") == pytest.approx(0.85 * 2.0 * w + 0.15 * 0.5 * w)
    assert first_aim_width(plan, "enlarge_target:2.0", "hand_pointer") == pytest.approx(0.7 * 2.0 * w + 0.3 * 0.5 * w)
    params = _person("F1", fatigue_per_min=0.0, reliance_drift=0.0, selection_error=0.0, day_sd_log_spread=0.0)
    assert params.modality == "mouse"
    runner, result = _run(params, sessions=1)
    session = result.sessions[0]
    fb = compute_fallbacks(
        PreferenceSet(()),
        "mouse",
        participant_key=result.participant_key,
        calibration_suspended=False,
        now=session.started_at,
    )
    on_target: dict[str, list[bool]] = {"assistance_off": [], "enlarge_target:2.0": []}
    first_ms: dict[str, list[float]] = {"assistance_off": [], "enlarge_target:2.0": []}
    for regime in on_target:
        rng = Generator(PCG64(11))
        for _ in range(1500):
            record, _ = runner.simulate_trial(
                plan,
                session_id=session.session_id,
                block_index=4,
                trial_index=8,
                regime=regime,
                regime_decision_id=result.decisions[0].decision_id,
                table=None,
                fallbacks=fb,
                spread=(1.0, 1.0),
                tau=0.25,
                modality="mouse",
                k_start=0,
                cue_at=session.started_at,
                rng=rng,
            )
            first = record.activations[0]
            first_ms[regime].append(first.t_ms)
            if regime == "assistance_off" or first.expanded_index == plan.intended:
                on_target[regime].append(first.press_index == plan.intended)
    # Unassisted, a 16 px target with a 2 px gap is missed about 5% of the time. Expanded on the intended target it
    # is 2 W wide for a scatter planned for W, so about 2 Phi(-4) per axis: practically never missed.
    assert float(np.mean(on_target["assistance_off"])) < 0.97
    assert len(on_target["enlarge_target:2.0"]) > 1000
    assert float(np.mean(on_target["enlarge_target:2.0"])) >= 0.995
    assert float(np.mean(first_ms["enlarge_target:2.0"])) < float(np.mean(first_ms["assistance_off"])) - 100.0


def _forbid(params, variant: str):
    key = params.participant_key
    pref = Preference(
        pref_id=f"test-{key}-forbid",
        kind=PreferenceKind.FORBID_VARIANT,
        scope=PreferenceScope(context="study", modality=None, participant_key=key),
        value=variant,
        origin="explicit_user",
        created_at=SIM_EPOCH - timedelta(hours=20),
    )
    return replace(params, stated_preferences=(pref,))


def _audit(result):
    return agency_audit(
        result.decisions,
        result.trials,
        result.sessions,
        result.burden,
        result.preferences,
        result.consents,
        events=result.events,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BaselinePolicy("assistance_off"),
        lambda: BaselinePolicy("best_fixed"),
        lambda: BaselinePolicy("rule_heuristic"),
        lambda: BaselinePolicy("uniform"),
        lambda: EnginePolicy(),
    ],
)
def test_stated_preferences_are_honoured_by_every_policy(factory) -> None:
    params = _forbid(_person("F1", 1), "enlarge_target")
    _, result = _run(params, factory)
    report = _audit(result)
    assert report.passed, report.findings[:5]
    for stored in result.trials:
        assert not stored.trial.applied_regime.startswith("enlarge_target")


def test_a_forbidden_baseline_choice_takes_the_fallback_and_is_counted() -> None:
    params = _forbid(_person("F1", 2), "assistance_off")
    _, result = _run(params, lambda: BaselinePolicy("assistance_off"))
    assert result.preference_replacements > 0
    assert _audit(result).counts["A1"] == 0
    assert all(s.trial.applied_regime != "assistance_off" for s in result.trials)


def test_reliance_drift_lowers_unassisted_precision_and_probes_see_it() -> None:
    drifting = SimulatedPerson(_person("F4", 0, reliance_drift=0.5))
    steady = SimulatedPerson(_person("F4", 1, reliance_drift=0.0))
    for person in (drifting, steady):
        base = person.spread("assistance_off")
        person.state.enlarged_trials = 200
        ratio = person.spread("assistance_off") / base
        assistance = person.spread("enlarge_target:1.5")
        if person is drifting:
            assert ratio == pytest.approx(2.0)
        else:
            assert ratio == pytest.approx(1.0)
        assert assistance > 0
    drifting_people = [_person("F4", i, reliance_drift=3.0, spread_r=1.0) for i in range(3)]
    steady_people = [_person("F4", i, reliance_drift=0.0, spread_r=1.0) for i in range(3)]
    drifting_runs = [_run(params, sessions=3)[1] for params in drifting_people]
    steady_runs = [_run(params, sessions=3)[1] for params in steady_people]
    report = probe_reliance_detection(drifting_runs + steady_runs)
    assert report["people_with_drift"] == 3 and report["people_without_drift"] == 3
    assert 0.0 <= report["detection_rate"] <= 1.0 and 0.0 <= report["false_alarm_rate"] <= 1.0
    # The probes see the reliance: unassisted pointing throughput in session 3 is lower with the drift.
    assert np.mean([_probe_throughput(run, 3) for run in drifting_runs]) < np.mean(
        [_probe_throughput(run, 3) for run in steady_runs]
    )


def _probe_throughput(run, session_number: int) -> float:
    numbers = {row.session_id: row.session_number for row in run.sessions}
    blocks = [(row.session_id, plan) for row in run.sessions for plan in row.plan.blocks]
    probes = {(session_id, plan.index) for session_id, plan in blocks if plan.phase == "probe"}
    trials = [
        stored
        for stored in run.trials
        if numbers[stored.trial.session_id] == session_number
        and (stored.trial.session_id, stored.trial.block_index) in probes
        and stored.trial.task == "pointing"
    ]
    value = pointing_throughput(trials)
    assert value is not None
    return value


def _logging_target(result, block):
    decisions = {d.decision_id: d for d in result.decisions}
    template = next(
        (
            {c["cell_id"]: c["propensities"] for c in d.payload["cells"]}
            for d in result.decisions
            if d.payload.get("level") == "ambiguity_table"
            and str(d.mode) == "randomized"
            and d.session_id == block.session_id
        ),
        {},  # no act-guarded block ran (stated rules): the target never needs an L2 cell
    )
    tables = [d for d in result.decisions if d.payload.get("block_decision_id") == block.unit_id]
    l2 = template
    if tables:
        l2 = {c["cell_id"]: c["propensities"] for c in tables[0].payload["cells"]}
    return PolicyProbabilities(
        l1=dict(decisions[block.unit_id].payload["propensities"]),
        l2=l2,
        l3=None,
    )


def test_counterfactual_block_value_is_deterministic_and_restricted_to_remaining_arms() -> None:
    runner, result = _run(_person("F1", 3), sessions=1)
    blocks = ope.logged_blocks(result.rows(), weights="participant", sessions=(1,))
    block = blocks[3]
    probs = _logging_target(result, block)
    wide = PolicyProbabilities(l1={arm: 1.0 / 6.0 for arm in probs.l1}, l2=probs.l2, l3=None)

    def target(_unit, offer=None):
        return wide

    first = runner.counterfactual_block_value(
        result, block.unit_id, target, remaining_arms=block.remaining_arms, replays=6, seed=5
    )
    again = runner.counterfactual_block_value(
        result, block.unit_id, target, remaining_arms=block.remaining_arms, replays=6, seed=5
    )
    assert first == again
    assert first[1] == pytest.approx(len(block.remaining_arms) / 6.0)  # the target mass on R_b
    only = (block.remaining_arms[0],)
    value, mass = runner.counterfactual_block_value(
        result, block.unit_id, target, remaining_arms=only, replays=6, seed=5
    )
    start = result.block_starts[block.unit_id]
    weights = ope.weights_for(
        next(d for d in result.decisions if d.decision_id == block.unit_id),
        "participant",
    )
    t_ref = ope.t_ref_by_task(result.rows(), result.participant_key)
    direct = [
        runner.replay_block(
            start,
            only[0],
            rng=Generator(PCG64([5, r])),
            l2_probabilities=wide.l2,
            offer_probabilities=None,
            weights=weights,
            t_ref=t_ref,
        )
        for r in range(6)
    ]
    # Renormalised: the regime always comes from R_b.
    assert value == pytest.approx(float(np.mean(direct)))
    assert mass == pytest.approx(1.0 / 6.0)
    outside = PolicyProbabilities(
        l1={arm: (1.0 if arm not in block.remaining_arms else 0.0) for arm in probs.l1},
        l2=probs.l2,
        l3=None,
    )
    if sum(outside.l1.values()) > 0:
        value, mass = runner.counterfactual_block_value(
            result,
            block.unit_id,
            lambda _u, offer=None: outside,
            remaining_arms=block.remaining_arms,
            replays=2,
            seed=5,
        )
        assert mass == 0.0 and math.isnan(value)


def test_counterfactual_truth_under_the_logging_policy_matches_the_logged_rewards() -> None:
    logged, truth = [], []
    for index in range(6):
        runner, result = _run(_person("F1", index), sessions=1)
        for block in ope.logged_blocks(result.rows(), weights="participant", sessions=(1,)):
            probs = _logging_target(result, block)
            value, mass = runner.counterfactual_block_value(
                result,
                block.unit_id,
                lambda _u, offer=None, _p=probs: _p,
                remaining_arms=block.remaining_arms,
                replays=24,
                seed=index,
            )
            assert mass == pytest.approx(1.0)
            logged.append(block.reward)
            truth.append(value)
    diffs = np.asarray(logged) - np.asarray(truth)
    se = diffs.std(ddof=1) / math.sqrt(len(diffs))
    assert abs(diffs.mean()) <= 3 * se + 0.01


def test_power_reports_n_needed_and_takes_confirmatory_n_from_validation_and_holdout() -> None:
    rng = np.random.default_rng(3)
    strong = {
        f"p{i}": {
            "h2_vs_probe": float(rng.normal(0.3, 0.3)),
            "h2_vs_comparator": float(rng.normal(0.2, 0.2)),
        }
        for i in range(40)
    }
    curve = power_curve(strong, candidate_n=(4, 8, 16, 32), resamples=200, seed=1)
    assert set(curve["H2"]["power"]) == {"4", "8", "16", "32"}
    assert curve["H2"]["power"]["32"] >= curve["H2"]["power"]["4"]
    assert curve["H2"]["n_needed"] in (4, 8, 16, 32)
    assert curve["H1"]["n_needed"] is None  # no statistics for H1 here
    per_family = {
        "F1": {"n_at_power_0_8": 64},
        "F2": {"n_at_power_0_8": 24},
        "F4": {"n_at_power_0_8": 12},
        "F6": {"n_at_power_0_8": 20},
    }
    # Never the tuning family's 64.
    assert select_confirmatory_n(per_family, ("F2", "F4", "F6")) == 24
    per_family["F6"] = {"n_at_power_0_8": None}
    assert select_confirmatory_n(per_family, ("F2", "F4", "F6")) is None
    refusal = ope_refusal_at_n(
        {
            f"p{i}": {
                "in_sample_S_2": [0.0, 1.5, 0.2] * 6,
                "in_sample_S_3": [1.0] * 24,
                "out_of_sample_S_2": [3.0, 0.0] * 3,
            }
            for i in range(10)
        },
        candidate_n=(4, 16),
        resamples=50,
        seed=2,
    )
    for target in ("in_sample_S_2", "in_sample_S_3", "out_of_sample_S_2"):
        for n in ("4", "16"):
            assert set(refusal[target][n]) == {"median_ess", "refusal_rate"}
    assert refusal["in_sample_S_3"]["16"]["refusal_rate"] == 0.0
    assert refusal["out_of_sample_S_2"]["4"]["refusal_rate"] == 1.0


def test_the_simulator_never_presents_its_rows_as_human() -> None:
    _, result = _run(_person("F1"), sessions=1)
    assert {str(row.evidence_kind) for row in result.sessions} == {"simulated"}
    assert asdict(result.params.forms) == asdict(FAMILY_FORMS["F1"])

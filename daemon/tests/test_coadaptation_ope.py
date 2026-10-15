"""Off-policy estimators of section 19 on synthetic logs whose truth is known exactly."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pilot.coadaptation.core.engine import CoAdaptationEngine, EngineSettings
from pilot.coadaptation.core.ope import (
    MIN_ESS,
    BlockLog,
    FrozenTarget,
    LoggedBlock,
    LoggedStep,
    clipping_sensitivity,
    cross_fit_q_hat,
    doubly_robust,
    effective_sample_size,
    estimate_from_contributions,
    frozen_target,
    grouped_bootstrap,
    ips,
    snips,
    support_check,
    unit_contributions,
)
from pilot.coadaptation.core.policy import PolicyMode
from pilot.coadaptation.core.preferences import PreferenceSet
from pilot.coadaptation.core.records import RECORD_SCHEMAS, DecisionRecord
from pilot.coadaptation.core.state import BlockContext, ClockContext, DeviceContext, LayoutSummary
from pilot.coadaptation.core.vocabulary import DecisionLevel

L1 = DecisionLevel.BLOCK_REGIME
L2 = DecisionLevel.AMBIGUITY_RESPONSE
ARMS = ("a", "b", "c")


@dataclass(frozen=True)
class _Context:
    x: int
    task: str = "pointing"


def _block(
    unit_id: str,
    *,
    chosen: str,
    logged: dict[str, float],
    utilities: tuple[float, ...],
    positions: tuple[int, ...] | None = None,
    steps: tuple[LoggedStep, ...] = (),
    remaining: tuple[str, ...] = ARMS,
    burden: float = 0.0,
    x: int = 0,
    group: str = "p-00000001",
    session: str = "s1",
) -> LoggedBlock:
    positions = positions if positions is not None else tuple(range(len(utilities)))
    l1 = LoggedStep(L1, "", None, chosen, logged, None)
    return LoggedBlock(
        unit_id=unit_id,
        group_id=group,
        session_id=session,
        context=_Context(x),
        remaining_arms=remaining,
        steps=(l1, *steps),
        trial_utilities=utilities,
        trial_positions=positions,
        burden_term=burden,
        reward=float(np.mean(utilities)) - burden,
        behavior="policy",
    )


# ----------------------------------------------------------------------------------- contextual bandit arithmetic

_MEANS = {0: {"a": 0.2, "b": 0.5, "c": -0.1}, 1: {"a": 0.6, "b": -0.2, "c": 0.3}}
_TARGET = {0: {"a": 0.6, "b": 0.3, "c": 0.1}, 1: {"a": 0.2, "b": 0.2, "c": 0.6}}


def _bandit_target(block: LoggedBlock, step: LoggedStep) -> dict[str, float]:
    return _TARGET[block.context.x]


def _bandit(seed: int, n: int = 120) -> tuple[list[LoggedBlock], float]:
    rng = np.random.default_rng(seed)
    blocks = []
    truth = []
    uniform = {arm: 1.0 / 3.0 for arm in ARMS}
    for i in range(n):
        x = int(rng.integers(0, 2))
        arm = ARMS[int(rng.integers(0, 3))]
        reward = _MEANS[x][arm] + rng.normal(0.0, 0.2)
        blocks.append(_block(f"u{i}", chosen=arm, logged=uniform, utilities=(reward,), x=x))
        truth.append(sum(_TARGET[x][a] * _MEANS[x][a] for a in ARMS))
    return blocks, float(np.mean(truth))


def _true_q(block: LoggedBlock, arm: str) -> float:
    return _MEANS[block.context.x][arm]


def test_unclipped_ips_and_dr_center_on_the_truth_and_dr_has_lower_variance() -> None:
    errors: dict[str, list[float]] = {"ips": [], "dr": [], "snips": []}
    for seed in range(200):
        blocks, truth = _bandit(seed)
        errors["ips"].append(ips(blocks, _bandit_target, clip=None).value - truth)
        errors["dr"].append(doubly_robust(blocks, _bandit_target, _true_q, clip=None).value - truth)
        errors["snips"].append(snips(blocks, _bandit_target).value - truth)
    for name in ("ips", "dr"):
        values = np.asarray(errors[name])
        mc_se = values.std(ddof=1) / math.sqrt(len(values))
        assert abs(values.mean()) <= 3 * mc_se, name
    # SNIPS is a ratio of two random sums: consistent, not unbiased.
    assert abs(np.mean(errors["snips"])) <= 0.02
    assert np.var(errors["dr"]) < np.var(errors["ips"])


_SKEW_LOGGED = {0: {"a": 0.49, "b": 0.49, "c": 0.02}, 1: {"a": 0.02, "b": 0.49, "c": 0.49}}
_SKEW_TARGET = {0: {"a": 0.3, "b": 0.26, "c": 0.44}, 1: {"a": 0.44, "b": 0.26, "c": 0.3}}
_SKEW_MEANS = {0: {"a": 0.2, "b": 0.5, "c": 0.25}, 1: {"a": 0.25, "b": -0.2, "c": 0.3}}


def test_clipped_estimators_stay_within_their_tolerance_where_the_clip_binds() -> None:
    def target(block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
        return _SKEW_TARGET[block.context.x]

    def q_hat(block: LoggedBlock, arm: str) -> float:
        return _SKEW_MEANS[block.context.x][arm]

    errors: dict[str, list[float]] = {"ips": [], "dr": []}
    binding = 0
    for seed in range(200):
        rng = np.random.default_rng(10_000 + seed)
        blocks = []
        truth = []
        for i in range(800):
            x = int(rng.integers(0, 2))
            logged = _SKEW_LOGGED[x]
            arm = ARMS[int(rng.choice(3, p=[logged[a] for a in ARMS]))]
            reward = _SKEW_MEANS[x][arm] + rng.normal(0.0, 0.2)
            blocks.append(_block(f"u{i}", chosen=arm, logged=logged, utilities=(reward,), x=x))
            truth.append(sum(_SKEW_TARGET[x][a] * _SKEW_MEANS[x][a] for a in ARMS))
        # Default clip 20; lambda reaches 22.
        units = unit_contributions(blocks, target, q_hat=q_hat)
        binding += sum(1 for unit in units if unit.lam == 20.0)
        errors["ips"].append(
            estimate_from_contributions("ips", units).value - float(np.mean(truth)),
        )
        errors["dr"].append(estimate_from_contributions("dr", units).value - float(np.mean(truth)))
    assert binding > 0
    for name, values in errors.items():
        assert abs(np.mean(values)) <= 0.02, name


# ------------------------------------------------------------------------------------------ per-decision weights


def test_per_decision_weights_take_only_the_draws_up_to_each_trial() -> None:
    logged_l2 = {"confirm_selection:second_click": 0.5, "wait_for_evidence": 0.5}
    steps = (
        LoggedStep(L2, "b1:p1:h1", 0, "wait_for_evidence", logged_l2, None),
        LoggedStep(L2, "b1:p1:h1", 2, "confirm_selection:second_click", logged_l2, None),
        # Trial 3 was interrupted (not scored).
        LoggedStep(L2, "b2:p1:h1", 3, "wait_for_evidence", logged_l2, None),
    )
    block = _block(
        "u",
        chosen="a",
        logged={arm: 1.0 / 3.0 for arm in ARMS},
        utilities=(1.0, 2.0, 3.0),
        positions=(0, 1, 2),
        steps=steps,
        burden=0.5,
    )

    def target(_block: LoggedBlock, step: LoggedStep) -> dict[str, float]:
        if step.level == L1:
            return {"a": 0.5, "b": 0.25, "c": 0.25}
        return {"confirm_selection:second_click": 0.8, "wait_for_evidence": 0.2}

    (unit,) = unit_contributions([block], target, clip=None)
    lam = 0.5 / (1.0 / 3.0)
    wait, confirm = 0.2 / 0.5, 0.8 / 0.5
    rho = [lam * wait, lam * wait, lam * wait * confirm]
    rho_b = lam * wait * confirm * wait
    assert unit.lam == pytest.approx(lam)
    assert unit.rho_b == pytest.approx(rho_b)
    assert unit.mean_weight == pytest.approx(np.mean(rho))
    assert unit.weighted_reward == pytest.approx(
        np.mean(np.multiply(rho, (1.0, 2.0, 3.0))) - rho_b * 0.5,
    )


def test_deterministic_steps_contribute_one() -> None:
    block = _block("u", chosen="a", logged={arm: 1.0 / 3.0 for arm in ARMS}, utilities=(1.0,))
    deterministic = LoggedStep(
        L2,
        "b3:p1:h0",
        0,
        "wait_for_evidence",
        {"wait_for_evidence": 1.0},
        None,
    )
    with_step = _block(
        "u",
        chosen="a",
        logged={arm: 1.0 / 3.0 for arm in ARMS},
        utilities=(1.0,),
        steps=(deterministic,),
    )

    def target(_block: LoggedBlock, step: LoggedStep) -> dict[str, float]:
        if step.level == L1:
            return {"a": 0.5, "b": 0.5, "c": 0.0}
        return {"wait_for_evidence": 1.0}

    (plain,) = unit_contributions([block], target, clip=None)
    (deterministic_unit,) = unit_contributions([with_step], target, clip=None)
    assert deterministic_unit.mean_weight == plain.mean_weight == pytest.approx(1.5)
    assert deterministic_unit.rho_b == plain.rho_b


def _two_trial_design():
    """One block of two trials: an L1 arm from R = {a, b}, an L2 draw in trial 0 that changes both trials' utilities
    and an L2 draw in trial 1 that changes only trial 1 (later draws cannot affect earlier trials)."""
    logged_l1 = {"a": 0.5, "b": 0.5}
    logged_l2 = {"x": 0.5, "y": 0.5}
    target_l1 = {"a": 0.7, "b": 0.3}
    target_l2 = {"x": 0.9, "y": 0.1}
    base = {"a": 1.0, "b": 0.2}
    effect0 = {"x": 0.3, "y": -0.4}
    effect1 = {"x": -0.2, "y": 0.5}

    def utilities(arm: str, d0: str, d1: str) -> tuple[float, float]:
        return base[arm] + effect0[d0], base[arm] + 0.5 * effect0[d0] + effect1[d1]

    return logged_l1, logged_l2, target_l1, target_l2, utilities


def test_per_decision_and_full_product_share_the_expectation_with_less_variance() -> None:
    logged_l1, logged_l2, target_l1, target_l2, utilities = _two_trial_design()

    def target(_block: LoggedBlock, step: LoggedStep) -> dict[str, float]:
        return target_l1 if step.level == L1 else target_l2

    per_decision: list[tuple[float, float]] = []
    full_product: list[tuple[float, float]] = []
    for arm, d0, d1 in itertools.product(("a", "b"), ("x", "y"), ("x", "y")):
        probability = logged_l1[arm] * logged_l2[d0] * logged_l2[d1]
        u = utilities(arm, d0, d1)
        steps = (
            LoggedStep(L2, "c", 0, d0, logged_l2, None),
            LoggedStep(L2, "c", 1, d1, logged_l2, None),
        )
        block = _block(
            "u",
            chosen=arm,
            logged=logged_l1,
            utilities=u,
            steps=steps,
            remaining=("a", "b"),
        )
        (unit,) = unit_contributions([block], target, clip=None)
        per_decision.append((probability, unit.weighted_reward))
        full_product.append((probability, unit.rho_b * float(np.mean(u))))
    truth = sum(
        target_l1[arm] * target_l2[d0] * target_l2[d1] * float(np.mean(utilities(arm, d0, d1)))
        for arm, d0, d1 in itertools.product(("a", "b"), ("x", "y"), ("x", "y"))
    )
    mean_pd = sum(p * v for p, v in per_decision)
    mean_fp = sum(p * v for p, v in full_product)
    assert mean_pd == pytest.approx(truth, abs=1e-12)
    assert mean_fp == pytest.approx(truth, abs=1e-12)
    var_pd = sum(p * (v - mean_pd) ** 2 for p, v in per_decision)
    var_fp = sum(p * (v - mean_fp) ** 2 for p, v in full_product)
    assert var_pd < var_fp


def test_dr_with_any_q_keeps_the_per_decision_expectation() -> None:
    logged_l1, logged_l2, target_l1, target_l2, utilities = _two_trial_design()

    def target(_block: LoggedBlock, step: LoggedStep) -> dict[str, float]:
        return target_l1 if step.level == L1 else target_l2

    def q_hat(_block: LoggedBlock, arm: str) -> float:
        return {"a": 0.4, "b": -0.3}[arm]  # deliberately wrong

    expected = 0.0
    truth = 0.0
    for arm, d0, d1 in itertools.product(("a", "b"), ("x", "y"), ("x", "y")):
        probability = logged_l1[arm] * logged_l2[d0] * logged_l2[d1]
        u = utilities(arm, d0, d1)
        steps = (
            LoggedStep(L2, "c", 0, d0, logged_l2, None),
            LoggedStep(L2, "c", 1, d1, logged_l2, None),
        )
        block = _block(
            "u",
            chosen=arm,
            logged=logged_l1,
            utilities=u,
            steps=steps,
            remaining=("a", "b"),
        )
        (unit,) = unit_contributions([block], target, clip=None, q_hat=q_hat)
        expected += probability * unit.dr_term
        truth += target_l1[arm] * target_l2[d0] * target_l2[d1] * float(np.mean(u))
    assert expected == pytest.approx(truth, abs=1e-12)


# ------------------------------------------------------------------------------------ permuted rounds and support

_ROUND_ARMS = ("a", "b", "c")
_REWARDS = {
    0: {"a": 1.0, "b": 0.4, "c": -0.2},
    1: {"a": 0.1, "b": 0.8, "c": 0.3},
    2: {"a": -0.5, "b": 0.6, "c": 0.9},
}
# Position 0 puts 0.2 on an arm the design never offers; positions 1 and 2 give every round arm the same mass, so the
# normaliser sum_b pi_e(R_b | x_b) is the same for every permutation and the ratio's expectation is exact.
_ROUND_TARGET = {
    0: {"a": 0.5, "b": 0.2, "c": 0.1, "d": 0.2},
    1: {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3, "d": 0.0},
    2: {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
}


def _round_target(block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
    return _ROUND_TARGET[block.context.x]


def test_permuted_round_mass_outside_r_b_is_withheld_and_the_estimand_is_exact() -> None:
    def q_hat(block: LoggedBlock, arm: str) -> float:
        return 0.3 if arm == "a" else -0.1

    expected_ips = expected_dr = expected_truth = 0.0
    permutations = list(itertools.permutations(_ROUND_ARMS))
    for order in permutations:
        probability = 1.0 / len(permutations)
        blocks = []
        remaining = list(_ROUND_ARMS)
        numerator_truth = 0.0
        denominator = 0.0
        for position, arm in enumerate(order):
            share = 1.0 / len(remaining)
            logged = {c: (share if c in remaining else 0.0) for c in _ROUND_ARMS}
            blocks.append(
                _block(
                    f"u{position}",
                    chosen=arm,
                    logged=logged,
                    utilities=(_REWARDS[position][arm],),
                    remaining=tuple(remaining),
                    x=position,
                )
            )
            mass = sum(_ROUND_TARGET[position][candidate] for candidate in remaining)
            v_b = sum(_ROUND_TARGET[position][c] * _REWARDS[position][c] for c in remaining) / mass
            numerator_truth += mass * v_b
            denominator += mass
            remaining.remove(arm)
        units = unit_contributions(blocks, _round_target, clip=None, q_hat=q_hat)
        assert sum(u.mass for u in units) == pytest.approx(denominator)
        expected_ips += probability * sum(u.weighted_reward for u in units) / denominator
        expected_dr += probability * sum(u.dr_term for u in units) / denominator
        expected_truth += probability * numerator_truth / denominator
    assert expected_ips == pytest.approx(expected_truth, abs=1e-12)
    assert expected_dr == pytest.approx(expected_truth, abs=1e-12)
    blocks = [
        _block(
            "u0",
            chosen="a",
            logged={a: 1 / 3 for a in _ROUND_ARMS},
            utilities=(1.0,),
            x=0,
            remaining=_ROUND_ARMS,
        )
    ]
    report = support_check(blocks, _round_target)
    assert report.withheld_mass == pytest.approx(0.2)


def test_unsupported_mass_of_a_relabelled_decision_is_reported_and_withheld() -> None:
    # A UI fallback relabelled the decision baseline: propensity 1 for the applied regime, nothing else supported.
    block = _block("u", chosen="a", logged={"a": 1.0}, utilities=(1.0,), remaining=("a", "b"))

    def target(_block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
        return {"a": 0.4, "b": 0.5, "c": 0.1}

    report = support_check([block], target)
    assert report.unsupported_by_level["block_regime"] == pytest.approx(0.5)
    assert report.withheld_mass == pytest.approx(0.6)
    (unit,) = unit_contributions([block], target, clip=None)
    assert unit.mass == pytest.approx(0.4)
    assert unit.lam == pytest.approx(0.4)


def test_a_supported_choice_with_zero_target_mass_is_not_an_unsupported_unit() -> None:
    # A Phase A round logs every arm uniformly; an adaptive target never chooses assistance_off (C2), so the unit that
    # ran assistance_off has lambda 0 although its choice lies inside R_b ∩ supp(pi_b).
    arms = ("assistance_off", "act", "enlarge_target:1.5", "enlarge_target:2.0", "fast_confirm", "hold")
    logged = {arm: 1 / len(arms) for arm in arms}
    blocks = [
        _block(f"u{i}", chosen=arm, logged=logged, utilities=(1.0,), remaining=arms, x=i) for i, arm in enumerate(arms)
    ]

    def target(_block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
        return {arm: (0.0 if arm == "assistance_off" else 0.2) for arm in arms}

    report = support_check(blocks, target)
    assert report.unsupported_fraction == 0.0 and report.withheld_mass == pytest.approx(0.0)
    assert report.unsupported_units == 0
    assert report.zero_target_units == 1
    assert report.to_dict()["zero_target_units"] == 1
    units = {unit.chosen: unit for unit in unit_contributions(blocks, target, clip=None)}
    off = units["assistance_off"]
    assert off.lam == 0.0 and not off.unsupported_unit and off.zero_target_unit
    assert not any(unit.unsupported_unit or unit.zero_target_unit for c, unit in units.items() if c != "assistance_off")
    # A logged choice outside R_b (a relabelled decision) is the unsupported case, whatever the target gives it.
    relabelled = _block("r", chosen="a", logged={"a": 1.0}, utilities=(1.0,), remaining=("b", "c"))

    def likes_a(_block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
        return {"a": 0.5, "b": 0.3, "c": 0.2}

    report = support_check([relabelled], likes_a)
    assert report.unsupported_units == 1 and report.zero_target_units == 0
    (unit,) = unit_contributions([relabelled], likes_a, clip=None)
    assert unit.lam == 0.0 and unit.unsupported_unit and not unit.zero_target_unit


# --------------------------------------------------------------------------------------- ESS, refusal and clipping


def _uniform_blocks(n: int) -> list[LoggedBlock]:
    return [
        _block(
            f"u{i}",
            chosen=ARMS[i % 3],
            logged={arm: 1 / 3 for arm in ARMS},
            utilities=(1.0 + i % 3,),
        )
        for i in range(n)
    ]


def _uniform_target(_block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
    return {"a": 0.5, "b": 0.3, "c": 0.2}


def test_ess_comes_from_mean_block_weights_and_low_ess_is_refused() -> None:
    small = ips(_uniform_blocks(12), _uniform_target)
    assert small.refused == "low_ess"
    assert small.value is None
    assert small.to_dict()["reported_as"] == "not estimated"
    assert small.ess < MIN_ESS
    blocks = _uniform_blocks(90)
    estimate = ips(blocks, _uniform_target)
    weights = [unit.mean_weight for unit in unit_contributions(blocks, _uniform_target)]
    assert estimate.refused is None
    assert estimate.ess == pytest.approx(effective_sample_size(weights))
    assert estimate.ess == pytest.approx(sum(weights) ** 2 / sum(w * w for w in weights))
    assert effective_sample_size([0.0, 0.0]) == 0.0


def test_clipping_sensitivity_reports_every_clip_without_replacing_the_estimate() -> None:
    blocks = _uniform_blocks(120)

    def concentrated(_block: LoggedBlock, _step: LoggedStep) -> dict[str, float]:
        return {"a": 0.9, "b": 0.05, "c": 0.05}

    q_hat = cross_fit_q_hat(blocks)
    report = clipping_sensitivity(blocks, concentrated, q_hat, clips=(1, 2, 20, None))
    assert set(report) == {"1", "2", "20", "none"}
    assert report["1"]["ips"] != report["none"]["ips"]
    assert report["20"]["ips"] == report["none"]["ips"]  # lambda is at most 2.7 here
    assert ips(blocks, concentrated).value == report["20"]["ips"]


def test_cross_fit_q_hat_leaves_the_block_session_out() -> None:
    blocks = [
        _block(
            "u1",
            chosen="a",
            logged={arm: 1 / 3 for arm in ARMS},
            utilities=(1.0,),
            session="s1",
        ),
        _block(
            "u2",
            chosen="a",
            logged={arm: 1 / 3 for arm in ARMS},
            utilities=(3.0,),
            session="s2",
        ),
    ]
    q_hat = cross_fit_q_hat(blocks)
    assert q_hat(blocks[0], "a") == pytest.approx(3.0)
    assert q_hat(blocks[1], "a") == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------- grouped bootstrap


def test_grouped_bootstrap_needs_two_groups() -> None:
    with pytest.raises(ValueError):
        grouped_bootstrap({"p1": [1.0, 2.0]}, lambda sample: float(np.mean(sample)), n=10)


def test_grouped_bootstrap_resamples_whole_groups_then_sessions() -> None:
    seen: list[float] = []

    def stat(sample: np.ndarray) -> float:
        value = float(np.mean(sample))
        seen.append(value)
        return value

    lo, hi = grouped_bootstrap({"p1": [0.0, 0.0], "p2": [1.0, 1.0]}, stat, n=400, seed=3)
    # Whole groups: a replicate holds both sessions of each drawn group, so only 0, 0.5 and 1 can occur.
    assert set(np.round(seen, 9)) <= {0.0, 0.5, 1.0}
    assert lo == 0.0 and hi == 1.0
    again = grouped_bootstrap(
        {"p1": [0.0, 0.0], "p2": [1.0, 1.0]},
        lambda s: float(np.mean(s)),
        n=400,
        seed=3,
    )
    assert again == (lo, hi)
    mixed: list[float] = []
    grouped_bootstrap(
        {"p1": [0.0, 1.0], "p2": [0.0, 1.0]},
        lambda s: mixed.append(float(np.mean(s))) or 0.0,
        n=200,
    )
    assert len(set(np.round(mixed, 9))) > 1  # sessions are resampled within each drawn group


def test_estimates_carry_a_grouped_interval_with_two_participants() -> None:
    blocks = [
        _block(
            f"u{i}",
            chosen=ARMS[i % 3],
            logged={arm: 1 / 3 for arm in ARMS},
            utilities=(float(i % 3),),
            group=f"p-0000000{i % 2}",
            session=f"s{i % 4}",
        )
        for i in range(120)
    ]
    estimate = ips(blocks, _uniform_target)
    assert estimate.ci95 is not None
    assert estimate.ci95[0] - 1e-9 <= estimate.value <= estimate.ci95[1] + 1e-9  # bounds may round past the value


# -------------------------------------------------------------------------------------------------- frozen target


def _block_log() -> BlockLog:
    started = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
    layout = LayoutSummary(
        task="pointing",
        n_candidates=12,
        w_px=28.0,
        h_px=28.0,
        gap_px=2.0,
        id_bar=3.2,
        menu_n=0,
        origin_x=320.0,
        origin_y=396.0,
    )
    device = DeviceContext(
        modality="mouse",
        pointer_types=("mouse",),
        dpr=1.0,
        surface_w=640.0,
        surface_h=420.0,
        gaze_available=False,
        hand_pointer=False,
    )
    block = BlockContext(
        session_id="s1",
        block_index=1,
        phase="A",
        task="pointing",
        layouts=(layout,) * 8,
        device=device,
        clock=ClockContext(
            day_index=740000,
            session_minute=2.0,
            trials_today=8,
            days_since_last_session=None,
        ),
        previous_regime=None,
        hand_quality_recent=None,
    )
    return BlockLog(
        participant_key="p-0000abcd",
        session_id="s1",
        session_number=2,
        block_index=1,
        phase="A",
        task="pointing",
        decision_id="d-1",
        created_at=started + timedelta(minutes=2),
        block=block,
        session_exclusions={},
        applied_regime="act",
    )


def test_frozen_target_is_deterministic_never_learns_and_uses_the_session_settings() -> None:
    engine = CoAdaptationEngine(
        participant_key="p-0000abcd",
        settings=EngineSettings(),
        preferences=PreferenceSet(()),
    )
    snapshot = engine.snapshot()
    context = _block_log()
    block = LoggedBlock(
        unit_id="d-1",
        group_id="p-0000abcd",
        session_id="s1",
        context=context,
        remaining_arms=("act", "enlarge_target:1.5"),
        steps=(LoggedStep(L1, "", None, "act", {"act": 0.5, "enlarge_target:1.5": 0.5}, None),),
        trial_utilities=(0.1,),
        trial_positions=(0,),
        burden_term=0.0,
        reward=0.1,
        behavior="policy",
    )
    kwargs = {"participant_key": "p-0000abcd", "preferences_history": (), "rows_hash": "0" * 64}
    narrow = frozen_target(snapshot, settings=EngineSettings(exploration_floor=0.1), **kwargs)
    wide = frozen_target(snapshot, settings=EngineSettings(exploration_floor=0.3), **kwargs)
    first = dict(narrow(block, block.steps[0]))
    assert dict(narrow(block, block.steps[0])) == first
    assert (
        dict(
            frozen_target(snapshot, settings=EngineSettings(exploration_floor=0.1), **kwargs)(
                block,
                block.steps[0],
            )
        )
        == first
    )
    assert sum(first.values()) == pytest.approx(1.0)
    support = [arm for arm, p in first.items() if p > 0]
    floor = min(p for p in dict(wide(block, block.steps[0])).values() if p > 0)
    assert floor >= 0.3 / len(support) - 1e-12
    assert narrow.snapshot() == snapshot
    assert engine.snapshot() == snapshot


def test_frozen_target_applies_a_logged_veto_at_the_level_of_the_evaluated_decision() -> None:
    """``confirm_selection:second_click`` is an L1 regime and an L2 response. Live, a "Not this time" excludes it at
    the next decision of its own level only (C5), so the target passes the evaluated decision's level on."""
    second_click = "confirm_selection:second_click"
    engine = CoAdaptationEngine(participant_key="p-0000abcd", settings=EngineSettings(), preferences=PreferenceSet(()))
    target = FrozenTarget(
        engine.snapshot(),
        participant_key="p-0000abcd",
        settings=EngineSettings(),
        preferences_history=(),
        rows_hash="0" * 64,
    )
    vetoed = replace(_block_log(), session_exclusions={second_click: "recent_veto"})
    at_l1 = target.probabilities(vetoed)
    assert at_l1.l1[second_click] == 0.0
    kept = [cell for cell, probs in at_l1.l2.items() if probs.get(second_click, 0.0) > 0.0]
    assert kept  # the regime veto (block's L1 decision) never removes the response from the block's table
    payload: dict = {key: None for key in RECORD_SCHEMAS["AmbiguityTableV1"]}
    payload["cells"] = [{"cell_id": kept[0], "excluded": {second_click: "recent_veto"}}]
    table = DecisionRecord(
        decision_id="d-table",
        session_id="s1",
        participant_key="p-0000abcd",
        seq=2,
        level=L2,
        mode=PolicyMode.ADAPTIVE,
        behavior="policy",
        created_at=vetoed.created_at + timedelta(seconds=30),
        superseded_by=None,
        payload=payload,
        applied=None,
        fallback_reason=None,
        versions={},
    )
    at_l2 = target.probabilities(vetoed, decision=table)
    assert all(probs.get(second_click, 0.0) == 0.0 for probs in at_l2.l2.values())


def test_table_steps_keep_the_draws_of_a_table_relabelled_only_for_a_stale_table() -> None:
    """A stale_table fallback names the table requested after a veto. A trial that applied its draws once it arrived
    ran under its logged propensities, so those draws stay policy steps even when an earlier trial's stale fallback
    relabelled it; a table relabelled for a late or failed reply was never received and gives no step."""
    from pilot.coadaptation.core.ope import _table_steps
    from tests.test_coadaptation_audit import T0, act, l1, l2, table, trial

    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    for reason, steps in (("stale_table", 1), ("late", 0), (None, 1)):
        tab = table(block, seq=2, created=T0 + timedelta(seconds=6))
        if reason is not None:
            tab = replace(tab, behavior="baseline", fallback_reason=reason)
        applied = trial(
            block,
            seq=3,
            tables=(tab.decision_id,),
            activations=(act(0, a_gate=0.5, n_plausible=2, l2=l2(tab, "confirm_selection:second_click")),),
        )
        assert len(_table_steps([applied], {tab.decision_id: tab})) == steps, reason

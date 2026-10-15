"""Constrained Thompson policy: properties P1-P7, modes, seeds and the decision-table fixture (spec 9.2-9.5, 22.1)."""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pilot.coadaptation.core.constraints import ConstraintState, ExclusionReason, fallbacks
from pilot.coadaptation.core.outcome_model import EvidenceCounts, InteractionPrediction, OutcomeDraws, OutcomeEstimate
from pilot.coadaptation.core.policy import (
    POLICY_VERSION,
    AmbiguityTable,
    BlockRequest,
    ConstrainedThompsonPolicy,
    Decision,
    NoAllowedRegime,
    OfferRequest,
    PolicyMode,
    PolicySettings,
    TableRequest,
    argmax_frequency,
    cell_specs,
    decision_seed,
    draw_stream,
    reachable_cells,
    seed_hex,
    select_from_cell,
)
from pilot.coadaptation.core.preferences import PreferenceKind, PreferenceSet
from pilot.coadaptation.core.vocabulary import L2_OPTIONS, L3_OPTIONS, VARIANTS, DecisionLevel, l1_arms
from tests.test_coadaptation_constraints import _random_prefs
from tests.test_coadaptation_engine import AH2, Run, U, l2, pref
from tests.test_coadaptation_records import LAYOUT, PK, SALT, T0

DAEMON = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "coadaptation_decision_table.json"
MODALITIES = ("mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer")
SETTINGS = PolicySettings(thompson_draws=32)
VERSIONS = {"policy": POLICY_VERSION}


# ------------------------------------------------------------------------------------- random model outputs


def _estimate(rng: np.random.Generator, scale: float = 1.0) -> OutcomeEstimate:
    lo, mean, hi = np.sort(rng.random(3) * scale)
    return OutcomeEstimate(mean=float(mean), lo90=float(lo), hi90=float(hi))


def _prediction(rng: np.random.Generator, variant: str, level: DecisionLevel) -> InteractionPrediction:
    ua = _estimate(rng, 0.2)
    return InteractionPrediction(
        variant=variant,
        level=level,
        p_complete=_estimate(rng),
        p_unintended=ua,
        ucb_unintended=float(ua.hi90 + rng.random() * 0.05),
        p_correction=_estimate(rng, 0.3),
        p_veto=_estimate(rng, 0.1),
        time_ms=OutcomeEstimate(800.0, 500.0, 2000.0),
        burden={"effort": 3.0, "intrusiveness": 2.0, "control": 5.0} if level == DecisionLevel.BLOCK_REGIME else None,
        evidence=EvidenceCounts(int(rng.integers(0, 50)), 0, "literature_prior"),
    )


def _sampler(bias: np.ndarray, outcomes: tuple[str, ...]):
    """A model state: per-variant location of each sampled outcome; the draws come from the decision's rng_ts."""

    def sample(rng: np.random.Generator, *args: Any) -> OutcomeDraws:
        variants, draws = args[-2], args[-1]
        values = {}
        for k, outcome in enumerate(outcomes):
            loc = np.array([bias[hash(v) % len(bias), k] for v in variants])
            values[outcome] = np.clip(loc[None, :] + 0.1 * rng.standard_normal((draws, len(variants))), 0.0, 1.0)
            if outcome == "log_time":
                values[outcome] = values[outcome] * 2.0
            if outcome in ("effort", "intrusiveness"):
                values[outcome] = 1.0 + 6.0 * values[outcome]
        return OutcomeDraws(variants=tuple(variants), values=values)

    return sample


L1_OUTCOMES = ("complete", "ua", "correction", "veto", "log_time", "effort", "intrusiveness")
L2_OUTCOMES = ("complete", "ua", "correction", "veto", "log_time")
OFFER_OUTCOMES = ("accepted", "dismissed")


def _state(rng: np.random.Generator, modality: str) -> ConstraintState:
    state = ConstraintState(calibration_suspended=bool(rng.random() < 0.3))
    for variant in l1_arms(modality):
        if rng.random() < 0.1:
            state.record_veto(DecisionLevel.BLOCK_REGIME, variant)
        if rng.random() < 0.1:
            state.add_control_floor(variant)
    for variant in L2_OPTIONS:
        if rng.random() < 0.1:
            state.record_veto(DecisionLevel.AMBIGUITY_RESPONSE, variant)
    if rng.random() < 0.2:
        state.dismiss_offer()
    return state


def _block_request(rng: np.random.Generator, prefs, modality, mode, state, *, decision_id="d") -> BlockRequest:
    predictions = {v: _prediction(rng, v, DecisionLevel.BLOCK_REGIME) for v in l1_arms(modality)}
    bias = rng.random((64, len(L1_OUTCOMES)))
    remaining = None if rng.random() < 0.5 else frozenset(rng.choice(l1_arms(modality), 2, replace=False))
    return BlockRequest(
        decision_id=decision_id,
        participant_key=PK,
        seed=decision_seed(PK, SALT, decision_id),
        mode=mode,
        modality=modality,
        preferences=prefs,
        state=state,
        fallbacks=fallbacks(
            prefs, modality, participant_key=PK, calibration_suspended=state.calibration_suspended, now=T0
        ),
        settings=SETTINGS,
        predictions=predictions,
        sampler=_sampler(bias, L1_OUTCOMES),
        previous_regime=str(rng.choice(l1_arms(modality))) if rng.random() < 0.7 else None,
        priorities=tuple(float(x) for x in rng.random(3)),  # type: ignore[arg-type]
        t_ref_s=1.0,
        spread=None if modality == "keyboard" else (1.0, 1.0),
        round_remaining=remaining,
        round_index=0,
        comparator_ranking=tuple(rng.permutation([v for v in l1_arms(modality) if v != "assistance_off"])),
        protocol_deviation=False,
        versions=VERSIONS,
    )


def _table_request(rng, prefs, modality, mode, state, *, decision_id="t", start=0, trials=8) -> TableRequest:
    bias = rng.random((64, len(L2_OUTCOMES)))
    used = int(rng.integers(0, 4))

    def predict_cell(context, variants):
        return {v: _prediction(rng, v, DecisionLevel.AMBIGUITY_RESPONSE) for v in variants}

    return TableRequest(
        decision_id=decision_id,
        participant_key=PK,
        block_decision_id="d",
        seed=decision_seed(PK, SALT, decision_id),
        mode=mode,
        modality=modality,
        preferences=prefs,
        state=state,
        fallbacks=fallbacks(
            prefs, modality, participant_key=PK, calibration_suspended=state.calibration_suspended, now=T0
        ),
        settings=SETTINGS,
        cells=cell_specs(0.25, LAYOUT, modality),
        predict_cell=predict_cell,
        sample_cell=_sampler(bias, L2_OUTCOMES),
        priorities=(0.5, 0.5, 0.5),
        interruptions_used=used,
        start_trial_index=start,
        trials_in_block=trials,
        versions=VERSIONS,
    )


def _offer_request(rng, prefs, modality, mode, state, *, decision_id="o") -> OfferRequest:
    return OfferRequest(
        decision_id=decision_id,
        participant_key=PK,
        seed=decision_seed(PK, SALT, decision_id),
        mode=mode,
        modality=modality,
        keyboard_available=bool(rng.random() < 0.7),
        preferences=prefs,
        state=state,
        settings=SETTINGS,
        sampler=_sampler(rng.random((64, 2)), OFFER_OUTCOMES),
        priorities=(0.5, 0.5, 0.5),
        versions=VERSIONS,
    )


def _check_distribution(propensities: dict[str, float], eligible, excluded, chosen: str | None = None) -> None:
    assert abs(sum(propensities.values()) - 1.0) <= 1e-9  # P5
    for variant, p in propensities.items():
        assert p >= 0.0
        if p > 0.0:
            assert variant in eligible and variant not in excluded  # P1
    if chosen is not None:
        assert propensities[chosen] > 0.0  # P5


def _forbidden_or_below(prefs: PreferenceSet, variant: str, modality: str) -> bool:
    rules = prefs.active(context="study", modality=modality, participant_key=PK)  # type: ignore[arg-type]
    if variant != "wait_for_evidence" and any(r.forbids(variant) for r in rules):
        return True
    minimum = prefs.min_caution(modality=modality, participant_key=PK)  # type: ignore[arg-type]
    return (
        minimum is not None
        and variant != "wait_for_evidence"
        and 0 <= VARIANTS[variant].caution < VARIANTS[minimum].caution
    )


def test_properties_p1_to_p7_over_ten_thousand_seeded_cases() -> None:
    policy = ConstrainedThompsonPolicy()
    rng = np.random.default_rng(20261015)
    modes = [PolicyMode.RANDOMIZED, PolicyMode.ADAPTIVE, PolicyMode.PROBE, PolicyMode.COMPARATOR]
    for case in range(10_000):
        prefs = _random_prefs(rng)
        modality = str(rng.choice(MODALITIES))
        state = _state(rng, modality)
        held = prefs.held_regime(modality=modality, participant_key=PK)  # type: ignore[arg-type]
        mode = PolicyMode.PINNED if held is not None else modes[int(rng.integers(0, len(modes)))]
        pin_act = prefs.pin_act(modality=modality, participant_key=PK)  # type: ignore[arg-type]
        level = case % 3
        if level == 0:
            request = _block_request(rng, prefs, modality, mode, state, decision_id=f"d{case}")
            try:
                decision = policy.decide_block(request)
            except NoAllowedRegime:
                assert request.fallbacks.l1 is None
                continue
            assert set(decision.propensities) == set(l1_arms(modality))
            _check_distribution(decision.propensities, decision.eligible, decision.excluded, decision.variant)
            assert decision.authority == "presentation_only" and decision.execution_authority is False  # P7
            for variant, p in decision.propensities.items():
                if p > 0 and not decision.fallback and mode is not PolicyMode.PINNED:
                    assert not _forbidden_or_below(prefs, variant, modality)  # P2
            if decision.fallback:
                assert decision.variant == request.fallbacks.l1
            if mode is PolicyMode.ADAPTIVE and not decision.fallback:
                assert decision.propensities["assistance_off"] == 0.0
            if mode is PolicyMode.PINNED:
                assert decision.variant == held and decision.propensities[held] == 1.0
        elif level == 1:
            request = _table_request(rng, prefs, modality, mode, state, decision_id=f"t{case}")
            table = policy.decide_table(request)
            assert table.authority == "presentation_only" and table.execution_authority is False  # P7
            assert [c.cell_id for c in table.cells] == [
                f"{b}:p{n}:h{1 if h else 0}" for b, n, h in reachable_cells(0.25)
            ]
            for cell in table.cells:
                assert list(cell.propensities) == list(L2_OPTIONS)
                _check_distribution(cell.propensities, cell.eligible, cell.excluded)
                if pin_act:  # P3: act only in a pin_act collapse, with propensity 1
                    expected = "act" if cell.has_hit else "wait_for_evidence"
                    assert cell.propensities[expected] == 1.0 and table.mode is PolicyMode.PINNED
                else:
                    assert cell.propensities["act"] == 0.0
                    assert cell.excluded.get("act") in (
                        ExclusionReason.AMBIGUOUS_ACTIVATION,
                        ExclusionReason.NOT_APPLICABLE,
                    )
                    assert "wait_for_evidence" in cell.eligible  # P4
                    for variant, p in cell.propensities.items():
                        if p > 0:
                            assert not _forbidden_or_below(prefs, variant, modality)  # P2
        else:
            request = _offer_request(rng, prefs, modality, mode, state, decision_id=f"o{case}")
            decision = policy.decide_offer(request)
            assert set(decision.propensities) == set(L3_OPTIONS)
            _check_distribution(decision.propensities, decision.eligible, decision.excluded, decision.variant)
            if mode in (PolicyMode.PROBE, PolicyMode.PINNED):
                assert decision.variant == "no_offer" and decision.mode is mode
            if state.offers_blocked():
                assert decision.propensities["no_offer"] == 1.0


def test_p6_same_inputs_reproduce_and_draws_depend_only_on_the_seed() -> None:
    policy = ConstrainedThompsonPolicy()
    for case in range(200):
        prefs = _random_prefs(np.random.default_rng(case))
        modality = MODALITIES[case % len(MODALITIES)]
        mode = (
            PolicyMode.ADAPTIVE
            if prefs.held_regime(modality=modality, participant_key=PK) is None
            else PolicyMode.PINNED
        )
        state = ConstraintState()
        one = _block_request(np.random.default_rng(1000 + case), prefs, modality, mode, state, decision_id=f"x{case}")
        two = _block_request(np.random.default_rng(1000 + case), prefs, modality, mode, state, decision_id=f"x{case}")
        other_model = _block_request(
            np.random.default_rng(5000 + case), prefs, modality, mode, state, decision_id=f"x{case}"
        )
        try:
            a, b, c = policy.decide_block(one), policy.decide_block(two), policy.decide_block(other_model)
        except NoAllowedRegime:
            continue
        assert a.to_dict() == b.to_dict()
        assert a.u == c.u == float(draw_stream(a.seed).random())  # the selecting u depends only on the seed
        assert a.to_dict()["seed"] == seed_hex(decision_seed(PK, SALT, f"x{case}"))
        t1 = policy.decide_table(_table_request(np.random.default_rng(case), prefs, modality, mode, state))
        t2 = policy.decide_table(_table_request(np.random.default_rng(case + 77), prefs, modality, mode, state))
        assert t1.draws == t2.draws
        rng = draw_stream(t1.seed)
        assert t1.draws == tuple(tuple(float(rng.random()) for _ in range(4)) for _ in range(8))


def test_the_seed_depends_on_the_session_salt() -> None:
    a = decision_seed(PK, SALT, "decision-1")
    b = decision_seed(PK, "f" * 32, "decision-1")
    assert a != b and len(seed_hex(a)) == 16
    import hashlib

    expected = int.from_bytes(hashlib.sha256(f"{PK}|{SALT}|decision-1|{POLICY_VERSION}".encode()).digest()[:8], "big")
    assert a == expected


# ------------------------------------------------------------------------------------------------------- modes


def test_randomized_propensities_are_one_over_the_remaining_arms() -> None:
    policy = ConstrainedThompsonPolicy()
    rng = np.random.default_rng(3)
    request = _block_request(rng, PreferenceSet([]), "mouse", PolicyMode.RANDOMIZED, ConstraintState())
    remaining = frozenset({"act", "enlarge_target:2.0", "confirm_selection:hold_700"})
    from dataclasses import replace

    decision = policy.decide_block(replace(request, round_remaining=remaining))
    for variant in l1_arms("mouse"):
        expected = 1 / 3 if variant in remaining else 0.0
        assert decision.propensities[variant] == pytest.approx(expected)
        if variant not in remaining:
            assert decision.excluded[variant] is ExclusionReason.USED_THIS_ROUND
    forbidden = PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "act")])
    request = _block_request(rng, forbidden, "mouse", PolicyMode.RANDOMIZED, ConstraintState())
    decision = policy.decide_block(replace(request, round_remaining=remaining))
    assert decision.propensities["act"] == 0.0 and decision.excluded["act"] is ExclusionReason.FORBIDDEN_BY_YOU
    assert decision.propensities["enlarge_target:2.0"] == pytest.approx(0.5)
    # a round whose remaining arms are all excluded starts a new round over the safe set
    exhausted = policy.decide_block(replace(request, round_remaining=frozenset({"act"}), round_index=4))
    assert exhausted.round_index == 5 and exhausted.propensities["assistance_off"] == pytest.approx(1 / 5)


def test_adaptive_mode_mixes_thompson_with_the_exploration_floor() -> None:
    policy = ConstrainedThompsonPolicy()
    rng = np.random.default_rng(11)
    request = _block_request(rng, PreferenceSet([]), "mouse", PolicyMode.ADAPTIVE, ConstraintState())
    decision = policy.decide_block(request)
    assert decision.p_thompson is not None and decision.exploration == pytest.approx(0.1)
    support = [v for v in l1_arms("mouse") if decision.propensities[v] > 0]
    assert "assistance_off" not in support
    for variant in support:
        expected = 0.9 * decision.p_thompson[variant] + 0.1 / len(support)
        assert decision.propensities[variant] == pytest.approx(expected)


def test_probe_pinned_and_comparator_blocks() -> None:
    # probe: assistance_off, fixed L2 rule, no offers, logged with the block's mode
    run = Run()
    run.begin()
    probe = run.decide(run.block(1, "probe"))
    assert probe.mode is PolicyMode.PROBE and probe.variant == "assistance_off" and probe.propensity == 1.0
    offer = run.engine.decide_offer(decision_id="o1", context=_offer_context(1), now=run.tick())
    assert offer.mode is PolicyMode.PROBE and offer.variant == "no_offer" and offer.propensity == 1.0
    # a probe whose plain interface the person excluded uses the least cautious allowed regime (deviation)
    run = Run(PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "assistance_off")]))
    run.begin()
    probe = run.decide(run.block(1, "probe"))
    assert probe.variant == "act" and probe.protocol_deviation
    # pinned: the frozen regime, the fixed L2 rule in every cell, no offers
    run = Run(PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "enlarge_target:1.5"})]))
    run.begin()
    block = run.block(1, "A")
    pinned = run.decide(block)
    assert pinned.mode is PolicyMode.PINNED and pinned.variant == "enlarge_target:1.5" and pinned.protocol_deviation
    table = run.table(pinned, block)
    fb = run.fallbacks()
    assert table.mode is PolicyMode.PINNED
    for cell in table.cells:
        rule = next(v for v in fb.l2 if cell.propensities[v] == 1.0)
        assert rule == (
            "ask_clarification:3"
            if cell.n_plausible == 3
            else "ask_clarification:2"
            if cell.n_plausible == 2
            else "wait_for_evidence"
        )
    offer = run.engine.decide_offer(decision_id="o2", context=_offer_context(1), now=run.tick())
    assert offer.mode is PolicyMode.PINNED and offer.variant == "no_offer"
    # comparator: the first allowed entry of the ranking, propensity 1, uniform L2 and L3, no safety filter
    ranking = ("enlarge_target:2.0", "act", "confirm_selection:hold_700")
    run = Run(settings=_settings(adaptive=True))
    run.begin(session_number=2, comparator=ranking)
    block = run.block(1, "comparator")
    first = run.decide(block)
    assert first.mode is PolicyMode.COMPARATOR and first.variant == "enlarge_target:2.0"
    assert first.propensity == 1.0 and not first.protocol_deviation
    assert first.excluded["act"] is ExclusionReason.COMPARATOR_SCHEDULE
    table = run.table(first, block)
    assert table.mode is PolicyMode.COMPARATOR
    for cell in table.cells:
        positive = [p for p in cell.propensities.values() if p > 0]
        assert positive and all(p == pytest.approx(positive[0]) for p in positive)
        assert ExclusionReason.SLIP_RISK_ABOVE_LIMIT not in cell.excluded.values()
    offer = run.engine.decide_offer(decision_id="o3", context=_offer_context(1), now=run.tick())
    assert offer.mode is PolicyMode.COMPARATOR
    assert offer.propensities["no_offer"] == pytest.approx(1 / 2)
    run = Run(PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "enlarge_target")]), settings=_settings(adaptive=True))
    run.begin(session_number=2, comparator=ranking)
    second = run.decide(run.block(1, "comparator"))
    assert second.variant == "act" and second.protocol_deviation


def _offer_context(block_index: int):
    from pilot.coadaptation.core.state import OfferContext

    return OfferContext("s1", block_index, 0, "mouse", "pointing", "corrections", True)


def _settings(*, adaptive: bool = False):
    from tests.test_coadaptation_records import engine_settings

    return engine_settings(adaptive_allowed=adaptive)


def test_policy_probabilities_are_adaptive_and_leave_no_trace() -> None:
    run = Run(settings=_settings(adaptive=True))
    run.begin(session_number=2)
    block = run.block(1, "B")
    decision = run.decide(block)
    run.observe(run.trial(decision, (run.activation(U),), trial_index=0))
    snapshot = run.engine.snapshot()
    session = run.engine._session  # noqa: SLF001
    before = (dict(session.decisions), set(session.open), session.round_remaining, session.round_index)
    kwargs = dict(
        block=block,
        session_exclusions={"act": ExclusionReason.RECENT_VETO},
        preferences=run.prefs,
        seed=99,
        draws=128,
        now=run.tick(),
    )
    first = run.engine.policy_probabilities(**kwargs)
    second = run.engine.policy_probabilities(**kwargs)
    assert first == second
    assert first.l1["act"] == 0.0 and first.l1["assistance_off"] == 0.0
    assert sum(first.l1.values()) == pytest.approx(1.0)
    assert all(cell["act"] == 0.0 for cell in first.l2.values()) and first.l3 is None
    assert run.engine.snapshot() == snapshot
    assert (dict(session.decisions), set(session.open), session.round_remaining, session.round_index) == before


def test_argmax_frequency_splits_ties_evenly() -> None:
    utilities = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0], [0.0, 0.0, 5.0]])
    expected = [(1 + 1 / 3) / 4, (0.5 + 1 / 3) / 4, (0.5 + 1 / 3 + 1) / 4]
    assert np.allclose(argmax_frequency(utilities), expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(argmax_frequency(np.array([[0.0, 1.0], [2.0, 1.0], [0.5, 0.0]])), [2 / 3, 1 / 3])


def test_switching_penalty_lowers_the_switch_rate() -> None:
    policy = ConstrainedThompsonPolicy()

    def run_loop(penalty: float) -> int:
        settings = PolicySettings(thompson_draws=64, switching_penalty=penalty, exploration_floor=0.0)
        rng = np.random.default_rng(5)
        previous = None
        switches = 0
        for step in range(150):
            request = _block_request(
                rng, PreferenceSet([]), "mouse", PolicyMode.ADAPTIVE, ConstraintState(), decision_id=f"sw{step}"
            )
            flat = np.full((64, len(L1_OUTCOMES)), 0.5)  # every regime is equally good up to sampling noise
            from dataclasses import replace

            request = replace(request, settings=settings, sampler=_sampler(flat, L1_OUTCOMES), previous_regime=previous)
            decision = policy.decide_block(request)
            if previous is not None and decision.variant != previous:
                switches += 1
            previous = decision.variant
        return switches

    assert run_loop(1.0) < run_loop(0.0)


# ------------------------------------------------------------------------------------------------ select_from_cell


def test_select_from_cell_semantics() -> None:
    order = ["a", "b", "c"]
    props = {"a": 0.2, "b": 0.0, "c": 0.8}
    assert select_from_cell(order, props, 0.0) == "a"
    assert select_from_cell(order, props, 0.2) == "c"  # strictly greater than u
    assert select_from_cell(order, props, 0.9999999999999999) == "c"
    assert select_from_cell(order, {"a": 0.5, "b": 0.5 - 1e-17, "c": 0.0}, 0.9999999999999999) == "b"
    with pytest.raises(ValueError):
        select_from_cell(order, {"a": 0.0}, 0.5)


def test_select_from_cell_matches_the_decision_table_fixture() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["seed"] == 20261015 and len(data["cases"]) == 200
    for case in data["cases"]:
        assert select_from_cell(case["order"], case["propensities"], case["u"]) == case["expected"]


def test_fixture_regeneration_reproduces_the_committed_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "decision_table.json"
    subprocess.run(
        [sys.executable, "-m", "pilot.coadaptation.core.policy", "--write-fixture", str(out)],
        cwd=DAEMON,
        check=True,
        capture_output=True,
    )
    # Compared as parsed JSON: line endings change with core.autocrlf, and the last float digits change with the
    # platform's math library.
    _assert_json_close(json.loads(out.read_text(encoding="utf-8")), json.loads(FIXTURE.read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------------- hooks, tables


def test_ablation_hooks_are_the_only_overridable_methods() -> None:
    hooks = {"_ambiguity_constraint", "_safety_filter"}
    for name, member in inspect.getmembers(ConstrainedThompsonPolicy, predicate=inspect.isfunction):
        if name.startswith("__") or name in hooks:
            continue
        assert getattr(member, "__final__", False), name
    for hook in hooks:
        assert not getattr(getattr(ConstrainedThompsonPolicy, hook), "__final__", False)

    class NoAmbiguityRule(ConstrainedThompsonPolicy):
        def _ambiguity_constraint(self, candidates, *, pin_act):
            return list(candidates), {}

    run = Run()
    run.engine.policy = NoAmbiguityRule()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    table = run.engine.decide_ambiguity_table(
        decision_id="t-ablation",
        block_decision_id=decision.decision_id,
        block=block,
        start_trial_index=0,
        trials_in_block=8,
        now=run.tick(),
    )
    # the ablation measures the price of rule C2: act now receives mass in cells with a literal hit
    assert any(cell.propensities["act"] > 0 for cell in table.cells if cell.has_hit)
    assert all(cell.propensities["act"] == 0 for cell in table.cells if not cell.has_hit)


def test_a_table_requested_after_a_veto_has_rows_for_the_remaining_trials() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    table = run.engine.decide_ambiguity_table(
        decision_id="after-veto",
        block_decision_id=decision.decision_id,
        block=block,
        start_trial_index=3,
        trials_in_block=5,
        now=run.tick(),
    )
    assert table.start_trial_index == 3 and len(table.draws) == 5
    assert table.draw(2, 0) is None and table.draw(3, 0) == table.draws[0][0] and table.draw(7, 3) is not None
    assert table.draw(8, 0) is None and table.variant_for("b1:p2:h1", 3, 4) == "wait_for_evidence"
    payload = table.to_dict()
    assert payload["level"] == "ambiguity_table" and payload["start_trial_index"] == 3
    assert payload["max_draws_per_trial"] == 4 and payload["after_max"] == "wait_for_evidence"
    assert [b["id"] for b in payload["bins"]] == ["b1", "b2", "b3"]
    for cell in payload["cells"]:
        assert cell["order"] == list(L2_OPTIONS) and cell["propensities"]["act"] == 0
        assert cell["excluded"]["act"] in ("ambiguous_activation", "not_applicable")


def test_recommendation_payload_shape() -> None:
    run = Run()
    run.begin()
    decision = run.decide(run.block(1), want="enlarge_target:1.5")
    payload = decision.to_dict()
    assert payload["params"] == {"scale": 1.5}
    assert list(payload["propensities"]) == list(l1_arms("mouse"))
    assert payload["spread"] == {"r": 1.0, "floor": 1.0}
    assert payload["priorities"] == {"fewer_mistakes": 0.5, "faster": 0.5, "fewer_interruptions": 0.5}
    assert payload["mode"] == "randomized" and payload["level"] == "block_regime"
    assert payload["explanation"]["summary_key"] == "coadaptation.explain.randomized"
    for prediction in payload["predictions"].values():
        assert set(prediction) == {
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
        }
    text = json.dumps(payload)
    for word in ("stress", "attention", "health"):
        assert word not in text
    assert AH2 and l2 and timedelta and Decision and AmbiguityTable  # imported helpers stay in use


def _assert_json_close(actual: object, expected: object, path: str = "$") -> None:
    """Equal JSON, with floats compared to 1e-9 relative: the last digits differ between platforms' math libraries."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (path, actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_close(a, e, f"{path}[{index}]")
    else:
        assert actual == expected, (path, actual, expected)

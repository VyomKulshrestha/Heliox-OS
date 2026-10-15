"""Agency audit A1-A12 (spec section 20): each invariant passes on clean rows and detects an injected violation."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pilot.coadaptation.core.audit import agency_audit, trial_interruptions
from pilot.coadaptation.core.consent import CoAdaptationPurpose, ConsentPeriod, ConsentRecord
from pilot.coadaptation.core.engine import EngineEvent, EngineSettings
from pilot.coadaptation.core.evidence import EvidenceKind
from pilot.coadaptation.core.policy import PolicyMode
from pilot.coadaptation.core.preferences import Preference, PreferenceKind, PreferenceScope
from pilot.coadaptation.core.records import (
    ActivationRecord,
    BurdenReport,
    DecisionRecord,
    L2Applied,
    RevertRecord,
    SessionRow,
    SpreadUsed,
    StoredTrial,
    TrialRecord,
    VetoRecord,
    derive_labels,
)
from pilot.coadaptation.core.state import DeviceContext, LayoutSummary, Rect
from pilot.coadaptation.core.study_plan import build_session_plan
from pilot.coadaptation.core.vocabulary import L2_OPTIONS, L3_OPTIONS, DecisionLevel, l1_arms

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
PK = "p-0000aaaa"
SID = "11111111-1111-4111-8111-111111111111"
RECTS = (
    Rect(100.0, 100.0, 28.0, 28.0),
    Rect(130.0, 100.0, 28.0, 28.0),
    Rect(160.0, 100.0, 28.0, 28.0),
)
LAYOUT = LayoutSummary(
    task="pointing",
    n_candidates=3,
    w_px=28.0,
    h_px=28.0,
    gap_px=2.0,
    id_bar=3.0,
    menu_n=0,
    origin_x=320.0,
    origin_y=400.0,
)
ARMS = l1_arms("mouse")
ALL_PURPOSES = frozenset(CoAdaptationPurpose)
_ids = iter(range(10_000))


def _uuid() -> str:
    return f"00000000-0000-4000-8000-{next(_ids):012d}"


# --------------------------------------------------------------------------------------------- record builders


def session(
    session_id: str = SID,
    *,
    number: int = 1,
    tau: float = 0.25,
    adaptive: bool = False,
    started: datetime = T0,
    gaze_accuracy: float | None = None,
) -> SessionRow:
    return SessionRow(
        session_id=session_id,
        participant_key=PK,
        context="study",
        day_index=740000 + number,
        session_number=number,
        protocol_id=None,
        protocol_sha256=None,
        evidence_kind=EvidenceKind.LIVE_HUMAN,
        started_at=started,
        ended_at=started + timedelta(minutes=30),
        end_reason="completed",
        device=DeviceContext(
            modality="mouse",
            pointer_types=("mouse",),
            dpr=1.0,
            surface_w=640.0,
            surface_h=420.0,
            gaze_available=False,
            hand_pointer=False,
        ),
        plan=build_session_plan(
            session_number=number,
            adaptive_allowed=adaptive,
            comparator_available=True,
            plan_seed=3,
            burden_every_blocks=1,
        ),
        layout_seed=1,
        seed_salt="0" * 32,
        gaze_check_accuracy=gaze_accuracy,
        comparator_ranking=("act",) if adaptive else (),
        settings=asdict(EngineSettings(ambiguity_threshold=tau, adaptive_allowed=adaptive)),
        protocol_deviations=(),
        versions={},
    )


def _explanation(mode: str) -> dict[str, Any]:
    return {
        "summary_key": f"coadaptation.explain.{mode}",
        "values": {},
        "contrasts": [],
        "excluded": [],
        "evidence_source": "literature_prior",
        "trials_person": 0,
        "trials_today": 0,
    }


def l1(
    *,
    seq: int,
    created: datetime,
    variant: str = "act",
    mode: str = "randomized",
    propensities: dict[str, float] | None = None,
    level: str = "block_regime",
    session_id: str = SID,
    fallback: bool = False,
    behavior: str = "policy",
    **payload: Any,
) -> DecisionRecord:
    options = ARMS if level == "block_regime" else L3_OPTIONS
    propensities = propensities or {arm: 1.0 / len(options) for arm in options}
    body = {
        "decision_id": _uuid(),
        "level": level,
        "mode": mode,
        "variant": variant,
        "params": {},
        "eligible": [arm for arm, p in propensities.items() if p > 0],
        "excluded": {arm: "used_this_round" for arm in options if propensities.get(arm, 0.0) == 0},
        "propensities": {arm: float(propensities.get(arm, 0.0)) for arm in options},
        "p_thompson": None,
        "exploration": 0.0,
        "fallback": fallback,
        "protocol_deviation": False,
        "round_index": 1,
        "spread": {"r": 1.0, "floor": 1.0},
        "priorities": {"fewer_mistakes": 0.5, "faster": 0.5, "fewer_interruptions": 0.5},
        "predictions": {},
        "explanation": _explanation(mode),
        "seed": "0" * 16,
        "versions": {},
        "authority": "presentation_only",
        "execution_authority": False,
    }
    body.update(payload)
    return DecisionRecord(
        decision_id=body["decision_id"],
        session_id=session_id,
        participant_key=PK,
        seq=seq,
        level=DecisionLevel(level),
        mode=PolicyMode(mode),
        behavior=behavior,  # type: ignore[arg-type]
        created_at=created,
        superseded_by=None,
        payload=body,
        applied=None,
        fallback_reason=None,
        versions={},
    )


def cell(cell_id: str, propensities: dict[str, float]) -> dict[str, Any]:
    zero = [v for v in L2_OPTIONS if not propensities.get(v)]
    return {
        "cell_id": cell_id,
        "bin": cell_id[:2],
        "n_plausible": int(cell_id[4]),
        "has_hit": cell_id.endswith("h1"),
        "order": list(L2_OPTIONS),
        "propensities": {v: float(propensities.get(v, 0.0)) for v in L2_OPTIONS},
        "eligible": [v for v, p in propensities.items() if p > 0],
        "excluded": {v: "ambiguous_activation" if v == "act" else "not_applicable" for v in zero},
    }


UNIFORM_B2 = {
    "confirm_selection:second_click": 1 / 3,
    "wait_for_evidence": 1 / 3,
    "ask_clarification:2": 1 / 3,
}


def table(
    block: DecisionRecord,
    *,
    seq: int,
    created: datetime,
    mode: str = "randomized",
    cells: list[dict[str, Any]] | None = None,
    draws: list[list[float]] | None = None,
    start: int = 0,
) -> DecisionRecord:
    body = {
        "decision_id": _uuid(),
        "level": "ambiguity_table",
        "block_decision_id": block.decision_id,
        "mode": mode,
        "ambiguity_threshold": 0.25,
        "bins": [
            {"id": "b1", "lo": 0.25, "hi": 0.4},
            {"id": "b2", "lo": 0.4, "hi": 0.6},
            {"id": "b3", "lo": 0.6, "hi": 1.0},
        ],
        "cells": cells or [cell("b2:p2:h1", UNIFORM_B2)],
        "start_trial_index": start,
        "draws": draws or [[0.1, 0.5, 0.9, 0.2] for _ in range(8 - start)],
        "max_draws_per_trial": 4,
        "after_max": "wait_for_evidence",
        "explanation": _explanation(mode),
        "seed": "0" * 16,
        "versions": {},
        "authority": "presentation_only",
        "execution_authority": False,
    }
    return DecisionRecord(
        decision_id=body["decision_id"],
        session_id=block.session_id,
        participant_key=PK,
        seq=seq,
        level=DecisionLevel.AMBIGUITY_RESPONSE,
        mode=PolicyMode(mode),
        behavior="policy",
        created_at=created,
        superseded_by=None,
        payload=body,
        applied=None,
        fallback_reason=None,
        versions={},
    )


def act(
    index: int,
    *,
    hit: int | None = 0,
    a_gate: float = 0.1,
    n_plausible: int = 1,
    l2: L2Applied | None = None,
    result: str = "committed",
    outcome: str | None = None,
    committed: int | None = None,
) -> ActivationRecord:
    outcome = outcome or {"committed": "committed", "none": "none"}.get(result, "cancelled")
    if committed is None and outcome == "committed":
        committed = hit
    return ActivationRecord(
        index=index,
        t_ms=500.0 + 400.0 * index,
        modality="mouse",
        x=100,
        y=100,
        hit_index=hit,
        a_gate=a_gate,
        n_plausible=n_plausible,
        press_index=hit if hit is not None else -1,
        expanded_index=None,
        hit_rect=RECTS[hit] if hit is not None else None,
        gaze=None,
        l2=l2,
        irreversible_rule=False,
        result=result,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        committed_index=committed if outcome == "committed" else None,
    )


def l2(
    decision: DecisionRecord,
    variant: str,
    *,
    source: str = "table",
    draw_index: int = 0,
    cell_id: str = "b2:p2:h1",
):
    return L2Applied(
        decision_id=decision.decision_id,
        cell_id=cell_id,
        draw_index=draw_index,
        variant=variant,
        source=source,  # type: ignore[arg-type]
        fallback_reason="late" if source == "fallback" else None,
    )


def trial(
    decision: DecisionRecord,
    *,
    seq: int,
    trial_index: int = 0,
    block_index: int = 1,
    activations: tuple[ActivationRecord, ...] = (),
    regime: str | None = None,
    applied: str | None = None,
    issued: datetime = T0,
    revert: RevertRecord | None = None,
    veto: VetoRecord | None = None,
    l1_fallback: str | None = None,
    committed: int | None = None,
    commit_t: float | None = None,
    reversible: tuple[bool, ...] = (True, True, True),
    recorded: datetime | None = None,
    session_id: str = SID,
    tables: tuple[str, ...] = (),
) -> StoredTrial:
    regime = regime or str(decision.payload["variant"])
    if committed is None:
        committed = next(
            (a.committed_index for a in reversed(activations) if a.outcome == "committed"),
            None,
        )
    if commit_t is None and committed is not None:
        commit_t = 500.0 + 400.0 * len(activations)
    record = TrialRecord(
        trial_id=_uuid(),
        session_id=session_id,
        block_index=block_index,
        trial_index=trial_index,
        task="pointing",
        regime=regime,
        regime_decision_id=decision.decision_id,
        table_decision_ids=tables,
        applied_regime=applied or regime,
        l1_fallback_reason=l1_fallback,  # type: ignore[arg-type]
        fallbacks_issued_at=issued,
        spread=SpreadUsed(r=1.0, floor=1.0),
        layout=LAYOUT,
        candidates=RECTS,
        surface_w=640.0,
        surface_h=420.0,
        dpr=1.0,
        reversible=reversible,
        cue_at=issued + timedelta(seconds=1),
        intended_index=0,
        modality="mouse",
        activations=activations,
        revert=revert,
        committed_index=committed,
        commit_t_ms=commit_t,
        commit_via="direct" if committed is not None else "none",
        corrections=sum(1 for a in activations if a.outcome == "cancelled"),
        time_ms=commit_t or 10000.0,
        limit_ms=10000.0,
        duration_ms=commit_t or 10000.0,
        movement=None,
        veto=veto,
        offers=(),
        hand=None,
        gaze=None,
        interrupted=False,
        interruption_reason=None,
        explanation_views=0,
        recorded_at=recorded or issued + timedelta(seconds=20 + seq),
        seq=seq,
    )
    return StoredTrial(record, "ok", 740001, None, derive_labels(record), (), ())


def consent(
    granted: datetime, purposes=ALL_PURPOSES, *, replaced: datetime | None = None, replaces=None
) -> ConsentPeriod:
    record = ConsentRecord(
        participant_key=PK,
        purposes=frozenset(purposes),
        consent_text_version="coadapt-consent-v1",
        consent_text_sha256="a" * 64,
        granted_at=granted,
        expires_at=granted + timedelta(days=90),
        retention_days=90,
        authorized=True,
        replaces_consent_id=replaces,
    )
    return ConsentPeriod(record=record, replaced_at=replaced)


def pref(
    kind: PreferenceKind,
    value: Any,
    created: datetime,
    *,
    participant: str | None = None,
) -> Preference:
    return Preference(
        pref_id=f"rule-{next(_ids)}",
        kind=kind,
        scope=PreferenceScope(context="study", modality=None, participant_key=participant),
        value=value,
        origin="explicit_user",
        created_at=created,
    )


def audit(decisions, trials, sessions=None, *, burden=(), prefs=(), consents=None, events=()):
    return agency_audit(
        decisions,
        trials,
        sessions or [session()],
        burden,
        prefs,
        consents or [consent(T0 - timedelta(days=1))],
        events=events,
    )


def clean() -> tuple[list[DecisionRecord], list[StoredTrial]]:
    """A randomized act block with its uniform table: one trial with an L2 confirm step that was cancelled, then
    a confident commit; a second trial with a direct commit."""
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    tab = table(block, seq=2, created=T0 + timedelta(seconds=5))
    first = trial(
        block,
        seq=3,
        tables=(tab.payload["decision_id"],),
        activations=(
            act(
                0,
                a_gate=0.5,
                n_plausible=2,
                l2=l2(tab, "confirm_selection:second_click"),
                result="confirming",
                outcome="cancelled",
            ),
            act(1),
        ),
    )
    second = trial(block, seq=4, trial_index=1, activations=(act(0),))
    return [block, tab], [first, second]


# --------------------------------------------------------------------------------------------- clean rows


def test_clean_rows_pass_every_invariant() -> None:
    decisions, trials = clean()
    report = audit(decisions, trials)
    assert report.passed, report.findings
    assert report.counts["A7"] == 1.0 and report.counts["A8"] == 1.0
    assert set(report.to_dict()["invariants"]) == {f"A{i}" for i in range(1, 13)}
    assert {
        "override_rate",
        "never_creations",
        "pause_adapting_uses",
        "offers",
        "burden_prompts_per_hour",
    } <= set(report.soft)


# --------------------------------------------------------------------------------------------- A1


def test_a1_fallback_trial_applying_a_forbidden_variant() -> None:
    decisions, trials = clean()
    rule = pref(PreferenceKind.FORBID_VARIANT, "enlarge_target", T0 + timedelta(seconds=1))
    issued = T0 + timedelta(seconds=10)
    fallback = trial(
        decisions[0],
        seq=5,
        trial_index=2,
        applied="enlarge_target:1.5",
        l1_fallback="late",
        issued=issued,
    )
    assert audit(decisions, [*trials, fallback], prefs=[rule]).counts["A1"] >= 1
    # A rule saved after the trial's fallbacks were issued could not have been held: not a violation.
    late_rule = replace(rule, created_at=issued + timedelta(seconds=1))
    assert audit(decisions, [*trials, fallback], prefs=[late_rule]).counts["A1"] == 0


def test_a1_revert_regime_forbidden_when_its_fallbacks_were_issued() -> None:
    decisions, trials = clean()
    issued = T0 + timedelta(seconds=10)
    rule = pref(
        PreferenceKind.FORBID_VARIANT,
        "confirm_selection:second_click",
        T0 + timedelta(seconds=2),
    )
    reverted = trial(
        decisions[0],
        seq=5,
        trial_index=2,
        veto=VetoRecord(decision_id=decisions[0].decision_id, level=DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord(
            reason="veto",
            from_activation=0,
            regime="confirm_selection:second_click",
            fallbacks_issued_at=issued,
        ),
    )
    assert audit(decisions, [*trials, reverted], prefs=[rule]).counts["A1"] >= 1


def test_a1_interrupting_l2_response_at_the_cap() -> None:
    decisions, trials = clean()
    cap = pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 1, T0 - timedelta(hours=1))
    tab = decisions[1]
    # The first trial's cancelled L2 confirm step already counted (k = 1 = cap): another confirm is not allowed.
    over = trial(
        decisions[0],
        seq=5,
        trial_index=2,
        tables=(tab.decision_id,),
        activations=(
            act(
                0,
                a_gate=0.5,
                n_plausible=2,
                l2=l2(tab, "confirm_selection:second_click"),
                result="confirming",
            ),
        ),
    )
    report = audit(decisions, [*trials, over], prefs=[cap])
    assert report.counts["A1"] >= 1
    a1 = [finding.detail for finding in report.findings if finding.invariant == "A1"]
    assert any("cap 1" in detail for detail in a1)


def test_a1_refresh_reverts_must_keep_a_frozen_or_pinned_regime() -> None:
    issued = T0 + timedelta(seconds=30)
    freeze = pref(
        PreferenceKind.FREEZE_ADAPTATION,
        {"regime": "confirm_selection:hold_700"},
        T0 + timedelta(seconds=20),
        participant=PK,
    )
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    to_plain = trial(
        block,
        seq=2,
        revert=RevertRecord(
            reason="refresh",
            from_activation=0,
            regime="assistance_off",
            fallbacks_issued_at=issued,
        ),
    )
    assert audit([block], [to_plain], prefs=[freeze]).counts["A1"] >= 1
    pin = pref(PreferenceKind.PIN_REGIME, "enlarge_target:1.5", T0, participant=PK)
    pinned = l1(
        seq=1,
        created=T0 + timedelta(seconds=5),
        variant="enlarge_target:1.5",
        mode="pinned",
        propensities={"enlarge_target:1.5": 1.0},
    )
    to_hold = trial(
        pinned,
        seq=2,
        revert=RevertRecord(
            reason="refresh",
            from_activation=0,
            regime="confirm_selection:hold_700",
            fallbacks_issued_at=issued,
        ),
    )
    assert audit([pinned], [to_hold], prefs=[pin]).counts["A1"] >= 1


def test_a1_exempts_a_not_this_time_revert_to_plain_in_a_frozen_block() -> None:
    freeze = pref(
        PreferenceKind.FREEZE_ADAPTATION,
        {"regime": "confirm_selection:hold_700"},
        T0,
        participant=PK,
    )
    frozen = l1(
        seq=1,
        created=T0 + timedelta(seconds=5),
        variant="confirm_selection:hold_700",
        mode="pinned",
        propensities={"confirm_selection:hold_700": 1.0},
    )
    vetoed = trial(
        frozen,
        seq=2,
        issued=T0 + timedelta(seconds=5),
        veto=VetoRecord(decision_id=frozen.decision_id, level=DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord(
            reason="veto",
            from_activation=0,
            regime="assistance_off",
            fallbacks_issued_at=T0 + timedelta(seconds=5),
        ),
    )
    report = audit([frozen], [vetoed], prefs=[freeze])
    assert report.counts["A1"] == 0
    assert report.counts["A11"] == 0


# --------------------------------------------------------------------------------------------- A2-A5


@pytest.mark.parametrize(("tau", "violations"), [(0.15, 1), (0.25, 0)])
def test_a2_uses_each_sessions_tau(tau: float, violations: int) -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    committed = trial(block, seq=2, activations=(act(0, a_gate=0.2),))
    assert audit([block], [committed], [session(tau=tau)]).counts["A2"] == violations


def test_a2_allows_an_l2_act_only_under_pin_act() -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    collapse = table(
        block,
        seq=2,
        created=T0 + timedelta(seconds=5),
        mode="pinned",
        cells=[cell("b2:p2:h1", {"act": 1.0})],
    )
    applied = trial(
        block,
        seq=3,
        tables=(collapse.decision_id,),
        activations=(act(0, a_gate=0.5, n_plausible=2, l2=l2(collapse, "act")),),
    )
    pin_act = pref(PreferenceKind.PIN_ACT, True, T0, participant=PK)
    with_rule = audit([block, collapse], [applied], prefs=[pin_act])
    assert with_rule.counts["A2"] == 0 and with_rule.counts["A5"] == 0
    without_rule = audit([block, collapse], [applied])
    assert without_rule.counts["A2"] == 1 and without_rule.counts["A5"] == 1


def test_a3_commits_need_an_explicit_input_and_never_happen_at_the_limit() -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    ghost = trial(
        block,
        seq=2,
        activations=(act(0, hit=None, result="none"),),
        committed=0,
        commit_t=900.0,
    )
    assert audit([block], [ghost]).counts["A3"] == 1
    at_limit = trial(block, seq=3, activations=(act(0),), commit_t=10000.0)
    assert audit([block], [at_limit]).counts["A3"] == 1


def test_a4_flags_exploration_mass_on_a_forbidden_variant() -> None:
    rule = pref(PreferenceKind.FORBID_VARIANT, "enlarge_target:2.0", T0)
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    assert audit([block], [], prefs=[rule]).counts["A4"] == 1
    later = replace(rule, created_at=T0 + timedelta(minutes=5))
    assert audit([block], [], prefs=[later]).counts["A4"] == 0


@pytest.mark.parametrize("mode", ["randomized", "probe", "adaptive", "comparator"])
def test_a5_flags_act_in_any_table_that_is_not_a_pin_act_collapse(mode: str) -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    bad = table(
        block,
        seq=2,
        created=T0 + timedelta(seconds=5),
        mode=mode,
        cells=[
            cell(
                "b2:p2:h1",
                {"act": 0.25, "confirm_selection:second_click": 0.25, "wait_for_evidence": 0.5},
            )
        ],
    )
    assert audit([block, bad], []).counts["A5"] == 1


# --------------------------------------------------------------------------------------------- A6


def test_a6_judges_every_row_against_the_consent_period_in_force() -> None:
    update = T0 + timedelta(days=2)
    first = consent(T0 - timedelta(days=1), {CoAdaptationPurpose.SELECTION_STUDY}, replaced=update)
    second = consent(
        update,
        {CoAdaptationPurpose.SELECTION_STUDY, CoAdaptationPurpose.ADAPTIVE_ASSISTANCE},
        replaces=first.record.consent_id,
    )
    s1 = session()
    s2 = session(
        "22222222-2222-4222-8222-222222222222",
        number=2,
        adaptive=True,
        started=update + timedelta(hours=1),
    )
    d1 = l1(seq=1, created=T0 + timedelta(seconds=5))
    d2 = l1(
        seq=1,
        created=update + timedelta(hours=1, seconds=5),
        mode="adaptive",
        session_id=s2.session_id,
    )
    t1 = trial(d1, seq=2)
    t2 = trial(
        d2,
        seq=2,
        block_index=8,
        session_id=s2.session_id,
        issued=update + timedelta(hours=1),
    )
    report = audit([d1, d2], [t1, t2], [s1, s2], consents=[first, second])
    assert report.counts["A6"] == 0, report.findings
    # Needs adaptive_assistance before it was granted.
    early = replace(d2, created_at=update - timedelta(minutes=1))
    assert audit([d1, early], [t1, t2], [s1, s2], consents=[first, second]).counts["A6"] >= 1
    outside = replace(d1, created_at=T0 - timedelta(days=3))
    assert audit([outside], [], [s1], consents=[first, second]).counts["A6"] >= 1
    report_row = BurdenReport(
        report_id="r1",
        session_id=s1.session_id,
        seq=9,
        block_index=1,
        scope="block",
        effort=3,
        intrusiveness=2,
        control=6,
        tlx=None,
        predictability=None,
        preferred_regime=None,
        skipped=(),
        recorded_at=T0 - timedelta(days=3),
    )
    assert audit([], [], [s1], burden=[report_row], consents=[first, second]).counts["A6"] == 1
    gaze_session = session(gaze_accuracy=0.8)
    assert audit([], [], [gaze_session], consents=[first, second]).counts["A6"] == 1


# --------------------------------------------------------------------------------------------- A7, A8, A10, A12


def test_a7_a8_and_a10_check_every_decision_payload() -> None:
    decisions, trials = clean()
    broken = replace(
        decisions[0],
        payload={**decisions[0].payload, "propensities": {arm: 0.15 for arm in ARMS}},
    )
    assert audit([broken, decisions[1]], trials).counts["A7"] < 1.0
    unexplained = replace(decisions[0], payload={**decisions[0].payload, "explanation": None})
    assert audit([unexplained, decisions[1]], trials).counts["A8"] < 1.0
    claiming = replace(decisions[0], payload={**decisions[0].payload, "execution_authority": True})
    assert audit([claiming, decisions[1]], trials).counts["A10"] == 1


def test_a12_flags_an_irreversible_candidate_committed_at_once() -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    irreversible = trial(block, seq=2, activations=(act(0),), reversible=(False, True, True))
    assert audit([block], [irreversible]).counts["A12"] == 1


# --------------------------------------------------------------------------------------------- A9


def _veto_case(
    first_mode: str,
    next_mode: str,
    next_props: dict[str, float],
    *,
    regime: str = "act",
):
    d1 = l1(
        seq=1,
        created=T0 + timedelta(seconds=5),
        mode=first_mode,
        variant=regime,
        propensities=None if first_mode != "pinned" else {regime: 1.0},
    )
    vetoed = trial(
        d1,
        seq=2,
        veto=VetoRecord(decision_id=d1.decision_id, level=DecisionLevel.BLOCK_REGIME),
    )
    d2 = l1(
        seq=3,
        created=T0 + timedelta(minutes=2),
        mode=next_mode,
        variant=max(next_props, key=next_props.get),
        propensities=next_props,
    )
    return [d1, d2], [vetoed]


def test_a9_fails_when_the_next_adaptive_or_randomized_decision_keeps_the_vetoed_variant() -> None:
    for mode in ("adaptive", "randomized"):
        decisions, trials = _veto_case("adaptive", mode, {"act": 0.5, "enlarge_target:1.5": 0.5})
        assert audit(decisions, trials).counts["A9"] == 1, mode
        decisions, trials = _veto_case("adaptive", mode, {"enlarge_target:1.5": 1.0})
        assert audit(decisions, trials).counts["A9"] == 0, mode


def test_a9_judges_the_revert_regime_a_veto_named_after_a_refresh() -> None:
    # A refresh reverted the trial to enlarge_target:1.5 before the veto, so the chip named that regime and C5
    # excludes it at the next L1 decision, never the cue regime act the refresh had replaced.
    d1 = l1(seq=1, created=T0 + timedelta(seconds=5), mode="adaptive", variant="act")
    refreshed = trial(
        d1,
        seq=2,
        revert=RevertRecord(reason="refresh", from_activation=0, regime="enlarge_target:1.5", fallbacks_issued_at=T0),
        veto=VetoRecord(decision_id=d1.decision_id, level=DecisionLevel.BLOCK_REGIME),
    )
    for props, flagged in (({"enlarge_target:1.5": 0.5, "act": 0.5}, 1), ({"act": 1.0}, 0)):
        d2 = l1(
            seq=3,
            created=T0 + timedelta(minutes=2),
            mode="adaptive",
            variant=max(props, key=props.get),
            propensities=props,
        )
        assert audit([d1, d2], [refreshed]).counts["A9"] == flagged, props


def test_a9_exempts_pinned_decisions_and_never_judges_probes_or_comparators() -> None:
    decisions, trials = _veto_case(
        "pinned", "pinned", {"confirm_selection:hold_700": 1.0}, regime="confirm_selection:hold_700"
    )
    report = audit(decisions, trials)
    assert report.counts["A9"] == 0
    assert any("pinned" in exemption.detail for exemption in report.exemptions)
    for mode in ("probe", "comparator"):
        decisions, trials = _veto_case("adaptive", mode, {"act": 1.0})
        report = audit(decisions, trials)
        assert report.counts["A9"] == 0
        assert any(mode in exemption.detail for exemption in report.exemptions)


def test_a9_exempts_a_vetoed_pin_act_collapse_and_never_evaluates_wait() -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    collapse = table(
        block,
        seq=2,
        created=T0 + timedelta(seconds=5),
        mode="pinned",
        cells=[cell("b2:p2:h1", {"act": 1.0})],
    )
    vetoed = trial(
        block,
        seq=3,
        tables=(collapse.decision_id,),
        activations=(act(0, a_gate=0.5, n_plausible=2, l2=l2(collapse, "act")),),
        veto=VetoRecord(decision_id=collapse.decision_id, level=DecisionLevel.AMBIGUITY_RESPONSE),
    )
    again = table(
        block,
        seq=4,
        created=T0 + timedelta(minutes=1),
        mode="pinned",
        cells=[cell("b2:p2:h1", {"act": 1.0})],
        start=1,
    )
    pin_act = pref(PreferenceKind.PIN_ACT, True, T0, participant=PK)
    report = audit([block, collapse, again], [vetoed], prefs=[pin_act])
    assert report.counts["A9"] == 0
    assert any("pinned" in exemption.detail for exemption in report.exemptions)
    tab = table(block, seq=2, created=T0 + timedelta(seconds=5))
    waited = trial(
        block,
        seq=3,
        tables=(tab.decision_id,),
        activations=(
            act(
                0,
                a_gate=0.5,
                n_plausible=2,
                l2=l2(tab, "wait_for_evidence", draw_index=0),
                result="none",
            ),
        ),
        veto=VetoRecord(decision_id=tab.decision_id, level=DecisionLevel.AMBIGUITY_RESPONSE),
    )
    nxt = table(block, seq=4, created=T0 + timedelta(minutes=1), start=1)
    report = audit([block, tab, nxt], [waited])
    assert report.counts["A9"] == 0
    assert any("wait_for_evidence" in exemption.detail for exemption in report.exemptions)


# --------------------------------------------------------------------------------------------- A11


def test_a11_cap_responses_need_the_count_to_reach_the_cap() -> None:
    decisions, trials = clean()
    tab = decisions[1]
    cap = pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2, T0 - timedelta(hours=1))
    early_cap = trial(
        decisions[0],
        seq=5,
        trial_index=2,
        activations=(
            act(
                0,
                a_gate=0.5,
                n_plausible=2,
                l2=l2(tab, "wait_for_evidence", source="cap"),
                result="none",
            ),
        ),
    )
    assert audit(decisions, [*trials, early_cap], prefs=[cap]).counts["A11"] == 1  # k = 1 < 2
    # The cancelled confirm step counts from its result (confirming), not its outcome: at cap 1 the cap is due.
    cap1 = replace(cap, value=1)
    assert audit(decisions, [*trials, early_cap], prefs=[cap1]).counts["A11"] == 0


def test_a11_counts_a_trials_l1_confirm_and_hold_steps_once() -> None:
    cap = pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2, T0 - timedelta(hours=1))
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    tab = table(block, seq=2, created=T0 + timedelta(seconds=5))
    # A minimum caution saved after the block's decisions leaves only hold_700, so the plain regime the study holds
    # from then on is hold_700 (the cap does not remove the only allowed regime, 9.2 C1).
    no_plain = pref(
        PreferenceKind.MIN_CAUTION,
        "confirm_selection:hold_700",
        T0 + timedelta(seconds=10),
    )
    issued = T0 + timedelta(seconds=20)
    # Trial 0 was reverted ("Not this time") to that plain regime; the hold was released and held again: two L1
    # hold steps in one trial count once (k = 1).
    reverted = trial(
        block,
        seq=3,
        issued=issued,
        veto=VetoRecord(decision_id=block.decision_id, level=DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord(
            reason="veto",
            from_activation=0,
            regime="confirm_selection:hold_700",
            fallbacks_issued_at=issued,
        ),
        activations=(
            act(0, result="held", outcome="cancelled"),
            act(1, result="held", outcome="committed"),
        ),
    )
    assert trial_interruptions(reverted.trial) == 1
    cap_response = trial(
        block,
        seq=4,
        trial_index=1,
        issued=issued,
        tables=(tab.decision_id,),
        activations=(
            act(
                0,
                a_gate=0.5,
                n_plausible=2,
                l2=l2(tab, "wait_for_evidence", source="cap"),
                result="none",
            ),
        ),
    )
    report = audit([block, tab], [reverted, cap_response], prefs=[no_plain, cap])
    assert report.counts["A11"] == 1  # only a per-step count (2) would have justified the cap
    table_draw = trial(
        block,
        seq=4,
        trial_index=1,
        issued=issued,
        tables=(tab.decision_id,),
        activations=(
            act(
                0,
                a_gate=0.5,
                n_plausible=2,
                l2=l2(tab, "confirm_selection:second_click"),
                result="confirming",
            ),
        ),
    )
    report = audit([block, tab], [reverted, table_draw], prefs=[no_plain, cap])
    assert report.counts["A11"] == 0, report.findings


def test_a11_flags_draws_and_reverts_the_rules_do_not_give() -> None:
    decisions, trials = clean()
    tab = decisions[1]
    asked = act(0, a_gate=0.5, n_plausible=2, l2=l2(tab, "ask_clarification:2"), result="asked")
    # u = 0.1
    wrong_draw = trial(
        decisions[0],
        seq=5,
        trial_index=2,
        tables=(tab.decision_id,),
        activations=(asked,),
    )
    assert audit(decisions, [*trials, wrong_draw]).counts["A11"] == 1
    bad_revert = trial(
        decisions[0],
        seq=5,
        trial_index=2,
        veto=VetoRecord(decision_id=decisions[0].decision_id, level=DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord(reason="veto", from_activation=0, regime="act", fallbacks_issued_at=T0),
    )
    assert audit(decisions, [*trials, bad_revert]).counts["A11"] == 1
    stored_mismatch = replace(trials[1], verification="draw_mismatch")
    assert audit(decisions, [trials[0], stored_mismatch]).counts["A11"] == 1


def test_a11_reads_the_c7_state_from_the_events() -> None:
    block = l1(seq=1, created=T0 + timedelta(seconds=5))
    fired = trial(block, seq=2, activations=(act(0),), recorded=T0 + timedelta(seconds=30))
    event = EngineEvent.create(
        kind="calibration_suspended",
        session_id=SID,
        trial_id=fired.trial.trial_id,
        created_at=T0 + timedelta(seconds=30),
    )
    issued = T0 + timedelta(minutes=1)
    cautious = trial(
        block,
        seq=3,
        trial_index=1,
        issued=issued,
        veto=VetoRecord(decision_id=block.decision_id, level=DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord(
            reason="veto",
            from_activation=0,
            regime="confirm_selection:hold_700",
            fallbacks_issued_at=issued,
        ),
    )
    assert audit([block], [fired, cautious], events=[event]).counts["A11"] == 0
    # Without events C7 counts as never suspended.
    assert audit([block], [fired, cautious]).counts["A11"] == 1
    plain = trial(
        block,
        seq=3,
        trial_index=1,
        issued=issued,
        veto=VetoRecord(decision_id=block.decision_id, level=DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord(
            reason="veto",
            from_activation=0,
            regime="assistance_off",
            fallbacks_issued_at=issued,
        ),
    )
    assert audit([block], [fired, plain], events=[event]).counts["A11"] == 1

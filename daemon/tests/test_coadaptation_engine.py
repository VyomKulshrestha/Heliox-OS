"""Engine: decisions, verification, labels, learning, replay and snapshots (spec 13.1.1, 22.1)."""

from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pilot.coadaptation.core.constraints import ExclusionReason, trial_interruptions
from pilot.coadaptation.core.engine import (
    CoAdaptationEngine,
    EngineEvent,
    EngineSettings,
    NoAllowedRegime,
    TrialObservation,
)
from pilot.coadaptation.core.outcome_model import ModelVariant
from pilot.coadaptation.core.policy import (
    AmbiguityTable,
    Decision,
    PolicyMode,
    decision_seed,
    draw_stream,
    select_from_cell,
)
from pilot.coadaptation.core.preferences import Preference, PreferenceKind, PreferenceScope, PreferenceSet
from pilot.coadaptation.core.records import (
    ActivationGaze,
    ActivationRecord,
    BurdenReport,
    L2Applied,
    OfferResponse,
    RevertRecord,
    SessionRow,
    SpreadUsed,
    StoredTrial,
    TrialRecord,
    VetoRecord,
)
from pilot.coadaptation.core.state import BlockContext, OfferContext
from pilot.coadaptation.core.target_posterior import compute_posterior
from pilot.coadaptation.core.vocabulary import DecisionLevel
from tests.test_coadaptation_records import (
    CANDS,
    LAYOUT,
    PK,
    SALT,
    SURFACE,
    T0,
    block_context,
    engine_settings,
    grid,
    keyboard_activation,
    layout_for,
    menu,
    pointer_activation,
    session_row,
    trial_record,
)

FIXTURE = Path(__file__).parent / "fixtures" / "coadaptation_cap_recount.json"
INTENDED = 5
# Integer endpoints on the shared 16 px grid (spread r = 1, mouse floor 1): see the cell of each in the comments.
U = (341, 200)  # the intended centre: a literal hit, A_gate <= tau
AH2 = (333, 193)  # an ambiguous hit with two plausible candidates (b1:p2:h1)
AH3 = (333, 192)  # an ambiguous hit with three plausible candidates (b2:p3:h1)
GAP = (332, 200)  # the gap between candidates 4 and 5: no literal hit (b3:p2:h0)
FAR = (120, 460)  # far from every candidate: no hit and n_plausible 1 (b3:p1:h0)
_ids = itertools.count()


def pref(kind: PreferenceKind, value: Any, *, at: datetime = T0 - timedelta(days=1), **scope: Any) -> Preference:
    return Preference(
        pref_id=f"pref-{next(_ids)}",
        kind=kind,
        scope=PreferenceScope(context=scope.pop("context", "study"), **scope),
        value=value,
        origin="explicit_user",
        created_at=at,
    )


class Run:
    """One participant's engine with the rows it produced, so a rebuild can replay them."""

    def __init__(
        self,
        prefs: PreferenceSet | None = None,
        *,
        settings: EngineSettings | None = None,
        variant: ModelVariant = ModelVariant.FULL,
    ) -> None:
        self.prefs = prefs or PreferenceSet([])
        self.initial_settings = settings or engine_settings()
        self.variant = variant
        self.engine = CoAdaptationEngine(
            participant_key=PK, settings=self.initial_settings, preferences=self.prefs, model_variant=variant
        )
        self.rows: list[Any] = []
        self.ends: dict[str, datetime] = {}
        self.now = T0
        self.row: SessionRow | None = None
        self.seq = 0
        self.last_issued: datetime | None = None
        self.spread = SpreadUsed(1.0, 1.0)  # the spread pair of the latest L1 decision (echoed by trials)

    # time and sessions
    def tick(self, seconds: float = 1.0) -> datetime:
        self.now += timedelta(seconds=seconds)
        return self.now

    def begin(self, session_id: str = "s1", **kwargs: Any) -> SessionRow:
        kwargs.setdefault("settings", self.initial_settings)
        self.row = session_row(session_id, started_at=self.now, **kwargs)
        self.engine.begin_session(self.row)
        self.rows.append(self.row)
        self.seq = 0
        self.last_issued = None
        return self.row

    def end(self) -> Any:
        ended = self.tick(5)
        assert self.row is not None
        self.ends[self.row.session_id] = ended
        return self.engine.end_session(ended_at=ended, reason="completed")

    def set_prefs(self, prefs: PreferenceSet) -> None:
        self.prefs = prefs
        self.engine.set_preferences(prefs)

    # decisions
    def block(self, index: int, phase: str = "A", *, layout=LAYOUT, previous: str | None = None) -> BlockContext:
        assert self.row is not None
        return block_context(self.row, index, phase, layout, previous=previous, minute=float(index))

    def decide(self, block: BlockContext, *, want: str | None = None) -> Decision:
        now = self.tick()
        decision_id = f"L1-{block.block_index}-{next(_ids)}"
        if want is not None:
            for attempt in range(600):
                candidate = f"L1-{block.block_index}-{attempt}-{next(_ids)}"
                probe = copy.deepcopy(self.engine)
                if probe.decide_block(decision_id=candidate, block=block, now=now).variant == want:
                    decision_id = candidate
                    break
            else:  # pragma: no cover - the search space always holds the regime
                raise AssertionError("no decision id gives the wanted regime")
        decision = self.engine.decide_block(decision_id=decision_id, block=block, now=now)
        if want is not None:
            assert decision.variant == want
        if decision.spread is not None:
            self.spread = SpreadUsed(r=decision.spread[0], floor=decision.spread[1])
        return decision

    def table(
        self,
        decision: Decision,
        block: BlockContext,
        needs: list[tuple[int, int, str, str]] = (),  # type: ignore[assignment]
        *,
        start: int = 0,
        trials: int = 8,
    ) -> AmbiguityTable:
        """A table whose draws give each needed (trial, draw index, cell, variant)."""
        now = self.tick()
        assert self.row is not None
        reference = copy.deepcopy(self.engine).decide_ambiguity_table(
            decision_id="reference",
            block_decision_id=decision.decision_id,
            block=block,
            start_trial_index=start,
            trials_in_block=trials,
            now=now,
        )
        decision_id = f"T-{block.block_index}-{next(_ids)}"
        for attempt in range(20000):
            candidate = f"T-{block.block_index}-{attempt}"
            rng = draw_stream(decision_seed(PK, self.row.seed_salt, candidate))
            rows = [[float(rng.random()) for _ in range(4)] for _ in range(trials)]
            ok = True
            for trial_index, draw_index, cell_id, variant in needs:
                cell = reference.cell(cell_id)
                assert cell is not None, cell_id
                u = rows[trial_index - start][draw_index]
                if select_from_cell(cell.order, cell.propensities, u) != variant:
                    ok = False
                    break
            if ok:
                decision_id = candidate
                break
        else:  # pragma: no cover
            raise AssertionError("no table id gives the needed draws")
        return self.engine.decide_ambiguity_table(
            decision_id=decision_id,
            block_decision_id=decision.decision_id,
            block=block,
            start_trial_index=start,
            trials_in_block=trials,
            now=now,
        )

    # trials
    def fallbacks(self) -> Any:
        assert self.row is not None
        return self.engine.fallbacks(self.row.device.modality, now=self.tick(0.5))

    def activation(
        self,
        endpoint: tuple[int, int],
        *,
        index: int = 0,
        press: int | None = None,
        spread: SpreadUsed | None = None,
        cands=CANDS,
        **kwargs: Any,
    ) -> ActivationRecord:
        if press is None:
            press = INTENDED if endpoint in (U, AH2, AH3) else -1
        return pointer_activation(
            cands,
            endpoint,
            index=index,
            spread=spread or self.spread,
            press_index=press,
            t_ms=400.0 + 300.0 * index,
            **kwargs,
        )

    def trial(
        self,
        decision: Decision,
        activations: tuple[ActivationRecord, ...],
        *,
        trial_index: int,
        fallbacks: Any = None,
        block_index: int | None = None,
        **overrides: Any,
    ) -> TrialRecord:
        assert self.row is not None
        fb = fallbacks or self.fallbacks()
        spread = None if decision.spread is None else SpreadUsed(r=decision.spread[0], floor=decision.spread[1])
        values: dict[str, Any] = {
            "activations": activations,
            "cands": CANDS,
            "layout": LAYOUT,
            "regime": decision.variant,
            "decision_id": decision.decision_id,
            "session_id": self.row.session_id,
            "block_index": block_index if block_index is not None else self._block_of(decision),
            "trial_index": trial_index,
            "issued_at": fb.issued_at,
            "recorded_at": self.tick(3),
            "spread": spread,
            "modality": self.row.device.modality,
            "seq": self.seq,
        }
        values.update(overrides)
        self.seq += 1
        return trial_record(**values)

    def _block_of(self, decision: Decision) -> int:
        session = self.engine._session  # noqa: SLF001 - test access to the block index of a decision
        for index, decision_id in session.block_decisions.items():
            if decision_id == decision.decision_id:
                return index
        return 1

    def observe(self, trial: TrialRecord) -> TrialObservation:
        observation = self.engine.observe_trial(trial)
        self.rows.append(observation.stored)
        self.rows.extend(observation.events)
        return observation

    def burden(self, block_index: int, **ratings: Any) -> BurdenReport:
        assert self.row is not None
        report = BurdenReport(
            report_id=f"b-{next(_ids)}",
            session_id=self.row.session_id,
            seq=self.seq,
            block_index=block_index,
            scope="block",
            effort=ratings.get("effort"),
            intrusiveness=ratings.get("intrusiveness"),
            control=ratings.get("control"),
            tlx=None,
            predictability=None,
            preferred_regime=None,
            skipped=(),
            recorded_at=self.tick(2),
        )
        self.seq += 1
        self.engine.observe_burden(report)
        self.rows.append(report)
        return report

    def replay(self, *, settings: EngineSettings | None = None) -> CoAdaptationEngine:
        engine = CoAdaptationEngine(
            participant_key=PK,
            settings=settings or self.initial_settings,
            preferences=self.prefs,
            model_variant=self.variant,
        )
        current: SessionRow | None = None
        for item in self.rows:
            if isinstance(item, SessionRow):
                if current is not None and current.session_id in self.ends:
                    engine.end_session(ended_at=self.ends[current.session_id], reason="completed")
                engine.begin_session(item)
                current = item
            elif isinstance(item, StoredTrial):
                engine.replay_trial(item)
            elif isinstance(item, BurdenReport):
                engine.observe_burden(item)
            elif isinstance(item, EngineEvent):
                engine.apply_event(item)
        if current is not None and current.session_id in self.ends:
            engine.end_session(ended_at=self.ends[current.session_id], reason="completed")
        return engine


def unambiguous(run: Run, decision: Decision, trial_index: int, **overrides: Any) -> TrialRecord:
    return run.trial(decision, (run.activation(U),), trial_index=trial_index, **overrides)


def l2(decision_id: str, cell: str, variant: str, *, source: str = "table", draw: int = 0, reason: str | None = None):
    return L2Applied(decision_id, cell, draw, variant, source, reason)  # type: ignore[arg-type]


RESULT_OF = {"confirm_selection:second_click": "confirming", "wait_for_evidence": "none"}


# ---------------------------------------------------------------------------------------------------- basics


def test_end_to_end_decisions_trials_labels_and_updates() -> None:
    run = Run()
    run.begin()
    before = run.engine.snapshot()
    regimes = []
    for index in range(1, 7):
        block = run.block(index)
        decision = run.decide(block)
        regimes.append(decision.variant)
        assert decision.mode is PolicyMode.RANDOMIZED and decision.level == DecisionLevel.BLOCK_REGIME
        assert decision.authority == "presentation_only" and decision.execution_authority is False
        assert decision.spread == (1.0, 1.0) or index > 1
        for trial_index in range(2):
            observation = run.observe(unambiguous(run, decision, trial_index))
            assert observation.stored.verification == "ok" and observation.learned
            assert observation.stored.labels is not None and observation.stored.labels.complete
    # one permuted round covers every arm exactly once
    assert sorted(regimes) == sorted(
        [
            "assistance_off",
            "act",
            "enlarge_target:1.5",
            "enlarge_target:2.0",
            "confirm_selection:second_click",
            "confirm_selection:hold_700",
        ]
    )
    assert run.engine.snapshot() != before
    status = run.engine.status()
    assert status.trials_seen == 12 and status.sessions_seen == 1 and status.last_day_index == 740000
    summary = run.end()
    assert summary.trials == 12 and summary.completion == 1.0 and summary.switch_rate == 1.0
    assert summary.regime_entropy_bits == pytest.approx(math.log2(6))


def test_every_returned_object_carries_the_authority_constants() -> None:
    run = Run(settings=engine_settings(adaptive_allowed=True))
    run.begin()
    block = run.block(1, "B")
    decision = run.decide(block)
    assert decision.mode is PolicyMode.ADAPTIVE
    table = run.table(decision, block)
    offer = run.engine.decide_offer(
        decision_id="offer-1",
        context=OfferContext("s1", 1, 2, "mouse", "pointing", "corrections", True),
        now=run.tick(),
    )
    for item in (decision, table, offer):
        assert item.authority == "presentation_only" and item.execution_authority is False
        payload = item.to_dict()
        assert payload["authority"] == "presentation_only" and payload["execution_authority"] is False
        text = json.dumps(payload)
        for forbidden in ("plan_id", "action_index", "ActionType", "coordinate"):
            assert forbidden not in text
    for prediction in decision.predictions.values():
        assert prediction.authority == "presentation_only"
    assert offer.level == DecisionLevel.OFFER and table.level == DecisionLevel.AMBIGUITY_RESPONSE


def test_decide_block_raises_when_nothing_is_allowed() -> None:
    prefs = PreferenceSet(
        [
            pref(PreferenceKind.FORBID_VARIANT, "act"),
            pref(PreferenceKind.FORBID_VARIANT, "enlarge_target"),
            pref(PreferenceKind.FORBID_VARIANT, "assistance_off"),
        ]
    )
    run = Run(prefs, settings=engine_settings(adaptive_allowed=True))
    run.begin()
    run.engine.apply_event(
        EngineEvent.create(kind="calibration_suspended", session_id="s1", trial_id="t0", created_at=run.now)
    )
    # C7 removes nothing here (confirm regimes are not acting): forbid the confirm family too, through a new set
    prefs = PreferenceSet([*prefs.all, pref(PreferenceKind.FORBID_VARIANT, "confirm_selection")])
    run.set_prefs(prefs)
    assert run.engine.fallbacks("mouse", now=run.tick()).l1 is None
    with pytest.raises(NoAllowedRegime):
        run.decide(run.block(1, "B"))


def test_duplicate_decision_ids_are_refused() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    run.engine.decide_block(decision_id="same", block=block, now=run.tick())
    with pytest.raises(ValueError, match="Duplicate decision_id"):
        run.engine.decide_block(decision_id="same", block=block, now=run.tick())


# ------------------------------------------------------------------------------------------------- verification


def test_l2_draw_verification_catches_a_tampered_variant() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    cell = "b1:p2:h1"
    table = run.table(decision, block, [(0, 0, cell, "ask_clarification:2")])
    applied = l2(table.decision_id, cell, "ask_clarification:2")
    good = run.activation(AH2, l2=applied, result="asked", outcome="committed", committed_index=INTENDED)
    observation = run.observe(run.trial(decision, (good,), trial_index=0, table_decision_ids=(table.decision_id,)))
    assert observation.stored.verification == "ok"
    actual = table.variant_for(cell, 1, 0)
    wrong = "confirm_selection:second_click" if actual != "confirm_selection:second_click" else "ask_clarification:2"
    tampered = run.activation(
        AH2, l2=replace(applied, variant=wrong), result=RESULT_OF.get(wrong, "asked"), outcome="committed"
    )
    observation = run.observe(run.trial(decision, (tampered,), trial_index=1))
    assert observation.stored.verification == "draw_mismatch" and not observation.learned


def test_posterior_mismatches_are_detected() -> None:
    run = Run(settings=engine_settings(gaze_allowed=True))
    run.begin(gaze_accuracy=0.8)
    decision = run.decide(run.block(1), want="assistance_off")

    def verify(trial: TrialRecord) -> str:
        return run.observe(trial).stored.verification

    base = run.activation(U)
    assert verify(run.trial(decision, (base,), trial_index=0)) == "ok"
    # the echoed spread differs from the L1 decision's
    assert verify(run.trial(decision, (base,), trial_index=1, spread=SpreadUsed(1.1, 1.0))) == "posterior_mismatch"
    # the hit's rendered rect differs from the rebuilt one
    moved = replace(base, hit_rect=replace(base.hit_rect, w=18.0))
    assert verify(run.trial(decision, (moved,), trial_index=2)) == "posterior_mismatch"
    # a gaze confidence altered after the posterior was computed (the centre region splits this grid at x = 233)
    gz = grid(centre=(233.0, 200.0))
    gz_layout = layout_for(gz)
    gaze = ActivationGaze(region="center", conf=0.9)
    with_gaze = run.activation((232, 197), press=5, cands=gz, gaze=gaze, gaze_accuracy=0.8)
    plain = run.activation((232, 197), press=5, cands=gz)
    assert with_gaze.a_gate > plain.a_gate  # gaze favours the neighbour here, so A_gate rises
    assert verify(run.trial(decision, (with_gaze,), trial_index=3, cands=gz, layout=gz_layout)) == "ok"
    altered = replace(with_gaze, gaze=ActivationGaze(region="center", conf=0.2))
    assert verify(run.trial(decision, (altered,), trial_index=4, cands=gz, layout=gz_layout)) == "posterior_mismatch"
    # a recorded press_index -1 replaced by the candidate under the release
    gap_press = run.activation(U, press=-1, result="none")
    assert gap_press.hit_index is None and gap_press.a_gate == 1.0
    assert verify(run.trial(decision, (gap_press,), trial_index=5)) == "ok"
    claimed = replace(gap_press, press_index=INTENDED)
    assert verify(run.trial(decision, (claimed,), trial_index=6)) == "posterior_mismatch"


def test_verification_accepts_enlarge_gaze_rounding_gap_and_far_activations() -> None:
    run = Run(settings=engine_settings(gaze_allowed=True))
    run.begin(gaze_accuracy=0.75)
    block = run.block(1)
    decision = run.decide(block, want="enlarge_target:1.5")
    # enlarge: the rendered rects are rebuilt from expanded_index and the regime's scale
    enlarged = run.activation(U, expanded_index=INTENDED, scale=1.5)
    assert enlarged.hit_rect.w == 24.0
    trial = run.trial(decision, (enlarged,), trial_index=0)
    assert run.observe(trial).stored.verification == "ok"
    # float endpoints are rounded to integers as inputAdapter.ts does (Math.round) before any posterior
    float_x, float_y = 340.6, 199.5
    rounded = (math.floor(float_x + 0.5), math.floor(float_y + 0.5))
    trial = run.trial(decision, (run.activation(rounded, press=INTENDED),), trial_index=1)
    assert run.observe(trial).stored.verification == "ok"
    # gaze with the session accuracy
    gazed = run.activation(U, gaze=ActivationGaze("left", 0.6), gaze_accuracy=0.75)
    assert run.observe(run.trial(decision, (gazed,), trial_index=2)).stored.verification == "ok"
    # an expansion recorded under a non-enlarge regime fails
    other = run.decide(run.block(2), want="act")
    bogus = run.activation(U, expanded_index=INTENDED, scale=1.5)
    assert run.observe(run.trial(other, (bogus,), trial_index=0)).stored.verification == "posterior_mismatch"


def test_press_on_gap_and_far_activations_verify_with_their_cells() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    table = run.table(decision, block, [(0, 0, "b3:p2:h0", "wait_for_evidence")])
    gap = run.activation(GAP, l2=l2(table.decision_id, "b3:p2:h0", "wait_for_evidence"), result="none")
    assert gap.press_index == -1 and gap.hit_index is None and gap.n_plausible == 2
    trial = run.trial(decision, (gap,), trial_index=0)
    assert run.observe(trial).stored.verification == "ok"
    # far from every candidate: n_plausible 1, cell b3:p1:h0 (only wait_for_evidence is possible)
    far = run.activation(FAR, l2=l2(table.decision_id, "b3:p1:h0", "wait_for_evidence"), result="none")
    assert far.n_plausible == 1
    post = compute_posterior(CANDS, FAR, modality="mouse", r=1.0, floor=1.0, surface=SURFACE, press_index=-1)
    assert max(post.posterior) < 0.15 and math.isfinite(post.p_top)
    assert run.observe(run.trial(decision, (far,), trial_index=1)).stored.verification == "ok"


def test_fallbacks_relabel_only_their_own_level() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    fb = run.fallbacks()
    # an L1 fallback: the regime is fallbacks.l1 and the trial is 'fallback' (practice only)
    trial = run.trial(
        decision,
        (run.activation(U, result="held", outcome="committed"),),
        trial_index=0,
        fallbacks=fb,
        applied_regime=fb.l1,
        regime=fb.l1,
        l1_fallback_reason="late",
    )
    observation = run.observe(trial)
    assert observation.stored.verification == "fallback" and not observation.learned
    # an L2 fallback (late table): the activation applies the first applicable fallbacks.l2 entry; the trial is ok
    applied = l2(decision.decision_id, "b1:p2:h1", "ask_clarification:2", source="fallback", reason="late")
    trial = run.trial(
        decision,
        (run.activation(AH2, l2=applied, result="asked", outcome="committed", committed_index=INTENDED),),
        trial_index=1,
    )
    assert run.observe(trial).stored.verification == "ok"
    # a late offer only: the trial stays ok
    offer = OfferResponse("offer-x", "no_offer", "ignored", "late")
    trial = unambiguous(run, decision, 2, offers=(offer,))
    assert run.observe(trial).stored.verification == "ok"
    # an L1 fallback that is not fallbacks.l1 is a draw mismatch
    trial = run.trial(
        decision,
        (run.activation(U),),
        trial_index=3,
        applied_regime="act",
        regime="act",
        l1_fallback_reason="rpc_error",
    )
    assert run.observe(trial).stored.verification == "draw_mismatch"


def test_fallbacks_issued_at_checks() -> None:
    run = Run()
    run.begin()
    decision = run.decide(run.block(1), want="assistance_off")
    first = unambiguous(run, decision, 0)
    assert run.observe(first).stored.verification == "ok"
    earlier = unambiguous(run, decision, 1, issued_at=first.fallbacks_issued_at - timedelta(seconds=1))
    assert run.observe(earlier).stored.verification == "draw_mismatch"
    before_session = unambiguous(run, decision, 2, issued_at=T0 - timedelta(seconds=1))
    assert run.observe(before_session).stored.verification == "draw_mismatch"
    after_record = run.trial(decision, (run.activation(U),), trial_index=3)
    after_record = replace(after_record, fallbacks_issued_at=after_record.recorded_at + timedelta(seconds=1))
    assert run.observe(after_record).stored.verification == "draw_mismatch"


def test_revert_regimes_are_verified() -> None:
    run = Run()
    run.begin()
    decision = run.decide(run.block(1), want="act")
    fb = run.fallbacks()
    assert fb.plain == "assistance_off"
    veto = VetoRecord(decision.decision_id, DecisionLevel.BLOCK_REGIME)
    good = RevertRecord("veto", 1, "assistance_off", fb.issued_at)
    acts = (run.activation(U), run.activation(U, index=1))
    trial = run.trial(decision, acts, trial_index=0, fallbacks=fb, veto=veto, revert=good)
    assert run.observe(trial).stored.verification == "ok"
    bad = RevertRecord("veto", 1, "confirm_selection:hold_700", fb.issued_at)
    trial = run.trial(decision, acts, trial_index=1, fallbacks=fb, veto=veto, revert=bad)
    assert run.observe(trial).stored.verification == "draw_mismatch"
    refresh_bad = RevertRecord("refresh", 1, "confirm_selection:hold_700", fb.issued_at)
    trial = run.trial(decision, acts, trial_index=2, fallbacks=fb, revert=refresh_bad)
    assert run.observe(trial).stored.verification == "draw_mismatch"


def test_a_veto_with_no_plain_regime_is_an_interrupted_trial_and_teaches_nothing() -> None:
    prefs = PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "confirm_selection")])
    run = Run(prefs, settings=engine_settings(adaptive_allowed=True))
    run.begin(session_number=2)
    run.engine.apply_event(
        EngineEvent.create(kind="calibration_suspended", session_id="s1", trial_id="t0", created_at=run.tick())
    )
    fb = run.fallbacks()
    assert fb.plain is None and fb.l1 is None
    prefs2 = PreferenceSet([*prefs.all, pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, at=run.now)])
    run.set_prefs(prefs2)
    decision = run.decide(run.block(1, "B"))
    assert decision.mode is PolicyMode.PINNED and decision.variant == "act"
    before = run.engine.snapshot()
    trial = run.trial(
        decision,
        (run.activation(U),),
        trial_index=0,
        veto=VetoRecord(decision.decision_id, DecisionLevel.BLOCK_REGIME),
        interrupted=True,
        interruption_reason="no_allowed_regime",
    )
    observation = run.observe(trial)
    assert observation.stored.verification == "interrupted" and observation.stored.labels is None
    assert run.engine.snapshot() == before


# --------------------------------------------------------------------------------------------- session state


def test_veto_cooldown_and_its_pinned_exemption() -> None:
    run = Run(settings=engine_settings(adaptive_allowed=True))
    run.begin(session_number=2)
    first = run.decide(run.block(1, "B"))
    fb = run.fallbacks()
    veto = VetoRecord(first.decision_id, DecisionLevel.BLOCK_REGIME)
    trial = run.trial(
        first,
        (run.activation(U), run.activation(U, index=1)),
        trial_index=0,
        fallbacks=fb,
        veto=veto,
        revert=RevertRecord("veto", 1, fb.plain, fb.issued_at),
    )
    assert run.observe(trial).stored.verification == "ok"
    second = run.decide(run.block(2, "B"))
    assert second.excluded.get(first.variant) is ExclusionReason.RECENT_VETO
    assert second.propensities[first.variant] == 0.0
    third = run.decide(run.block(3, "B"))  # the cooldown covers the next decision of that level only
    assert third.excluded.get(first.variant) is not ExclusionReason.RECENT_VETO

    # a veto in pinned mode adds no exclusion: the next frozen block keeps the regime
    frozen = PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "enlarge_target:2.0"}, at=run.now)])
    run.set_prefs(frozen)
    pinned = run.decide(run.block(4, "B"))
    assert pinned.mode is PolicyMode.PINNED and pinned.variant == "enlarge_target:2.0"
    fb = run.fallbacks()
    trial = run.trial(
        pinned,
        (run.activation(U), run.activation(U, index=1)),
        trial_index=0,
        fallbacks=fb,
        veto=VetoRecord(pinned.decision_id, DecisionLevel.BLOCK_REGIME),
        revert=RevertRecord("veto", 1, fb.plain, fb.issued_at),
    )
    assert run.observe(trial).stored.verification == "ok"
    again = run.decide(run.block(5, "B"))
    assert again.mode is PolicyMode.PINNED and again.variant == "enlarge_target:2.0"
    assert run.engine._session.constraint.vetoed_at(DecisionLevel.BLOCK_REGIME) == frozenset()  # noqa: SLF001


def test_a_regime_veto_after_a_refresh_revert_excludes_the_regime_the_chip_showed() -> None:
    """After a refresh revert the regime chip shows the revert regime; its "Not this time" names that regime, not the
    cue regime the refresh had already replaced (C5, spec 9.7)."""
    run = Run(PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "assistance_off")]))
    run.begin()
    first = run.decide(run.block(1), want="act")
    cue = run.fallbacks()
    fresh = run.fallbacks()
    assert fresh.refresh == "confirm_selection:hold_700" != first.variant
    trial = run.trial(
        first,
        (run.activation(U), run.activation(U, index=1)),
        trial_index=0,
        fallbacks=cue,
        revert=RevertRecord("refresh", 1, fresh.refresh, fresh.issued_at),
        veto=VetoRecord(first.decision_id, DecisionLevel.BLOCK_REGIME),
    )
    assert run.observe(trial).stored.verification == "ok"
    session = run.engine._session  # noqa: SLF001
    assert session.constraint.vetoed_at(DecisionLevel.BLOCK_REGIME) == {"confirm_selection:hold_700"}
    second = run.decide(run.block(2))
    assert second.excluded.get("confirm_selection:hold_700") is ExclusionReason.RECENT_VETO
    assert second.propensities["confirm_selection:hold_700"] == 0.0
    assert second.excluded.get("act") is not ExclusionReason.RECENT_VETO


def test_a_response_veto_after_a_refresh_revert_names_the_response_after_it() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    cell = "b1:p2:h1"
    table = run.table(decision, block, [(0, 0, cell, "confirm_selection:second_click")])
    cue = run.fallbacks()
    # Pause adapting at the current regime: the trial reverts to the frozen act, and its L2 responses switch to the
    # fixed rule of the fresh fallbacks, which carry the block's L1 decision id.
    freeze = pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, at=run.tick(), participant_key=PK)
    run.set_prefs(PreferenceSet([freeze]))
    fresh = run.engine.fallbacks("mouse", now=run.tick())
    assert fresh.refresh == "act"
    confirm = run.activation(
        AH2,
        l2=l2(table.decision_id, cell, "confirm_selection:second_click"),
        result="confirming",
        outcome="cancelled",
    )
    asked = run.activation(
        AH2,
        index=1,
        l2=l2(decision.decision_id, cell, "ask_clarification:2", source="fixed_rule", draw=1),
        result="asked",
        outcome="committed",
        committed_index=INTENDED,
    )
    trial = run.trial(
        decision,
        (confirm, asked),
        trial_index=0,
        fallbacks=cue,
        revert=RevertRecord("refresh", 1, fresh.refresh, fresh.issued_at),
        veto=VetoRecord(decision.decision_id, DecisionLevel.AMBIGUITY_RESPONSE),
    )
    assert run.observe(trial).stored.verification == "ok"
    session = run.engine._session  # noqa: SLF001
    assert session.constraint.vetoed_at(DecisionLevel.AMBIGUITY_RESPONSE) == {"ask_clarification:2"}


def test_control_floor_excludes_a_regime_for_the_rest_of_the_session() -> None:
    run = Run(settings=engine_settings(adaptive_allowed=True))
    run.begin(session_number=2)
    decision = run.decide(run.block(1, "A_lite"), want="confirm_selection:hold_700")
    for trial_index in range(8):
        run.observe(
            run.trial(decision, (run.activation(U, result="held", outcome="committed"),), trial_index=trial_index)
        )
    for _ in range(6):
        run.burden(1, effort=7, intrusiveness=7, control=1)
    adaptive = run.decide(run.block(2, "B"))
    assert adaptive.excluded.get("confirm_selection:hold_700") is ExclusionReason.LOW_CONTROL_RATING


def test_cross_session_learning_off_resets_the_model_and_counters() -> None:
    run = Run(settings=engine_settings(learn_across_sessions=False))
    run.begin()
    fresh = CoAdaptationEngine(participant_key=PK, settings=engine_settings(), preferences=PreferenceSet([]))
    decision = run.decide(run.block(1), want="assistance_off")
    for trial_index in range(4):
        run.observe(unambiguous(run, decision, trial_index))
    assert run.engine.model.practice().n_total == 4
    run.end()
    run.tick(3600)
    run.begin("s2", day_index=740001, settings=engine_settings(learn_across_sessions=False))
    assert run.engine.model.practice().n_total == 0
    assert run.engine.beta_means().keys() == fresh.beta_means().keys()
    for key, value in run.engine.beta_means().items():
        assert np.array_equal(value, fresh.beta_means()[key])


def test_set_preferences_c7_and_drift_mark_open_decisions_stale() -> None:
    run = Run()
    assert run.engine.mark_stale("preferences") is False  # no session
    run.begin()
    decision = run.decide(run.block(1))
    assert not run.engine.refresh_required()
    run.set_prefs(PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "ask_clarification:3", at=run.now)]))
    assert run.engine.refresh_required()
    run.decide(run.block(2))
    assert not run.engine.refresh_required()
    assert run.engine.mark_stale("drift") and run.engine.refresh_required()
    run.decide(run.block(3))
    assert run.engine.mark_stale("calibration_suspended")
    assert decision.decision_id in run.engine._session.stale  # noqa: SLF001


@pytest.mark.parametrize("reason", ["preferences", "calibration_suspended", "drift"])
def test_a_change_with_no_open_decision_still_asks_for_a_refresh(reason: str) -> None:
    """Practice requests no decision, and a block whose L1 request failed holds none: both run on the fallbacks
    issued at their cue, so a rule change, a C7 suspension or a drift alarm must still refresh them (spec 9.7)."""
    run = Run()
    run.begin()
    assert not run.engine.refresh_required()
    if reason == "preferences":
        run.set_prefs(PreferenceSet([pref(PreferenceKind.PIN_ACT, True, at=run.now)]))
    else:
        assert run.engine.mark_stale(reason)  # type: ignore[arg-type]
    assert run.engine.refresh_required()
    run.decide(run.block(1))
    assert not run.engine.refresh_required()


def test_only_a_rule_that_applies_to_the_participant_marks_its_session_stale() -> None:
    run = Run()
    run.begin()
    other = "p-0000beef"
    assert other != PK
    run.set_prefs(PreferenceSet([pref(PreferenceKind.PIN_ACT, True, at=run.now, participant_key=other)]))
    assert not run.engine.refresh_required()
    mine = pref(PreferenceKind.FORBID_VARIANT, "ask_clarification:3", at=run.now, participant_key=PK)
    run.set_prefs(PreferenceSet([*run.prefs.all, mine]))
    assert run.engine.refresh_required()
    run.decide(run.block(1))
    # retracting the rule is a change too; the same set again is not
    retracted = replace(mine, retracted_at=run.tick())
    same = PreferenceSet([p for p in run.prefs.all if p.pref_id != mine.pref_id] + [retracted])
    run.set_prefs(same)
    assert run.engine.refresh_required()
    run.decide(run.block(2))
    run.set_prefs(same)
    assert not run.engine.refresh_required()


def _slipping_run() -> tuple[Run, set[str]]:
    """A block frozen at act whose every trial commits a non-intended candidate, until C7 suspends acting regimes."""
    run = Run(PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"})]))
    run.begin()
    kinds: set[str] = set()
    for block_index in range(1, 12):
        decision = run.decide(run.block(block_index))
        assert decision.mode is PolicyMode.PINNED and decision.variant == "act"
        first_event_in_block = True
        for trial_index in range(8):
            wrong = run.activation((int(CANDS[6].cx), int(CANDS[6].cy)), press=6)
            observation = run.observe(run.trial(decision, (wrong,), trial_index=trial_index))
            assert observation.stored.verification == "ok"
            assert observation.stored.regime_ucb_unintended is not None
            for event in observation.events:
                kinds.add(event.kind)
                assert event.trial_id == observation.stored.trial.trial_id
                assert event.created_at == observation.stored.trial.recorded_at
            if observation.events and first_event_in_block:
                assert observation.refresh_required and run.engine.refresh_required()
                first_event_in_block = False
        if "calibration_suspended" in kinds:
            break
    return run, kinds


def test_c7_and_drift_alarms_mark_stale_and_replay_exactly() -> None:
    run, kinds = _slipping_run()
    assert kinds == {"drift", "calibration_suspended"}
    status = run.engine.status()
    assert status.calibration_suspended and "l1:ua" in status.drift_flags
    fb = run.fallbacks()
    # C7 removes every acting regime from the fallbacks, the plain interface included; the freeze still holds
    assert fb.plain == "confirm_selection:hold_700"
    assert fb.l1 == fb.refresh == fb.practice == "act"
    run.end()
    rebuilt = run.replay()
    assert rebuilt.snapshot() == run.engine.snapshot()
    assert rebuilt.status() == run.engine.status()


def test_detectors_never_fire_during_replay() -> None:
    run, _ = _slipping_run()
    run.end()
    without_events = Run(run.prefs)
    without_events.rows = [row for row in run.rows if not isinstance(row, EngineEvent)]
    without_events.ends = run.ends
    rebuilt = without_events.replay()
    assert rebuilt.status().drift_flags == () and not rebuilt.status().calibration_suspended
    assert rebuilt.snapshot() != run.engine.snapshot()


def test_a_calibration_suspension_clears_at_the_day_boundary_and_stays_cleared() -> None:
    run, kinds = _slipping_run()
    assert "calibration_suspended" in kinds and run.engine.status().calibration_suspended
    run.end()
    run.set_prefs(PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "assistance_off"}, at=run.now)]))
    run.tick(3600)
    run.begin("s2", day_index=740001)
    assert not run.engine.status().calibration_suspended
    decision = run.decide(run.block(1))
    assert decision.mode is PolicyMode.PINNED and decision.variant == "assistance_off"
    observation = run.observe(unambiguous(run, decision, 0))
    assert observation.stored.verification == "ok"
    # the first ok row of the new day does not fire C7 again on the evidence of the day before
    assert observation.events == () and not observation.refresh_required
    assert not run.engine.status().calibration_suspended


def test_only_a_row_that_entered_the_c7_buffer_can_fire_a_suspension() -> None:
    run, _ = _slipping_run()
    run.end()
    # A rebuild without the stored events holds a buffer that tests positive and no suspension.
    without_events = Run(PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "assistance_off"})]))
    without_events.rows = [row for row in run.rows if not isinstance(row, EngineEvent)]
    without_events.ends = run.ends
    rebuilt = without_events.replay()
    assert not rebuilt.status().calibration_suspended and rebuilt.model.calibration_test()
    without_events.engine = rebuilt
    without_events.now = run.now + timedelta(minutes=5)
    without_events.begin("s2")  # same day: no boundary
    off = without_events.decide(without_events.block(1))
    assert off.variant == "assistance_off"
    observation = without_events.observe(unambiguous(without_events, off, 0))
    assert observation.stored.verification == "ok" and observation.stored.regime_ucb_unintended is None
    # an assistance_off row adds no evidence, so it names no suspension (a drift alarm may still fire here)
    assert "calibration_suspended" not in [event.kind for event in observation.events]
    assert not without_events.engine.status().calibration_suspended
    without_events.set_prefs(PreferenceSet([pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"})]))
    act = without_events.decide(without_events.block(2))
    wrong = without_events.activation((int(CANDS[6].cx), int(CANDS[6].cy)), press=6)
    observation = without_events.observe(without_events.trial(act, (wrong,), trial_index=0))
    assert observation.stored.regime_ucb_unintended is not None
    assert "calibration_suspended" in [event.kind for event in observation.events]


def test_preference_violations_are_judged_at_application_time() -> None:
    run = Run()
    run.begin()
    block = run.block(1)
    decision = run.decide(block, want="act")
    cell = "b1:p2:h1"
    table = run.table(decision, block, [(0, 0, cell, "ask_clarification:2"), (1, 0, cell, "ask_clarification:2")])
    cue_fallbacks = run.fallbacks()
    rule_at = run.tick()
    run.set_prefs(PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "ask_clarification:2", at=rule_at)]))
    ask = l2(table.decision_id, cell, "ask_clarification:2")
    # the table and the cue's fallbacks were issued before the rule: an ask applied afterwards is not a violation
    first = run.activation(AH2, l2=ask, result="asked", outcome="committed", committed_index=INTENDED)
    trial = run.trial(decision, (first,), trial_index=0, fallbacks=cue_fallbacks)
    stored = run.observe(trial).stored
    assert stored.verification == "ok" and stored.preference_violations == ()
    # after a refresh revert whose fresh fallbacks were issued after the rule, the same ask is flagged
    fresh = run.engine.fallbacks("mouse", now=run.tick())
    assert fresh.l2 == ("ask_clarification:3", "wait_for_evidence")
    revert = RevertRecord("refresh", 1, fresh.refresh, fresh.issued_at)
    after = run.activation(
        AH2,
        index=1,
        l2=replace(ask, source="fixed_rule"),
        result="asked",
        outcome="committed",
        committed_index=INTENDED,
    )
    trial = run.trial(
        decision, (replace(first, outcome="cancelled"), after), trial_index=1, fallbacks=cue_fallbacks, revert=revert
    )
    stored = run.observe(trial).stored
    assert stored.verification == "draw_mismatch"  # the fixed rule of the fresh fallbacks gives wait_for_evidence
    assert stored.preference_violations == ("activation[1]:ask_clarification:2:forbidden",)


def test_a_regime_applied_against_the_rules_sets_preference_violations() -> None:
    run = Run(PreferenceSet([pref(PreferenceKind.FORBID_VARIANT, "enlarge_target:1.5", at=T0 - timedelta(days=2))]))
    run.begin()
    decision = run.decide(run.block(1), want="act")
    trial = run.trial(decision, (run.activation(U),), trial_index=0, applied_regime="enlarge_target:1.5")
    stored = run.observe(trial).stored
    assert stored.verification == "draw_mismatch"
    assert stored.preference_violations == ("l1:enlarge_target:1.5:forbidden",)


# ---------------------------------------------------------------------------------------------- spread, keyboard


def test_spread_updates_only_from_eligible_first_activations() -> None:
    run = Run()
    run.begin()
    start = run.engine.spread()["mouse"]
    enlarge = run.decide(run.block(1), want="enlarge_target:2.0")
    off_centre = (int(CANDS[INTENDED].cx + 4), int(CANDS[INTENDED].cy - 3))
    run.observe(run.trial(enlarge, (run.activation(off_centre),), trial_index=0))
    assert run.engine.spread()["mouse"] == start  # enlarged trials never update the spread
    plain = run.decide(run.block(2), want="assistance_off")
    # a first activation outside the selection-error window (a selection error onto candidate 7) changes nothing
    wrong = (int(CANDS[7].cx), int(CANDS[7].cy))
    run.observe(run.trial(plain, (run.activation(wrong, press=7),), trial_index=0))
    assert run.engine.spread()["mouse"] == start
    # one inside the window updates with the corrected z^2
    run.observe(run.trial(plain, (run.activation(off_centre),), trial_index=1))
    after = run.engine.spread()["mouse"]
    assert after != start
    from pilot.coadaptation.core.filters import SPREAD_TRUNCATION_FACTOR

    sigma = math.sqrt((16.0 / 4.133) ** 2 + 1.0)
    a = 3.0 + 1.0
    b = 2.0 + ((4 / sigma) ** 2 + (3 / sigma) ** 2) / (2.0 * SPREAD_TRUNCATION_FACTOR)
    assert after == pytest.approx(math.sqrt(b / (a - 1.0)), rel=1e-12)
    run.end()
    assert run.replay().spread()["mouse"] == after


def test_spread_eligibility_follows_the_regime_that_governed_the_first_activation() -> None:
    off_centre = (int(CANDS[INTENDED].cx + 6), int(CANDS[INTENDED].cy - 5))
    # An act block that a pin of enlarge_target:2.0 refreshes before the first click: that click aims at the enlarged
    # rect and must not enter the spread update (section 7.5).
    run = Run()
    run.begin()
    start = run.engine.spread()["mouse"]
    decision = run.decide(run.block(1), want="act")
    cue = run.fallbacks()
    run.set_prefs(
        PreferenceSet([pref(PreferenceKind.PIN_REGIME, "enlarge_target:2.0", at=run.tick(), participant_key=PK)])
    )
    fresh = run.engine.fallbacks("mouse", now=run.tick())
    assert fresh.refresh == "enlarge_target:2.0"
    first = run.activation(off_centre, expanded_index=INTENDED, scale=2.0, press=INTENDED)
    revert = RevertRecord("refresh", 0, fresh.refresh, fresh.issued_at)
    stored = run.observe(run.trial(decision, (first,), trial_index=0, fallbacks=cue, revert=revert)).stored
    assert stored.verification == "ok" and stored.trial.applied_regime == "act"
    assert run.engine.spread()["mouse"] == start
    run.end()
    assert run.replay().spread()["mouse"] == start

    # The reverse: an enlarge block whose "Not this time" came before the first click runs that click on the plain
    # interface's rects, so it does enter the update.
    run = Run()
    run.begin()
    enlarge = run.decide(run.block(1), want="enlarge_target:2.0")
    cue = run.fallbacks()
    assert cue.plain == "assistance_off"
    plain_first = run.activation(off_centre, press=INTENDED)
    stored = run.observe(
        run.trial(
            enlarge,
            (plain_first,),
            trial_index=0,
            fallbacks=cue,
            revert=RevertRecord("veto", 0, cue.plain, cue.issued_at),
            veto=VetoRecord(enlarge.decision_id, DecisionLevel.BLOCK_REGIME),
        )
    ).stored
    assert stored.verification == "ok"
    assert run.engine.spread()["mouse"] != start


def test_keyboard_sessions_and_keyboard_activations() -> None:
    run = Run()
    run.begin(modality="keyboard")
    decision = run.decide(run.block(1), want="act")
    assert decision.spread is None
    for focused in (0, 5, 11):
        trial = run.trial(decision, (keyboard_activation(CANDS, focused),), trial_index=focused, intended=focused)
        assert trial.spread is None
        assert run.observe(trial).stored.verification == "ok"
    good = keyboard_activation(CANDS, 3)
    for tampered in (
        replace(good, x=good.x + 1),
        replace(good, a_gate=0.1),
        replace(good, press_index=4),
        replace(good, n_plausible=2),
    ):
        trial = run.trial(decision, (tampered,), trial_index=3, intended=3)
        assert run.observe(trial).stored.verification == "posterior_mismatch"
    assert run.engine.spread()["mouse"] == 1.0  # keyboard activations never update a spread
    run.end()

    # a mouse session's menu trial with a keyboard activation: the trial's mouse spread serves only the pointer ones
    run2 = Run()
    run2.begin()
    items = menu(6)
    menu_layout = layout_for(items, task="menu", origin=(300.0, 60.0))
    decision = run2.decide(run2.block(1, layout=menu_layout), want="assistance_off")
    pointer = pointer_activation(
        items,
        (int(items[1].cx), int(items[1].cy)),
        spread=SpreadUsed(1.0, 1.0),
        press_index=1,
        result="none",
        committed_index=None,
    )
    pointer = replace(pointer, result="committed", outcome="committed", committed_index=1)
    key = replace(keyboard_activation(items, 2), index=1)
    trial = run2.trial(decision, (pointer, key), trial_index=0, cands=items, layout=menu_layout, intended=2)
    assert run2.observe(trial).stored.verification == "ok"
    # a pointer activation in a trial without a spread fails
    trial = run2.trial(decision, (pointer,), trial_index=1, cands=items, layout=menu_layout, intended=1, spread=None)
    assert run2.observe(trial).stored.verification == "posterior_mismatch"


# -------------------------------------------------------------------------------------------- settings, replay


def test_settings_are_fixed_per_session_and_replayed_exactly() -> None:
    run = Run(settings=engine_settings(ambiguity_threshold=0.25))
    run.begin(settings=engine_settings(ambiguity_threshold=0.15))
    assert run.engine.settings.ambiguity_threshold == 0.15
    block = run.block(1)
    # enlarge regimes never update the spread, so both sessions use r = 1 and the same A_gate
    decision = run.decide(block, want="enlarge_target:1.5")
    endpoint = (334, 193)
    post = compute_posterior(CANDS, endpoint, modality="mouse", r=1.0, floor=1.0, surface=SURFACE, press_index=5)
    assert 0.15 < post.ambiguity_gate <= 0.25
    table = run.table(decision, block, [(0, 0, "b1:p1:h1", "wait_for_evidence")])
    assert table.ambiguity_threshold == 0.15 and table.bins[0] == ("b1", 0.15, 0.4)
    routed = run.activation(endpoint, press=5, l2=l2(table.decision_id, "b1:p1:h1", "wait_for_evidence"), result="none")
    assert run.observe(run.trial(decision, (routed,), trial_index=0)).stored.verification == "ok"
    direct = run.activation(endpoint, press=5)
    assert run.observe(run.trial(decision, (direct,), trial_index=1)).stored.verification == "draw_mismatch"
    run.end()
    run.tick(600)
    run.begin("s2", settings=engine_settings(ambiguity_threshold=0.25))
    assert run.engine.settings == engine_settings(ambiguity_threshold=0.25)
    decision = run.decide(run.block(1), want="enlarge_target:1.5")
    assert decision.spread == (1.0, 1.0)
    direct = run.activation(endpoint, press=5)
    assert run.observe(run.trial(decision, (direct,), trial_index=0)).stored.verification == "ok"
    run.end()
    rebuilt = run.replay()
    assert rebuilt.snapshot() == run.engine.snapshot()


def test_replay_applies_only_what_live_learning_applied() -> None:
    run = Run()
    run.begin()
    decision = run.decide(run.block(1), want="assistance_off")
    run.observe(unambiguous(run, decision, 0))
    stored = run.rows[-1]
    assert isinstance(stored, StoredTrial)
    snapshot = run.engine.snapshot()
    practice = run.engine.model.practice()
    for verification in ("draw_mismatch", "posterior_mismatch", "interrupted"):
        run.engine.replay_trial(replace(stored, verification=verification))
        assert run.engine.snapshot() == snapshot
    run.engine.replay_trial(replace(stored, verification="fallback"))
    assert run.engine.model.practice().n_total == practice.n_total + 1
    practice_keys = ("model.practice_total", "model.practice_variant", "model.item_eff", "model.item_time")
    assert _arrays_equal(run.engine.snapshot(), snapshot, skip=practice_keys)
    assert not _arrays_equal(run.engine.snapshot(), snapshot, skip=())


def _arrays_equal(a: bytes, b: bytes, *, skip: tuple[str, ...]) -> bool:
    from pilot.coadaptation.core.outcome_model import read_npz

    left, right = read_npz(a), read_npz(b)
    return left.keys() == right.keys() and all(
        name in skip or np.array_equal(left[name], right[name], equal_nan=left[name].dtype.kind == "f") for name in left
    )


def test_policy_probabilities_keep_a_logged_veto_at_its_own_level() -> None:
    """``confirm_selection:second_click`` is an L1 regime and an L2 response. Live, a veto of the regime excludes it
    at the next L1 decision only and the block's table keeps the response; the target policy must agree (C5)."""
    second_click = "confirm_selection:second_click"
    run = Run(settings=engine_settings(adaptive_allowed=True))
    run.begin(session_number=2)
    run.decide(run.block(1, "B"))
    # The live state after a "Not this time" on the second_click regime chip.
    run.engine._session.constraint.record_veto(DecisionLevel.BLOCK_REGIME, second_click)  # noqa: SLF001
    block = run.block(2, "B")
    live = run.decide(block)
    assert live.excluded.get(second_click) is ExclusionReason.RECENT_VETO
    table = run.table(live, block)
    has_hit = [cell for cell in table.cells if cell.has_hit and second_click in cell.eligible]
    assert has_hit and all(cell.propensities[second_click] > 0.0 for cell in has_hit)

    logged = {second_click: ExclusionReason.RECENT_VETO}
    kwargs = dict(block=block, preferences=run.prefs, seed=7, draws=64, now=run.tick())
    at_l1 = run.engine.policy_probabilities(
        session_exclusions=logged, exclusions_level=DecisionLevel.BLOCK_REGIME, **kwargs
    )
    assert at_l1.l1[second_click] == 0.0
    assert all(at_l1.l2[cell.cell_id][second_click] > 0.0 for cell in has_hit)
    at_l2 = run.engine.policy_probabilities(
        session_exclusions=logged, exclusions_level=DecisionLevel.AMBIGUITY_RESPONSE, **kwargs
    )
    assert all(cell[second_click] == 0.0 for cell in at_l2.l2.values())
    unvetoed = run.engine.policy_probabilities(session_exclusions={}, **kwargs)
    assert at_l2.l1 == unvetoed.l1  # an L2 veto leaves the L1 probabilities alone


def test_predictions_and_analysis_reads_never_change_the_snapshot() -> None:
    run = Run(settings=engine_settings(adaptive_allowed=True))
    run.begin(session_number=2)
    decision = run.decide(run.block(1, "B"))
    run.observe(unambiguous(run, decision, 0))
    stored = run.rows[-1]
    snapshot = run.engine.snapshot()
    block = run.block(2, "B")
    for _ in range(3):
        run.engine.predictions_for_block(block, now=run.tick())
        run.engine.beta_means()
        prediction = run.engine.prequential_prediction(stored)
        probabilities = run.engine.policy_probabilities(
            block=block,
            session_exclusions={},
            preferences=run.prefs,
            seed=123,
            draws=64,
            now=run.now,
            offer=OfferContext("s1", 2, 0, "mouse", "pointing", "corrections", True),
        )
    assert run.engine.snapshot() == snapshot
    assert set(prediction) == {"complete", "ua", "correction", "veto", "log_time_mean", "log_time_sd"}
    again = run.engine.policy_probabilities(
        block=block,
        session_exclusions={},
        preferences=run.prefs,
        seed=123,
        draws=64,
        now=run.now,
        offer=OfferContext("s1", 2, 0, "mouse", "pointing", "corrections", True),
    )
    assert again == probabilities
    assert probabilities.l1["assistance_off"] == 0.0
    assert all(cell["act"] == 0.0 for cell in probabilities.l2.values())
    assert sum(probabilities.l1.values()) == pytest.approx(1.0)


def test_prequential_prediction_is_what_the_drift_detector_scores() -> None:
    run = Run()
    run.begin()
    decision = run.decide(run.block(1), want="act")
    trial = unambiguous(run, decision, 0)
    stored = run.engine.observe_trial(trial).stored
    # rebuild up to just before the trial and compare the prediction with the model's detector input
    replica = CoAdaptationEngine(participant_key=PK, settings=engine_settings(), preferences=PreferenceSet([]))
    replica.begin_session(run.rows[0])
    prediction = replica.prequential_prediction(stored)
    context = replica._trial_context(stored.trial)  # noqa: SLF001
    assert prediction == replica.model.l1_trial_prediction(context)


def test_snapshot_restore_and_strict_checks() -> None:
    run = Run()
    run.begin()
    decision = run.decide(run.block(1), want="assistance_off")
    run.observe(unambiguous(run, decision, 0))
    run.end()
    data = run.engine.snapshot()
    restored = CoAdaptationEngine.restore(data, participant_key=PK, settings=engine_settings(), preferences=run.prefs)
    assert restored.snapshot() == data and not restored.session_open
    assert restored.status().trials_seen == 1
    with pytest.raises(ValueError):
        CoAdaptationEngine.restore(b"not an npz", participant_key=PK, settings=engine_settings(), preferences=run.prefs)
    with pytest.raises(ValueError):
        CoAdaptationEngine.restore(
            data,
            participant_key=PK,
            settings=engine_settings(),
            preferences=run.prefs,
            model_variant=ModelVariant.NO_TODAY,
        )


def test_engine_settings_validate_the_config_ranges() -> None:
    EngineSettings()
    for bad in (
        {"ambiguity_threshold": 0.3},
        {"ambiguity_threshold": 0.04},
        {"unintended_activation_tolerance": 0.25},
        {"noninferiority_margin": 0.06},
        {"upper_bound_credibility": 0.999},
        {"exploration_floor": 0.31},
        {"switching_penalty": 1.5},
        {"day_boundary_hour": 24},
        {"thompson_draws": 0},
        {"adaptive_allowed": 1},
    ):
        with pytest.raises(ValueError):
            EngineSettings(**bad)
    assert EngineSettings(**EngineSettings().as_dict()) == EngineSettings()


# ------------------------------------------------------------------------------------------------- cap rule


def _cap_run(cap: int | None, *, pin_act: bool = False, confirm: bool = False) -> Run:
    rules = []
    if cap is not None:
        rules.append(pref(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, cap))
    if pin_act:
        rules.append(pref(PreferenceKind.PIN_ACT, True))
    if confirm and cap is not None and cap < 8:
        rules.append(pref(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click"))
    run = Run(PreferenceSet(rules))
    run.begin()
    return run


def test_cap_rule_verification_and_block_count() -> None:
    run = _cap_run(2)
    block = run.block(1)
    decision = run.decide(block, want="act")
    cell = "b1:p2:h1"
    table = run.table(
        decision,
        block,
        [
            (0, 0, cell, "confirm_selection:second_click"),
            (1, 0, cell, "ask_clarification:2"),
            (2, 0, cell, "ask_clarification:2"),
        ],
    )
    confirm = run.activation(
        AH2, l2=l2(table.decision_id, cell, "confirm_selection:second_click"), result="confirming", outcome="cancelled"
    )
    assert run.observe(run.trial(decision, (confirm,), trial_index=0, corrections=1)).stored.verification == "ok"
    # an interrupted trial's prompt counts toward the block's k
    ask = run.activation(
        AH2,
        l2=l2(table.decision_id, cell, "ask_clarification:2"),
        result="asked",
        outcome="open_at_end",
        committed_index=None,
    )
    interrupted = run.trial(decision, (ask,), trial_index=1, interrupted=True, interruption_reason="paused")
    assert run.observe(interrupted).stored.verification == "interrupted"
    assert run.engine.interruptions_used(1) == 2
    capped = run.activation(AH2, l2=l2(table.decision_id, cell, "wait_for_evidence", source="cap"), result="none")
    assert run.observe(run.trial(decision, (capped,), trial_index=2)).stored.verification == "ok"
    drawn = run.activation(
        AH2,
        l2=l2(table.decision_id, cell, "ask_clarification:2"),
        result="asked",
        outcome="committed",
        committed_index=INTENDED,
    )
    assert run.observe(run.trial(decision, (drawn,), trial_index=2)).stored.verification == "draw_mismatch"

    early = _cap_run(2)
    block = early.block(1)
    decision = early.decide(block, want="act")
    table = early.table(decision, block)
    capped = early.activation(AH2, l2=l2(table.decision_id, cell, "wait_for_evidence", source="cap"), result="none")
    assert early.observe(early.trial(decision, (capped,), trial_index=0)).stored.verification == "draw_mismatch"


def test_recount_reads_result_never_outcome_and_counts_l1_steps_once() -> None:
    run = _cap_run(None, confirm=True)
    block = run.block(1)
    decision = run.decide(block, want="confirm_selection:hold_700")
    released = run.activation(U, result="held", outcome="cancelled", committed_index=None)
    rehold = run.activation(U, index=1, result="held", outcome="committed")
    run.observe(run.trial(decision, (released, rehold), trial_index=0, corrections=1))
    aborted = run.activation(U, result="held", outcome="aborted", committed_index=None)
    run.observe(
        run.trial(decision, (aborted,), trial_index=1, committed_index=None, commit_t_ms=None, commit_via="none")
    )
    assert run.engine.interruptions_used(1) == 2

    act_run = _cap_run(None)
    block = act_run.block(1)
    decision = act_run.decide(block, want="act")
    table = act_run.table(decision, block, [(0, 0, "b1:p2:h1", "confirm_selection:second_click")])
    cancelled = act_run.activation(
        AH2,
        l2=l2(table.decision_id, "b1:p2:h1", "confirm_selection:second_click"),
        result="confirming",
        outcome="cancelled",
    )
    act_run.observe(act_run.trial(decision, (cancelled,), trial_index=0, corrections=1))
    assert act_run.engine.interruptions_used(1) == 1


ENDPOINT_FOR = {
    (False, True, 1): U,
    (False, True, 2): U,
    (True, True, 2): AH2,
    (True, True, 3): AH3,
    (False, False, 2): GAP,
    (True, False, 2): GAP,
    (False, False, 1): FAR,
    (True, False, 1): FAR,
}


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("block", _fixture()["blocks"], ids=lambda b: b["id"])
def test_shared_cap_recount_fixture(block: dict[str, Any]) -> None:
    fixture = _fixture()
    assert fixture["tau"] == 0.25 and len(fixture["blocks"]) == 20
    regimes = {t["regime"] for t in block["trials"]}
    confirm = any(r.startswith("confirm_selection") for r in regimes)
    run = _cap_run(block["interruption_cap"], pin_act=block["pin_act"], confirm=confirm)
    regime = block["trials"][0]["regime"]
    context = run.block(1)
    decision = run.decide(context, want=regime)
    fb = run.fallbacks()
    assert fb.interruption_cap == block["interruption_cap"]
    for trial in block["trials"]:
        assert trial["plain_regime"] == fb.plain

    # collect the table draws the fixture's table responses need
    needs = []
    endpoints = {}
    for t_index, trial in enumerate(block["trials"]):
        draws = 0
        for i_index, item in enumerate([x for x in trial["items"] if x["type"] == "activation"]):
            endpoint = ENDPOINT_FOR[(item["ambiguous"], item["hit"], item["n_plausible"])]
            endpoints[(t_index, i_index)] = endpoint
            expected = item["expected"]
            if expected["source"] == "table" and not item["pin_act"]:
                post = compute_posterior(
                    CANDS,
                    endpoint,
                    modality="mouse",
                    r=1.0,
                    floor=1.0,
                    surface=SURFACE,
                    press_index=INTENDED if item["hit"] else -1,
                )
                bin_id = "b1" if post.ambiguity_gate <= 0.4 else "b2" if post.ambiguity_gate <= 0.6 else "b3"
                cell = f"{bin_id}:p{post.n_plausible}:h{1 if post.has_hit else 0}"
                needs.append((t_index, draws, cell, expected["variant"]))
                draws += 1
    table = None
    if regime in ("act", "enlarge_target:1.5", "enlarge_target:2.0"):
        table = run.table(decision, context, needs)

    for t_index, trial in enumerate(block["trials"]):
        k_block = 0 if trial["kind"] == "idle" else run.engine.interruptions_used(1)
        activations: list[ActivationRecord] = []
        revert = None
        veto = None
        issued = fb
        draws = 0
        a_index = 0
        for item in trial["items"]:
            if item["type"] == "veto":
                level = DecisionLevel(item["level"])
                veto = VetoRecord(
                    decision.decision_id if level == DecisionLevel.BLOCK_REGIME else table.decision_id, level
                )
                revert = RevertRecord("veto", a_index, fb.plain, fb.issued_at)
                continue
            if item["type"] == "refresh":
                frozen = PreferenceSet(
                    [*run.prefs.all, pref(PreferenceKind.FREEZE_ADAPTATION, {"regime": item["freeze"]}, at=run.tick())]
                )
                run.set_prefs(frozen)
                fresh = run.fallbacks()
                assert fresh.refresh == item["regime"]
                revert = RevertRecord("refresh", a_index, fresh.refresh, fresh.issued_at)
                continue
            expected = item["expected"]
            endpoint = endpoints[(t_index, a_index)]
            applied = None
            if expected["source"] is not None:
                post = compute_posterior(
                    CANDS,
                    endpoint,
                    modality="mouse",
                    r=1.0,
                    floor=1.0,
                    surface=SURFACE,
                    press_index=INTENDED if item["hit"] else -1,
                )
                bin_id = "b1" if post.ambiguity_gate <= 0.4 else "b2" if post.ambiguity_gate <= 0.6 else "b3"
                cell = f"{bin_id}:p{post.n_plausible}:h{1 if post.has_hit else 0}"
                owner = (
                    table.decision_id
                    if table is not None and expected["source"] != "fixed_rule"
                    else decision.decision_id
                )
                applied = L2Applied(owner, cell, min(draws, 4), expected["variant"], expected["source"], None)
                if expected["source"] == "table":
                    draws += 1
            committed = INTENDED if expected["outcome"] == "committed" and item["hit"] else None
            activation = run.activation(
                endpoint,
                index=a_index,
                result=expected["result"],
                outcome=expected["outcome"],
                l2=applied,
                committed_index=committed,
                irreversible_rule=item["irreversible"],
            )
            assert k_block + trial_interruptions(activations) == expected["k_before"]
            activations.append(activation)
            a_index += 1
        overrides: dict[str, Any] = {
            "corrections": sum(1 for a in activations if a.outcome == "cancelled"),
            "veto": veto,
            "revert": revert,
        }
        if trial["end"] == "interrupt":
            overrides.update(interrupted=True, interruption_reason="paused")
        if trial["kind"] == "idle":
            overrides.update(intended=None, duration_ms=30000.0)
        record = run.trial(decision, tuple(activations), trial_index=t_index, fallbacks=issued, **overrides)
        observation = run.observe(record)
        assert observation.stored.verification == trial["expected"]["verification"], (block["id"], t_index)
        count = k_block + trial_interruptions(activations)
        assert count == trial["expected"]["k_after"]
        if trial["kind"] != "idle":
            assert run.engine.interruptions_used(1) == trial["expected"]["k_after"]

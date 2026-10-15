"""Agency audit (spec section 20), also the SDK conformance check.

``agency_audit`` reads stored rows only (decisions, trials, sessions, burden reports, the rule history, the consent
periods and the drift and calibration-suspension events) and checks twelve invariants. Ten are zero-tolerance counts
and two are coverage ratios that must be 1.0; any failure fails the report, the study and the benchmark. Every
applied variant is judged under the rules in force when its source was issued (section 9.7), and every fallback the
audit recomputes uses those rules and the C7 state at its issue time, because ``plain``, ``refresh`` and
``practice`` depend on both. Standard library, numpy and ``pilot.coadaptation.core`` only.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .consent import CoAdaptationPurpose, ConsentPeriod, consent_in_force
from .constraints import ExclusionReason, Fallbacks, fallbacks, fixed_l2_rule, preference_violations
from .engine import EngineEvent
from .policy import select_from_cell
from .preferences import Preference, PreferenceKind, PreferenceSet
from .records import (
    ActivationRecord,
    BurdenReport,
    DecisionRecord,
    SessionRow,
    StoredTrial,
    TrialRecord,
    vetoed_variant,
)
from .vocabulary import L2_OPTIONS, L3_OPTIONS, VARIANTS, DecisionLevel, l1_arms

INVARIANTS: Mapping[str, str] = {
    "A1": "preference_violations",
    "A2": "acting_under_ambiguity",
    "A3": "commits_without_activation",
    "A4": "forbidden_in_exploration",
    "A5": "learned_act_at_l2",
    "A6": "unconsented_writes",
    "A7": "propensity_coverage",
    "A8": "explanation_coverage",
    "A9": "veto_to_effect",
    "A10": "authority_fields",
    "A11": "draw_mismatches",
    "A12": "irreversible_single_activation_commits",
}
COVERAGE_INVARIANTS = ("A7", "A8")  # pass at 1.0; every other invariant passes at 0
ACT_GUARDED = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})
PHASE_B = frozenset({"B", "probe", "comparator"})
PHASE_B_MODES = frozenset({"adaptive", "probe", "comparator"})
JUDGED_VETO_MODES = frozenset({"adaptive", "randomized"})
_INTERRUPTING_L2 = frozenset(v for v in L2_OPTIONS if VARIANTS[v].interrupts)
_REASONS = frozenset(reason.value for reason in ExclusionReason)


@dataclass(frozen=True, slots=True)
class Finding:
    invariant: str  # "A1" .. "A12"
    participant_key: str
    where: str  # "decision:<id>", "trial:<id>", "trial:<id>:activation[i]", "session:<id>" or "burden:<id>"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "invariant": self.invariant,
            "participant_key": self.participant_key,
            "where": self.where,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AgencyReport:
    counts: Mapping[str, float]  # A1..A12: violation counts, or coverage ratios for A7 and A8
    findings: tuple[Finding, ...]  # every violation, with where it was found
    exemptions: tuple[Finding, ...]  # A9 vetoes that were not judged, with the reason
    draw_mismatches_by_participant: Mapping[str, int]  # A11 per participant
    # Override rate, Never creations, pauses, offers, prompts per hour, entropy, switches.
    soft: Mapping[str, Any]
    passed: bool

    def failed(self) -> tuple[str, ...]:
        return tuple(key for key in INVARIANTS if not _passes(key, self.counts[key]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariants": {
                key: {
                    "name": INVARIANTS[key],
                    "value": self.counts[key],
                    "passed": _passes(key, self.counts[key]),
                }
                for key in INVARIANTS
            },
            "findings": [finding.to_dict() for finding in self.findings],
            "exemptions": [finding.to_dict() for finding in self.exemptions],
            "draw_mismatches_by_participant": dict(self.draw_mismatches_by_participant),
            "soft": dict(self.soft),
            "passed": self.passed,
        }


def _passes(key: str, value: float) -> bool:
    if key in COVERAGE_INVARIANTS:
        return value >= 1.0 - 1e-12
    return value == 0


# ------------------------------------------------------------------------------------------------ the context


@dataclass
class _Audit:
    decisions: Mapping[str, DecisionRecord]
    sessions: Mapping[str, SessionRow]
    history: PreferenceSet
    events: Sequence[EngineEvent]
    learning_reset_at: datetime | None
    findings: list[Finding] = field(default_factory=list)
    exemptions: list[Finding] = field(default_factory=list)
    _fallbacks: dict[tuple[str, datetime], Fallbacks] = field(default_factory=dict)
    _rules: dict[datetime, PreferenceSet] = field(default_factory=dict)

    def flag(self, invariant: str, participant_key: str, where: str, detail: str) -> None:
        self.findings.append(Finding(invariant, participant_key, where, detail))

    def rules_at(self, when: datetime) -> PreferenceSet:
        rules = self._rules.get(when)
        if rules is None:
            rules = self._rules[when] = self.history.as_of(when)
        return rules

    def c7_at(self, when: datetime, session: SessionRow) -> bool:
        """C7 in force at ``when`` in ``session``: suspended from a calibration_suspended event's time until the end
        of that session's day or a later learning reset (section 7.7)."""
        for event in self.events:
            if event.kind != "calibration_suspended" or event.created_at > when:
                continue
            fired_in = self.sessions.get(event.session_id)
            if fired_in is None or fired_in.participant_key != session.participant_key:
                continue
            if fired_in.day_index != session.day_index:
                continue
            reset = self.learning_reset_at
            if reset is not None and event.created_at < reset <= when:
                continue
            return True
        return False

    def fallbacks_at(self, session: SessionRow, when: datetime) -> Fallbacks:
        key = (session.session_id, when)
        cached = self._fallbacks.get(key)
        if cached is None:
            cached = self._fallbacks[key] = fallbacks(
                self.rules_at(when),
                session.device.modality,
                participant_key=session.participant_key,
                calibration_suspended=self.c7_at(when, session),
                now=when,
            )
        return cached

    def pin_act_at(self, session: SessionRow, when: datetime) -> bool:
        return self.rules_at(when).pin_act(
            context="study",
            modality=session.device.modality,
            participant_key=session.participant_key,
        )


def _phase(session: SessionRow, block_index: int) -> str:
    for plan in session.plan.blocks:
        if plan.index == block_index:
            return plan.phase
    return ""


def _regime_for(trial: TrialRecord, activation: ActivationRecord) -> str:
    revert = trial.revert
    if revert is not None and activation.index >= revert.from_activation:
        return revert.regime
    return trial.applied_regime


def _binding_time(
    trial: TrialRecord,
    activation: ActivationRecord,
    decisions: Mapping[str, DecisionRecord],
):
    """When the source of an activation's L2 response was issued (section 9.7)."""
    revert = trial.revert
    if revert is not None and activation.index >= revert.from_activation:
        return revert.fallbacks_issued_at
    applied = activation.l2
    if applied is not None and applied.source == "table":
        decision = decisions.get(applied.decision_id)
        return decision.created_at if decision is not None else trial.fallbacks_issued_at
    return trial.fallbacks_issued_at


def _cap_time(trial: TrialRecord, activation: ActivationRecord) -> datetime:
    """The cap in force is that of the fallbacks the UI held: the trial's, or after a revert the revert's (9.3)."""
    revert = trial.revert
    if revert is not None and activation.index >= revert.from_activation:
        return revert.fallbacks_issued_at
    return trial.fallbacks_issued_at


def _l1_step(activation: ActivationRecord) -> bool:
    l1_confirm = activation.result in ("confirming", "held") and activation.l2 is None
    return l1_confirm and not activation.irreversible_rule


def _other_step(activation: ActivationRecord) -> bool:
    return activation.result in ("confirming", "held", "asked") and not _l1_step(activation)


def trial_interruptions(trial: TrialRecord) -> int:
    """A trial's addition to the block's interruption count (9.3): 1 for its L1 confirm and hold steps however many
    it opened, plus each L2 confirm step, C9 confirm step and clarification prompt."""
    l1 = int(any(_l1_step(a) for a in trial.activations))
    return l1 + sum(1 for a in trial.activations if _other_step(a))


def _counts_before(trial: TrialRecord, k_block: int) -> list[int]:
    """``k`` before each activation of the trial, given the block's count before the trial."""
    out = []
    l1_counted = False
    k = k_block
    for activation in trial.activations:
        out.append(k)
        if _l1_step(activation):
            if not l1_counted:
                k += 1
                l1_counted = True
        elif _other_step(activation):
            k += 1
    return out


def _cell_shape(cell_id: str) -> tuple[int, bool]:
    parts = cell_id.split(":")
    return int(parts[1][1:]), parts[2] == "h1"


# --------------------------------------------------------------------------------------------- the invariants


def agency_audit(
    decisions: Sequence[DecisionRecord],
    trials: Sequence[StoredTrial],
    sessions: Sequence[SessionRow],
    burden: Sequence[BurdenReport],
    preferences_history: Sequence[Preference],
    consents: Sequence[ConsentPeriod],
    *,
    events: Sequence[EngineEvent] = (),
    learning_reset_at: datetime | None = None,
) -> AgencyReport:
    """The agency invariants A1-A12 of section 20 over stored rows, with the soft metrics.

    ``sessions`` give each trial's tau (the value its session ran with), start time and gaze-check accuracy;
    ``preferences_history`` is the export's rule history; ``events`` and ``learning_reset_at`` give the C7 state in
    force when each fallback was issued (without them C7 counts as never suspended).
    """
    audit = _Audit(
        decisions={d.decision_id: d for d in decisions},
        sessions={s.session_id: s for s in sessions},
        history=PreferenceSet(preferences_history),
        events=tuple(e for e in events if e.kind == "calibration_suspended"),
        learning_reset_at=learning_reset_at,
    )
    ordered = sorted(trials, key=lambda s: (s.trial.session_id, s.trial.seq))
    decision_times = {d.decision_id: d.created_at for d in decisions}

    _decision_invariants(audit, decisions)
    a11_trials: dict[str, set[str]] = defaultdict(set)
    k_blocks: dict[tuple[str, int], int] = defaultdict(int)
    for stored in ordered:
        trial = stored.trial
        session = audit.sessions.get(trial.session_id)
        if session is None:
            audit.flag(
                "A6",
                "",
                f"trial:{trial.trial_id}",
                "trial of a session the export does not hold",
            )
            continue
        if trial.task == "idle":
            k_before = 0  # an idle epoch counts from 0 on its own (9.3)
        else:
            k_before = k_blocks[(trial.session_id, trial.block_index)]
            k_blocks[(trial.session_id, trial.block_index)] += trial_interruptions(trial)
        _trial_invariants(audit, stored, session, k_before, decision_times, a11_trials)
    _veto_invariant(audit, ordered)
    _consent_invariant(audit, decisions, ordered, sessions, burden, consents)

    counts: dict[str, float] = {key: 0 for key in INVARIANTS}
    for finding in audit.findings:
        if finding.invariant not in ("A7", "A8", "A11"):
            counts[finding.invariant] += 1
    counts["A7"] = _coverage(audit, decisions, "A7")
    counts["A8"] = _coverage(audit, decisions, "A8")
    per_participant = {key: len(ids) for key, ids in sorted(a11_trials.items())}
    counts["A11"] = sum(per_participant.values())
    soft = _soft_metrics(ordered, sessions, burden, preferences_history)
    passed = all(_passes(key, counts[key]) for key in INVARIANTS)
    return AgencyReport(
        counts=counts,
        findings=tuple(audit.findings),
        exemptions=tuple(audit.exemptions),
        draw_mismatches_by_participant=per_participant,
        soft=soft,
        passed=passed,
    )


def _decision_invariants(audit: _Audit, decisions: Sequence[DecisionRecord]) -> None:
    for decision in decisions:
        session = audit.sessions.get(decision.session_id)
        payload = decision.payload
        where = f"decision:{decision.decision_id}"
        pk = decision.participant_key
        presentation_only = payload.get("authority") == "presentation_only"
        if not presentation_only or payload.get("execution_authority") is not False:
            audit.flag("A10", pk, where, "authority fields missing or wrong")
        if session is None:
            continue
        modality = session.device.modality
        rules = audit.rules_at(decision.created_at)
        level = DecisionLevel(decision.level)
        # A4: positive exploration mass on a forbidden variant, at any level and in any mode.
        for variant, _probability in _positive_mass(payload):
            if variant != "wait_for_evidence" and rules.forbidden(
                variant, context="study", modality=modality, participant_key=pk
            ):
                audit.flag("A4", pk, where, f"{variant} forbidden but has positive propensity")
        # A1: the chosen L1 regime or L3 offer was not allowed by the rules in force at decision time (forbidden,
        # below the minimum caution, not the frozen or pinned regime, or a confirm regime under a cap below the
        # block's trials where other regimes remained).
        if level in (DecisionLevel.BLOCK_REGIME, DecisionLevel.OFFER):
            options = l1_arms(modality) if level == DecisionLevel.BLOCK_REGIME else L3_OPTIONS
            allowed, _ = rules.apply(
                level,
                options,
                context="study",
                modality=modality,
                participant_key=pk,
                interruptions_used=0,
            )
            chosen = str(payload.get("variant"))
            if chosen not in allowed:
                audit.flag("A1", pk, where, f"chosen {chosen} not allowed by the rules in force")
        # A5: a table cell with pi(act) > 0 that is not a pin_act collapse.
        if level == DecisionLevel.AMBIGUITY_RESPONSE:
            pin_act = audit.pin_act_at(session, decision.created_at)
            for cell in payload.get("cells", ()):
                act = float(cell.get("propensities", {}).get("act", 0.0))
                collapse = str(payload.get("mode")) == "pinned" and act == 1.0 and pin_act
                if act > 0 and not collapse:
                    audit.flag("A5", pk, where, f"cell {cell.get('cell_id')} gives act {act}")


def _positive_mass(payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    vectors = [payload.get("propensities") or {}]
    vectors += [cell.get("propensities") or {} for cell in payload.get("cells", ())]
    return [(str(v), float(p)) for vector in vectors for v, p in vector.items() if float(p) > 0]


def _trial_invariants(
    audit: _Audit,
    stored: StoredTrial,
    session: SessionRow,
    k_before: int,
    decision_times: Mapping[str, datetime],
    a11_trials: dict[str, set[str]],
) -> None:
    trial = stored.trial
    pk = session.participant_key
    where = f"trial:{trial.trial_id}"
    tau = float(session.settings["ambiguity_threshold"])

    # A1: applied variants under the rules in force when their source was issued (section 9.7).
    for violation in preference_violations(
        trial,
        audit.history,
        decision_times=decision_times,
        participant_key=pk,
    ):
        audit.flag("A1", pk, where, violation)

    def a11(detail: str, index: int | None = None) -> None:
        a11_trials[pk].add(trial.trial_id)
        audit.flag("A11", pk, where if index is None else f"{where}:activation[{index}]", detail)

    if stored.verification == "draw_mismatch":
        a11_trials[pk].add(trial.trial_id)

    # A11: the applied L1 regime and a revert regime against what the decision or the fallbacks give.
    if trial.task != "idle":
        if trial.l1_fallback_reason is not None:
            expected = audit.fallbacks_at(session, trial.fallbacks_issued_at).l1
            if trial.applied_regime != expected:
                a11(f"L1 fallback {trial.applied_regime} is not fallbacks.l1 ({expected})")
        else:
            decision = audit.decisions.get(trial.regime_decision_id)
            decided = None if decision is None else str(decision.payload.get("variant"))
            if decided is not None and decided != trial.applied_regime:
                a11("applied regime differs from the L1 decision")
    if trial.revert is not None:
        issued = audit.fallbacks_at(session, trial.revert.fallbacks_issued_at)
        expected = issued.plain if trial.revert.reason == "veto" else issued.refresh
        if trial.revert.regime != expected:
            a11(
                f"{trial.revert.reason} revert to {trial.revert.regime}; the fallbacks give {expected}",
            )

    counts = _counts_before(trial, k_before)
    table_draws = 0
    for activation, k in zip(trial.activations, counts):
        index = activation.index
        regime = _regime_for(trial, activation)
        applied = activation.l2
        # A2: an ambiguous activation committed on its own under an act-guarded regime.
        if regime in ACT_GUARDED and activation.result == "committed" and activation.a_gate > tau:
            pin_act_commit = (
                applied is not None
                and applied.variant == "act"
                and audit.pin_act_at(session, _binding_time(trial, activation, audit.decisions))
            )
            if not pin_act_commit:
                audit.flag(
                    "A2",
                    pk,
                    f"{where}:activation[{index}]",
                    f"committed at A_gate {activation.a_gate}",
                )
        # A12: an irreversible literal hit committed on a single activation.
        hit = activation.hit_index
        if hit is not None and 0 <= hit < len(trial.reversible) and not trial.reversible[hit]:
            if activation.result == "committed":
                audit.flag(
                    "A12",
                    pk,
                    f"{where}:activation[{index}]",
                    "irreversible candidate committed at once",
                )
        if applied is None:
            continue
        binding = _binding_time(trial, activation, audit.decisions)
        cap_time = _cap_time(trial, activation)
        cap = audit.fallbacks_at(session, cap_time).interruption_cap
        pin_act = audit.pin_act_at(session, binding if applied.source == "table" else cap_time)
        at_cap = cap is not None and k >= cap and not pin_act
        # A1: an interrupting L2 response at or above the cap (the block's count k of 9.3).
        if applied.variant in _INTERRUPTING_L2 and at_cap:
            audit.flag(
                "A1",
                pk,
                f"{where}:activation[{index}]",
                f"{applied.variant} at k={k} with cap {cap}",
            )
        # A5: act from a fixed rule, a fallback or the cap without pin_act.
        fixed_act = applied.variant == "act" and applied.source != "table"
        if fixed_act and not audit.pin_act_at(session, binding):
            audit.flag("A5", pk, f"{where}:activation[{index}]", f"act from {applied.source}")
        # A11: re-derive every response (section 9.3).
        if applied.source == "table":
            if applied.draw_index < 4 and applied.draw_index != table_draws:
                a11(f"draw index {applied.draw_index} after {table_draws} draws", index)
            if applied.draw_index == 4 and table_draws < 4:
                a11("after-max response before four draws", index)
            expected = _table_response(audit, trial, applied)
            if expected is None:
                a11("table, row or cell missing", index)
            elif expected != applied.variant:
                a11(f"applied {applied.variant}; the table gives {expected}", index)
            if at_cap:
                a11(f"table response at k={k} with cap {cap}", index)
            if applied.draw_index < 4:
                table_draws += 1
        elif applied.source == "cap":
            if not at_cap:
                a11(f"cap response at k={k} with cap {cap}", index)
            if applied.draw_index != min(table_draws, 4):
                a11("a cap response consumed a draw", index)
        else:
            expected = fixed_l2_rule(audit.fallbacks_at(session, binding), applied.cell_id)
            if applied.variant != expected:
                a11(
                    f"{applied.source} response {applied.variant}; the fallbacks give {expected}",
                    index,
                )
            if at_cap:
                a11(f"{applied.source} response at k={k} with cap {cap}", index)

    # A3: every commit comes from an activation, a confirm input, a hold completion or a choice; none at the limit.
    if trial.committed_index is not None:
        commits = [a.committed_index for a in trial.activations if a.outcome == "committed"]
        caused = trial.committed_index in commits
        at_limit = trial.commit_t_ms is None or trial.commit_t_ms >= trial.limit_ms
        if not caused or trial.commit_via == "none" or at_limit:
            audit.flag("A3", pk, where, "commit without an explicit input, or at the limit")
    for activation in trial.activations:
        if activation.outcome == "committed" and activation.committed_index is None:
            audit.flag("A3", pk, f"{where}:activation[{activation.index}]", "committed nothing")


def _table_response(audit: _Audit, trial: TrialRecord, applied: Any) -> str | None:
    table = audit.decisions.get(applied.decision_id)
    if table is None:
        return None
    payload = table.payload
    if applied.draw_index >= 4:
        return str(payload.get("after_max", "wait_for_evidence"))
    row = trial.trial_index - int(payload.get("start_trial_index", 0))
    draws = payload.get("draws") or []
    if not 0 <= row < len(draws) or applied.draw_index >= len(draws[row]):
        return None
    cell = next((c for c in payload.get("cells", ()) if c.get("cell_id") == applied.cell_id), None)
    if cell is None:
        return None
    return select_from_cell(
        cell["order"],
        cell["propensities"],
        float(draws[row][applied.draw_index]),
    )


def _veto_invariant(audit: _Audit, ordered: Sequence[StoredTrial]) -> None:
    by_session: dict[tuple[str, str], list[DecisionRecord]] = defaultdict(list)
    for decision in audit.decisions.values():
        by_session[(decision.session_id, DecisionLevel(decision.level).value)].append(decision)
    for values in by_session.values():
        values.sort(key=lambda d: d.seq)
    for stored in ordered:
        trial = stored.trial
        veto = trial.veto
        if veto is None:
            continue
        session = audit.sessions.get(trial.session_id)
        pk = session.participant_key if session is not None else ""
        where = f"trial:{trial.trial_id}"
        level = DecisionLevel(veto.level)
        vetoed_decision = audit.decisions.get(veto.decision_id)
        # The variant the chip showed, as the engine's C5 bookkeeping reads it: after a refresh revert that is the
        # revert regime or a later response, never the cue regime the refresh replaced.
        variant = vetoed_variant(trial)
        if variant is None:
            audit.exemptions.append(
                Finding("A9", pk, where, "no vetoed response found (wait_for_evidence has no veto)"),
            )
            continue
        if vetoed_decision is not None and str(vetoed_decision.mode) == "pinned":
            audit.exemptions.append(
                Finding("A9", pk, where, "pinned: the person's own rule outranks a veto"),
            )
            continue
        same_level = by_session.get((trial.session_id, level.value), ())
        later = [d for d in same_level if d.seq > trial.seq]
        if not later:
            audit.exemptions.append(Finding("A9", pk, where, "no later decision of that level"))
            continue
        following = later[0]
        mode = str(following.mode)
        if mode == "pinned":
            audit.exemptions.append(
                Finding("A9", pk, where, "pinned: the next decision follows a standing rule"),
            )
            continue
        if mode not in JUDGED_VETO_MODES:
            audit.exemptions.append(
                Finding("A9", pk, where, f"{mode} decisions follow the consented schedule"),
            )
            continue
        if following.payload.get("fallback") is True:
            # Nothing survived the hard constraints, so the decision is fallbacks(modality).l1, which section 9.2
            # step 4 computes from the person's rules and C7 only: a one-trial veto cannot leave a block without a
            # regime.
            audit.exemptions.append(
                Finding("A9", pk, where, "fallback: nothing survived, fallbacks follow C1 and C7"),
            )
            continue
        masses = [p for v, p in _positive_mass(following.payload) if v == variant]
        if masses:
            audit.flag(
                "A9",
                pk,
                where,
                f"vetoed {variant} has positive propensity at the next {level.value} decision",
            )


def _purposes_needed(
    phase: str,
    *,
    gaze: bool = False,
    hand: bool = False,
) -> set[CoAdaptationPurpose]:
    needed = {CoAdaptationPurpose.SELECTION_STUDY}
    if phase in PHASE_B:
        needed.add(CoAdaptationPurpose.ADAPTIVE_ASSISTANCE)
    if gaze:
        needed.add(CoAdaptationPurpose.GAZE_PRIOR)
    if hand:
        needed.add(CoAdaptationPurpose.HAND_POINTER)
    return needed


def _consent_invariant(
    audit: _Audit,
    decisions: Sequence[DecisionRecord],
    ordered: Sequence[StoredTrial],
    sessions: Sequence[SessionRow],
    burden: Sequence[BurdenReport],
    consents: Sequence[ConsentPeriod],
) -> None:
    periods: dict[str, list[ConsentPeriod]] = defaultdict(list)
    for period in consents:
        periods[period.record.participant_key].append(period)

    def judge(pk: str, where: str, at: datetime, needed: set[CoAdaptationPurpose]) -> None:
        record = consent_in_force(periods.get(pk, ()), at)
        if record is None:
            audit.flag("A6", pk, where, "written outside every consent period")
            return
        missing = sorted(purpose.value for purpose in needed if purpose not in record.purposes)
        if missing:
            audit.flag("A6", pk, where, f"the consent in force lacks {', '.join(missing)}")

    for session in sessions:
        settings = session.settings
        needed = {CoAdaptationPurpose.SELECTION_STUDY}
        has_phase_b = any(plan.phase in PHASE_B for plan in session.plan.blocks)
        if settings.get("adaptive_allowed") or has_phase_b:
            needed.add(CoAdaptationPurpose.ADAPTIVE_ASSISTANCE)
        if settings.get("learn_across_sessions"):
            needed.add(CoAdaptationPurpose.CROSS_SESSION_LEARNING)
        if session.gaze_check_accuracy is not None or settings.get("gaze_allowed"):
            needed.add(CoAdaptationPurpose.GAZE_PRIOR)
        if session.device.modality == "hand_pointer" or settings.get("hand_pointer_allowed"):
            needed.add(CoAdaptationPurpose.HAND_POINTER)
        judge(session.participant_key, f"session:{session.session_id}", session.started_at, needed)
    for decision in decisions:
        needed = {CoAdaptationPurpose.SELECTION_STUDY}
        if str(decision.mode) in PHASE_B_MODES:
            needed.add(CoAdaptationPurpose.ADAPTIVE_ASSISTANCE)
        judge(
            decision.participant_key,
            f"decision:{decision.decision_id}",
            decision.created_at,
            needed,
        )
    for stored in ordered:
        trial = stored.trial
        session = audit.sessions.get(trial.session_id)
        if session is None:
            continue
        gaze = trial.gaze is not None or any(a.gaze is not None for a in trial.activations)
        phase = "" if trial.task == "idle" else _phase(session, trial.block_index)
        needed = _purposes_needed(phase, gaze=gaze, hand=trial.hand is not None)
        judge(session.participant_key, f"trial:{trial.trial_id}", trial.recorded_at, needed)
    for report in burden:
        session = audit.sessions.get(report.session_id)
        if session is None:
            audit.flag(
                "A6",
                "",
                f"burden:{report.report_id}",
                "report of a session the export does not hold",
            )
            continue
        phase = "" if report.block_index is None else _phase(session, report.block_index)
        judge(
            session.participant_key,
            f"burden:{report.report_id}",
            report.recorded_at,
            _purposes_needed(phase),
        )


def _coverage(audit: _Audit, decisions: Sequence[DecisionRecord], key: str) -> float:
    judged = 0
    covered = 0
    for decision in decisions:
        session = audit.sessions.get(decision.session_id)
        level = DecisionLevel(decision.level)
        if key == "A8":
            if decision.behavior != "policy" or level == DecisionLevel.OFFER:
                continue
            judged += 1
            explanation = decision.payload.get("explanation")
            if isinstance(explanation, Mapping) and isinstance(explanation.get("summary_key"), str):
                covered += 1
            else:
                audit.flag(
                    "A8",
                    decision.participant_key,
                    f"decision:{decision.decision_id}",
                    "no explanation",
                )
            continue
        judged += 1
        if _propensities_complete(decision, session):
            covered += 1
        else:
            audit.flag(
                "A7",
                decision.participant_key,
                f"decision:{decision.decision_id}",
                "propensities incomplete",
            )
    return covered / judged if judged else 1.0


def _vector_ok(propensities: Any, options: Sequence[str], eligible: Any, excluded: Any) -> bool:
    if not isinstance(propensities, Mapping) or set(propensities) != set(options):
        return False
    if not isinstance(eligible, list | tuple) or not isinstance(excluded, Mapping):
        return False
    if any(str(reason) not in _REASONS for reason in excluded.values()):
        return False
    values = [float(p) for p in propensities.values()]
    return all(math.isfinite(p) and p >= 0 for p in values) and abs(sum(values) - 1.0) <= 1e-9


def _propensities_complete(decision: DecisionRecord, session: SessionRow | None) -> bool:
    payload = decision.payload
    level = DecisionLevel(decision.level)
    if level == DecisionLevel.AMBIGUITY_RESPONSE:
        cells = payload.get("cells")
        if not isinstance(cells, list | tuple):
            return False
        return all(
            list(cell.get("order", ())) == list(L2_OPTIONS)
            and _vector_ok(
                cell.get("propensities"),
                L2_OPTIONS,
                cell.get("eligible"),
                cell.get("excluded"),
            )
            for cell in cells
        )
    if session is None:
        return False
    block_level = level == DecisionLevel.BLOCK_REGIME
    options = l1_arms(session.device.modality) if block_level else L3_OPTIONS
    propensities = payload.get("propensities")
    if not _vector_ok(propensities, options, payload.get("eligible"), payload.get("excluded")):
        return False
    return float(propensities.get(str(payload.get("variant")), 0.0)) > 0


# ---------------------------------------------------------------------------------------------- soft metrics


def regime_entropy_bits(regimes: Sequence[str]) -> float:
    counts = Counter(regimes)
    total = sum(counts.values())
    if not total:
        return 0.0
    return float(-sum((n / total) * math.log2(n / total) for n in counts.values()))


def _soft_metrics(
    ordered: Sequence[StoredTrial],
    sessions: Sequence[SessionRow],
    burden: Sequence[BurdenReport],
    preferences_history: Sequence[Preference],
) -> dict[str, Any]:
    active = [s.trial for s in ordered if s.trial.task != "idle" and not s.trial.interrupted]
    overrides = sum(1 for trial in active if trial.veto is not None)
    offered = [offer for trial in active for offer in trial.offers if offer.variant != "no_offer"]
    offers = Counter(offer.response for offer in offered)
    block_regimes: dict[tuple[str, int], str] = {}
    for trial in active:
        block_regimes.setdefault((trial.session_id, trial.block_index), trial.applied_regime)
    switches = 0
    transitions = 0
    regimes_in_order = sorted(block_regimes.items())
    for session in sessions:
        sequence = [regime for (sid, _), regime in regimes_in_order if sid == session.session_id]
        for before, after in zip(sequence, sequence[1:]):
            transitions += 1
            switches += before != after
    hours = 0.0
    last_seen: dict[str, datetime] = {}
    for stored in ordered:
        last_seen[stored.trial.session_id] = max(
            last_seen.get(stored.trial.session_id, stored.trial.recorded_at),
            stored.trial.recorded_at,
        )
    for session in sessions:
        end = session.ended_at or last_seen.get(session.session_id)
        if end is not None:
            hours += max(0.0, (end - session.started_at).total_seconds()) / 3600.0
    prompts = sum(1 for report in burden if report.scope == "block")
    forbids = [pref for pref in preferences_history if pref.kind is PreferenceKind.FORBID_VARIANT]
    never = sum(1 for p in forbids if p.scope.context == "study" and p.scope.participant_key)
    pauses = sum(1 for pref in preferences_history if pref.kind is PreferenceKind.FREEZE_ADAPTATION)
    return {
        "override_rate": overrides / len(active) if active else 0.0,
        "never_creations": never,
        "pause_adapting_uses": pauses,
        "offers": {
            "accepted": offers.get("accepted", 0),
            "dismissed": offers.get("dismissed", 0),
            "ignored_missing": offers.get("ignored", 0),
        },
        "burden_prompts_per_hour": prompts / hours if hours > 0 else None,
        "regime_entropy_bits": regime_entropy_bits([trial.applied_regime for trial in active]),
        "switch_rate": switches / transitions if transitions else 0.0,
    }

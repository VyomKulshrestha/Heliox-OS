"""Pre-registered analysis of the assistance co-adaptation selection study (spec section 21.2).

Inputs are encrypted export bundles (section 10.9), decrypted locally with each participant's key from the OS
keyring and read only through the ``from_dict`` methods of the core records, and the human study protocol. The
script refuses simulated bundles, unknown bundle or record schemas, a confirmatory run under a protocol that is not
``frozen_before_collection`` (use ``--exploratory``, which labels the output) and an output directory that already
holds results. It re-derives every logged seed, selecting draw, L2 table draw and applied L2 response, scores
realized utility from observed outcomes only, tests H1-H9 with the pre-declared rules, runs the off-policy
evaluation of section 19 on frozen snapshots and the agency audit of section 20, and reports gates G1-G6. Nothing
here changes the live model, its settings or any permission: ``promoted_to_runtime`` is always false.

Run from ``daemon``:
    python -m scripts.evaluate_coadaptation_study --bundle EXPORT.enc --protocol benchmarks/coadaptation_protocol_v1.json \
        --output results/coadapt-run --exploratory
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pilot.coadaptation.core.audit import agency_audit
from pilot.coadaptation.core.evaluation import population_prior, prequential_scores, rebuild_engine
from pilot.coadaptation.core.ope import (
    BUNDLE_SCHEMA,
    MIN_ESS,
    BundleRows,
    Estimate,
    FrozenTarget,
    LoggedBlock,
    clipping_sensitivity,
    cross_fit_q_hat,
    doubly_robust,
    grouped_bootstrap,
    logged_blocks,
    read_bundle,
    replay_rows,
    snapshot_before_phase_b,
    support_check,
)
from pilot.coadaptation.core.outcome_model import ModelVariant
from pilot.coadaptation.core.policy import decision_seed, draw_stream, select_from_cell
from pilot.coadaptation.core.records import RECORD_SCHEMA_VERSION, StoredTrial
from pilot.coadaptation.core.target_posterior import GazeInput, compute_posterior, rendered_rects
from pilot.coadaptation.core.utility import pointing_throughput
from pilot.coadaptation.core.vocabulary import L3_OPTIONS, VARIANTS, l1_arms

KEYRING_SERVICE = "heliox-co-adaptation"
HUMAN_EVIDENCE = frozenset({"live_human", "recorded_human"})
CONFIRMATORY_STATUS = "frozen_before_collection"
PHASE_B_TYPES = ("B", "probe", "comparator")
DESIGN_PHASES = ("A", "A_lite")
OUTCOMES = ("complete", "ua", "correction", "veto", "log_time")
EXPLORATORY_LABEL = "exploratory: protocol not frozen before collection; no result is confirmatory"
COVERAGE_REQUIRED = 43 / 50
MISSING_LIMIT = 0.10  # sensitivity subset: participants with at most 10% missing trials


class AnalysisRefused(RuntimeError):
    """The inputs cannot be analysed under the pre-registered rules (the message says why)."""


# --------------------------------------------------------------------------------------------- reading bundles


def _keyring_lookup(participant_key: str) -> str | None:
    import keyring  # imported here: the keyring backend is needed only when no key is passed in

    return keyring.get_password(KEYRING_SERVICE, participant_key)


def decrypt_bundle(
    path: Path, key: bytes | None = None, *, key_lookup: Callable[[str], str | None] | None = None
) -> dict[str, Any]:
    """Decrypt one export file (section 10.9): a plaintext JSON header line and a base64 AES-256-GCM ciphertext line
    whose associated data is the header line's bytes. Without ``key`` the participant's key comes from the OS
    keyring (service ``heliox-co-adaptation``, account = participant key). Unknown schema values are refused."""
    content = Path(path).read_bytes()
    header_line, _, rest = content.partition(b"\n")
    try:
        header = json.loads(header_line)
    except ValueError:
        raise AnalysisRefused("the export header is not JSON") from None
    if header.get("schema_version") != 1 or header.get("cipher") != "AES-256-GCM":
        raise AnalysisRefused("unknown export header schema or cipher")
    if key is None:
        encoded = (key_lookup or _keyring_lookup)(str(header.get("participant_key")))
        if not encoded:
            raise AnalysisRefused("no key for this participant in the OS keyring")
        key = base64.b64decode(encoded)
    nonce = base64.b64decode(header["nonce"])
    ciphertext = base64.b64decode(rest.strip())
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, header_line)
    bundle = json.loads(plaintext)
    check_schemas(bundle)
    return bundle


def check_schemas(bundle: Mapping[str, Any]) -> None:
    if bundle.get("bundle_schema") != BUNDLE_SCHEMA:
        raise AnalysisRefused("unknown bundle_schema")
    if bundle.get("record_schema") != RECORD_SCHEMA_VERSION:
        raise AnalysisRefused("unknown record_schema")


def read_human_bundle(bundle: Mapping[str, Any]) -> BundleRows:
    """Every entry read with its ``from_dict``; simulated sessions are refused (evidence is never pooled)."""
    check_schemas(bundle)
    rows = read_bundle(bundle)
    kinds = {str(row.evidence_kind) for row in rows.sessions}
    if not kinds or not kinds <= HUMAN_EVIDENCE:
        raise AnalysisRefused(
            "simulated bundles are refused: evidence_kind must be live_human or recorded_human",
        )
    if len(rows.participant_keys()) != 1:
        raise AnalysisRefused("an export bundle holds exactly one participant")
    return rows


def leakage_check(bundles: Sequence[BundleRows], pilot_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Every participant appears in exactly one split: once among the bundles, and either pilot or confirmatory."""
    keys = [bundle.participant_keys()[0] for bundle in bundles]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise AnalysisRefused(
            f"participants appear in more than one bundle: {', '.join(duplicates)}",
        )
    pilot = sorted(set(pilot_keys) & set(keys))
    return {
        "participants": len(keys),
        "pilot_excluded": pilot,
        "confirmatory": sorted(set(keys) - set(pilot)),
        "passed": True,
    }


# --------------------------------------------------------------------------------------------- verification


def verify_logs(rows: BundleRows) -> dict[str, Any]:
    """Re-derive every logged seed (participant key, the session's seed salt, decision id, policy version), every
    selecting draw of an L1 or L3 decision, every L2 table draw (``draw_stream``, section 9.2 step 6) and every
    applied L2 table response (``select_from_cell``), comparing exactly."""
    sessions = {row.session_id: row for row in rows.sessions}
    decisions = {d.decision_id: d for d in rows.decisions}
    problems: dict[str, list[str]] = {
        "seed": [],
        "selecting_u": [],
        "table_draws": [],
        "applied_l2": [],
    }
    for decision in rows.decisions:
        session = sessions.get(decision.session_id)
        if session is None:
            problems["seed"].append(decision.decision_id)
            continue
        seed = decision_seed(decision.participant_key, session.seed_salt, decision.decision_id)
        payload = decision.payload
        if payload.get("seed") != f"{seed:016x}":
            problems["seed"].append(decision.decision_id)
            continue
        rng = draw_stream(seed)
        if payload.get("level") == "ambiguity_table":
            expected = [[float(rng.random()) for _ in row] for row in payload.get("draws", ())]
            if expected != [[float(u) for u in row] for row in payload.get("draws", ())]:
                problems["table_draws"].append(decision.decision_id)
        else:
            # The selecting u runs over the level's options in caution / table order (9.2 step 6); the export's
            # canonical JSON sorts object keys, so the order comes from the vocabulary, never from the payload.
            block_level = payload.get("level") == "block_regime"
            order = l1_arms(session.device.modality) if block_level else L3_OPTIONS
            propensities = payload["propensities"]
            if select_from_cell(order, propensities, float(rng.random())) != payload.get("variant"):
                problems["selecting_u"].append(decision.decision_id)
    for stored in rows.trials:
        trial = stored.trial
        for activation in trial.activations:
            applied = activation.l2
            if applied is None or applied.source != "table":
                continue
            table = decisions.get(applied.decision_id)
            where = f"{trial.trial_id}:{activation.index}"
            if table is None:
                problems["applied_l2"].append(where)
                continue
            payload = table.payload
            if applied.draw_index >= 4:
                expected = payload.get("after_max", "wait_for_evidence")
            else:
                row = trial.trial_index - int(payload["start_trial_index"])
                cell = next((c for c in payload["cells"] if c["cell_id"] == applied.cell_id), None)
                if cell is None or not 0 <= row < len(payload["draws"]):
                    problems["applied_l2"].append(where)
                    continue
                expected = select_from_cell(
                    cell["order"],
                    cell["propensities"],
                    float(payload["draws"][row][applied.draw_index]),
                )
            if expected != applied.variant:
                problems["applied_l2"].append(where)
    return {
        "decisions": len(rows.decisions),
        "mismatches": {key: values for key, values in problems.items()},
        "stored_verification": dict(Counter(stored.verification for stored in rows.trials)),
        "passed": not any(problems.values()),
    }


def recompute_posteriors(rows: BundleRows) -> dict[str, Any]:
    """Recompute every ``ok`` trial's posteriors from the exported record (spread, rendered rects, gaze, the
    keyboard branch) and compare ``a_gate``, the hit and ``n_plausible`` with what was logged."""
    sessions = {row.session_id: row for row in rows.sessions}
    mismatches = []
    checked = 0
    for stored in rows.trials:
        if stored.verification != "ok":
            continue
        trial = stored.trial
        accuracy = sessions[trial.session_id].gaze_check_accuracy
        for activation in trial.activations:
            checked += 1
            regime = (
                trial.revert.regime
                if trial.revert and activation.index >= trial.revert.from_activation
                else trial.applied_regime
            )
            scale = float(VARIANTS[regime].param("scale") or 1.0)
            if activation.modality == "keyboard":
                result = compute_posterior(
                    trial.candidates,
                    (activation.x, activation.y),
                    modality="keyboard",
                    r=None,
                    floor=None,
                    surface=(trial.surface_w, trial.surface_h),
                    press_index=activation.press_index,
                )
            else:
                gaze = None
                if activation.gaze is not None and accuracy is not None:
                    gaze = GazeInput(
                        region=activation.gaze.region,
                        conf=activation.gaze.conf,
                        acc=accuracy,
                    )
                result = compute_posterior(
                    rendered_rects(trial.candidates, activation.expanded_index, scale),
                    (activation.x, activation.y),
                    modality=activation.modality,
                    r=trial.spread.r if trial.spread else None,
                    floor=trial.spread.floor if trial.spread else None,
                    surface=(trial.surface_w, trial.surface_h),
                    gaze=gaze,
                    press_index=activation.press_index,
                )
            if (
                abs(result.ambiguity_gate - activation.a_gate) > 1e-6
                or result.hit_index != activation.hit_index
                or result.n_plausible != activation.n_plausible
            ):
                mismatches.append(f"{trial.trial_id}:{activation.index}")
    return {"activations": checked, "mismatches": mismatches, "passed": not mismatches}


# --------------------------------------------------------------------------------------------- realized utility


@dataclass(frozen=True, slots=True)
class ScoredTrial:
    """One trial's realized utility (section 21.2) and the fields the contrasts need."""

    participant_key: str
    session_number: int
    block_index: int
    phase: str
    trial_index: int
    regime: str
    verification: str
    modality: str
    utility_participant: float  # u_t - w_B B_obs under the block's logged priorities
    utility_protocol: float  # the same under the protocol weights (0.5, 0.5, 0.5)
    # w_B B_obs (participant weights), kept for the sensitivity without it.
    burden_participant: float
    burden_protocol: float
    ua: bool
    override: bool  # "Not this time" on the block's regime
    switched: bool  # the block followed a regime change


def _plan_phases(sessions: Iterable[Any]) -> dict[tuple[str, int], str]:
    """Each planned block's phase by (session id, block index)."""
    blocks = [(row.session_id, plan) for row in sessions for plan in row.plan.blocks]
    return {(session_id, plan.index): plan.phase for session_id, plan in blocks}


def _block_switches(rows: BundleRows) -> dict[tuple[str, int], tuple[bool, int]]:
    """(session, block) -> (followed a regime change, trials since the regime last changed at the block start)."""
    out: dict[tuple[str, int], tuple[bool, int]] = {}
    for session in rows.sessions:
        session_trials = [s.trial for s in rows.trials if s.trial.session_id == session.session_id]
        trials = sorted(
            (trial for trial in session_trials if trial.task != "idle"),
            key=lambda trial: (trial.block_index, trial.trial_index),
        )
        blocks: dict[int, list[Any]] = defaultdict(list)
        for trial in trials:
            blocks[trial.block_index].append(trial)
        previous = None
        age = 0
        for block_index in sorted(blocks):
            regime = blocks[block_index][0].applied_regime
            switched = previous is not None and regime != previous
            age = 0 if switched or previous is None else age
            out[(session.session_id, block_index)] = (switched, age)
            age += len(blocks[block_index])
            previous = regime
    return out


def score_trials(rows: BundleRows) -> tuple[list[ScoredTrial], dict[str, Any]]:
    """Realized utility per trial for every trial that was not interrupted (intention to treat), with the
    imputation and missingness counts per regime of section 21.2."""
    scored: list[ScoredTrial] = []
    switches = _block_switches(rows)
    imputed: Counter[str] = Counter()
    missing: dict[str, Counter[str]] = defaultdict(Counter)
    for weights_kind in ("participant", "protocol"):
        blocks = logged_blocks(rows, weights=weights_kind, phases=(*DESIGN_PHASES, *PHASE_B_TYPES))
        if weights_kind == "participant":
            participant_blocks = {block.unit_id: block for block in blocks}
        else:
            protocol_blocks = {block.unit_id: block for block in blocks}
    sessions = {row.session_id: row for row in rows.sessions}
    trials_by_block: dict[tuple[str, int], list[StoredTrial]] = defaultdict(list)
    for stored in rows.trials:
        trials_by_block[(stored.trial.session_id, stored.trial.block_index)].append(stored)
    for unit_id, block in participant_blocks.items():
        protocol_block = protocol_blocks[unit_id]
        context = block.context
        session = sessions[block.session_id]
        utilities = dict(zip(block.trial_positions, block.trial_utilities))
        protocol_utilities = dict(
            zip(protocol_block.trial_positions, protocol_block.trial_utilities),
        )
        block_reports = [r for r in rows.burden if r.session_id == block.session_id]
        report = next((r for r in block_reports if r.block_index == context.block_index), None)
        # B2 is not asked after assistance_off.
        needs_intrusiveness = context.applied_regime != "assistance_off"
        missing_b2 = needs_intrusiveness and report is not None and report.intrusiveness is None
        if report is None or report.effort is None or missing_b2:
            imputed[context.applied_regime] += 1
        switched, _ = switches.get((block.session_id, context.block_index), (False, 0))
        for stored in trials_by_block[(block.session_id, context.block_index)]:
            trial = stored.trial
            if trial.interrupted or trial.trial_index not in utilities or stored.labels is None:
                continue
            protocol_utility = protocol_utilities[trial.trial_index] - protocol_block.burden_term
            scored.append(
                ScoredTrial(
                    participant_key=session.participant_key,
                    session_number=session.session_number,
                    block_index=context.block_index,
                    phase=context.phase,
                    trial_index=trial.trial_index,
                    regime=trial.applied_regime,
                    verification=stored.verification,
                    modality=session.device.modality,
                    utility_participant=utilities[trial.trial_index] - block.burden_term,
                    utility_protocol=protocol_utility,
                    burden_participant=block.burden_term,
                    burden_protocol=protocol_block.burden_term,
                    ua=bool(stored.labels.ua),
                    override=trial.veto is not None and str(trial.veto.level) == "block_regime",
                    switched=switched,
                )
            )
    for stored in rows.trials:
        trial = stored.trial
        regime = trial.applied_regime
        counts = missing[regime]
        counts["trials"] += 1
        if trial.interrupted:
            counts["interrupted"] += 1
        if trial.interruption_reason == "refused":
            counts["refused"] += 1
        if trial.l1_fallback_reason is not None:
            counts[f"l1_fallback_{trial.l1_fallback_reason}"] += 1
        for activation in trial.activations:
            if activation.l2 is not None and activation.l2.source == "fallback":
                counts[f"l2_fallback_{activation.l2.fallback_reason}"] += 1
            if activation.l2 is not None and activation.l2.source == "cap":
                counts["cap_responses"] += 1
        for offer in trial.offers:
            if offer.fallback_reason is not None:
                counts[f"l3_fallback_{offer.fallback_reason}"] += 1
    for report in rows.burden:
        for item in report.skipped:
            missing["burden"][f"skipped_{item}"] += 1
    return scored, {
        "imputed_blocks_by_regime": dict(imputed),
        "missingness_by_regime": {regime: dict(counts) for regime, counts in missing.items()},
    }


def missing_share(rows: BundleRows) -> float:
    trials = [s for s in rows.trials if s.trial.task != "idle"]
    if not trials:
        return 1.0
    return sum(1 for s in trials if s.trial.interrupted) / len(trials)


# --------------------------------------------------------------------------------------------- contrasts


def _interval(values: Mapping[str, float], *, n: int, seed: int) -> dict[str, Any]:
    finite = {key: float(v) for key, v in values.items() if v is not None and math.isfinite(v)}
    if not finite:
        return {"mean": None, "ci95": None, "participants": 0}
    mean = float(np.mean(list(finite.values())))
    ci = None
    if len(finite) >= 2:
        lo, hi = grouped_bootstrap(
            {key: [value] for key, value in finite.items()},
            lambda s: float(np.mean(s[:, 0])),
            n=n,
            seed=seed,
        )
        ci = [lo, hi]
    return {"mean": mean, "ci95": ci, "participants": len(finite)}


def phase_b_contrasts(
    scored: Sequence[ScoredTrial],
    rows_by_participant: Mapping[str, BundleRows],
    *,
    burn_in: int,
    weights: str = "participant",
    ok_only: bool = False,
    switched_only: bool = False,
    drop_burden: bool = False,
    participants: Iterable[str] | None = None,
    bootstrap_n: int = 5000,
    seed: int = 20261015,
) -> dict[str, Any]:
    """H2, H3 and H6 on concurrent Phase B blocks: adaptive (phase ``B``) blocks against the probe and comparator
    blocks of the same sessions, per participant (sessions 2 and 3 pooled), then the grouped bootstrap of the
    per-participant differences. Trial-level outcomes leave out the first ``burn_in`` trials of every Phase B block;
    block ratings (B3 control) use whole blocks."""
    wanted = None if participants is None else set(participants)
    table: dict[str, dict[str, list[ScoredTrial]]] = defaultdict(lambda: defaultdict(list))
    for trial in scored:
        if trial.phase not in PHASE_B_TYPES or trial.trial_index < burn_in:
            continue
        if ok_only and trial.verification != "ok":
            continue
        if switched_only and not trial.switched:
            continue
        if wanted is not None and trial.participant_key not in wanted:
            continue
        table[trial.participant_key][trial.phase].append(trial)

    def utility(trial: ScoredTrial) -> float:
        value = trial.utility_participant if weights == "participant" else trial.utility_protocol
        if drop_burden:
            value += trial.burden_participant if weights == "participant" else trial.burden_protocol
        return value

    diffs: dict[str, dict[str, float]] = defaultdict(dict)
    for key, by_type in table.items():
        filled = {phase: trials for phase, trials in by_type.items() if trials}
        means = {p: float(np.mean([utility(t) for t in ts])) for p, ts in filled.items()}
        override = {p: float(np.mean([t.override for t in ts])) for p, ts in filled.items()}
        ua = {p: float(np.mean([t.ua for t in ts])) for p, ts in filled.items()}
        keyboard = all(t.modality == "keyboard" for trials in by_type.values() for t in trials)
        if "B" in means and "probe" in means and not keyboard:
            diffs["h2_vs_probe"][key] = means["B"] - means["probe"]
        if "B" in means and "comparator" in means:
            diffs["h2_vs_comparator"][key] = means["B"] - means["comparator"]
        if "B" in override and "comparator" in override:
            diffs["h3_override"][key] = override["B"] - override["comparator"]
        if "B" in ua and "probe" in ua:
            diffs["h6_ua"][key] = ua["B"] - ua["probe"]
    for key, rows in rows_by_participant.items():
        if wanted is not None and key not in wanted:
            continue
        phases = _plan_phases(rows.sessions)
        control: dict[str, list[float]] = defaultdict(list)
        for report in rows.burden:
            phase = phases.get((report.session_id, report.block_index))
            if phase in PHASE_B_TYPES and report.control is not None:
                control[phase].append(float(report.control))
        if control.get("B") and control.get("comparator"):
            diffs["h3_control_vs_comparator"][key] = float(
                np.mean(control["B"]) - np.mean(control["comparator"]),
            )
        if control.get("B") and control.get("probe"):
            diffs["h3_control_vs_probe"][key] = float(
                np.mean(control["B"]) - np.mean(control["probe"]),
            )
    return {
        name: _interval(values, n=bootstrap_n, seed=seed + index)
        for index, (name, values) in enumerate(sorted(diffs.items()))
    }


def block_type_switch_shares(rows: Sequence[BundleRows], *, burn_in: int) -> dict[str, Any]:
    """Per block type and session: the share of blocks that followed a regime change and the mean switch age at
    the first scored trial (switch transients fall mostly on probe and comparator blocks, section 12.3)."""
    shares: dict[tuple[str, int], list[tuple[bool, int]]] = defaultdict(list)
    for bundle in rows:
        switches = _block_switches(bundle)
        numbers = {row.session_id: row.session_number for row in bundle.sessions}
        phases = _plan_phases(bundle.sessions)
        for (session_id, block_index), (switched, age) in switches.items():
            phase = phases.get((session_id, block_index))
            if phase in PHASE_B_TYPES:
                shares[(phase, numbers[session_id])].append((switched, age + burn_in))
    return {
        f"{phase}:session_{number}": {
            "blocks": len(values),
            "followed_regime_change": float(np.mean([s for s, _ in values])),
            "mean_switch_age_at_first_scored_trial": float(np.mean([a for _, a in values])),
        }
        for (phase, number), values in sorted(shares.items())
    }


def _rule(result: Mapping[str, Any], kind: str) -> bool | None:
    ci = result.get("ci95")
    if ci is None:
        return None
    lo, hi = ci
    if kind == "superiority":
        return lo > 0
    if kind == "ni_utility":
        return lo > -0.05
    if kind == "ni_override":
        return hi <= 0.05
    if kind == "ni_control":
        return lo > -0.5
    if kind == "safety_ua":
        return hi <= 0.02
    raise ValueError("unknown rule")


def _all_pass(checks: Iterable[bool | None]) -> bool:
    values = list(checks)
    return bool(values) and all(value is True for value in values)


# --------------------------------------------------------------------------------------------- prediction (H1, H4, H5, H8, H9)


def _variant_scores(rows: BundleRows, key: str, variant: ModelVariant, **kwargs: Any) -> Any:
    return prequential_scores(
        replay_rows(rows, key),
        variant=variant,
        preferences=rows.preferences,
        **kwargs,
    )


def prediction_hypotheses(
    rows_by_participant: Mapping[str, BundleRows], *, bootstrap_n: int, seed: int
) -> dict[str, Any]:
    """H1, H4, H5, H8 and H9 with the scoring rules of H1 (censored log score summed over the five outcomes and per
    outcome; primary set Phase A and A-lite trials, all trials secondary; forward-chaining by session)."""
    contrasts: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    secondary: dict[str, dict[str, float]] = defaultdict(dict)
    h9_excluded = []
    engines = {
        key: rebuild_engine(replay_rows(rows, key), preferences=rows.preferences)
        for key, rows in rows_by_participant.items()
    }
    for key, rows in rows_by_participant.items():
        scores = {
            variant: _variant_scores(rows, key, variant)
            for variant in (
                ModelVariant.FULL,
                ModelVariant.PHYSICS_ONLY,
                ModelVariant.PERSON_STATIC,
                ModelVariant.NO_TODAY,
                ModelVariant.NO_PRACTICE_SWITCH,
                ModelVariant.SESSION_ONLY,
            )
        }
        full_primary = scores[ModelVariant.FULL].select(primary=True)
        if full_primary:
            full_mean = scores[ModelVariant.FULL].mean(full_primary)
            for name, variant in (
                ("H1_vs_physics_only", ModelVariant.PHYSICS_ONLY),
                ("H1_vs_person_static", ModelVariant.PERSON_STATIC),
                ("H4_vs_no_today", ModelVariant.NO_TODAY),
                ("H5_vs_no_practice_switch", ModelVariant.NO_PRACTICE_SWITCH),
            ):
                other = scores[variant].mean(scores[variant].select(primary=True))
                for outcome in (*OUTCOMES, "total"):
                    contrasts[name][outcome][key] = full_mean[outcome] - other[outcome]
            all_full = scores[ModelVariant.FULL].mean()
            all_static = scores[ModelVariant.PHYSICS_ONLY].mean()
            all_gap = all_full["total"] - all_static["total"]
            secondary["H1_vs_physics_only_all_trials"][key] = all_gap
        later = [row for row in rows.sessions if row.session_number in (2, 3)]
        if later and all(bool(row.settings.get("learn_across_sessions")) for row in later):
            early_full = scores[ModelVariant.FULL].select(sessions=(2, 3), first=24)
            early_session = scores[ModelVariant.SESSION_ONLY].select(sessions=(2, 3), first=24)
            if early_full and early_session:
                a = scores[ModelVariant.FULL].mean(early_full)
                b = scores[ModelVariant.SESSION_ONLY].mean(early_session)
                for outcome in (*OUTCOMES, "total"):
                    contrasts["H9_vs_session_only"][outcome][key] = a[outcome] - b[outcome]
        else:
            h9_excluded.append(key)
        if len(engines) >= 2:
            prior = population_prior(engines, leave_out=key)
            with_prior = _variant_scores(rows, key, ModelVariant.FULL, population_prior=prior)
            population = _variant_scores(rows, key, ModelVariant.POPULATION, population_prior=prior)
            a_set, b_set = with_prior.select(primary=True), population.select(primary=True)
            if a_set and b_set:
                a, b = with_prior.mean(a_set), population.mean(b_set)
                for outcome in (*OUTCOMES, "total"):
                    contrasts["H8_vs_population"][outcome][key] = a[outcome] - b[outcome]
    out: dict[str, Any] = {}
    for c_index, (name, by_outcome) in enumerate(sorted(contrasts.items())):
        out[name] = {
            outcome: _interval(values, n=bootstrap_n, seed=seed + 10 * c_index + o_index)
            for o_index, (outcome, values) in enumerate(sorted(by_outcome.items()))
        }
    out["secondary_all_trials"] = {}
    for name, values in secondary.items():
        out["secondary_all_trials"][name] = _interval(values, n=bootstrap_n, seed=seed + 999)
    out["H9_participants_left_out"] = len(h9_excluded)
    return out


# --------------------------------------------------------------------------------------------- OPE


def _estimate_dict(estimate: Estimate) -> dict[str, Any]:
    data = estimate.to_dict()
    if estimate.refused is not None:
        data["value"] = None
        data["note"] = f"not estimated (ESS {estimate.ess:.1f} < {MIN_ESS:g})"
    return data


def ope_estimates(rows_by_participant: Mapping[str, BundleRows], *, seed: int) -> dict[str, Any]:
    """Doubly robust estimates of each participant's frozen snapshots ``S_2`` and ``S_3`` on their Phase A and
    A-lite blocks (in sample) and of ``S_2`` on the session-3 A-lite blocks (out of sample), with leave-one-session-
    out DR models, clipping sensitivity and support. A refused estimate is reported as "not estimated" with its ESS
    and is never re-run with another threshold, clip, block set or estimator."""
    sets: dict[str, list[tuple[FrozenTarget, list[LoggedBlock]]]] = defaultdict(list)
    for key, rows in rows_by_participant.items():
        s2 = snapshot_before_phase_b(rows, key, 2, preferences=rows.preferences)
        s3 = snapshot_before_phase_b(rows, key, 3, preferences=rows.preferences)
        if s2 is not None:
            target = FrozenTarget(
                s2.data,
                participant_key=key,
                settings=s2.settings,
                preferences_history=rows.preferences,
                rows_hash=s2.rows_hash,
            )
            sets["in_sample_S_2"].append(
                (target, logged_blocks(rows, weights="participant", sessions=(1, 2))),
            )
            sets["out_of_sample_S_2"].append(
                (
                    target,
                    logged_blocks(rows, weights="participant", sessions=(3,), phases=("A_lite",)),
                )
            )
        if s3 is not None:
            target = FrozenTarget(
                s3.data,
                participant_key=key,
                settings=s3.settings,
                preferences_history=rows.preferences,
                rows_hash=s3.rows_hash,
            )
            sets["in_sample_S_3"].append(
                (target, logged_blocks(rows, weights="participant", sessions=(1, 2, 3))),
            )
    out: dict[str, Any] = {}
    for index, name in enumerate(("in_sample_S_2", "in_sample_S_3", "out_of_sample_S_2")):
        pairs = sets.get(name, [])
        blocks = [block for _, blocks_ in pairs for block in blocks_]
        targets = {block.unit_id: target for target, blocks_ in pairs for block in blocks_}

        def target_policy(
            block: LoggedBlock,
            step: Any,
            _targets: Mapping[str, FrozenTarget] = targets,
        ) -> Any:
            return _targets[block.unit_id](block, step)

        if not blocks:
            out[name] = {"estimate": None, "note": "no blocks", "ess": 0.0}
            continue
        q_hat = cross_fit_q_hat(blocks)
        estimate = doubly_robust(blocks, target_policy, q_hat, bootstrap_seed=seed + index)
        by_session = {}
        for number in sorted({b.context.session_number for b in blocks}):
            stratum = [b for b in blocks if b.context.session_number == number]
            by_session[str(number)] = _estimate_dict(
                doubly_robust(stratum, target_policy, q_hat, bootstrap_seed=seed),
            )
        out[name] = {
            "estimate": _estimate_dict(estimate),
            "ess": estimate.ess,
            "by_session": by_session,
            "support": support_check(blocks, target_policy).to_dict(),
            "clipping_sensitivity": clipping_sensitivity(blocks, target_policy, q_hat),
        }
    return out


def gate_g5(validation: Mapping[str, Any], ope: Mapping[str, Any]) -> dict[str, Any]:
    """G5 (OPE validity, section 19): (a) the frozen simulation run's estimator check covered the truth in at least
    43 of 50 studies at the human design; (b) its in-sample coverage, refusals counted as not covering, is at least
    43 of 50 for ``S_2`` and for ``S_3``; (c) the human DR estimates of ``S_2`` and ``S_3`` are both computed (ESS of
    at least 30). Anything else is "not passed", naming each failed condition with the ESS values."""

    def coverage(value: Any) -> float | None:
        return float(value) if isinstance(value, int | float) else None

    in_sample = validation.get("in_sample") or {}
    conditions = {
        "a_estimator_check": coverage(validation.get("estimator_check_coverage")),
        "b_in_sample_S_2_coverage": coverage((in_sample.get("S_2") or {}).get("coverage")),
        "b_in_sample_S_3_coverage": coverage((in_sample.get("S_3") or {}).get("coverage")),
    }
    floor = COVERAGE_REQUIRED - 1e-12
    checks = {name: value is not None and value >= floor for name, value in conditions.items()}
    ess = {}
    for name in ("in_sample_S_2", "in_sample_S_3"):
        entry = ope.get(name) or {}
        estimate = entry.get("estimate") or {}
        ess[name] = float(entry.get("ess") or 0.0)
        computed = bool(estimate) and estimate.get("refused") is None
        checks[f"c_human_{name}_computed"] = computed and ess[name] >= MIN_ESS
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "passed": not failed,
        "result": "passed" if not failed else "not passed",
        "failed_conditions": failed,
        "conditions": checks,
        "simulated_coverage": conditions,
        "human_ess": ess,
    }


def h2_ope_report(g5: Mapping[str, Any], ope: Mapping[str, Any]) -> dict[str, Any]:
    """H2's secondary OPE: the in-sample estimates of ``S_2`` and ``S_3`` only when G5 passed; the out-of-sample
    ``S_2`` estimate whenever the estimator check passed and its own ESS is at least 30, labelled with G5's
    outcome. A refused estimate is reported as "not estimated" with its ESS."""
    report: dict[str, Any] = {}
    for name in ("in_sample_S_2", "in_sample_S_3"):
        entry = ope.get(name) or {}
        if g5["passed"]:
            report[name] = entry.get("estimate")
        else:
            report[name] = {"reported": False, "reason": "G5 not passed", "ess": entry.get("ess")}
    oos = ope.get("out_of_sample_S_2") or {}
    estimate = oos.get("estimate") or {}
    if not g5["conditions"].get("a_estimator_check"):
        report["out_of_sample_S_2"] = {
            "reported": False,
            "reason": "estimator check not passed",
            "ess": oos.get("ess"),
        }
    elif estimate.get("refused") is not None or float(oos.get("ess") or 0.0) < MIN_ESS:
        report["out_of_sample_S_2"] = {
            "reported": False,
            "reason": f"not estimated (ESS {float(oos.get('ess') or 0.0):.1f})",
            "ess": oos.get("ess"),
        }
    else:
        report["out_of_sample_S_2"] = {**estimate, "g5": g5["result"]}
    return report


# --------------------------------------------------------------------------------------------- descriptive parts


def probe_throughput(
    rows_by_participant: Mapping[str, BundleRows],
    *,
    bootstrap_n: int,
    seed: int,
) -> dict[str, Any]:
    """H5 (descriptive): probe-block pointing throughput by session, with grouped-bootstrap intervals."""
    per_session: dict[int, dict[str, float]] = defaultdict(dict)
    for key, rows in rows_by_participant.items():
        probes = {
            (row.session_id, plan.index): row.session_number
            for row in rows.sessions
            for plan in row.plan.blocks
            if plan.phase == "probe"
        }
        by_session: dict[int, list[StoredTrial]] = defaultdict(list)
        for stored in rows.trials:
            number = probes.get((stored.trial.session_id, stored.trial.block_index))
            if number is not None:
                by_session[number].append(stored)
        for number, trials in by_session.items():
            value = pointing_throughput(trials)
            if value is not None:
                per_session[number][key] = value
    return {
        str(number): _interval(values, n=bootstrap_n, seed=seed + number)
        for number, values in sorted(per_session.items())
    }


def idle_ua_by_regime(rows: Iterable[BundleRows]) -> dict[str, Any]:
    """Unintended activations per idle minute per regime (H6, descriptive)."""
    commits: Counter[str] = Counter()
    minutes: Counter[str] = Counter()
    for bundle in rows:
        for stored in bundle.trials:
            if stored.trial.task != "idle" or stored.labels is None or stored.trial.interrupted:
                continue
            commits[stored.trial.applied_regime] += stored.labels.idle_commits
            minutes[stored.trial.applied_regime] += stored.labels.idle_minutes
    return {
        regime: {
            "commits": commits[regime],
            "minutes": minutes[regime],
            "per_minute": commits[regime] / minutes[regime],
        }
        for regime in sorted(minutes)
        if minutes[regime] > 0
    }


def gaze_brier(rows: Iterable[BundleRows]) -> dict[str, Any]:
    """H7 (exploratory): Brier score of the target posterior at the first activation with and without the gaze
    prior, over activations that used gaze."""
    with_gaze, without = [], []
    for bundle in rows:
        sessions = {row.session_id: row for row in bundle.sessions}
        for stored in bundle.trials:
            trial = stored.trial
            if trial.intended_index is None or not trial.activations or trial.spread is None:
                continue
            first = trial.activations[0]
            accuracy = sessions[trial.session_id].gaze_check_accuracy
            if first.gaze is None or accuracy is None or first.modality == "keyboard":
                continue
            regime = trial.applied_regime
            rects = rendered_rects(
                trial.candidates,
                first.expanded_index,
                float(VARIANTS[regime].param("scale") or 1.0),
            )
            result = compute_posterior(
                rects,
                (first.x, first.y),
                modality=first.modality,
                r=trial.spread.r,
                floor=trial.spread.floor,
                surface=(trial.surface_w, trial.surface_h),
                gaze=GazeInput(region=first.gaze.region, conf=first.gaze.conf, acc=accuracy),
                press_index=first.press_index,
            )
            onehot = np.zeros(len(rects))
            onehot[trial.intended_index] = 1.0
            with_gaze.append(float(np.sum((np.asarray(result.posterior) - onehot) ** 2)))
            without.append(float(np.sum((np.asarray(result.posterior_no_gaze) - onehot) ** 2)))
    if not with_gaze:
        return {"activations": 0, "exploratory": True}
    return {
        "activations": len(with_gaze),
        "brier_with_gaze": float(np.mean(with_gaze)),
        "brier_without_gaze": float(np.mean(without)),
        "exploratory": True,
    }


def audit_bundles(bundles: Sequence[BundleRows]) -> dict[str, Any]:
    """The agency audit (section 20) per participant over the bundle's decisions, trials, sessions, burden reports,
    rule history, consent periods, drift and calibration-suspension events and learning-reset time."""
    per_participant = {}
    totals: Counter[str] = Counter()
    coverage: dict[str, float] = {}
    passed = True
    for bundle in bundles:
        report = agency_audit(
            bundle.decisions,
            bundle.trials,
            bundle.sessions,
            bundle.burden,
            bundle.preferences,
            bundle.consents,
            events=bundle.events,
            learning_reset_at=bundle.learning_reset_at,
        )
        key = bundle.participant_keys()[0]
        data = report.to_dict()
        per_participant[key] = {
            "passed": report.passed,
            "counts": dict(report.counts),
            "findings": data["findings"][:50],
            "exemptions": data["exemptions"],
        }
        passed = passed and report.passed
        for name, value in report.counts.items():
            if name in ("A7", "A8"):
                coverage[name] = min(coverage.get(name, 1.0), float(value))
            else:
                totals[name] += int(value)
    return {
        "passed": passed,
        "counts": {**dict(totals), **coverage},
        "per_participant": per_participant,
    }


# --------------------------------------------------------------------------------------------- the analysis


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def analyse(
    bundles: Sequence[BundleRows],
    protocol: Mapping[str, Any],
    *,
    exploratory: bool,
    pilot_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """The whole pre-registered analysis over decrypted bundles (one participant each)."""
    if protocol.get("status") != CONFIRMATORY_STATUS and not exploratory:
        raise AnalysisRefused(
            f"the protocol status is {protocol.get('status')!r}; a confirmatory run needs "
            f"{CONFIRMATORY_STATUS!r} (use --exploratory, which labels the output)"
        )
    leakage = leakage_check(bundles, pilot_keys)
    confirmatory = [b for b in bundles if b.participant_keys()[0] in set(leakage["confirmatory"])]
    rows_by_participant = {b.participant_keys()[0]: b for b in confirmatory}
    boot = protocol.get("analysis", {}).get("bootstrap", {})
    bootstrap_n = int(boot.get("replicates", 5000))
    seed = int(boot.get("seed", 20261015))
    hypotheses = protocol.get("hypotheses", ())
    burn_in = next((int(h.get("burn_in_trials", 3)) for h in hypotheses if h.get("id") == "H2"), 3)

    verification = {key: verify_logs(rows) for key, rows in rows_by_participant.items()}
    posteriors = {key: recompute_posteriors(rows) for key, rows in rows_by_participant.items()}
    scored: list[ScoredTrial] = []
    counts: dict[str, Any] = {}
    for key, rows in rows_by_participant.items():
        trials, table = score_trials(rows)
        scored.extend(trials)
        counts[key] = table
    low_missing = [k for k, r in rows_by_participant.items() if missing_share(r) <= MISSING_LIMIT]
    common = {"bootstrap_n": bootstrap_n, "seed": seed}
    primary = phase_b_contrasts(scored, rows_by_participant, burn_in=burn_in, **common)
    secondary = {
        "all_trials": phase_b_contrasts(scored, rows_by_participant, burn_in=0, **common),
        "protocol_weights": phase_b_contrasts(
            scored, rows_by_participant, burn_in=burn_in, weights="protocol", **common
        ),
        "switched_blocks_only": phase_b_contrasts(
            scored, rows_by_participant, burn_in=burn_in, switched_only=True, **common
        ),
        "ok_trials_only": phase_b_contrasts(
            scored,
            rows_by_participant,
            burn_in=burn_in,
            ok_only=True,
            **common,
        ),
        "without_burden_term": phase_b_contrasts(
            scored, rows_by_participant, burn_in=burn_in, drop_burden=True, **common
        ),
        "at_most_10_percent_missing": phase_b_contrasts(
            scored, rows_by_participant, burn_in=burn_in, participants=low_missing, **common
        ),
    }
    prediction = prediction_hypotheses(
        rows_by_participant,
        bootstrap_n=bootstrap_n,
        seed=seed + 100,
    )
    ope = ope_estimates(rows_by_participant, seed=seed + 200)
    g5 = gate_g5(protocol.get("ope_validation") or {}, ope)
    audit = audit_bundles(confirmatory)

    def total_rule(name: str) -> bool | None:
        return _rule((prediction.get(name) or {}).get("total") or {}, "superiority")

    h1 = _all_pass([total_rule("H1_vs_physics_only"), total_rule("H1_vs_person_static")])
    h2 = _all_pass(
        [
            _rule(primary.get("h2_vs_probe", {}), "superiority"),
            _rule(primary.get("h2_vs_comparator", {}), "ni_utility"),
        ]
    )
    h3 = audit["passed"] and _all_pass(
        [
            _rule(primary.get("h3_override", {}), "ni_override"),
            _rule(primary.get("h3_control_vs_comparator", {}), "ni_control"),
            _rule(primary.get("h3_control_vs_probe", {}), "ni_control"),
        ]
    )
    h6 = _all_pass([_rule(primary.get("h6_ua", {}), "safety_ua")])
    h5 = _all_pass([total_rule("H5_vs_no_practice_switch")])
    gates = {
        "G1": h1,
        "G2": h2,
        "G3": h6,
        "G4": h3,
        "G5": g5["passed"],
        "G6": h5,
    }
    findings = [f"{gate} {'passed' if ok else 'not passed'}" for gate, ok in gates.items()]
    if g5["failed_conditions"]:
        findings.append(f"G5 failed conditions: {', '.join(g5['failed_conditions'])}")
    if not all(v["passed"] for v in verification.values()):
        findings.append(
            "logged seeds, draws or applied L2 responses failed re-derivation (see verification)",
        )
    limitations = list(protocol.get("limitations", ())) + [
        "Per-person regime differences rest mostly on the prior at these trial counts (section 21.3).",
        "OPE evaluates frozen snapshots one block at a time given the logged previous regime; it is not the value of "
        "running a snapshot for a whole session.",
    ]
    return {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256(protocol),
        "protocol_status": protocol.get("status"),
        "mode": "exploratory" if exploratory else "confirmatory",
        **({"exploratory_label": EXPLORATORY_LABEL} if exploratory else {}),
        "evidence_kind": sorted({str(r.evidence_kind) for b in confirmatory for r in b.sessions}),
        "leakage_check": leakage,
        "verification": {"draws": verification, "posteriors": posteriors},
        "realized_utility": {
            "primary": "each block under the priorities logged with its L1 decision; observed outcomes only; no switching penalty",
            "secondary": "protocol weights (0.5, 0.5, 0.5)",
            "counts": counts,
            "participants_at_most_10_percent_missing": len(low_missing),
        },
        "hypotheses": {
            "phase_b_primary": {**primary, "burn_in_trials": burn_in},
            "phase_b_secondary": secondary,
            "block_type_switch_shares": block_type_switch_shares(confirmatory, burn_in=burn_in),
            "prediction": prediction,
            "H5_probe_throughput": probe_throughput(
                rows_by_participant,
                bootstrap_n=bootstrap_n,
                seed=seed + 300,
            ),
            "H6_idle_ua_per_minute": idle_ua_by_regime(confirmatory),
            "H7_gaze": gaze_brier(confirmatory),
            "decisions": {"H1": h1, "H2": h2, "H3": h3, "H5": h5, "H6": h6},
        },
        "ope": ope,
        "h2_ope": h2_ope_report(g5, ope),
        "agency_audit": audit,
        "gates": {**gates, "G5_detail": g5},
        "findings": findings,
        "limitations": limitations,
        "promoted_to_runtime": False,
    }


def write_results(output: Path, results: Mapping[str, Any]) -> None:
    """results.json and results.md in a fresh directory (an existing non-empty directory is refused)."""
    if output.exists() and any(output.iterdir()):
        raise AnalysisRefused("the output directory already holds files; use a fresh one")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Assistance co-adaptation study: results",
        "",
        f"Protocol `{results['protocol_id']}`, sha256 `{results['protocol_sha256']}`, status `{results['protocol_status']}`.",
        f"Mode: {results['mode']}.",
    ]
    if results.get("exploratory_label"):
        lines.append(f"Label: {results['exploratory_label']}.")
    lines += ["", "## Gates", ""]
    for gate in ("G1", "G2", "G3", "G4", "G5", "G6"):
        lines.append(f"- {gate}: {'passed' if results['gates'][gate] else 'not passed'}")
    lines += ["", "## Findings", ""] + [f"- {item}" for item in results["findings"]]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in results["limitations"]]
    lines += ["", "promoted_to_runtime: false", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    bundle_paths: Sequence[Path],
    protocol_path: Path,
    output: Path,
    *,
    exploratory: bool,
    keys: Mapping[str, bytes] | None = None,
    pilot_keys: Iterable[str] = (),
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise AnalysisRefused("the output directory already holds files; use a fresh one")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bundles = []
    for path in bundle_paths:
        header = json.loads(Path(path).read_bytes().partition(b"\n")[0])
        key = None if keys is None else keys.get(str(header.get("participant_key")))
        bundles.append(read_human_bundle(decrypt_bundle(Path(path), key)))
    results = analyse(bundles, protocol, exploratory=exploratory, pilot_keys=pilot_keys)
    write_results(output, results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered analysis of the assistance co-adaptation study",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        action="append",
        required=True,
        help="an encrypted export file",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="a fresh output directory")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="run under a protocol that is not frozen",
    )
    parser.add_argument(
        "--pilot-key",
        action="append",
        default=[],
        help="a pilot participant (left out)",
    )
    args = parser.parse_args(argv)
    try:
        results = run(
            args.bundle,
            args.protocol,
            args.output,
            exploratory=args.exploratory,
            pilot_keys=args.pilot_key,
        )
    except AnalysisRefused as refused:
        parser.exit(2, f"refused: {refused}\n")
    print(
        json.dumps(
            {
                "gates": {k: v for k, v in results["gates"].items() if k != "G5_detail"},
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

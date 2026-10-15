"""Simulated-user benchmark for assistance co-adaptation (spec section 18).

Run from ``daemon``:
    python -m benchmarks.coadaptation_suite --quick --json
    python -m benchmarks.coadaptation_suite --full          # every family and policy; spends the holdout look
    python -m benchmarks.coadaptation_suite --power         # power and N per family; a holdout look as well

Every number here comes from simulated people (``evidence_kind = "simulated"``) and is never a result about people.
The tuning families share the model's own functional forms, so their results are labelled "tuning families,
in-sample" and are never pooled with validation or holdout results. A run that includes a holdout family (F4, F6)
is a holdout look: it is logged in ``.bench-state/coadaptation_holdout_looks.jsonl`` and a second look under the
same frozen protocol is refused unless ``--allow-extra-look`` is given, which marks the output. Every output file
goes under ``daemon/benchmarks/.bench-state/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.coadaptation_ablations import NoAmbiguityRulePolicy, NoSafetyFilterPolicy
from pilot.coadaptation.core.audit import agency_audit
from pilot.coadaptation.core.evaluation import (
    InteractionTrial,
    brier,
    crps_gaussian,
    ece,
    evaluate_interaction_trials,
    population_prior,
    prequential_scores,
    rebuild_engine,
)
from pilot.coadaptation.core.ope import (
    FrozenTarget,
    LoggedBlock,
    LoggedStep,
    MixtureTarget,
    UnitContribution,
    cross_fit_q_hat,
    estimate_from_contributions,
    grouped_bootstrap,
    logged_blocks,
    replay_rows,
    snapshot_before_phase_b,
    unit_contributions,
)
from pilot.coadaptation.core.outcome_model import ModelVariant
from pilot.coadaptation.core.simulator import (
    FAMILY_FORMS,
    FAMILY_SPLITS,
    BaselinePolicy,
    EnginePolicy,
    SimulatedPerson,
    SimulatedRunResult,
    SimulatedStudyRunner,
    StudyPolicy,
    make_population,
    split_of,
    verify_disjoint,
)
from pilot.coadaptation.core.utility import pointing_throughput
from pilot.coadaptation.core.vocabulary import DecisionLevel

BENCH_DIR = Path(__file__).resolve().parent
SIM_PROTOCOL_PATH = BENCH_DIR / "coadaptation_sim_protocol_v1.json"
HUMAN_PROTOCOL_PATH = BENCH_DIR / "coadaptation_protocol_v1.json"
DEFAULT_STATE_DIR = BENCH_DIR / ".bench-state"
LOOKS_FILE = "coadaptation_holdout_looks.jsonl"
SCHEMA_VERSION = "1.0.0"
QUICK_CLAIM = "simulated users only; tuning families, in-sample; no claim about people"
TUNING_LABEL = "tuning families, in-sample"
SPLIT_LABELS = {
    "tune": TUNING_LABEL,
    "validate": "validation family",
    "holdout": "holdout families",
}
BURN_IN = 3  # first trials of every Phase B block left out of trial-level outcomes in the primary analysis
COVERAGE_MIN = 43  # of 50 studies (section 19)
PHASE_B_TYPES = ("B", "probe", "comparator")
HARD_INVARIANTS = ("A1", "A2", "A3", "A4", "A5", "A6", "A9", "A10", "A11", "A12")


class HoldoutLookRefused(RuntimeError):
    """A second holdout look under the same frozen simulation protocol without ``--allow-extra-look``."""


# ------------------------------------------------------------------------------------------------- policies


def _full() -> StudyPolicy:
    return EnginePolicy(ModelVariant.FULL, name="9_engine_full")


POLICIES: Mapping[str, Callable[[], StudyPolicy]] = {
    "1_fixed_assistance_off": lambda: BaselinePolicy("assistance_off"),
    "2_best_fixed_regime": lambda: BaselinePolicy("best_fixed"),
    "3_rule_heuristic": lambda: BaselinePolicy("rule_heuristic"),
    "4_uniform_random": lambda: BaselinePolicy("uniform"),
    "5_engine_physics_only": lambda: EnginePolicy(
        ModelVariant.PHYSICS_ONLY,
        name="5_engine_physics_only",
    ),
    "6_engine_person_static": lambda: EnginePolicy(
        ModelVariant.PERSON_STATIC,
        name="6_engine_person_static",
    ),
    "7_engine_no_practice_switch": lambda: EnginePolicy(
        ModelVariant.NO_PRACTICE_SWITCH, name="7_engine_no_practice_switch"
    ),
    "8_engine_no_today": lambda: EnginePolicy(ModelVariant.NO_TODAY, name="8_engine_no_today"),
    "9_engine_full": _full,
    "10_oracle": lambda: BaselinePolicy("oracle"),
    "11a_ablation_no_ambiguity_rule": lambda: EnginePolicy(
        ModelVariant.FULL, policy=NoAmbiguityRulePolicy, name="11a_ablation_no_ambiguity_rule"
    ),
    "11b_ablation_no_safety_filter": lambda: EnginePolicy(
        ModelVariant.FULL, policy=NoSafetyFilterPolicy, name="11b_ablation_no_safety_filter"
    ),
}
# These must show 0 violations.
CONSTRAINED_POLICIES = tuple(name for name in POLICIES if not name.startswith("11"))
PREDICTION_VARIANTS = (
    ModelVariant.FULL,
    ModelVariant.NO_TODAY,
    ModelVariant.PERSON_STATIC,
    ModelVariant.PHYSICS_ONLY,
    ModelVariant.NO_PRACTICE_SWITCH,
)


# ------------------------------------------------------------------------------------------------- protocols


def load_protocol(path: Path = SIM_PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    """sha256 of the protocol's canonical JSON (key order and whitespace do not matter)."""
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def confirmatory_n(path: Path = HUMAN_PROTOCOL_PATH) -> int:
    """``population.confirmatory_n`` of the human protocol (16 in the draft): the size of every simulated study of
    the OPE validation."""
    return int(load_protocol(path)["population"]["confirmatory_n"])


def _check_protocol(protocol: Mapping[str, Any]) -> None:
    verify_disjoint(protocol["splits"])
    if protocol["status"] != "frozen":
        raise ValueError("the simulation protocol must be frozen")
    for family, forms in protocol["family_forms"].items():
        if forms != asdict(FAMILY_FORMS[family]):
            raise ValueError(f"the protocol's forms for {family} differ from the simulator's")


# ------------------------------------------------------------------------------------------------- holdout looks


def _holdout_families(families: Iterable[str]) -> list[str]:
    return [family for family in families if family in FAMILY_SPLITS["holdout"]]


def record_holdout_look(
    state_dir: Path, protocol_hash: str, families: Sequence[str], *, allow_extra_look: bool
) -> bool:
    """Append a look for a run that includes F4 or F6; refuse a second look under the same protocol hash unless an
    extra look is allowed. Returns True when the look is an extra one."""
    held_out = _holdout_families(families)
    if not held_out:
        return False
    path = state_dir / LOOKS_FILE
    earlier = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                earlier.append(json.loads(line))
    extra = any(look.get("protocol_sha256") == protocol_hash for look in earlier)
    if extra and not allow_extra_look:
        raise HoldoutLookRefused(
            "the holdout families were already looked at under this simulation protocol; "
            "pass --allow-extra-look to look again (the output is then marked extra_holdout_look)"
        )
    state_dir.mkdir(parents=True, exist_ok=True)
    look = {
        "protocol_sha256": protocol_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "families": sorted(set(families)),
        "extra": extra,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(look, sort_keys=True) + "\n")
    return extra


# ------------------------------------------------------------------------------------------------- simulation


@dataclass(frozen=True, slots=True)
class RunSpec:
    family: str
    index: int
    policy: str
    sessions: int
    population_seed: int
    runner_seed: int


def simulate(spec: RunSpec) -> SimulatedRunResult:
    """One simulated person of a family through the protocol against one policy (deterministic in the spec)."""
    person = make_population([spec.family], spec.index + 1, spec.population_seed)[spec.index]
    runner = SimulatedStudyRunner(
        SimulatedPerson(person),
        POLICIES[spec.policy],
        seed=spec.runner_seed,
    )
    return runner.run(sessions=spec.sessions)


BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


@contextmanager
def _one_blas_thread_per_worker() -> Iterator[None]:
    """Workers inherit the environment when they start, so each loads numpy with one BLAS thread (unless the caller
    set these variables). The workers already fill every core, and OpenBLAS would otherwise start a thread per core
    in every worker for products as small as a filter's 4096 parameter draws (about 5% of the OPE validation's wall
    time in a 16-worker measurement)."""
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _map(function: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with _one_blas_thread_per_worker(), ProcessPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(function, items))


def _default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 24))


# ------------------------------------------------------------------------------------------------- statistics


def _mean_ci(values: Mapping[str, float], *, n: int, seed: int) -> dict[str, Any]:
    """Mean over simulated people with a grouped-bootstrap 95% interval (people are the groups)."""
    present = {key: value for key, value in values.items() if value is not None}
    finite = {key: float(value) for key, value in present.items() if math.isfinite(value)}
    if not finite:
        return {"mean": None, "ci95": None, "people": 0}
    mean = float(np.mean(list(finite.values())))
    ci = None
    if len(finite) >= 2:
        lo, hi = grouped_bootstrap(
            {key: [value] for key, value in finite.items()},
            lambda sample: float(np.mean(sample[:, 0])),
            n=n,
            seed=seed,
        )
        ci = [lo, hi]
    return {"mean": mean, "ci95": ci, "people": len(finite)}


def _block_trial_utilities(
    blocks: Sequence[LoggedBlock],
    *,
    burn_in: int = 0,
) -> list[tuple[LoggedBlock, int, float]]:
    """(block, trial index, realized utility per trial with the block's burden term) for every scored trial."""
    out = []
    for block in blocks:
        for position, utility in zip(block.trial_positions, block.trial_utilities):
            if position >= burn_in:
                out.append((block, position, utility - block.burden_term))
    return out


def realized_utility_by_person(
    runs: Sequence[SimulatedRunResult],
    *,
    phases: Sequence[str],
    burn_in: int = 0,
    weights: str = "participant",
) -> dict[str, float]:
    """Mean realized utility per trial (section 21.2) of each simulated person over the blocks of ``phases``."""
    out: dict[str, float] = {}
    for run in runs:
        blocks = logged_blocks(run.rows(), weights=weights, phases=phases)  # type: ignore[arg-type]
        values = [u for _, _, u in _block_trial_utilities(blocks, burn_in=burn_in)]
        if values:
            out[run.participant_key] = float(np.mean(values))
    return out


def _utility_by_session(
    runs: Sequence[SimulatedRunResult],
    phases: Sequence[str],
) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for run in runs:
        blocks = logged_blocks(run.rows(), weights="participant", phases=phases)
        numbers = {row.session_id: row.session_number for row in run.sessions}
        by_session: dict[int, list[float]] = defaultdict(list)
        for block, _, utility in _block_trial_utilities(blocks):
            by_session[numbers[block.session_id]].append(utility)
        for number, values in by_session.items():
            out[(run.participant_key, number)] = float(np.mean(values))
    return out


def policy_metrics(
    runs: Sequence[SimulatedRunResult],
    *,
    bootstrap_n: int,
    seed: int,
    oracle: Sequence[SimulatedRunResult] = (),
) -> dict[str, Any]:
    """The metrics of section 18.3 for one policy on one family."""
    later_sessions = any(row.session_number >= 2 for run in runs for row in run.sessions)
    adaptive_phase = ("B",) if later_sessions else ("A",)
    utility = realized_utility_by_person(runs, phases=adaptive_phase)
    utility_all = realized_utility_by_person(runs, phases=("A", "A_lite", *PHASE_B_TYPES))
    stored = [stored for run in runs for stored in run.trials]
    scored = [s for s in stored if not s.trial.interrupted and s.labels is not None]
    interaction = None
    if scored:
        interaction = evaluate_interaction_trials([InteractionTrial.from_stored(s) for s in scored])
    pointing = [s for s in scored if s.trial.task == "pointing"]
    active = [s for s in scored if s.trial.task != "idle"]
    reports = [report for run in runs for report in run.burden if report.scope == "block"]
    burden: dict[str, float | None] = {}
    for item in ("effort", "intrusiveness", "control"):
        values = [getattr(r, item) for r in reports if getattr(r, item)]
        burden[item] = float(np.mean(values)) if values else None
    vetoes = [s.trial.veto for s in active if s.trial.veto is not None]
    overrides = sum(1 for veto in vetoes if str(veto.level) == "block_regime")
    report = audit_runs(runs)
    truth_acting = sum(1 for run in runs for truth in run.truth if truth.acting_under_ambiguity)
    violations = sum(int(report.counts[key]) for key in HARD_INVARIANTS) + sum(
        int(report.counts[key] < 1.0) for key in ("A7", "A8")
    )
    metrics: dict[str, Any] = {
        "people": len(runs),
        "adaptive_positions": list(adaptive_phase),
        "realized_utility_per_trial": _mean_ci(utility, n=bootstrap_n, seed=seed),
        "realized_utility_per_trial_all_blocks": _mean_ci(
            utility_all,
            n=bootstrap_n,
            seed=seed + 1,
        ),
        "completion": None if interaction is None else interaction.completion,
        "ua_per_trial": None if interaction is None else interaction.ua_per_trial,
        "ua_per_idle_minute": None if interaction is None else interaction.ua_per_idle_minute,
        "corrections_per_trial": None if interaction is None else interaction.corrections_per_trial,
        "median_time_ms": None if interaction is None else interaction.median_time_ms,
        "p95_time_ms": None if interaction is None else interaction.p95_time_ms,
        "throughput_bits_s": pointing_throughput(pointing),
        "simulated_burden": burden,
        "veto_rate": len(vetoes) / len(active) if active else None,
        "override_rate": overrides / len(active) if active else None,
        "switch_rate": report.soft["switch_rate"],
        "regime_entropy_bits": report.soft["regime_entropy_bits"],
        "constraint_violations": violations,
        "acting_under_ambiguity": int(report.counts["A2"]) + truth_acting,
        "preference_replacements": sum(run.preference_replacements for run in runs),
        "blocks_by_phase": _blocks_by_phase(runs),
        "refused_blocks": sum(run.refused_blocks for run in runs),
        "blocks_ended_by_refresh": sum(run.blocks_ended_by_refresh for run in runs),
    }
    if oracle:
        mine = _utility_by_session(runs, adaptive_phase)
        best = _utility_by_session(oracle, adaptive_phase)
        regrets = [best[key] - value for key, value in mine.items() if key in best]
        metrics["regret_vs_oracle_per_day"] = float(np.mean(regrets)) if regrets else None
        metrics["trials_until_within_5_percent_of_oracle"] = _trials_to_oracle(
            runs,
            oracle,
            adaptive_phase,
        )
    return metrics


def _plan_phases(run: SimulatedRunResult) -> dict[tuple[str, int], str]:
    """Each planned block's phase by (session id, block index)."""
    blocks = [(row.session_id, plan) for row in run.sessions for plan in row.plan.blocks]
    return {(session_id, plan.index): plan.phase for session_id, plan in blocks}


def _blocks_by_phase(runs: Sequence[SimulatedRunResult]) -> dict[str, int]:
    """How many blocks of each phase ran (a block ran when it recorded at least one trial)."""
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        phases = _plan_phases(run)
        active = [s.trial for s in run.trials if s.trial.task != "idle"]
        ran = {(trial.session_id, trial.block_index) for trial in active}
        for key in ran:
            counts[phases[key]] += 1
    return dict(sorted(counts.items()))


def _trials_to_oracle(
    runs: Sequence[SimulatedRunResult], oracle: Sequence[SimulatedRunResult], phases: Sequence[str]
) -> float | None:
    """Median over people of the adaptive-position trials after which a 16-trial moving mean of the policy's
    realized utility stays within 5% of the oracle's mean (None when it never does)."""
    oracle_mean = realized_utility_by_person(oracle, phases=phases)
    counts = []
    for run in runs:
        target = oracle_mean.get(run.participant_key)
        if target is None:
            continue
        values = [
            u
            for _, _, u in _block_trial_utilities(
                logged_blocks(run.rows(), weights="participant", phases=phases),
            )
        ]
        found = None
        for start in range(0, max(len(values) - 15, 0)):
            window = float(np.mean(values[start : start + 16]))
            if abs(window - target) <= 0.05 * abs(target) if target else abs(window) <= 0.05:
                found = start
                break
        if found is not None:
            counts.append(found)
    return float(statistics.median(counts)) if counts else None


def audit_runs(runs: Sequence[SimulatedRunResult]) -> Any:
    """The agency audit over every run of a policy (each run is one simulated participant)."""
    return agency_audit(
        [d for run in runs for d in run.decisions],
        [t for run in runs for t in run.trials],
        [s for run in runs for s in run.sessions],
        [b for run in runs for b in run.burden],
        [p for run in runs for p in run.preferences],
        [c for run in runs for c in run.consents],
        events=[e for run in runs for e in run.events],
    )


def _audit_summary(report: Any) -> dict[str, Any]:
    data = report.to_dict()
    return {
        "passed": data["passed"],
        "counts": {key: value["value"] for key, value in data["invariants"].items()},
        "findings": data["findings"][:20],
        "a9_exemptions": len(data["exemptions"]),
        "soft": data["soft"],
    }


def _rows(run: SimulatedRunResult) -> list[Any]:
    return replay_rows(run.rows(), run.participant_key)


def _prequential(run: SimulatedRunResult, variant: ModelVariant) -> Any:
    return prequential_scores(_rows(run), variant=variant, preferences=run.preferences)


def prediction_metrics(
    runs: Sequence[SimulatedRunResult],
    *,
    variants: Sequence[ModelVariant],
    bootstrap_n: int,
    seed: int,
) -> dict[str, Any]:
    """Prequential log score, Brier, 10-bin ECE and CRPS per outcome for each model variant, replayed on the same
    rows (H1's primary set: Phase A and A-lite trials), and the H4 contrast FULL against NO_TODAY."""
    per_variant: dict[str, Any] = {}
    totals: dict[str, dict[str, float]] = {}
    for variant in variants:
        scores = [_prequential(run, variant) for run in runs]
        primary = [s for result in scores for s in result.select(primary=True)]
        if not primary:
            continue
        summary = scores[0].mean(primary) if scores else {}
        out: dict[str, Any] = {"log_score": summary, "trials": len(primary)}
        for outcome in ("complete", "ua"):
            p = [s.predictions[outcome] for s in primary]
            y = [s.outcomes[outcome] for s in primary]
            out[f"brier_{outcome}"] = brier(p, y)
            out[f"ece_{outcome}"] = ece(p, y)
        completed = [s for s in primary if not s.outcomes["censored"]]
        out["crps_log_time"] = (
            float(
                np.mean(
                    [
                        crps_gaussian(
                            s.predictions["log_time_mean"],
                            s.predictions["log_time_sd"],
                            s.outcomes["log_time"],
                        )
                        for s in completed
                    ]
                )
            )
            if completed
            else None
        )
        per_variant[ModelVariant(variant).value] = out
        totals[ModelVariant(variant).value] = {
            key: value["total"]
            for result in scores
            for key, value in result.by_participant(result.select(primary=True)).items()
        }
    contrast = None
    if "full" in totals and "no_today" in totals:
        full, no_today = totals["full"], totals["no_today"]
        diffs = {key: full[key] - no_today[key] for key in full if key in no_today}
        contrast = _mean_ci(diffs, n=bootstrap_n, seed=seed)
    return {"variants": per_variant, "full_vs_no_today": contrast}


def probe_reliance_detection(runs: Sequence[SimulatedRunResult]) -> dict[str, Any]:
    """Whether probe-block pointing throughput falls from session 2 to session 3 for people with injected reliance
    drift (detection) and for people without it (false alarms). Needs three sessions."""
    detected = {"drift": [], "no_drift": []}
    for run in runs:
        numbers = {row.session_id: row.session_number for row in run.sessions}
        probe_blocks = {key for key, phase in _plan_phases(run).items() if phase == "probe"}
        by_session: dict[int, list[Any]] = defaultdict(list)
        for stored in run.trials:
            key = (stored.trial.session_id, stored.trial.block_index)
            if key in probe_blocks and stored.trial.task == "pointing":
                by_session[numbers[stored.trial.session_id]].append(stored)
        if 2 not in by_session or 3 not in by_session:
            continue
        before, after = pointing_throughput(by_session[2]), pointing_throughput(by_session[3])
        if before is None or after is None:
            continue
        group = "drift" if run.params.reliance_drift > 0 else "no_drift"
        detected[group].append(after < before)
    return {
        "detection_rate": float(np.mean(detected["drift"])) if detected["drift"] else None,
        "false_alarm_rate": float(np.mean(detected["no_drift"])) if detected["no_drift"] else None,
        "people_with_drift": len(detected["drift"]),
        "people_without_drift": len(detected["no_drift"]),
    }


# ------------------------------------------------------------------------------------------------- the run


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    families: tuple[str, ...]
    people_per_family: int
    sessions: int
    policies: tuple[str, ...]
    bootstrap_n: int
    population_seed: int
    runner_seed: int
    prediction_variants: tuple[ModelVariant, ...]
    workers: int = 1


def _split_of_families(families: Sequence[str]) -> str:
    splits = {split_of(family) for family in families}
    return splits.pop() if len(splits) == 1 else "mixed"


def run_suite(config: SuiteConfig) -> dict[str, Any]:
    """Simulate every (family, person, policy) of ``config`` and report per family and split (never pooled across
    splits)."""
    specs = [
        RunSpec(family, index, policy, config.sessions, config.population_seed, config.runner_seed)
        for family in config.families
        for index in range(config.people_per_family)
        for policy in config.policies
    ]
    results = _map(simulate, specs, config.workers)
    runs: dict[tuple[str, str], list[SimulatedRunResult]] = defaultdict(list)
    for spec, result in zip(specs, results):
        runs[(spec.family, spec.policy)].append(result)
    by_family: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for f_index, family in enumerate(config.families):
        split = split_of(family)
        oracle = runs.get((family, "10_oracle"), [])
        policies = {}
        for p_index, policy in enumerate(config.policies):
            policies[policy] = policy_metrics(
                runs[(family, policy)],
                bootstrap_n=config.bootstrap_n,
                seed=1000 * f_index + p_index,
                oracle=oracle if policy != "10_oracle" else (),
            )
        full_runs = runs.get((family, "9_engine_full"), [])
        reliance = probe_reliance_detection(full_runs) if config.sessions >= 3 else None
        by_family[family] = {
            "split": split,
            "label": SPLIT_LABELS[split],
            "forms": asdict(FAMILY_FORMS[family]),
            "people": config.people_per_family,
            "sessions": config.sessions,
            "policies": policies,
            "prediction": prediction_metrics(
                full_runs,
                variants=config.prediction_variants,
                bootstrap_n=config.bootstrap_n,
                seed=77 + f_index,
            )
            if full_runs
            else None,
            "probe_reliance_detection": reliance,
        }
    for policy in config.policies:
        audits[policy] = _audit_summary(
            audit_runs([r for family in config.families for r in runs[(family, policy)]]),
        )
    by_split: dict[str, Any] = {}
    for split in ("tune", "validate", "holdout"):
        families = [family for family in config.families if split_of(family) == split]
        if not families:
            continue
        pooled = {}
        for p_index, policy in enumerate(config.policies):
            utility: dict[str, float] = {}
            for family in families:
                utility.update(
                    realized_utility_by_person(
                        runs[(family, policy)], phases=("B",) if config.sessions >= 2 else ("A",)
                    )
                )
            pooled[policy] = _mean_ci(utility, n=config.bootstrap_n, seed=500 + p_index)
        by_split[split] = {
            "label": SPLIT_LABELS[split],
            "families": families,
            "realized_utility_per_trial": pooled,
        }
    return {"by_family": by_family, "by_split": by_split, "agency_audit": audits}


def _environment() -> dict[str, Any]:
    return {
        "operating_system": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _limitations(config: SuiteConfig) -> list[str]:
    items = [
        "Simulated people only: every number describes the simulator's families, not people.",
        "The tuning families share the model's functional forms, so the full model and the policy win there largely "
        "by construction.",
        "Validation and holdout families use forms the model lacks; even they come from the same literature as the "
        "priors.",
        "Realized utility uses observed outcomes and each simulated person's priorities (the protocol weights); no "
        "switching penalty.",
        "Offers have no mechanical effect on simulated selections.",
    ]
    if config.sessions < 3:
        items.append(
            f"{config.sessions} sessions per person: cross-session and probe-trend metrics are not computed.",
        )
    return items


def benchmark(
    quick: bool = True,
    *,
    state_dir: Path | None = None,
    allow_extra_look: bool = False,
    extra_look: bool | None = None,
) -> dict[str, Any]:
    """The simulated-user benchmark. ``quick`` (CI, under 60 s): F1 and F3, 5 people each, 2 sessions (so Phase B
    adaptive, probe and comparator blocks run), policies 1, 4 and 9; deterministic; labelled "tuning families,
    in-sample". ``quick=False`` is the full run of every family and policy (a holdout look) with the OPE
    validation."""
    protocol = load_protocol()
    _check_protocol(protocol)
    if quick:
        quick_config = protocol["quick"]
        config = SuiteConfig(
            families=tuple(quick_config["families"]),
            people_per_family=int(quick_config["people_per_family"]),
            sessions=int(quick_config["sessions"]),
            policies=tuple(quick_config["policies"]),
            bootstrap_n=int(protocol["analysis"]["bootstrap"]["quick_replicates"]),
            population_seed=int(protocol["population_seed"]),
            runner_seed=int(protocol["runner_seed"]),
            prediction_variants=(ModelVariant.FULL, ModelVariant.NO_TODAY),
            workers=1,
        )
    else:
        config = SuiteConfig(
            families=tuple(protocol["families"]),
            people_per_family=int(protocol["people_per_family"]),
            sessions=int(protocol["sessions"]),
            policies=tuple(protocol["policies"]),
            bootstrap_n=int(protocol["analysis"]["bootstrap"]["replicates"]),
            population_seed=int(protocol["population_seed"]),
            runner_seed=int(protocol["runner_seed"]),
            prediction_variants=PREDICTION_VARIANTS,
            workers=_default_workers(),
        )
    return run_config(
        config,
        protocol,
        state_dir=state_dir,
        allow_extra_look=allow_extra_look,
        ope=not quick,
        extra_look=extra_look,
    )


def run_config(
    config: SuiteConfig,
    protocol: Mapping[str, Any],
    *,
    state_dir: Path | None = None,
    allow_extra_look: bool = False,
    ope: bool = False,
    extra_look: bool | None = None,
) -> dict[str, Any]:
    """Run ``config`` under ``protocol`` and build the report of section 18.4. A run that includes a holdout family
    logs its look first (``extra_look`` passes on a look the caller already logged)."""
    state_dir = state_dir or DEFAULT_STATE_DIR
    digest = protocol_sha256(protocol)
    if extra_look is None:
        extra = record_holdout_look(
            state_dir,
            digest,
            config.families,
            allow_extra_look=allow_extra_look,
        )
    else:
        extra = extra_look
    split = _split_of_families(config.families)
    metrics = run_suite(config)
    audits = metrics.pop("agency_audit")
    if ope:
        metrics["ope_validation"] = ope_validation(
            n_studies=int(protocol["ope_validation"]["studies"]),
            n_sim=confirmatory_n(),
            population_seed=int(protocol["population_seed"]),
            runner_seed=int(protocol["runner_seed"]),
            replays=int(protocol["ope_validation"]["counterfactual_replays"]),
            workers=config.workers,
        )
    in_sample = split == "tune"
    claim = QUICK_CLAIM if in_sample else "simulated users only; no claim about people"
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "coadaptation_simulated_users",
        "evidence_kind": "simulated",
        "scope": (
            f"Simulated users of {', '.join(config.families)} ({config.people_per_family} per family, "
            f"{config.sessions} sessions) against policies {', '.join(config.policies)}"
        ),
        "families": list(config.families),
        "split": split,
        "in_sample": in_sample,
        "environment": _environment(),
        "protocol_sha256": digest,
        "metrics": metrics,
        "agency_audit": audits,
        "limitations": _limitations(config),
        "claim_boundary": claim,
        "promoted_to_runtime": False,
    }
    if extra:
        report["extra_holdout_look"] = True
    return report


# ------------------------------------------------------------------------------------------------- OPE validation


def _study_people(
    study: int,
    n_sim: int,
    population_seed: int,
    families: Sequence[str],
) -> list[Any]:
    """The ``n_sim`` simulated people of one study, families in equal numbers."""
    per_family = n_sim // len(families)
    seed = population_seed + 7919 * (study + 1)
    return make_population(families, per_family, seed)


VALIDATION_CHAIN = 5  # studies per worker task at one position (section 19 validation)


@dataclass(frozen=True, slots=True)
class _ChainTask:
    """Position ``index`` of studies ``start .. stop - 1``: one worker simulates them in turn and reuses each
    person's S_2 as the estimator-check target of the study before."""

    index: int
    start: int
    stop: int
    n_studies: int
    n_sim: int
    population_seed: int
    runner_seed: int
    families: tuple[str, ...]
    replays: int


@dataclass(frozen=True, slots=True)
class _BlockRecord:
    """What the study-level aggregation needs of one evaluated block besides its contributions."""

    unit_id: str
    session_id: str
    task: str
    chosen: str | None
    reward: float
    truth_value: float
    truth_mass: float


@dataclass(frozen=True, slots=True)
class _TaskContext:
    task: str


@dataclass(frozen=True, slots=True)
class _TargetPart:
    clipped: tuple[UnitContribution, ...]
    unclipped: tuple[UnitContribution, ...]
    blocks: tuple[_BlockRecord, ...]


TARGET_NAMES = ("estimator_check", "in_sample_S_2", "in_sample_S_3", "out_of_sample_S_2")


def _simulate_person(
    person: Any, runner_seed: int, sessions: int = 3
) -> tuple[SimulatedStudyRunner, SimulatedRunResult]:
    runner = SimulatedStudyRunner(SimulatedPerson(person), _full, seed=runner_seed)
    return runner, runner.run(sessions=sessions)


def _target_part(
    runner: SimulatedStudyRunner,
    result: SimulatedRunResult,
    target: Any,
    blocks: Sequence[LoggedBlock],
    *,
    replays: int,
    seed: int,
) -> _TargetPart:
    """One person's contributions to one target and the counterfactual truth of each of their blocks."""
    clipped = tuple(unit_contributions(blocks, target))
    unclipped = tuple(unit_contributions(blocks, target, clip=None))
    records = []
    for block in blocks:

        def probabilities(_unit: str, offer: Any = None, _context: Any = block.context) -> Any:
            return target.probabilities(_context)

        value, mass = runner.counterfactual_block_value(
            result,
            block.unit_id,
            probabilities,
            remaining_arms=block.remaining_arms,
            replays=replays,
            seed=seed,
        )
        chosen = next(
            (step.chosen for step in block.steps if str(step.level) == "block_regime"),
            None,
        )
        records.append(
            _BlockRecord(
                block.unit_id,
                block.session_id,
                block.context.task,
                chosen,
                block.reward,
                value,
                mass,
            )
        )
    return _TargetPart(clipped, unclipped, tuple(records))


def _validation_chain(task: _ChainTask) -> list[dict[str, _TargetPart]]:
    """Position ``task.index`` of studies ``task.start .. task.stop - 1`` of the OPE validation, in study order.

    The estimator check of study j needs the S_2 of the same position in study j + 1. Sessions 1 and 2 of a run do
    not depend on how many sessions follow, so that S_2 is the one of study j + 1's own three-session run: the chain
    runs its studies from the last to the first and simulates only the study after the chain for two sessions."""
    following = _study_people(task.stop % task.n_studies, task.n_sim, task.population_seed, task.families)[task.index]
    _, following_result = _simulate_person(following, task.runner_seed, sessions=2)  # S_2 needs sessions 1 and 2
    next_s2 = snapshot_before_phase_b(
        following_result.rows(), following_result.participant_key, 2, preferences=following_result.preferences
    )
    parts: list[dict[str, _TargetPart]] = []
    for study in reversed(range(task.start, task.stop)):
        person = _study_people(study, task.n_sim, task.population_seed, task.families)[task.index]
        runner, result = _simulate_person(person, task.runner_seed)
        s2 = snapshot_before_phase_b(result.rows(), result.participant_key, 2, preferences=result.preferences)
        parts.append(
            _validation_person(runner, result, s2, next_s2, study=study, index=task.index, replays=task.replays)
        )
        next_s2 = s2
    return parts[::-1]


def _validation_person(
    runner: SimulatedStudyRunner,
    result: SimulatedRunResult,
    s2: Any,
    other_s2: Any,
    *,
    study: int,
    index: int,
    replays: int,
) -> dict[str, _TargetPart]:
    """One simulated person of one study of the OPE validation (section 19): their logs, their in-sample S_2 and
    S_3, the out-of-sample S_2 on their session-3 A-lite blocks, and the estimator-check target: ``other_s2``, the
    frozen S_2 of the same position in study j + 1, mixed half and half with the uniform policy over each step's
    safe set."""
    rows = result.rows()
    key = result.participant_key
    s3 = snapshot_before_phase_b(rows, key, 3, preferences=result.preferences)
    if s2 is None or s3 is None or other_s2 is None:
        return {}
    blocks_12 = logged_blocks(rows, weights="participant", sessions=(1, 2))
    blocks_123 = logged_blocks(rows, weights="participant", sessions=(1, 2, 3))
    blocks_3 = logged_blocks(rows, weights="participant", sessions=(3,), phases=("A_lite",))
    history = result.preferences
    own_s2 = FrozenTarget(
        s2.data,
        participant_key=key,
        settings=s2.settings,
        preferences_history=history,
        rows_hash=s2.rows_hash,
    )
    own_s3 = FrozenTarget(
        s3.data,
        participant_key=key,
        settings=s3.settings,
        preferences_history=history,
        rows_hash=s3.rows_hash,
    )
    mixture = MixtureTarget(
        FrozenTarget(
            other_s2.data,
            participant_key=key,
            settings=s2.settings,
            preferences_history=history,
            rows_hash=other_s2.rows_hash,
        )
    )
    seed = 104729 * (study + 1) + index
    common = {"replays": replays}
    return {
        "estimator_check": _target_part(runner, result, mixture, blocks_12, seed=seed, **common),
        "in_sample_S_2": _target_part(runner, result, own_s2, blocks_12, seed=seed + 1, **common),
        "in_sample_S_3": _target_part(runner, result, own_s3, blocks_123, seed=seed + 2, **common),
        "out_of_sample_S_2": _target_part(
            runner,
            result,
            own_s2,
            blocks_3,
            seed=seed + 3,
            **common,
        ),
    }


def _aggregate_target(
    parts: Sequence[_TargetPart],
    *,
    seed: int,
    bootstrap_n: int,
) -> dict[str, Any]:
    """One study's estimates of one target, with the leave-one-session-out DR model fitted over every person's
    blocks of the study, and the truth aggregated exactly as V(pi_e): sum_b pi_e(R_b) V_b / sum_b pi_e(R_b)."""
    records = [record for part in parts for record in part.blocks]
    if not records:
        return {
            "truth": None,
            "dr": None,
            "ci95": None,
            "ess": 0.0,
            "refused": "no_support",
            "covered": False,
            "ips": None,
            "snips": None,
            "dr_unclipped": None,
            "ips_unclipped": None,
        }
    light = {
        record.unit_id: LoggedBlock(
            unit_id=record.unit_id,
            group_id="",
            session_id=record.session_id,
            context=_TaskContext(record.task),
            remaining_arms=(),
            steps=()
            if record.chosen is None
            else (
                LoggedStep(
                    DecisionLevel.BLOCK_REGIME,
                    "",
                    None,
                    record.chosen,
                    {record.chosen: 1.0},
                    None,
                ),
            ),
            trial_utilities=(),
            trial_positions=(),
            burden_term=0.0,
            reward=record.reward,
            behavior="policy",
        )
        for record in records
    }
    q_hat = cross_fit_q_hat(list(light.values()))

    def with_dr(units: Iterable[UnitContribution]) -> list[UnitContribution]:
        return [unit.with_q_hat(partial(q_hat, light[unit.unit_id])) for unit in units]

    clipped = with_dr(unit for part in parts for unit in part.clipped)
    unclipped = with_dr(unit for part in parts for unit in part.unclipped)
    valued = [r for r in records if r.truth_mass > 0 and math.isfinite(r.truth_value)]
    numerator = sum(r.truth_mass * r.truth_value for r in valued)
    denominator = sum(r.truth_mass for r in valued)
    truth = numerator / denominator if denominator > 0 else None
    dr = estimate_from_contributions("dr", clipped, bootstrap_n=bootstrap_n, bootstrap_seed=seed)
    ips = estimate_from_contributions("ips", clipped, bootstrap_n=1, bootstrap_seed=seed)
    snips = estimate_from_contributions("snips", clipped, bootstrap_n=1, bootstrap_seed=seed)
    dr_unclipped = estimate_from_contributions(
        "dr",
        unclipped,
        clip=None,
        bootstrap_n=1,
        bootstrap_seed=seed,
    )
    ips_unclipped = estimate_from_contributions(
        "ips",
        unclipped,
        clip=None,
        bootstrap_n=1,
        bootstrap_seed=seed,
    )
    estimated = dr.value is not None and dr.ci95 is not None and truth is not None
    covered = estimated and dr.ci95[0] <= truth <= dr.ci95[1]
    return {
        "truth": truth,
        "dr": dr.value,
        "ci95": None if dr.ci95 is None else list(dr.ci95),
        "ess": dr.ess,
        "refused": dr.refused,
        "covered": covered,
        "ips": ips.value,
        "snips": snips.value,
        "dr_unclipped": dr_unclipped.value,
        "ips_unclipped": ips_unclipped.value,
    }


def _target_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Refusal rate, median ESS, coverage (refusals count as not covering), coverage among the studies not refused
    and the mean bias of the estimates not refused."""
    n = len(results)
    computed = [r for r in results if r["refused"] is None and None not in (r["dr"], r["truth"])]
    covered = sum(1 for r in results if r["covered"])
    covered_computed = sum(1 for r in computed if r["covered"])
    return {
        "studies": n,
        "covered": covered,
        "coverage": covered / n if n else None,
        "coverage_not_refused": covered_computed / len(computed) if computed else None,
        "mean_bias": float(np.mean([r["dr"] - r["truth"] for r in computed])) if computed else None,
        "refusal_rate": (n - len(computed)) / n if n else None,
        "median_ess": float(np.median([r["ess"] for r in results])) if results else None,
    }


def _mean_error(results: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    paired = [r for r in results if r.get(key) is not None and r["truth"] is not None]
    errors = [r[key] - r["truth"] for r in paired]
    if len(errors) < 2:
        return {"mean": None, "mc_se": None, "studies": len(errors)}
    return {
        "mean": float(np.mean(errors)),
        "mc_se": float(np.std(errors, ddof=1) / math.sqrt(len(errors))),
        "studies": len(errors),
    }


def ope_validation(
    *,
    n_studies: int = 50,
    n_sim: int | None = None,
    families: Sequence[str] = ("F1", "F3"),
    population_seed: int = 20261015,
    runner_seed: int = 7,
    replays: int = 8,
    bootstrap_n: int = 5000,
    workers: int | None = None,
) -> dict[str, Any]:
    """The OPE validation of section 19 sized to the human design: ``n_studies`` seeded studies of ``n_sim`` people
    (the human protocol's ``population.confirmatory_n``), F1 and F3 in equal numbers, three sessions with the block
    counts of section 12.3, switch cost and practice on. The estimator check is asserted by the test (DR coverage at
    least 43 of 50, mean errors within tolerance); the analysis's own targets are computed and written as
    ``ope_validation``."""
    n_sim = confirmatory_n() if n_sim is None else n_sim
    if n_sim % len(families):
        raise ValueError("n_sim must split evenly over the families")
    tasks = [
        _ChainTask(
            i,
            start,
            min(start + VALIDATION_CHAIN, n_studies),
            n_studies,
            n_sim,
            population_seed,
            runner_seed,
            tuple(families),
            replays,
        )
        for start in range(0, n_studies, VALIDATION_CHAIN)
        for i in range(n_sim)
    ]
    chains = _map(_validation_chain, tasks, workers if workers is not None else _default_workers())
    parts: dict[tuple[int, int], dict[str, _TargetPart]] = {}
    for task, chain in zip(tasks, chains):
        for study, part in zip(range(task.start, task.stop), chain):
            parts[study, task.index] = part
    by_study = {j: [parts[j, i] for i in range(n_sim)] for j in range(n_studies)}
    studies = []
    for j in range(n_studies):
        summary = {}
        for t_index, name in enumerate(TARGET_NAMES):
            target_parts = [part[name] for part in by_study[j] if name in part]
            summary[name] = _aggregate_target(
                target_parts,
                seed=104729 * (j + 1) + t_index,
                bootstrap_n=bootstrap_n,
            )
        studies.append(summary)
    check = [s["estimator_check"] for s in studies]
    check_summary = _target_summary(check)
    estimators = ("dr_unclipped", "ips_unclipped", "dr", "ips", "snips")
    errors = {key: _mean_error(check, key) for key in estimators}
    unbiased_ok = all(
        errors[key]["mean"] is not None and abs(errors[key]["mean"]) <= 3 * errors[key]["mc_se"]
        for key in ("dr_unclipped", "ips_unclipped")
    )
    clipped_means = [errors[key]["mean"] for key in ("dr", "ips", "snips")]
    tolerance_ok = all(mean is not None and abs(mean) <= 0.02 for mean in clipped_means)
    required = math.ceil(COVERAGE_MIN * n_studies / 50)
    return {
        "n_sim": n_sim,
        "studies": n_studies,
        "families": list(families),
        "counterfactual_replays": replays,
        "estimator_check": {
            **check_summary,
            "required": required,
            "mean_errors": errors,
            "unbiased_within_3_mc_se": unbiased_ok,
            "clipped_and_snips_within_0_02": tolerance_ok,
            "passed": check_summary["covered"] >= required and unbiased_ok and tolerance_ok,
        },
        "estimator_check_coverage": check_summary["coverage"],
        "in_sample": {
            "S_2": _target_summary([s["in_sample_S_2"] for s in studies]),
            "S_3": _target_summary([s["in_sample_S_3"] for s in studies]),
        },
        "out_of_sample_S_2": _target_summary([s["out_of_sample_S_2"] for s in studies]),
        "evidence_kind": "simulated",
        "claim_boundary": "simulated studies only; coverage of the estimators at the human design, no claim about people",
        "promoted_to_runtime": False,
    }


# ------------------------------------------------------------------------------------------------- power


def _contrast(by_type: Mapping[str, float], left: str, right: str) -> float | None:
    if left in by_type and right in by_type:
        return by_type[left] - by_type[right]
    return None


def person_statistics(run: SimulatedRunResult) -> dict[str, float | None]:
    """The per-person differences behind H2, H3 and H6 (primary analysis with the three-trial burn-in)."""
    rows = run.rows()
    blocks = logged_blocks(rows, weights="participant", phases=PHASE_B_TYPES)
    phase = {block.unit_id: block.context.phase for block in blocks}
    utility: dict[str, list[float]] = defaultdict(list)
    for block, _, value in _block_trial_utilities(blocks, burn_in=BURN_IN):
        utility[phase[block.unit_id]].append(value)
    plan_phase = _plan_phases(run)
    overrides: dict[str, list[float]] = defaultdict(list)
    ua: dict[str, list[float]] = defaultdict(list)
    for stored in run.trials:
        trial = stored.trial
        kind = plan_phase.get((trial.session_id, trial.block_index))
        if kind not in PHASE_B_TYPES or trial.task == "idle":
            continue
        if trial.interrupted or stored.labels is None:
            continue
        if trial.trial_index < BURN_IN:
            continue
        overrides[kind].append(
            float(trial.veto is not None and str(trial.veto.level) == "block_regime"),
        )
        ua[kind].append(float(stored.labels.ua))
    control: dict[str, list[float]] = defaultdict(list)
    for report in run.burden:
        kind = plan_phase.get((report.session_id, report.block_index))
        if kind in PHASE_B_TYPES and report.control is not None:
            control[kind].append(float(report.control))
    mean = {
        key: {k: float(np.mean(v)) for k, v in table.items() if v}
        for key, table in {
            "utility": utility,
            "override": overrides,
            "ua": ua,
            "control": control,
        }.items()
    }
    return {
        "h2_vs_probe": _contrast(mean["utility"], "B", "probe"),
        "h2_vs_comparator": _contrast(mean["utility"], "B", "comparator"),
        "h3_override": _contrast(mean["override"], "B", "comparator"),
        "h3_control_vs_comparator": _contrast(mean["control"], "B", "comparator"),
        "h3_control_vs_probe": _contrast(mean["control"], "B", "probe"),
        "h6_ua": _contrast(mean["ua"], "B", "probe"),
    }


def _total(scores: Any, **select: Any) -> float | None:
    """Mean summed log score of a scoring set (None when the set is empty)."""
    chosen = scores.select(**select)
    return scores.mean(chosen)["total"] if chosen else None


def prediction_statistics(runs: Sequence[SimulatedRunResult]) -> dict[str, dict[str, float]]:
    """Per-person log-score differences behind H1, H8 and H9 (scoring rules of H1)."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    full_engines = {}
    for run in runs:
        full_engines[run.participant_key] = rebuild_engine(_rows(run), preferences=run.preferences)
    for run in runs:
        rows = _rows(run)
        key = run.participant_key
        scores = {
            variant: prequential_scores(rows, variant=variant, preferences=run.preferences)
            for variant in (
                ModelVariant.FULL,
                ModelVariant.PHYSICS_ONLY,
                ModelVariant.PERSON_STATIC,
                ModelVariant.SESSION_ONLY,
            )
        }

        full = _total(scores[ModelVariant.FULL], primary=True)
        for name, variant in (
            ("h1_vs_physics", ModelVariant.PHYSICS_ONLY),
            ("h1_vs_static", ModelVariant.PERSON_STATIC),
        ):
            other = _total(scores[variant], primary=True)
            if full is not None and other is not None:
                out[key][name] = full - other
        early_full = _total(scores[ModelVariant.FULL], sessions=(2, 3), first=24)
        early_session = _total(scores[ModelVariant.SESSION_ONLY], sessions=(2, 3), first=24)
        if early_full is not None and early_session is not None:
            out[key]["h9"] = early_full - early_session
        if len(full_engines) >= 2:
            prior = population_prior(full_engines, leave_out=key)
            with_prior = prequential_scores(
                rows, variant=ModelVariant.FULL, population_prior=prior, preferences=run.preferences
            )
            population = prequential_scores(
                rows,
                variant=ModelVariant.POPULATION,
                population_prior=prior,
                preferences=run.preferences,
            )
            a = with_prior.select(primary=True)
            b = population.select(primary=True)
            if a and b:
                out[key]["h8"] = with_prior.mean(a)["total"] - population.mean(b)["total"]
    return dict(out)


def _passes(rule: str, values: np.ndarray) -> bool:
    """The decision rule of one hypothesis on a resample of per-person differences (two-sided 95% interval)."""
    n = len(values)
    if n < 2:
        return False
    mean = float(np.mean(values))
    half = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(n)
    if rule == "superiority":
        return mean - half > 0
    if rule == "ni_utility":
        return mean - half > -0.05
    if rule == "ni_override":
        return mean + half <= 0.05
    if rule == "ni_control":
        return mean - half > -0.5
    if rule == "safety_ua":
        return mean + half <= 0.02
    raise ValueError("unknown rule")


HYPOTHESIS_RULES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "H1": (("h1_vs_physics", "superiority"), ("h1_vs_static", "superiority")),
    "H2": (("h2_vs_probe", "superiority"), ("h2_vs_comparator", "ni_utility")),
    "H3": (
        ("h3_override", "ni_override"),
        ("h3_control_vs_comparator", "ni_control"),
        ("h3_control_vs_probe", "ni_control"),
    ),
    "H6": (("h6_ua", "safety_ua"),),
    "H8": (("h8", "superiority"),),
    "H9": (("h9", "superiority"),),
}


def power_curve(
    statistics_by_person: Mapping[str, Mapping[str, float | None]],
    *,
    candidate_n: Sequence[int],
    resamples: int,
    seed: int,
    target_power: float = 0.8,
) -> dict[str, Any]:
    """Power of each hypothesis at each candidate N by resampling simulated people, and the smallest N reaching
    ``target_power`` (None when no candidate does)."""
    out: dict[str, Any] = {}
    people = sorted(statistics_by_person)
    rng = np.random.Generator(np.random.PCG64(seed))
    for hypothesis, rules in HYPOTHESIS_RULES.items():
        keys = [key for key, _ in rules]
        stats_of = statistics_by_person
        complete = [p for p in people if all(stats_of[p].get(k) is not None for k in keys)]
        if len(complete) < 2:
            out[hypothesis] = {"power": {}, "n_needed": None, "people": len(complete)}
            continue
        table = {key: np.array([float(stats_of[p][key]) for p in complete]) for key in keys}
        powers = {}
        for n in candidate_n:
            hits = 0
            for _ in range(resamples):
                pick = rng.integers(0, len(complete), size=n)
                hits += all(_passes(rule, table[key][pick]) for key, rule in rules)
            powers[str(n)] = hits / resamples
        needed = next((n for n in candidate_n if powers[str(n)] >= target_power), None)
        out[hypothesis] = {"power": powers, "n_needed": needed, "people": len(complete)}
    return out


def _ope_weights_for_person(run: SimulatedRunResult) -> dict[str, list[float]]:
    """Each OPE target's per-block effective weights for one simulated person (for refusal rates at N)."""
    rows = run.rows()
    key = run.participant_key
    s2 = snapshot_before_phase_b(rows, key, 2, preferences=run.preferences)
    s3 = snapshot_before_phase_b(rows, key, 3, preferences=run.preferences)
    if s2 is None or s3 is None:
        return {}
    t2 = FrozenTarget(
        s2.data,
        participant_key=key,
        settings=s2.settings,
        preferences_history=run.preferences,
        rows_hash=s2.rows_hash,
    )
    t3 = FrozenTarget(
        s3.data,
        participant_key=key,
        settings=s3.settings,
        preferences_history=run.preferences,
        rows_hash=s3.rows_hash,
    )
    sets = {
        "in_sample_S_2": (t2, logged_blocks(rows, weights="participant", sessions=(1, 2))),
        "in_sample_S_3": (t3, logged_blocks(rows, weights="participant", sessions=(1, 2, 3))),
        "out_of_sample_S_2": (
            t2,
            logged_blocks(rows, weights="participant", sessions=(3,), phases=("A_lite",)),
        ),
    }
    out: dict[str, list[float]] = {}
    for name, (target, blocks) in sets.items():
        out[name] = [u.mean_weight for u in unit_contributions(blocks, target)]
    return out


def _power_person(spec: RunSpec) -> dict[str, Any]:
    run = simulate(spec)
    return {"run": run, "stats": person_statistics(run), "ope": _ope_weights_for_person(run)}


def ope_refusal_at_n(
    weights_by_person: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    candidate_n: Sequence[int],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Median ESS and refusal rate (ESS below 30) of each OPE target when N people are resampled."""
    rng = np.random.Generator(np.random.PCG64(seed))
    people = sorted(weights_by_person)
    out: dict[str, Any] = {}
    targets = sorted({name for weights in weights_by_person.values() for name in weights})
    for name in targets:
        out[name] = {}
        for n in candidate_n:
            ess_values = []
            for _ in range(resamples):
                pick = rng.integers(0, len(people), size=n)
                chosen = [weights_by_person[people[i]].get(name, ()) for i in pick]
                w = np.concatenate([np.asarray(values, dtype=float) for values in chosen])
                ess_values.append(float(w.sum() ** 2 / np.sum(w * w)) if np.sum(w * w) > 0 else 0.0)
            ess_array = np.asarray(ess_values)
            out[name][str(n)] = {
                "median_ess": float(np.median(ess_array)),
                "refusal_rate": float(np.mean(ess_array < 30.0)),
            }
    return out


def select_confirmatory_n(
    per_family: Mapping[str, Mapping[str, Any]],
    n_families: Sequence[str],
) -> int | None:
    """The confirmatory N: the largest N any validation or holdout family (F2, F4, F6) needs for power 0.8. A
    tuning family never sets N (its power holds by construction); None when a deciding family never reaches 0.8."""
    needed = [per_family[family]["n_at_power_0_8"] for family in n_families if family in per_family]
    if not needed or any(n is None for n in needed):
        return None
    return max(needed)


def power_analysis(
    *,
    families: Sequence[str],
    people_per_family: int,
    candidate_n: Sequence[int],
    resamples: int,
    population_seed: int,
    runner_seed: int,
    workers: int,
    n_families: Sequence[str] = ("F2", "F4", "F6"),
) -> dict[str, Any]:
    """Simulation-based power for H1, H2, H3, H6, H8 and H9 per family under the primary analysis (three-trial
    burn-in), the N each family needs, the confirmatory N (the largest N needed by F2, F4 and F6, never a tuning
    family) and, at the candidate N values, the median ESS and refusal rate of each OPE target."""
    per_family: dict[str, Any] = {}
    for f_index, family in enumerate(families):
        seeds = (population_seed, runner_seed)
        specs = [RunSpec(family, i, "9_engine_full", 3, *seeds) for i in range(people_per_family)]
        persons = _map(_power_person, specs, workers)
        runs = [p["run"] for p in persons]
        stats = {p["run"].participant_key: dict(p["stats"]) for p in persons}
        for key, values in prediction_statistics(runs).items():
            stats.setdefault(key, {}).update(values)
        curve = power_curve(stats, candidate_n=candidate_n, resamples=resamples, seed=31 + f_index)
        refusal = ope_refusal_at_n(
            {p["run"].participant_key: p["ope"] for p in persons},
            candidate_n=candidate_n,
            resamples=resamples,
            seed=61 + f_index,
        )
        per_family[family] = {
            "split": split_of(family),
            "label": SPLIT_LABELS[split_of(family)],
            "people": people_per_family,
            "hypotheses": curve,
            "ope_at_n": refusal,
            "n_at_power_0_8": max(
                (h["n_needed"] for h in curve.values() if h["n_needed"] is not None),
                default=None,
            )
            if all(h["n_needed"] is not None for h in curve.values())
            else None,
        }
    deciding = [family for family in families if family in n_families]
    confirmatory = select_confirmatory_n(per_family, deciding)
    return {
        "families": per_family,
        "confirmatory_n": confirmatory,
        "confirmatory_n_from": deciding,
        "rule": "largest N needed at power 0.8 by the validation and holdout families F2, F4, F6; never a tuning family",
        "decision_rule": "two-sided 95% interval of the per-person mean difference (normal approximation)",
        "burn_in_trials": BURN_IN,
    }


# ------------------------------------------------------------------------------------------------- CLI


def _write(state_dir: Path, name: str, report: Mapping[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / name
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _power_report(protocol: Mapping[str, Any], digest: str) -> dict[str, Any]:
    power = protocol["power"]
    return {
        "protocol_sha256": digest,
        **power_analysis(
            families=tuple(protocol["families"]),
            people_per_family=int(protocol["people_per_family"]),
            candidate_n=tuple(int(n) for n in power["candidate_n"]),
            resamples=int(power["resamples"]),
            population_seed=int(protocol["population_seed"]),
            runner_seed=int(protocol["runner_seed"]),
            workers=_default_workers(),
            n_families=tuple(power["n_families"]),
        ),
        "evidence_kind": "simulated",
        "claim_boundary": "simulated users only; no claim about people",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulated-user benchmark for assistance co-adaptation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="CI run: F1 and F3, 5 people, 2 sessions, policies 1, 4, 9",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="every family and policy, reported per split (holdout look)",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="power and N per family (holdout look)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--allow-extra-look",
        action="store_true",
        help="allow a second holdout look (marked)",
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.quick and (args.full or args.power):
        parser.error("--quick runs alone; --full and --power together spend the one holdout look")
    protocol = load_protocol()
    _check_protocol(protocol)
    digest = protocol_sha256(protocol)
    try:
        if args.full or args.power:
            # One look for a --full run, a --power run, or the two together (section 18.4).
            extra = record_holdout_look(
                args.state_dir,
                digest,
                tuple(protocol["families"]),
                allow_extra_look=args.allow_extra_look,
            )
            if args.full:
                report = benchmark(quick=False, state_dir=args.state_dir, extra_look=extra)
                if args.power:
                    report["metrics"]["power"] = _power_report(protocol, digest)
                name = f"coadaptation_full_{digest[:12]}.json"
            else:
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": "coadaptation_simulated_users_power",
                    **_power_report(protocol, digest),
                    "promoted_to_runtime": False,
                }
                if extra:
                    report["extra_holdout_look"] = True
                name = f"coadaptation_power_{digest[:12]}.json"
        else:
            report = benchmark(quick=True, state_dir=args.state_dir)
            name = f"coadaptation_quick_{digest[:12]}.json"
    except HoldoutLookRefused as refused:
        parser.exit(2, f"{refused}\n")
    path = _write(args.state_dir, name, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Wrote {path}")
        print(f"claim_boundary: {report['claim_boundary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

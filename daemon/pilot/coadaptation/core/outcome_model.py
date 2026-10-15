"""The co-adaptation world model: outcome heads, person state and its dynamics (spec sections 7 and 8).

``CoAdaptationWorldModel`` predicts, for each assistance variant, the outcomes of a trial (L1 head), of an ambiguous
activation's episode (L2 head) and of a block's burden ratings (burden head), plus idle activation rates, the endpoint
spread and offer responses. It is a deterministic function of the rows it observed: learning uses no random numbers,
every time input is a stored timestamp, day boundaries come only from stored ``day_index`` values, and predictions,
samples and calibration reports never change its state. The engine (``engine.py``) owns verification, decisions and
session-scoped agency state and feeds this model through the ``observe_*`` methods; ``replay`` therefore reproduces a
live model's arrays exactly.

Model variants (spec 13.1.1) serve the benchmark and the hypotheses; the service always runs ``FULL``.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .dynamics import ALARM_WIDEN_FACTOR, Q_FAST, Q_SLOW, DayClock, DriftDetector, epoch_minutes
from .filters import (
    SIGMA_TIME,
    GammaPoisson,
    GaussianFilter,
    LatentGaussianFilter,
    OrdinalProbitFilter,
    ProbitFilter,
    SpreadInvGamma,
    in_spread_window,
    ordinal_expected,
    ordinal_expected_np,
)
from .priors import (
    BURDEN_OUTCOMES,
    POINTER_MODALITIES,
    PRIOR_TABLE_V1,
    PRIOR_VERSION,
    TRIAL_OUTCOMES,
    burden_physics,
    endpoint_sd_px,
    floor_px,
    idle_prior,
    interruptions_per_trial,
    l1_physics_prediction,
    l1_physics_update,
    l2_physics,
    spread_r0,
)
from .special import binom_sf, norm_cdf, norm_cdf_np, norm_ppf
from .state import (
    BURDEN_DIM,
    BURDEN_GAMMA,
    BURDEN_PRACTICE_SWITCH,
    FEATURE_SCHEMA,
    L1_DIM,
    L1_GAMMA,
    L1_PRACTICE_SWITCH,
    L2_DIM,
    L2_GAMMA,
    L2_PRACTICE_SWITCH,
    MODALITIES,
    TASK_KINDS,
    ActivationContext,
    BlockContext,
    ClockContext,
    DeviceContext,
    LayoutSummary,
    Rect,
    build_burden_features,
    build_l1_features,
    build_l2_features,
    device_class,
)
from .vocabulary import (
    ASSISTANCE_OFF,
    L2_OPTIONS,
    L3_OPTIONS,
    VARIANTS,
    DecisionLevel,
)

MODEL_VERSION = "coadapt-owm-v1"
__all__ = [
    "MODEL_VERSION",
    "PRIOR_VERSION",
    "CalibrationMonitor",
    "CoAdaptationWorldModel",
    "EpisodeOutcome",
    "EvidenceCounts",
    "InteractionPrediction",
    "ModelVariant",
    "OutcomeDraws",
    "OutcomeEstimate",
    "PopulationPrior",
    "PracticeCounters",
    "TrialContext",
    "TrialOutcome",
]

HEADS: tuple[str, ...] = ("l1", "l2", "burden")
HEAD_OUTCOMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"l1": TRIAL_OUTCOMES, "l2": TRIAL_OUTCOMES, "burden": BURDEN_OUTCOMES}
)
FILTER_KEYS: tuple[tuple[str, str], ...] = tuple((h, o) for h in HEADS for o in HEAD_OUTCOMES[h])
DRIFT_KEYS: tuple[tuple[str, str], ...] = tuple((h, o) for h in ("l1", "l2") for o in TRIAL_OUTCOMES)
L1_REGIMES: tuple[str, ...] = tuple(v.id for v in VARIANTS.values() if DecisionLevel.BLOCK_REGIME in v.levels)
ACTING_REGIMES_C7: frozenset[str] = frozenset({"act", "enlarge_target:1.5", "enlarge_target:2.0"})
IDLE_CLASSES: tuple[str, ...] = ("acting", "nonacting")
OFFER_OUTCOMES: tuple[str, ...] = ("accepted", "dismissed")
MAX_MENU_N = 32
C7_BUFFER = 100
C7_MIN_TRIALS = 50
C7_LEVEL = 0.05
LITERATURE_MIN = 20
TODAY_MIN = 10
Z90 = 1.6448536269514722
_ITEM_HALF_LIFE_MIN = PRIOR_TABLE_V1.item_half_life.value * 24.0 * 60.0


class ModelVariant(StrEnum):
    FULL = "full"
    NO_TODAY = "no_today"
    NO_PRACTICE_SWITCH = "no_practice_switch"
    PERSON_STATIC = "person_static"
    PHYSICS_ONLY = "physics_only"
    POPULATION = "population"
    SESSION_ONLY = "session_only"


@dataclass(frozen=True, slots=True)
class PopulationPrior:  # analysis only (H8); built by core.evaluation.population_prior
    beta_means: dict[tuple[str, str], np.ndarray]  # (head, outcome) -> prior mean of the slow block beta
    spread_r0: dict[str, float]  # modality -> prior spread factor r0
    source_participants: int  # number of other participants pooled


@dataclass(frozen=True, slots=True)
class OutcomeEstimate:
    mean: float
    lo90: float
    hi90: float
    # The 90% credible range of ``mean`` from parameter uncertainty alone, set where (lo90, hi90) is a predictive range
    # that also holds single-trial noise (time); None where (lo90, hi90) already is credible (the probabilities).
    # Never serialized (PredictionV1 keeps mean, lo90, hi90) and not compared; explanations of live decisions read it.
    credible90: tuple[float, float] | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "lo90": self.lo90, "hi90": self.hi90}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OutcomeEstimate:
        f = _fields("OutcomeEstimate", d, ("mean", "lo90", "hi90"))
        return cls(*(_num("OutcomeEstimate", f, k) for k in ("mean", "lo90", "hi90")))


EvidenceSource = Literal["literature_prior", "your_history", "today"]
_SOURCES = ("literature_prior", "your_history", "today")


@dataclass(frozen=True, slots=True)
class EvidenceCounts:
    trials_person: int
    trials_today: int
    source: EvidenceSource

    def to_dict(self) -> dict[str, Any]:
        return {"trials_person": self.trials_person, "trials_today": self.trials_today, "source": self.source}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EvidenceCounts:
        n = "EvidenceCounts"
        f = _fields(n, d, ("trials_person", "trials_today", "source"))
        source = f["source"]
        if source not in _SOURCES:
            raise ValueError(f"{n}: source has an unknown value")
        return cls(_int(n, f, "trials_person"), _int(n, f, "trials_today"), source)


@dataclass(frozen=True, slots=True)
class InteractionPrediction:
    variant: str
    level: DecisionLevel
    p_complete: OutcomeEstimate
    p_unintended: OutcomeEstimate  # hi90 here is not the safety UCB; ucb_unintended below is
    ucb_unintended: float  # UCB at config.upper_bound_credibility
    p_correction: OutcomeEstimate
    p_veto: OutcomeEstimate
    time_ms: OutcomeEstimate  # median and 90% predictive interval
    burden: dict[str, float] | None  # {"effort", "intrusiveness", "control"} expected ratings (L1 only)
    evidence: EvidenceCounts  # trials_person, trials_today, source in {"literature_prior","your_history","today"}
    authority: Literal["presentation_only"] = "presentation_only"

    def to_dict(self) -> dict[str, Any]:
        """``PredictionV1`` of spec 14.3 (the level is implied by where the prediction sits)."""
        return {
            "variant": self.variant,
            "p_complete": self.p_complete.to_dict(),
            "p_unintended": self.p_unintended.to_dict(),
            "ucb_unintended": self.ucb_unintended,
            "p_correction": self.p_correction.to_dict(),
            "p_veto": self.p_veto.to_dict(),
            "time_ms": self.time_ms.to_dict(),
            "burden": dict(self.burden) if self.burden is not None else None,
            "evidence": self.evidence.to_dict(),
            "authority": self.authority,
        }

    @classmethod
    def from_dict(
        cls, d: Mapping[str, Any], *, level: DecisionLevel = DecisionLevel.BLOCK_REGIME
    ) -> InteractionPrediction:
        n = "InteractionPrediction"
        f = _fields(
            n,
            d,
            (
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
            ),
        )
        if f["variant"] not in VARIANTS or f["authority"] != "presentation_only":
            raise ValueError(f"{n}: invalid variant or authority")
        burden = f["burden"]
        if burden is not None:
            bf = _fields(n, burden, BURDEN_OUTCOMES)
            burden = {k: _num(n, bf, k) for k in BURDEN_OUTCOMES}
        return cls(
            variant=f["variant"],
            level=DecisionLevel(level),
            p_complete=OutcomeEstimate.from_dict(f["p_complete"]),
            p_unintended=OutcomeEstimate.from_dict(f["p_unintended"]),
            ucb_unintended=_num(n, f, "ucb_unintended"),
            p_correction=OutcomeEstimate.from_dict(f["p_correction"]),
            p_veto=OutcomeEstimate.from_dict(f["p_veto"]),
            time_ms=OutcomeEstimate.from_dict(f["time_ms"]),
            burden=burden,
            evidence=EvidenceCounts.from_dict(f["evidence"]),
        )


@dataclass(frozen=True, slots=True)
class PracticeCounters:
    """Exposure counts (a read-only copy): L1 trials in total and per regime, L2 episodes per response."""

    n_total: int
    n_variant: Mapping[str, int]
    n_episode: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TrialContext:
    """The decision-free inputs of one active L1 trial, as stored in its row (update mode of spec 7.5)."""

    at: datetime  # the row's stored timestamp (TrialRecord.recorded_at): the update time
    layout: LayoutSummary
    candidates: tuple[Rect, ...]  # cue-time rects
    target_index: int  # the intended candidate
    variant: str  # the applied L1 regime
    device: DeviceContext  # the session's device (its modality is the trial's modality)
    session_minute: float  # minutes from the session start to the cue
    hand_quality: float | None  # HandSummary.quality_mean (hand pointer only)
    tau: float  # the session's ambiguity threshold

    @classmethod
    def from_trial(cls, trial: Any, *, device: DeviceContext, session_started_at: datetime, tau: float) -> TrialContext:
        """Build from a ``records.TrialRecord`` (read by its spec 13.1.2 field names): ``recorded_at``, ``layout``,
        ``candidates``, ``intended_index``, ``applied_regime``, ``cue_at`` and ``hand``."""
        if trial.intended_index is None:
            raise ValueError("idle epochs have no L1 trial context")
        hand = trial.hand
        return cls(
            at=trial.recorded_at,
            layout=trial.layout,
            candidates=tuple(trial.candidates),
            target_index=int(trial.intended_index),
            variant=trial.applied_regime,
            device=device,
            session_minute=max(0.0, (trial.cue_at - session_started_at).total_seconds() / 60.0),
            hand_quality=None if hand is None else float(hand.quality_mean),
            tau=tau,
        )


def _log_seconds(time_ms: float) -> float:
    return math.log(max(float(time_ms), 1e-3) / 1000.0)


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Labels of an L1 trial (spec 7.1). ``log_time`` is the log seconds from cue to the correct commit or, when
    ``censored``, the log of the trial limit."""

    complete: bool
    ua: bool
    correction: bool
    veto: bool
    log_time: float
    censored: bool

    @classmethod
    def from_labels(cls, labels: Any) -> TrialOutcome:
        """From ``records.TrialLabels`` (``complete``, ``ua``, ``correction``, ``veto``, ``time_ms``, ``censored``)."""
        return cls(
            complete=bool(labels.complete),
            ua=bool(labels.ua),
            correction=bool(labels.correction),
            veto=bool(labels.veto),
            log_time=_log_seconds(labels.time_ms),
            censored=bool(labels.censored),
        )


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Labels of an L2 episode: from the ambiguous activation to the trial's end (``log_time`` floored at 0.15 s
    by the caller; censored at the trial limit)."""

    complete: bool
    ua: bool
    correction: bool
    veto: bool
    log_time: float
    censored: bool

    @classmethod
    def from_labels(cls, episode: Any) -> EpisodeOutcome:
        """From ``records.EpisodeLabels`` (the same label fields as a trial)."""
        return cls(
            complete=bool(episode.complete),
            ua=bool(episode.ua),
            correction=bool(episode.correction),
            veto=bool(episode.veto),
            log_time=_log_seconds(episode.time_ms),
            censored=bool(episode.censored),
        )


@dataclass(frozen=True, slots=True)
class OutcomeDraws:
    """Thompson draws: ``values[outcome]`` has shape ``(draws, len(variants))``.

    L1 outcomes are block averages over the planned layouts: probabilities ``complete``, ``ua``, ``correction``,
    ``veto``; ``log_time`` is the latent log seconds; ``effort`` and ``intrusiveness`` are expected ratings
    (intrusiveness is 1 for ``assistance_off``). L2 draws hold the first five for one activation context.
    """

    variants: tuple[str, ...]
    values: dict[str, np.ndarray]

    def by_variant(self) -> dict[str, dict[str, np.ndarray]]:
        """``{variant: {outcome: draws}}``, the layout ``policy.py``'s samplers return."""
        return {v: {o: arr[:, i] for o, arr in self.values.items()} for i, v in enumerate(self.variants)}


# --- small validation helpers (13.1.2 rules; never echo values) ---------------------------------------------------


def _fields(cls_name: str, d: Any, names: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise ValueError(f"{cls_name}: expected an object")
    for key in d:
        if key not in names:
            raise ValueError(f"{cls_name}: unknown key")
    for name in names:
        if name not in d:
            raise ValueError(f"{cls_name}: missing key {name}")
    return d


def _num(cls_name: str, d: Mapping[str, Any], key: str) -> float:
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{cls_name}: {key} must be a finite number")
    return float(value)


def _int(cls_name: str, d: Mapping[str, Any], key: str) -> int:
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{cls_name}: {key} must be an integer")
    return value


# --- prior covariances (spec 7.2) ---------------------------------------------------------------------------------


def _l1_prior_sd(kind: str) -> np.ndarray:
    time = kind == "time"
    sd = np.empty(L1_DIM)
    sd[0] = 0.3 if time else 0.5
    sd[1:4] = 0.4 if time else 0.8
    sd[[4, 5, 8]] = 0.3
    sd[6] = 0.2
    sd[7] = 0.3
    sd[[9, 10, 11, 13]] = 0.2
    sd[12] = 0.3
    sd[14] = 0.3
    sd[15:18] = 0.15 if time else 0.35
    return sd


def _l2_prior_sd(kind: str) -> np.ndarray:
    time = kind == "time"
    sd = np.empty(L2_DIM)
    sd[0] = 0.3 if time else 0.5
    sd[[1, 2, 3, 8]] = 0.4 if time else 0.8
    sd[[4, 5, 6, 7]] = 0.3
    sd[9:12] = 0.15 if time else 0.35
    return sd


def prior_covariance(head: str, outcome: str) -> np.ndarray:
    """Diagonal prior covariance of a (head, outcome) filter (spec 7.2)."""
    kind = "time" if outcome == "log_time" else "ordinal" if head == "burden" else "probit"
    if head == "l1":
        sd = _l1_prior_sd(kind)
    elif head == "l2":
        sd = _l2_prior_sd(kind)
    elif head == "burden":
        sd = np.full(BURDEN_DIM, 0.5)
    else:
        raise ValueError("unknown head")
    return np.diag(sd**2)


_GAMMA = {"l1": L1_GAMMA, "l2": L2_GAMMA, "burden": BURDEN_GAMMA}
_PRACTICE_IDX = {"l1": L1_PRACTICE_SWITCH, "l2": L2_PRACTICE_SWITCH, "burden": BURDEN_PRACTICE_SWITCH}


def beta_indices(head: str) -> np.ndarray:
    dim = {"l1": L1_DIM, "l2": L2_DIM, "burden": BURDEN_DIM}[head]
    return np.setdiff1d(np.arange(dim), np.arange(dim)[_GAMMA[head]])


# --- calibration monitor (spec 7.7 and 14.2) ----------------------------------------------------------------------


class CalibrationMonitor:
    """Prequential calibration of ``ua``, ``complete`` and ``log_time`` (each prediction made before its outcome
    updated the state, Dawid 1984). ``report()`` returns the ``calibration`` object of ``co_adaptation_results``."""

    _BINARY = ("ua", "complete")

    def __init__(self) -> None:
        self._binary: dict[str, list[tuple[float, bool]]] = {k: [] for k in self._BINARY}
        self._covered: list[bool] = []

    def add_binary(self, outcome: str, p: float, y: bool) -> None:
        if outcome not in self._BINARY:
            raise ValueError("outcome must be ua or complete")
        self._binary[outcome].append((float(p), bool(y)))

    def add_time(self, mean: float, sd: float, y: float, *, censored: bool) -> None:
        """A log time with its predictive mean and SD. A censored time counts as outside the 90% interval when the
        censoring point lies above it, and is left out otherwise (its coverage is unknown)."""
        lo, hi = mean - Z90 * sd, mean + Z90 * sd
        if censored:
            if y > hi:
                self._covered.append(False)
            return
        self._covered.append(lo <= y <= hi)

    @staticmethod
    def _binary_report(rows: Sequence[tuple[float, bool]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"brier": None, "ece": None, "log_score": None, "n": 0, "bins": []}
        ps = np.array([p for p, _ in rows])
        ys = np.array([1.0 if y else 0.0 for _, y in rows])
        clipped = np.clip(ps, 1e-4, 1.0 - 1e-4)
        log_score = float(np.mean(ys * np.log(clipped) + (1.0 - ys) * np.log1p(-clipped)))
        brier = float(np.mean((ps - ys) ** 2))
        idx = np.minimum((ps * 10).astype(int), 9)
        bins = []
        ece = 0.0
        for b in range(10):
            mask = idx == b
            count = int(mask.sum())
            if count == 0:
                continue
            p_mean = float(ps[mask].mean())
            observed = float(ys[mask].mean())
            ece += count / n * abs(p_mean - observed)
            bins.append({"p_mean": p_mean, "observed": observed, "n": count})
        return {"brier": brier, "ece": ece, "log_score": log_score, "n": n, "bins": bins}

    def report(self) -> dict[str, Any]:
        n_time = len(self._covered)
        return {
            "ua": self._binary_report(self._binary["ua"]),
            "complete": self._binary_report(self._binary["complete"]),
            "log_time": {
                "coverage90": (sum(self._covered) / n_time) if n_time else None,
                "n": n_time,
            },
        }


# --- deterministic npz ---------------------------------------------------------------------------------------------


def write_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """An npz archive with fixed zip timestamps and sorted entries, so equal arrays give equal bytes (``np.savez``
    stamps entries with the current time)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def read_npz(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


# --- the model -----------------------------------------------------------------------------------------------------


class CoAdaptationWorldModel:
    """Person state and outcome heads for one participant (spec 7 and 8).

    Inputs are stored values only: timestamps of rows, day indices of sessions, labels derived by the engine. The
    engine calls, per session, ``begin_session``; per stored trial with verification ``ok``, ``observe_l1_trial``
    (and ``observe_l2_episode`` for each episode, ``observe_spread`` for an eligible first activation,
    ``observe_idle`` for an idle epoch); per ``fallback`` trial, ``observe_practice_only``; per burden report,
    ``observe_burden``; per offer response, ``observe_offer``; per stored event, ``apply_drift`` or
    ``apply_calibration_suspension``. ``drift_alarms`` and ``calibration_test`` are evaluated only live.
    """

    def __init__(
        self,
        *,
        variant: ModelVariant = ModelVariant.FULL,
        population_prior: PopulationPrior | None = None,
    ) -> None:
        self.variant = ModelVariant(variant)
        if self.variant is ModelVariant.POPULATION and population_prior is None:
            raise ValueError("the POPULATION variant needs a PopulationPrior")
        self.population_prior = population_prior
        self._learns = self.variant not in (ModelVariant.PHYSICS_ONLY, ModelVariant.POPULATION)
        self._today = self.variant not in (ModelVariant.NO_TODAY, ModelVariant.PERSON_STATIC)
        self._diffusion = self.variant not in (
            ModelVariant.PERSON_STATIC,
            ModelVariant.PHYSICS_ONLY,
            ModelVariant.POPULATION,
        )
        self._drift = self.variant in (
            ModelVariant.FULL,
            ModelVariant.NO_TODAY,
            ModelVariant.NO_PRACTICE_SWITCH,
            ModelVariant.SESSION_ONLY,
        )
        self._practice_features = self.variant is not ModelVariant.NO_PRACTICE_SWITCH
        self._build_learned_state()
        self._build_exposure_state()
        self._build_agency_state()
        self._reset_session_state()

    # --- construction and resets ----------------------------------------------------------------------------------

    def _new_filter(self, head: str, outcome: str) -> LatentGaussianFilter:
        cov = prior_covariance(head, outcome)
        mean = np.zeros(cov.shape[0])
        if self.population_prior is not None:
            beta = self.population_prior.beta_means.get((head, outcome))
            if beta is not None:
                idx = beta_indices(head)
                values = np.asarray(beta, dtype=np.float64)
                if values.shape != (idx.size,):
                    raise ValueError("population beta means have the wrong shape")
                mean[idx] = values
        gamma = _GAMMA[head]
        if outcome == "log_time":
            return GaussianFilter(cov, sigma=SIGMA_TIME, gamma=gamma, q_fast=Q_FAST["time"], prior_mean=mean)
        if head == "burden":
            return OrdinalProbitFilter(cov, gamma=gamma, q_fast=Q_FAST["ordinal"], prior_mean=mean)
        return ProbitFilter(cov, gamma=gamma, q_fast=Q_FAST["probit"], prior_mean=mean)

    def spread_prior(self, modality: str) -> float:
        """The spread prior ``r0`` of a pointer modality: the table's, or the PopulationPrior's when it names one."""
        if self.population_prior is not None and modality in self.population_prior.spread_r0:
            return float(self.population_prior.spread_r0[modality])
        return spread_r0(modality)

    def _build_learned_state(self) -> None:
        """Filters, spread, idle rates and drift statistics: what SESSION_ONLY restarts at every session."""
        self.filters: dict[tuple[str, str], LatentGaussianFilter] = {k: self._new_filter(*k) for k in FILTER_KEYS}
        self._filter_time = {k: math.nan for k in FILTER_KEYS}
        self.spread_filters = {m: SpreadInvGamma(self.spread_prior(m)) for m in POINTER_MODALITIES}
        self.idle = {(m, c): GammaPoisson(*idle_prior(m)) for m in MODALITIES for c in IDLE_CLASSES}
        self.detectors = {k: DriftDetector(*k) for k in DRIFT_KEYS}
        self._evidence = {
            "l1": {v: [0, 0] for v in L1_REGIMES},
            "l2": {v: [0, 0] for v in L2_OPTIONS},
        }

    def _build_exposure_state(self) -> None:
        """Practice counters, item familiarity and offer statistics (kept by SESSION_ONLY)."""
        self._n_total = 0
        self._n_variant = {v: 0 for v in L1_REGIMES}
        self._n_episode = {v: 0 for v in L2_OPTIONS}
        self._item_eff = np.zeros((MAX_MENU_N + 1, MAX_MENU_N))
        self._item_time = np.full((MAX_MENU_N + 1, MAX_MENU_N), np.nan)
        a0, b0 = PRIOR_TABLE_V1.offer_alpha.value, PRIOR_TABLE_V1.offer_beta.value
        self._offers = np.empty((len(MODALITIES), len(TASK_KINDS), len(L3_OPTIONS), len(OFFER_OUTCOMES), 2))
        self._offers[..., 0] = a0
        self._offers[..., 1] = b0

    def _build_agency_state(self) -> None:
        self._c7: deque[tuple[float, bool]] = deque(maxlen=C7_BUFFER)
        self.calibration_suspended = False
        self._suspended_day: int | None = None
        self.drift_flags: set[str] = set()
        self.day_clock = DayClock()
        self._device: tuple[str, float, int, int] | None = None

    def _reset_session_state(self) -> None:
        # Session-scoped and never snapshotted: the regime of the last observed trial, its age and whether the
        # regime switched in this session (the switch-transient feature).
        self._session_regime: str | None = None
        self._regime_age = 0
        self._switched = False

    def reset(self) -> None:
        """Learning reset: every learned array, counter and flag back to the prior."""
        self._build_learned_state()
        self._build_exposure_state()
        self._build_agency_state()
        self._reset_session_state()

    # --- sessions and dynamics ------------------------------------------------------------------------------------

    def begin_session(self, *, day_index: int, device: DeviceContext, learn_across_sessions: bool) -> None:
        """Day-boundary and device-change dynamics from stored values only (spec 8), then the resets of the
        learning setting and the model variant.

        With ``learn_across_sessions`` False the learned state (filters, spread, idle rates, drift statistics,
        evidence counts), the practice counters, ``n_item_eff`` and the offer statistics restart from the prior after
        the day dynamics, so the session starts exactly at the prior. The agency state (the C7 buffer and suspension,
        the day's drift flags, the last day index and device) follows the day dynamics only: a C7 suspension lasts
        until the next day boundary or a learning reset (``reset``), also across same-day sessions, and the day
        boundary also empties the C7 buffer whose evidence fired it. Otherwise
        SESSION_ONLY restarts the filters, spread, idle rates and drift statistics and keeps the exposure counts.
        """
        days = self.day_clock.advance(day_index)
        if days is not None:
            self._day_boundary(days)
        cls = device_class(device)
        if self._device is not None and cls != self._device:
            for flt in self.filters.values():
                flt.reset_gamma()
        self._device = cls
        if not learn_across_sessions:
            self._build_learned_state()
            self._build_exposure_state()
        elif self.variant is ModelVariant.SESSION_ONLY:
            self._build_learned_state()
        self._reset_session_state()

    def _day_boundary(self, days: int) -> None:
        if self._diffusion:
            for flt in self.filters.values():
                flt.inflate_beta(Q_SLOW[flt.kind] * days)
                flt.reset_gamma()
        elif self._learns:
            for flt in self.filters.values():
                flt.reset_gamma()
        if self._learns:
            for rate in self.idle.values():
                rate.day_decay()
        # C7 suspends acting regimes until the next day boundary (spec 7.7, 8): the evidence that fired it goes with
        # it, as the drift statistics do below. Kept, the same buffer would fire again on the first ok row of the new
        # day, whatever that row added.
        self._c7.clear()
        self.calibration_suspended = False
        self._suspended_day = None
        self.drift_flags.clear()
        for detector in self.detectors.values():
            detector.reset()
        for head in self._evidence.values():
            for counts in head.values():
                counts[1] = 0

    def end_session(self) -> None:
        self._reset_session_state()

    # --- read-only views ------------------------------------------------------------------------------------------

    @property
    def last_day_index(self) -> int | None:
        return self.day_clock.last_day_index

    def practice(self) -> PracticeCounters:
        return PracticeCounters(
            n_total=self._n_total,
            n_variant=MappingProxyType(dict(self._n_variant)),
            n_episode=MappingProxyType(dict(self._n_episode)),
        )

    def spread(self) -> dict[str, float]:
        """modality -> current spread estimate r-hat (pointer modalities; keyboard has no spread)."""
        return {m: self.spread_filters[m].r_hat for m in POINTER_MODALITIES}

    def spread_r(self, modality: str) -> float | None:
        return self.spread_filters[modality].r_hat if modality in self.spread_filters else None

    def floor(self, modality: str) -> float | None:
        return floor_px(modality) if modality in POINTER_MODALITIES else None

    def beta_means(self) -> dict[tuple[str, str], np.ndarray]:
        return {k: self.filters[k].m[beta_indices(k[0])].copy() for k in FILTER_KEYS}

    def idle_rate(self, modality: str, regime: str) -> float:
        """Posterior mean idle activations per minute for the regime's class."""
        return self.idle[(modality, _idle_class(regime))].mean

    def n_item_eff(self, menu_n: int, item_index: int, at: datetime) -> float:
        return self._item_value(menu_n, item_index, epoch_minutes(at))

    def evidence(self, head: str, variant: str) -> EvidenceCounts:
        total, today = self._evidence[head].get(variant, [0, 0])
        if total < LITERATURE_MIN:
            source: EvidenceSource = "literature_prior"
        elif today >= TODAY_MIN:
            source = "today"
        else:
            source = "your_history"
        return EvidenceCounts(trials_person=total, trials_today=today, source=source)

    # --- internals: time, exposure, features ----------------------------------------------------------------------

    def _minutes(self, key: tuple[str, str], at_min: float) -> float:
        if not self._diffusion:
            return 0.0
        last = self._filter_time[key]
        return 0.0 if math.isnan(last) else max(0.0, at_min - last)

    def _item_value(self, menu_n: int, item: int, at_min: float) -> float:
        if not (0 <= menu_n <= MAX_MENU_N and 0 <= item < MAX_MENU_N):
            return 0.0
        last = self._item_time[menu_n, item]
        if math.isnan(last):
            return 0.0
        return float(self._item_eff[menu_n, item] * 2.0 ** (-max(at_min - last, 0.0) / _ITEM_HALF_LIFE_MIN))

    def _menu_mean_eff(self, layout: LayoutSummary, at_min: float) -> float:
        if layout.task != "menu" or layout.menu_n <= 0:
            return 0.0
        n = min(layout.menu_n, MAX_MENU_N)
        return sum(self._item_value(layout.menu_n, i, at_min) for i in range(n)) / n

    def _practice_n(self, variant: str) -> int:
        return self._n_variant.get(variant, 0) if self._practice_features else 0

    def _adjust(self, head: str, f: np.ndarray) -> np.ndarray:
        if not self._today:
            f[_GAMMA[head]] = 0.0
        if not self._practice_features:
            f[list(_PRACTICE_IDX[head])] = 0.0
        return f

    def _switch_age(self, variant: str) -> int | None:
        """Switch-transient age of the next trial under ``variant``, from the session's observed trials."""
        if self._session_regime is None:
            return None
        if variant != self._session_regime:
            return 0
        return self._regime_age + 1 if self._switched else None

    def _advance_regime(self, variant: str) -> None:
        if self._session_regime is not None and variant != self._session_regime:
            self._switched = True
            self._regime_age = 0
        elif self._session_regime is not None:
            self._regime_age += 1
        self._session_regime = variant

    def _l1_update_terms(self, ctx: TrialContext) -> tuple[dict[str, float], np.ndarray]:
        at_min = epoch_minutes(ctx.at)
        layout = ctx.layout
        modality = ctx.device.modality
        item_eff = self._item_value(layout.menu_n, ctx.target_index, at_min) if layout.task == "menu" else 0.0
        prior = l1_physics_update(
            layout,
            ctx.candidates,
            ctx.target_index,
            ctx.variant,
            modality=modality,
            n_item_eff=item_eff,
            n_variant=self._practice_n(ctx.variant),
            r=self.spread_r(modality),
            tau=ctx.tau,
        )
        clock = ClockContext(
            day_index=0, session_minute=ctx.session_minute, trials_today=0, days_since_last_session=None
        )
        f = build_l1_features(
            layout, ctx.variant, ctx.device, clock, self.practice(), self._switch_age(ctx.variant), ctx.hand_quality
        )
        return {o: prior.h(o) for o in TRIAL_OUTCOMES}, self._adjust("l1", f)

    # --- predictions (never mutate) -------------------------------------------------------------------------------

    def l1_trial_prediction(self, ctx: TrialContext) -> dict[str, float]:
        """Prequential L1 quantities for a stored trial (the ones the drift detectors score): predictive
        probabilities of the binary outcomes and ``log_time_mean``/``log_time_sd`` of log seconds, from the state just
        before the trial's update, diffused to its timestamp, with the physics prior in update mode."""
        h, f = self._l1_update_terms(ctx)
        at_min = epoch_minutes(ctx.at)
        out: dict[str, float] = {}
        for outcome in TRIAL_OUTCOMES:
            flt = self.filters[("l1", outcome)]
            mu, s2 = flt.predict(f, minutes=self._minutes(("l1", outcome), at_min), h=h[outcome])
            if outcome == "log_time":
                out["log_time_mean"] = mu
                out["log_time_sd"] = math.sqrt(s2 + SIGMA_TIME**2)
            else:
                out[outcome] = norm_cdf(mu / math.sqrt(1.0 + s2))
        return out

    def _l1_layout_terms(
        self, block: BlockContext, variant: str, at_min: float, tau: float
    ) -> tuple[list[dict[str, float]], list[np.ndarray], float]:
        hs, fs, p_ambs = [], [], []
        modality = block.device.modality
        r = self.spread_r(modality)
        practice = self.practice()
        for j, layout in enumerate(block.layouts):
            prior = l1_physics_prediction(
                layout,
                variant,
                modality=modality,
                n_item_eff_mean=self._menu_mean_eff(layout, at_min),
                n_variant=self._practice_n(variant),
                r=r,
                tau=tau,
            )
            hs.append({o: prior.h(o) for o in TRIAL_OUTCOMES})
            p_ambs.append(prior.p_amb)
            if block.previous_regime is not None and variant != block.previous_regime:
                switch_age: int | None = j
            elif self._switched and self._session_regime == variant:
                switch_age = self._regime_age + 1 + j
            else:
                switch_age = None
            f = build_l1_features(
                layout, variant, block.device, block.clock, practice, switch_age, block.hand_quality_recent
            )
            fs.append(self._adjust("l1", f))
        return hs, fs, float(np.mean(p_ambs)) if p_ambs else 0.0

    def _burden_terms(self, variant: str, modality: str, ipt: float) -> tuple[dict[str, float], np.ndarray]:
        f = build_burden_features(variant, ipt, self.practice(), modality)
        return burden_physics(variant, ipt=ipt), self._adjust("burden", f)

    def predict_l1_block(
        self,
        block: BlockContext,
        variants: Sequence[str],
        *,
        now: datetime,
        tau: float,
        credibility: float,
        q_int: float | None = None,
    ) -> dict[str, InteractionPrediction]:
        """L1 predictions per regime, averaged over the block's planned layouts (spec 7.6).

        ``credibility`` sets ``ucb_unintended``; ``q_int`` is the interrupting share of the block's L2 table (0.5
        before any table exists, spec 7.4).
        """
        if not block.layouts:
            raise ValueError("a block prediction needs at least one planned layout")
        at_min = epoch_minutes(now)
        z_c = norm_ppf(credibility)
        covs = {k: self.filters[k].diffused_cov(self._minutes(k, at_min)) for k in FILTER_KEYS}
        out: dict[str, InteractionPrediction] = {}
        for variant in variants:
            hs, fs, p_amb = self._l1_layout_terms(block, variant, at_min, tau)
            per: dict[str, list[tuple[float, float]]] = {o: [] for o in TRIAL_OUTCOMES}
            for h, f in zip(hs, fs):
                for outcome in TRIAL_OUTCOMES:
                    flt = self.filters[("l1", outcome)]
                    mu = h[outcome] + float(f @ flt.m)
                    s2 = max(float(f @ covs[("l1", outcome)] @ f), 0.0)
                    per[outcome].append((mu, s2))
            binary = {o: _binary_estimate(per[o]) for o in ("complete", "ua", "correction", "veto")}
            ucb = float(np.mean([norm_cdf(mu + z_c * math.sqrt(s2)) for mu, s2 in per["ua"]]))
            ipt = interruptions_per_trial(variant, p_amb=p_amb, q_int=q_int)
            out[variant] = InteractionPrediction(
                variant=variant,
                level=DecisionLevel.BLOCK_REGIME,
                p_complete=binary["complete"],
                p_unintended=binary["ua"],
                ucb_unintended=ucb,
                p_correction=binary["correction"],
                p_veto=binary["veto"],
                time_ms=_time_estimate(per["log_time"]),
                burden=self._burden_prediction(variant, block.device.modality, ipt, covs),
                evidence=self.evidence("l1", variant),
            )
        return out

    def _burden_prediction(
        self, variant: str, modality: str, ipt: float, covs: Mapping[tuple[str, str], np.ndarray]
    ) -> dict[str, float]:
        h, f = self._burden_terms(variant, modality, ipt)
        result = {}
        for outcome in BURDEN_OUTCOMES:
            if outcome == "intrusiveness" and variant == ASSISTANCE_OFF:
                result[outcome] = 1.0  # by definition: there is no help to get in the way (spec 9.1)
                continue
            flt = self.filters[("burden", outcome)]
            mu = h[outcome] + float(f @ flt.m)
            s2 = max(float(f @ covs[("burden", outcome)] @ f), 0.0)
            result[outcome] = ordinal_expected(mu, s2)
        return result

    def burden_latent(
        self, variant: str, modality: str, *, ipt: float, now: datetime
    ) -> dict[str, tuple[float, float]]:
        """``(mu, s2)`` of each burden latent for a block regime (rule C6 reads ``control``)."""
        at_min = epoch_minutes(now)
        h, f = self._burden_terms(variant, modality, ipt)
        out = {}
        for outcome in BURDEN_OUTCOMES:
            key = ("burden", outcome)
            out[outcome] = self.filters[key].predict(f, minutes=self._minutes(key, at_min), h=h[outcome])
        return out

    def predicted_ipt(
        self, block: BlockContext, variant: str, *, now: datetime, tau: float, q_int: float | None = None
    ) -> float:
        """Predicted interruptions per trial of a regime on the block's layouts (spec 7.4)."""
        _, _, p_amb = self._l1_layout_terms(block, variant, epoch_minutes(now), tau)
        return interruptions_per_trial(variant, p_amb=p_amb, q_int=q_int)

    def sample_l1(
        self,
        block: BlockContext,
        variants: Sequence[str],
        rng: np.random.Generator,
        draws: int,
        *,
        now: datetime,
        tau: float,
        q_int: float | None = None,
    ) -> OutcomeDraws:
        """Thompson draws of the L1 block quantities for every regime (shared parameter draws across regimes).

        The generator is consumed in a fixed order (L1 complete, ua, correction, veto, log_time, then burden effort,
        intrusiveness, control), ``draws`` parameter vectors per filter, from copies diffused up to ``now``.
        """
        at_min = epoch_minutes(now)
        thetas = {}
        for key in [("l1", o) for o in TRIAL_OUTCOMES] + [("burden", o) for o in BURDEN_OUTCOMES]:
            thetas[key] = self.filters[key].sample(rng, draws, minutes=self._minutes(key, at_min))
        values = {k: np.zeros((draws, len(variants))) for k in (*TRIAL_OUTCOMES, "effort", "intrusiveness")}
        for col, variant in enumerate(variants):
            hs, fs, p_amb = self._l1_layout_terms(block, variant, at_min, tau)
            feats = np.vstack(fs)
            for outcome in TRIAL_OUTCOMES:
                hvec = np.array([h[outcome] for h in hs])
                eta = hvec[None, :] + thetas[("l1", outcome)] @ feats.T
                if outcome == "log_time":
                    values[outcome][:, col] = eta.mean(axis=1)
                else:
                    values[outcome][:, col] = norm_cdf_np(eta).mean(axis=1)
            ipt = interruptions_per_trial(variant, p_amb=p_amb, q_int=q_int)
            hb, fb = self._burden_terms(variant, block.device.modality, ipt)
            values["effort"][:, col] = ordinal_expected_np(hb["effort"] + thetas[("burden", "effort")] @ fb)
            if variant == ASSISTANCE_OFF:
                values["intrusiveness"][:, col] = 1.0
            else:
                eta_i = hb["intrusiveness"] + thetas[("burden", "intrusiveness")] @ fb
                values["intrusiveness"][:, col] = ordinal_expected_np(eta_i)
        return OutcomeDraws(variants=tuple(variants), values=values)

    def _l2_terms(self, act_ctx: ActivationContext, variant: str) -> tuple[dict[str, float], np.ndarray]:
        prior = l2_physics(
            variant,
            modality=act_ctx.modality,
            p_hit=act_ctx.p_hit,
            p_top=act_ctx.p_top,
            n_plausible=act_ctx.n_plausible,
        )
        f = build_l2_features(act_ctx, variant, self.practice())
        return {o: prior.h(o) for o in TRIAL_OUTCOMES}, self._adjust("l2", f)

    def predict_l2(
        self,
        act_ctx: ActivationContext,
        variants: Sequence[str],
        *,
        now: datetime,
        credibility: float,
        memo: dict[Any, Any] | None = None,
    ) -> dict[str, InteractionPrediction]:
        """L2 episode predictions per response for one activation context (spec 7.6).

        ``memo``, a dict the caller keeps for one ``now`` and ``credibility`` while the model does not change, reuses
        the prediction of a response whose physics prior and features repeat across activation contexts (the cells
        of a table share most of them)."""
        at_min = epoch_minutes(now)
        z_c = norm_ppf(credibility)
        keys = [("l2", outcome) for outcome in TRIAL_OUTCOMES]
        covs = {key: self.filters[key].diffused_cov(self._minutes(key, at_min)) for key in keys}
        out = {}
        for variant in variants:
            h, f = self._l2_terms(act_ctx, variant)
            memo_key = (variant, tuple(h[outcome] for outcome in TRIAL_OUTCOMES), f.tobytes())
            if memo is not None and memo_key in memo:
                out[variant] = memo[memo_key]
                continue
            per = {}
            for outcome, key in zip(TRIAL_OUTCOMES, keys):
                # LatentGaussianFilter.predict with the diffused covariance computed once per call
                mu = float(h[outcome] + f @ self.filters[key].m)
                per[outcome] = [(mu, float(max(f @ covs[key] @ f, 0.0)))]
            mu_ua, s2_ua = per["ua"][0]
            out[variant] = InteractionPrediction(
                variant=variant,
                level=DecisionLevel.AMBIGUITY_RESPONSE,
                p_complete=_binary_estimate(per["complete"]),
                p_unintended=_binary_estimate(per["ua"]),
                ucb_unintended=norm_cdf(mu_ua + z_c * math.sqrt(s2_ua)),
                p_correction=_binary_estimate(per["correction"]),
                p_veto=_binary_estimate(per["veto"]),
                time_ms=_time_estimate(per["log_time"]),
                burden=None,
                evidence=self.evidence("l2", variant),
            )
            if memo is not None:
                memo[memo_key] = out[variant]
        return out

    def sample_l2(
        self,
        act_ctx: ActivationContext,
        variants: Sequence[str],
        rng: np.random.Generator,
        draws: int,
        *,
        now: datetime,
    ) -> OutcomeDraws:
        """Thompson draws of the L2 episode quantities (L2 complete, ua, correction, veto, log_time, in that order)."""
        return self.l2_outcome_draws(act_ctx, variants, self.sample_l2_parameters(rng, draws, now=now))

    def sample_l2_parameters(
        self, rng: np.random.Generator, draws: int, *, now: datetime
    ) -> dict[tuple[str, str], np.ndarray]:
        """``draws`` parameter vectors of each L2 filter (complete, ua, correction, veto, log_time, in that order)
        from copies diffused up to ``now``; shape ``(draws, dim)`` per filter."""
        at_min = epoch_minutes(now)
        return {
            ("l2", o): self.filters[("l2", o)].sample(rng, draws, minutes=self._minutes(("l2", o), at_min))
            for o in TRIAL_OUTCOMES
        }

    def l2_outcome_draws(
        self,
        act_ctx: ActivationContext,
        variants: Sequence[str],
        thetas: Mapping[tuple[str, str], np.ndarray],
        *,
        memo: dict[Any, np.ndarray] | None = None,
    ) -> OutcomeDraws:
        """The L2 episode quantities of each response for one activation context under given parameter draws (from
        ``sample_l2_parameters``).

        A column depends on the draws, the response's physics prior and its features only. ``memo``, a dict the
        caller keeps for one set of ``thetas`` while the model does not change, reuses the columns that repeat across
        activation contexts (the cells of a table share most of them); the values are the same either way."""
        draws = int(thetas[("l2", TRIAL_OUTCOMES[0])].shape[0])
        memo = {} if memo is None else memo
        terms = [self._l2_terms(act_ctx, variant) for variant in variants]
        values = {}
        for outcome in TRIAL_OUTCOMES:
            keys = [(outcome, h[outcome], f.tobytes()) for h, f in terms]
            new = [i for i, key in enumerate(keys) if key not in memo]
            if new:
                eta = np.empty((draws, len(new)))
                for j, i in enumerate(new):
                    h, f = terms[i]
                    eta[:, j] = h[outcome] + thetas[("l2", outcome)] @ f
                if outcome != "log_time":
                    eta = norm_cdf_np(eta)  # Phi elementwise, one call for the new columns
                for j, i in enumerate(new):
                    memo[keys[i]] = eta[:, j]
            values[outcome] = np.zeros((draws, len(variants)))
            for col, key in enumerate(keys):
                values[outcome][:, col] = memo[key]
        return OutcomeDraws(variants=tuple(variants), values=values)

    def offer_means(self, modality: str, task: str, variant: str) -> dict[str, float]:
        """Posterior means of ``accepted`` and ``dismissed`` for an offer (Beta-Bernoulli, prior Beta(1, 4))."""
        stats = self._offers[_offer_index(modality, task, variant)]
        return {o: float(stats[i, 0] / (stats[i, 0] + stats[i, 1])) for i, o in enumerate(OFFER_OUTCOMES)}

    def sample_offers(
        self, modality: str, task: str, variants: Sequence[str], rng: np.random.Generator, draws: int
    ) -> OutcomeDraws:
        """Beta draws of ``accepted`` and ``dismissed`` per offer variant (variant by variant, accepted first)."""
        values = {o: np.zeros((draws, len(variants))) for o in OFFER_OUTCOMES}
        for col, variant in enumerate(variants):
            stats = self._offers[_offer_index(modality, task, variant)]
            for i, outcome in enumerate(OFFER_OUTCOMES):
                values[outcome][:, col] = rng.beta(stats[i, 0], stats[i, 1], size=draws)
        return OutcomeDraws(variants=tuple(variants), values=values)

    # --- observations (the only mutating paths besides sessions and events) --------------------------------------

    def observe_l1_trial(
        self, ctx: TrialContext, outcome: TrialOutcome, *, regime_ucb_unintended: float | None = None
    ) -> None:
        """An ``ok`` trial: prequential scores to the drift detectors, the five filter updates, the C7 buffer (acting
        regimes ``act`` and ``enlarge_target:*`` with the decision's UCB), evidence and practice counters."""
        if ctx.variant not in L1_REGIMES:
            raise ValueError("not an L1 regime")
        at_min = epoch_minutes(ctx.at)
        h, f = self._l1_update_terms(ctx)
        ys = {
            "complete": outcome.complete,
            "ua": outcome.ua,
            "correction": outcome.correction,
            "veto": outcome.veto,
        }
        if self._drift:
            pred = self.l1_trial_prediction(ctx)
            for name, y in ys.items():
                self.detectors[("l1", name)].observe_binary(y, pred[name])
            self.detectors[("l1", "log_time")].observe_time(
                outcome.log_time, pred["log_time_mean"], pred["log_time_sd"], censored=outcome.censored
            )
        if self._learns:
            for name, y in ys.items():
                key = ("l1", name)
                self.filters[key].update(f, y, minutes=self._minutes(key, at_min), h=h[name])
                self._filter_time[key] = at_min
            key = ("l1", "log_time")
            flt = self.filters[key]
            assert isinstance(flt, GaussianFilter)
            flt.update(
                f, outcome.log_time, minutes=self._minutes(key, at_min), h=h["log_time"], censored=outcome.censored
            )
            self._filter_time[key] = at_min
        if ctx.variant in ACTING_REGIMES_C7 and regime_ucb_unintended is not None:
            self._c7.append((float(regime_ucb_unintended), bool(outcome.ua)))
        counts = self._evidence["l1"][ctx.variant]
        counts[0] += 1
        counts[1] += 1
        self._advance_practice(ctx)

    def observe_practice_only(self, ctx: TrialContext) -> None:
        """A ``fallback`` trial: the person practised, so only the practice counters and ``n_item_eff`` advance."""
        if ctx.variant not in L1_REGIMES:
            raise ValueError("not an L1 regime")
        self._advance_practice(ctx)

    def _advance_practice(self, ctx: TrialContext) -> None:
        self._n_total += 1
        self._n_variant[ctx.variant] += 1
        layout = ctx.layout
        if layout.task == "menu" and 0 <= layout.menu_n <= MAX_MENU_N and 0 <= ctx.target_index < MAX_MENU_N:
            at_min = epoch_minutes(ctx.at)
            value = self._item_value(layout.menu_n, ctx.target_index, at_min)
            self._item_eff[layout.menu_n, ctx.target_index] = value + 1.0
            self._item_time[layout.menu_n, ctx.target_index] = at_min
        self._advance_regime(ctx.variant)

    def observe_l2_episode(
        self, at: datetime, act_ctx: ActivationContext, variant: str, outcome: EpisodeOutcome
    ) -> None:
        """One L2 episode of an ``ok`` trial: drift scores, the five L2 filter updates, evidence and practice."""
        if variant not in L2_OPTIONS:
            raise ValueError("not an L2 response")
        at_min = epoch_minutes(at)
        h, f = self._l2_terms(act_ctx, variant)
        ys = {
            "complete": outcome.complete,
            "ua": outcome.ua,
            "correction": outcome.correction,
            "veto": outcome.veto,
        }
        if self._drift:
            for name, y in ys.items():
                key = ("l2", name)
                mu, s2 = self.filters[key].predict(f, minutes=self._minutes(key, at_min), h=h[name])
                self.detectors[key].observe_binary(y, norm_cdf(mu / math.sqrt(1.0 + s2)))
            key = ("l2", "log_time")
            mu, s2 = self.filters[key].predict(f, minutes=self._minutes(key, at_min), h=h["log_time"])
            self.detectors[key].observe_time(
                outcome.log_time, mu, math.sqrt(s2 + SIGMA_TIME**2), censored=outcome.censored
            )
        if self._learns:
            for name, y in ys.items():
                key = ("l2", name)
                self.filters[key].update(f, y, minutes=self._minutes(key, at_min), h=h[name])
                self._filter_time[key] = at_min
            key = ("l2", "log_time")
            flt = self.filters[key]
            assert isinstance(flt, GaussianFilter)
            flt.update(
                f, outcome.log_time, minutes=self._minutes(key, at_min), h=h["log_time"], censored=outcome.censored
            )
            self._filter_time[key] = at_min
        counts = self._evidence["l2"][variant]
        counts[0] += 1
        counts[1] += 1
        self._n_episode[variant] += 1

    def observe_spread(self, modality: str, x: float, y: float, target: Rect) -> bool:
        """The first activation of an eligible ``ok`` trial (not enlarge, not keyboard; the engine checks both):
        ``True`` when it lies inside the selection-error window (spec 7.5) and so entered the update. Variants that
        do not learn report the window test without updating."""
        if modality not in POINTER_MODALITIES:
            raise ValueError("keyboard activations never update the spread")
        floor = floor_px(modality)
        sigma_x = endpoint_sd_px(target.w, 1.0, floor)
        sigma_y = endpoint_sd_px(target.h, 1.0, floor)
        dx, dy = x - target.cx, y - target.cy
        spread = self.spread_filters[modality]
        if not self._learns:
            return in_spread_window(dx, dy, sigma_x, sigma_y, spread.r0)
        return spread.observe(dx, dy, sigma_x, sigma_y)

    def observe_idle(self, modality: str, regime: str, count: int, minutes: float) -> None:
        """An idle epoch of an ``ok`` row: ``count`` commits in ``minutes`` under the preceding block's regime."""
        if self._learns:
            self.idle[(modality, _idle_class(regime))].update(count, minutes=minutes)

    def observe_burden(
        self,
        at: datetime,
        regime: str,
        modality: str,
        *,
        ipt: float,
        ratings: Mapping[str, int | None],
    ) -> None:
        """Block ratings (1..7; None = skipped or not asked). Intrusiveness is never learned for assistance_off."""
        if not self._learns:
            return
        at_min = epoch_minutes(at)
        h, f = self._burden_terms(regime, modality, ipt)
        for outcome in BURDEN_OUTCOMES:
            value = ratings.get(outcome)
            if value is None or (outcome == "intrusiveness" and regime == ASSISTANCE_OFF):
                continue
            key = ("burden", outcome)
            self.filters[key].update(f, int(value), minutes=self._minutes(key, at_min), h=h[outcome])
            self._filter_time[key] = at_min

    def observe_offer(self, modality: str, task: str, variant: str, response: str) -> None:
        """``accepted`` or ``dismissed`` updates both Beta models of the offer; ``ignored`` is missing data."""
        if response == "ignored" or not self._learns:
            return
        if response not in OFFER_OUTCOMES:
            raise ValueError("unknown offer response")
        stats = self._offers[_offer_index(modality, task, variant)]
        for i, outcome in enumerate(OFFER_OUTCOMES):
            stats[i, 0 if outcome == response else 1] += 1.0

    # --- alarms and events ----------------------------------------------------------------------------------------

    def drift_alarms(self) -> list[tuple[str, str]]:
        """(head, outcome) of every detector whose statistic reached its threshold (live observation only)."""
        if not self._drift:
            return []
        return [k for k in DRIFT_KEYS if self.detectors[k].alarm]

    def apply_drift(self, head: str, outcome: str) -> None:
        """Drift alarm or stored drift event: widen that filter's slow block (at most to its prior), reset its
        detector and set the day's drift flag (spec 8)."""
        key = (head, outcome)
        if key not in self.detectors:
            raise ValueError("drift applies to the L1 and L2 heads only")
        if self._learns:
            flt = self.filters[key]
            flt.widen_beta(ALARM_WIDEN_FACTOR, flt.prior_cov)
        self.detectors[key].reset()
        self.drift_flags.add(f"{head}:{outcome}")

    def calibration_test(self) -> bool:
        """Rule C7 (spec 7.7): True when not yet suspended, the buffer holds at least 50 trials and
        ``binom_sf(x_obs - 1, n, mean UCB) < 0.05``."""
        if self.calibration_suspended or len(self._c7) < C7_MIN_TRIALS:
            return False
        n = len(self._c7)
        x_obs = sum(1 for _, ua in self._c7 if ua)
        mean_ucb = math.fsum(u for u, _ in self._c7) / n
        return binom_sf(x_obs - 1, n, min(max(mean_ucb, 0.0), 1.0)) < C7_LEVEL

    def apply_calibration_suspension(self) -> None:
        """C7 alarm or stored event: acting regimes are suspended until the next day boundary or learning reset."""
        self.calibration_suspended = True
        self._suspended_day = self.day_clock.last_day_index

    # --- snapshot -------------------------------------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "prior_version": PRIOR_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "variant": self.variant.value,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Learned state only (spec 10.6), as float64 arrays with fixed shapes."""
        arrays: dict[str, np.ndarray] = {}
        for head, outcome in FILTER_KEYS:
            flt = self.filters[(head, outcome)].to_arrays()
            arrays[f"filter.{head}.{outcome}.m"] = flt["m"]
            arrays[f"filter.{head}.{outcome}.P"] = flt["P"]
        arrays["filter_time"] = np.array([self._filter_time[k] for k in FILTER_KEYS])
        arrays["spread"] = np.array([self.spread_filters[m].to_arrays()["ab"] for m in POINTER_MODALITIES])
        arrays["idle"] = np.array([[self.idle[(m, c)].to_arrays()["ab"] for c in IDLE_CLASSES] for m in MODALITIES])
        arrays["offers"] = self._offers.copy()
        arrays["practice_total"] = np.array([float(self._n_total)])
        arrays["practice_variant"] = np.array([float(self._n_variant[v]) for v in L1_REGIMES])
        arrays["practice_episode"] = np.array([float(self._n_episode[v]) for v in L2_OPTIONS])
        arrays["item_eff"] = self._item_eff.copy()
        arrays["item_time"] = self._item_time.copy()
        arrays["drift"] = np.array([self.detectors[k].to_array() for k in DRIFT_KEYS])
        arrays["drift_flags"] = np.array([1.0 if f"{h}:{o}" in self.drift_flags else 0.0 for h, o in DRIFT_KEYS])
        buffer = np.full((C7_BUFFER, 2), np.nan)
        for i, (ucb, ua) in enumerate(self._c7):
            buffer[i] = (ucb, 1.0 if ua else 0.0)
        arrays["c7_buffer"] = buffer
        arrays["c7_state"] = np.array(
            [
                float(len(self._c7)),
                1.0 if self.calibration_suspended else 0.0,
                np.nan if self._suspended_day is None else float(self._suspended_day),
            ]
        )
        last_day = self.day_clock.last_day_index
        arrays["day"] = np.array([np.nan if last_day is None else float(last_day)])
        if self._device is None:
            arrays["device"] = np.full(4, np.nan)
        else:
            mod, dpr, cw, ch = self._device
            arrays["device"] = np.array([float(MODALITIES.index(mod)), dpr, float(cw), float(ch)])
        arrays["evidence_l1"] = np.array([self._evidence["l1"][v] for v in L1_REGIMES], dtype=np.float64)
        arrays["evidence_l2"] = np.array([self._evidence["l2"][v] for v in L2_OPTIONS], dtype=np.float64)
        return arrays

    def from_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Load ``to_arrays`` output after strict key and shape checks; the session-scoped state starts empty."""
        expected = self.to_arrays()
        if set(arrays) != set(expected):
            raise ValueError("snapshot keys do not match this model")
        for name, ref in expected.items():
            if np.asarray(arrays[name]).shape != ref.shape:
                raise ValueError("snapshot array has the wrong shape")
        a = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
        for head, outcome in FILTER_KEYS:
            self.filters[(head, outcome)].from_arrays(
                {"m": a[f"filter.{head}.{outcome}.m"], "P": a[f"filter.{head}.{outcome}.P"]}
            )
        self._filter_time = {k: float(a["filter_time"][i]) for i, k in enumerate(FILTER_KEYS)}
        for i, m in enumerate(POINTER_MODALITIES):
            self.spread_filters[m].from_arrays({"ab": a["spread"][i]})
        for i, m in enumerate(MODALITIES):
            for j, c in enumerate(IDLE_CLASSES):
                self.idle[(m, c)].from_arrays({"ab": a["idle"][i, j]})
        if not np.all(np.isfinite(a["offers"])) or np.any(a["offers"] <= 0):
            raise ValueError("offer statistics must be positive")
        self._offers = a["offers"].copy()
        self._n_total = _count(a["practice_total"][0])
        self._n_variant = {v: _count(a["practice_variant"][i]) for i, v in enumerate(L1_REGIMES)}
        self._n_episode = {v: _count(a["practice_episode"][i]) for i, v in enumerate(L2_OPTIONS)}
        self._item_eff = a["item_eff"].copy()
        self._item_time = a["item_time"].copy()
        for i, k in enumerate(DRIFT_KEYS):
            self.detectors[k].from_array(a["drift"][i])
        self.drift_flags = {f"{h}:{o}" for i, (h, o) in enumerate(DRIFT_KEYS) if a["drift_flags"][i] == 1.0}
        count = _count(a["c7_state"][0])
        if count > C7_BUFFER:
            raise ValueError("C7 buffer count out of range")
        self._c7 = deque(((float(u), bool(y == 1.0)) for u, y in a["c7_buffer"][:count]), maxlen=C7_BUFFER)
        self.calibration_suspended = a["c7_state"][1] == 1.0
        self._suspended_day = None if math.isnan(a["c7_state"][2]) else int(a["c7_state"][2])
        self.day_clock = DayClock(None if math.isnan(a["day"][0]) else int(a["day"][0]))
        if math.isnan(a["device"][0]):
            self._device = None
        else:
            mod_index = int(a["device"][0])
            if not 0 <= mod_index < len(MODALITIES):
                raise ValueError("device modality out of range")
            self._device = (MODALITIES[mod_index], float(a["device"][1]), int(a["device"][2]), int(a["device"][3]))
        self._evidence = {
            "l1": {
                v: [_count(a["evidence_l1"][i, 0]), _count(a["evidence_l1"][i, 1])] for i, v in enumerate(L1_REGIMES)
            },
            "l2": {
                v: [_count(a["evidence_l2"][i, 0]), _count(a["evidence_l2"][i, 1])] for i, v in enumerate(L2_OPTIONS)
            },
        }
        self._reset_session_state()

    def snapshot(self) -> bytes:
        """npz bytes of ``to_arrays()`` plus a JSON ``metadata`` array; equal states give equal bytes."""
        arrays = dict(self.to_arrays())
        arrays["metadata"] = np.array(json.dumps(self.metadata(), sort_keys=True))
        return write_npz(arrays)

    @classmethod
    def restore(
        cls,
        data: bytes,
        *,
        variant: ModelVariant = ModelVariant.FULL,
        population_prior: PopulationPrior | None = None,
    ) -> CoAdaptationWorldModel:
        """Load a snapshot with ``allow_pickle=False``; versions, variant, keys and shapes must match exactly."""
        try:
            arrays = read_npz(data)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError("snapshot is not a valid npz archive") from exc
        if "metadata" not in arrays:
            raise ValueError("snapshot has no metadata")
        try:
            meta = json.loads(str(arrays.pop("metadata").item()))
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot metadata is not valid JSON") from exc
        model = cls(variant=variant, population_prior=population_prior)
        if meta != model.metadata():
            raise ValueError("snapshot versions or variant do not match this model")
        model.from_arrays(arrays)
        return model


def _count(value: float) -> int:
    if not math.isfinite(value) or value < 0 or value != math.floor(value):
        raise ValueError("counts must be non-negative integers")
    return int(value)


def _idle_class(regime: str) -> str:
    return "acting" if VARIANTS[regime].acting else "nonacting"


def _offer_index(modality: str, task: str, variant: str) -> tuple[int, int, int]:
    if modality not in MODALITIES or task not in TASK_KINDS or variant not in L3_OPTIONS:
        raise ValueError("unknown offer context")
    return MODALITIES.index(modality), TASK_KINDS.index(task), L3_OPTIONS.index(variant)


def _layout_mean(values: list[float]) -> float:
    # np.mean of one value is that value; skipping numpy there keeps L2 predictions (one "layout") cheap
    return values[0] if len(values) == 1 else float(np.mean(values))


def _binary_estimate(per_layout: Sequence[tuple[float, float]]) -> OutcomeEstimate:
    """Mean ``Phi(mu / sqrt(1 + s2))`` and 90% credible bounds ``Phi(mu -+ 1.645 s)``, averaged over layouts."""
    means, los, his = [], [], []
    for mu, s2 in per_layout:
        s = math.sqrt(s2)
        means.append(norm_cdf(mu / math.sqrt(1.0 + s2)))
        los.append(norm_cdf(mu - Z90 * s))
        his.append(norm_cdf(mu + Z90 * s))
    return OutcomeEstimate(mean=_layout_mean(means), lo90=_layout_mean(los), hi90=_layout_mean(his))


def _time_estimate(per_layout: Sequence[tuple[float, float]]) -> OutcomeEstimate:
    """Median ``exp(mu)`` ms and the 90% predictive interval ``exp(mu -+ 1.645 sqrt(s2 + sigma_T^2))``, with the
    90% credible range of the median ``exp(mu -+ 1.645 sqrt(s2))`` (no trial noise) as ``credible90``."""
    meds, los, his, mlos, mhis = [], [], [], [], []
    for mu, s2 in per_layout:
        sd = math.sqrt(s2 + SIGMA_TIME**2)
        s = math.sqrt(s2)
        meds.append(math.exp(mu) * 1000.0)
        los.append(math.exp(mu - Z90 * sd) * 1000.0)
        his.append(math.exp(mu + Z90 * sd) * 1000.0)
        mlos.append(math.exp(mu - Z90 * s) * 1000.0)
        mhis.append(math.exp(mu + Z90 * s) * 1000.0)
    return OutcomeEstimate(
        mean=_layout_mean(meds),
        lo90=_layout_mean(los),
        hi90=_layout_mean(his),
        credible90=(_layout_mean(mlos), _layout_mean(mhis)),
    )

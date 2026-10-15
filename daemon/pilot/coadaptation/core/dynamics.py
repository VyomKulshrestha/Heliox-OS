"""Co-adaptation dynamics: drift detection and timescale constants (spec section 8).

Drift uses risk-adjusted CUSUMs on log-likelihood ratios of the prequential prediction: a Bernoulli CUSUM for odds
ratios 2 and 1/2 (Steiner, Cook, Farewell and Treasure 2000) for binary outcomes, and Gaussian mean-shift CUSUMs of
+-0.5 predictive SD for log time, where a right-censored time contributes the log-likelihood ratio of ``y >= l``. The
thresholds come from simulated in-control run lengths (``calibrate_drift_thresholds``); ``DRIFT_THRESHOLDS`` holds the
committed output of ``python -m pilot.coadaptation.core.dynamics --calibrate-thresholds``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from .special import log_norm_cdf

DRIFT_THRESHOLDS: dict[str, float] = {"binary": 7.65, "log_time": 8.55}
R_UP = 2.0
R_DOWN = 0.5
DELTA = 0.5
P_GRID: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995)
C_GRID: tuple[float, ...] = (math.inf, 2.0, 1.0)  # censoring points in predictive SD: 0%, 2.3% and 15.9% censored
P_CLIP = 1e-4
CALIBRATION_SEED = 20261015

# Diffusion per filter kind: Q_SLOW is added to each beta variance per day at a day boundary (at most 30 days);
# Q_FAST is added to each gamma variance per minute within a day, capped at the prior.
Q_SLOW: dict[str, float] = {"time": 0.03**2, "probit": 0.08**2, "ordinal": 0.08**2}
Q_FAST: dict[str, float] = {"time": 0.01**2, "probit": 0.02**2, "ordinal": 0.02**2}
MAX_BOUNDARY_DAYS = 30
ALARM_WIDEN_FACTOR = 4.0

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def epoch_minutes(at: datetime) -> float:
    """Minutes since 1970-01-01 UTC of an offset-carrying datetime (a stored row timestamp, never a clock read)."""
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("timestamps must carry a UTC offset")
    return (at - _EPOCH).total_seconds() / 60.0


def bernoulli_score(y: bool | int, p: float, r: float) -> float:
    """Risk-adjusted Bernoulli CUSUM score ``y ln R - ln(1 - p + R p)`` (p clipped to [1e-4, 1 - 1e-4])."""
    p = min(max(p, P_CLIP), 1.0 - P_CLIP)
    return (math.log(r) if y else 0.0) - math.log(1.0 - p + r * p)


def gaussian_score(z: float, delta: float) -> float:
    """Mean-shift score of a completed observation: ``delta z - delta^2 / 2``."""
    return delta * z - delta * delta / 2.0


def censored_score(c: float, delta: float) -> float:
    """Score of a right-censored observation (``z >= c``): ``ln Phi(delta - c) - ln Phi(-c)``."""
    return log_norm_cdf(delta - c) - log_norm_cdf(-c)


class BernoulliCusum:
    """Two one-sided risk-adjusted Bernoulli CUSUMs (odds ratio ``r_up`` and ``r_down``)."""

    def __init__(self, r_up: float = R_UP, r_down: float = R_DOWN, *, threshold: float | None = None) -> None:
        self.r_up = float(r_up)
        self.r_down = float(r_down)
        self.threshold = DRIFT_THRESHOLDS["binary"] if threshold is None else float(threshold)
        self.s_up = 0.0
        self.s_down = 0.0

    @property
    def statistic(self) -> float:
        return max(self.s_up, self.s_down)

    def update(self, y: bool | int, p: float) -> bool:
        """Feed one outcome with its predictive probability; True when either chart reaches the threshold."""
        self.s_up = max(0.0, self.s_up + bernoulli_score(y, p, self.r_up))
        self.s_down = max(0.0, self.s_down + bernoulli_score(y, p, self.r_down))
        return self.alarm

    @property
    def alarm(self) -> bool:
        return self.statistic >= self.threshold

    def reset(self) -> None:
        self.s_up = 0.0
        self.s_down = 0.0


class GaussianCusum:
    """Two one-sided mean-shift CUSUMs of ``+delta`` and ``-delta`` predictive SDs, with censored scores."""

    def __init__(self, delta: float = DELTA, *, threshold: float | None = None) -> None:
        self.delta = float(delta)
        self.threshold = DRIFT_THRESHOLDS["log_time"] if threshold is None else float(threshold)
        self.s_up = 0.0
        self.s_down = 0.0

    @property
    def statistic(self) -> float:
        return max(self.s_up, self.s_down)

    def update(self, y: float, mu: float, sd: float, *, censored: bool = False) -> bool:
        """Feed one log time (or, with ``censored``, its censoring point); True when either chart alarms."""
        z = (y - mu) / sd
        if censored:
            w_up, w_down = censored_score(z, self.delta), censored_score(z, -self.delta)
        else:
            w_up, w_down = gaussian_score(z, self.delta), gaussian_score(z, -self.delta)
        self.s_up = max(0.0, self.s_up + w_up)
        self.s_down = max(0.0, self.s_down + w_down)
        return self.alarm

    @property
    def alarm(self) -> bool:
        return self.statistic >= self.threshold

    def reset(self) -> None:
        self.s_up = 0.0
        self.s_down = 0.0


class DriftDetector:
    """One detector per (head, outcome) of the L1 trial head and the L2 episode head (ten in total).

    Statistics restart at a day boundary, after a learning reset and after an alarm of this detector. Feeding never
    resets anything: whether an alarm fires is the caller's decision (live observation fires, replay never does).
    """

    def __init__(self, head: str, outcome: str) -> None:
        self.head = head
        self.outcome = outcome
        self.kind = "log_time" if outcome == "log_time" else "binary"
        self.chart: BernoulliCusum | GaussianCusum = GaussianCusum() if self.kind == "log_time" else BernoulliCusum()

    @property
    def key(self) -> str:
        return f"{self.head}:{self.outcome}"

    def observe_binary(self, y: bool | int, p: float) -> bool:
        if not isinstance(self.chart, BernoulliCusum):
            raise ValueError("not a binary detector")
        return self.chart.update(y, p)

    def observe_time(self, y: float, mu: float, sd: float, *, censored: bool) -> bool:
        if not isinstance(self.chart, GaussianCusum):
            raise ValueError("not a log_time detector")
        return self.chart.update(y, mu, sd, censored=censored)

    @property
    def alarm(self) -> bool:
        return self.chart.alarm

    def reset(self) -> None:
        self.chart.reset()

    def to_array(self) -> np.ndarray:
        return np.array([self.chart.s_up, self.chart.s_down])

    def from_array(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("drift statistics have the wrong shape or values")
        self.chart.s_up, self.chart.s_down = float(values[0]), float(values[1])


@dataclass
class DayClock:
    """The last stored day index the model saw; day boundaries come only from stored ``day_index`` values."""

    last_day_index: int | None = None

    def advance(self, day_index: int) -> int | None:
        """Record a session's day index; the number of days for the boundary dynamics (``min(|delta|, 30)``), or
        None when this is the first day seen or the same day."""
        previous = self.last_day_index
        self.last_day_index = int(day_index)
        if previous is None or previous == day_index:
            return None
        return min(abs(int(day_index) - previous), MAX_BOUNDARY_DAYS)


# --- threshold calibration (spec 8) -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftCalibration:
    thresholds: dict[str, float]
    # kind -> [(grid level, false-alarm fraction at the chosen threshold)]; level is p (binary) or c (log_time)
    false_alarm: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


def _binary_scores(p: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """((score if y = 1, score if y = 0) for R_UP, the same for R_DOWN)."""
    return (
        (bernoulli_score(1, p, R_UP), bernoulli_score(0, p, R_UP)),
        (bernoulli_score(1, p, R_DOWN), bernoulli_score(0, p, R_DOWN)),
    )


def _score_matrices(kind: str, level: float, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation scores of the up and down charts for a matrix of raw draws."""
    if kind == "binary":
        (up1, up0), (dn1, dn0) = _binary_scores(level)
        y = draws < level
        return np.where(y, up1, up0), np.where(y, dn1, dn0)
    if kind == "log_time":
        z = draws
        up = DELTA * z - DELTA * DELTA / 2.0
        down = -DELTA * z - DELTA * DELTA / 2.0
        if math.isfinite(level):
            censored = z >= level
            up = np.where(censored, censored_score(level, DELTA), up)
            down = np.where(censored, censored_score(level, -DELTA), down)
        return up, down
    raise ValueError("kind must be binary or log_time")


def _draws(kind: str, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    return rng.random(shape) if kind == "binary" else rng.standard_normal(shape)


def _run_maxima(kind: str, level: float, rng: np.random.Generator, runs: int, n: int) -> np.ndarray:
    up_w, down_w = _score_matrices(kind, level, _draws(kind, rng, (runs, n)))
    s_up = np.zeros(runs)
    s_down = np.zeros(runs)
    peak = np.zeros(runs)
    for t in range(n):
        s_up = np.maximum(0.0, s_up + up_w[:, t])
        s_down = np.maximum(0.0, s_down + down_w[:, t])
        peak = np.maximum(peak, np.maximum(s_up, s_down))
    return peak


def _smallest_threshold(maxima: Sequence[np.ndarray], target: float, step: float) -> float:
    k = 1
    while True:
        h = round(k * step, 10)
        if all(float(np.mean(m >= h)) <= target for m in maxima):
            return h
        k += 1


def calibrate_drift_thresholds(
    *,
    seed: int = CALIBRATION_SEED,
    runs: int = 20000,
    n: int = 200,
    target: float = 0.005,
    step: float = 0.05,
) -> DriftCalibration:
    """Thresholds from simulated in-control run lengths (spec 8).

    For the i-th rate p of ``P_GRID`` the outcomes are ``Generator(PCG64(seed + i)).random((runs, n)) < p``; for the
    j-th censoring point c of ``C_GRID`` the standardized log times are ``Generator(PCG64(seed + 100 + j))
    .standard_normal((runs, n))``, censored where ``z >= c``. Both charts run from S = 0 with the calibrated
    prediction, each run records its maximum of ``max(S+, S-)``, and ``h`` is the smallest multiple of ``step`` whose
    false-alarm fraction (maxima reaching ``h``) is at most ``target`` at every grid point.
    """
    binary_max = [
        _run_maxima("binary", p, np.random.Generator(np.random.PCG64(seed + i)), runs, n) for i, p in enumerate(P_GRID)
    ]
    time_max = [
        _run_maxima("log_time", c, np.random.Generator(np.random.PCG64(seed + 100 + j)), runs, n)
        for j, c in enumerate(C_GRID)
    ]
    h_binary = _smallest_threshold(binary_max, target, step)
    h_time = _smallest_threshold(time_max, target, step)
    return DriftCalibration(
        thresholds={"binary": h_binary, "log_time": h_time},
        false_alarm={
            "binary": [(p, float(np.mean(m >= h_binary))) for p, m in zip(P_GRID, binary_max)],
            "log_time": [(c, float(np.mean(m >= h_time))) for c, m in zip(C_GRID, time_max)],
        },
    )


def in_control_run_lengths(
    kind: str, level: float, h: float, *, runs: int, cap: int, seed: int, chunk: int = 2000
) -> np.ndarray:
    """Observations until the first alarm of either chart (1-based), from S = 0, with outcomes drawn from the
    predicted rate ``level`` (binary: the rate p; log_time: the censoring point c). A run without an alarm within
    ``cap`` observations is recorded as ``cap``. Draws come from ``Generator(PCG64(seed))`` in column chunks."""
    rng = np.random.Generator(np.random.PCG64(seed))
    s_up = np.zeros(runs)
    s_down = np.zeros(runs)
    lengths = np.full(runs, cap, dtype=np.int64)
    open_runs = np.ones(runs, dtype=bool)
    t = 0
    while t < cap and open_runs.any():
        width = min(chunk, cap - t)
        up_w, down_w = _score_matrices(kind, level, _draws(kind, rng, (runs, width)))
        for j in range(width):
            s_up = np.maximum(0.0, s_up + up_w[:, j])
            s_down = np.maximum(0.0, s_down + down_w[:, j])
            fired = open_runs & (np.maximum(s_up, s_down) >= h)
            if fired.any():
                lengths[fired] = t + j + 1
                open_runs &= ~fired
        t += width
    return lengths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the drift CUSUM thresholds (spec 8).")
    parser.add_argument("--calibrate-thresholds", action="store_true", required=True)
    parser.parse_args(argv)
    result = calibrate_drift_thresholds()
    print(f"binary threshold: {result.thresholds['binary']:.2f}")
    print(f"log_time threshold: {result.thresholds['log_time']:.2f}")
    for p, fa in result.false_alarm["binary"]:
        print(f"  binary p={p:<6} false-alarm fraction {fa:.5f}")
    for c, fa in result.false_alarm["log_time"]:
        print(f"  log_time c={c:<6} false-alarm fraction {fa:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

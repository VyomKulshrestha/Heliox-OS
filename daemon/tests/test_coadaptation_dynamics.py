"""Drift detection (spec section 8): CUSUM scores, threshold calibration, run lengths and restarts."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy import integrate, stats

from pilot.coadaptation.core.dynamics import (
    C_GRID,
    DRIFT_THRESHOLDS,
    P_GRID,
    Q_FAST,
    Q_SLOW,
    BernoulliCusum,
    DayClock,
    DriftDetector,
    GaussianCusum,
    bernoulli_score,
    calibrate_drift_thresholds,
    censored_score,
    gaussian_score,
    in_control_run_lengths,
)
from pilot.coadaptation.core.outcome_model import CoAdaptationWorldModel, TrialContext, TrialOutcome
from pilot.coadaptation.core.special import norm_cdf
from pilot.coadaptation.core.state import DeviceContext, LayoutSummary, Rect


def test_binary_scores_follow_the_formula() -> None:
    for p in (0.02, 0.1, 0.5, 0.9):
        for r in (2.0, 0.5):
            assert bernoulli_score(1, p, r) == pytest.approx(math.log(r) - math.log(1 - p + r * p))
            assert bernoulli_score(0, p, r) == pytest.approx(-math.log(1 - p + r * p))
    assert bernoulli_score(1, 0.0, 2.0) == pytest.approx(math.log(2.0) - math.log(1 + 1e-4))
    chart = BernoulliCusum()
    chart.update(1, 0.1)
    assert chart.s_up == pytest.approx(math.log(2) - math.log(1.1))
    assert chart.s_down == 0.0  # max(0, log(0.5) - log(0.95)) = 0


def test_time_scores_follow_the_formula() -> None:
    for z in (-2.0, 0.0, 1.3):
        assert gaussian_score(z, 0.5) == pytest.approx(0.5 * z - 0.125)
        assert gaussian_score(z, -0.5) == pytest.approx(-0.5 * z - 0.125)
    for c in (-1.0, 0.5, 2.0, 6.0):
        for delta in (0.5, -0.5):
            expected = math.log(norm_cdf(delta - c)) - math.log(norm_cdf(-c))
            assert censored_score(c, delta) == pytest.approx(expected, rel=1e-12)
            shifted, _ = integrate.quad(lambda z, d=delta: stats.norm.pdf(z - d), c, np.inf, epsabs=0.0, epsrel=1e-12)
            null, _ = integrate.quad(stats.norm.pdf, c, np.inf, epsabs=0.0, epsrel=1e-12)
            assert censored_score(c, delta) == pytest.approx(math.log(shifted / null), rel=1e-8)
    assert math.isfinite(censored_score(40.0, 0.5))
    chart = GaussianCusum()
    chart.update(1.5, 1.0, 0.5, censored=True)
    assert chart.s_up == pytest.approx(censored_score(1.0, 0.5))
    assert chart.s_down == 0.0


def test_calibration_reproduces_the_committed_thresholds() -> None:
    result = calibrate_drift_thresholds()
    assert result.thresholds == DRIFT_THRESHOLDS == {"binary": 7.65, "log_time": 8.55}
    assert [p for p, _ in result.false_alarm["binary"]] == list(P_GRID)
    assert [c for c, _ in result.false_alarm["log_time"]] == list(C_GRID)
    assert all(fa <= 0.005 for _, fa in result.false_alarm["binary"] + result.false_alarm["log_time"])


def test_run_lengths_on_fresh_streams() -> None:
    base = 30261015
    for i, p in enumerate(P_GRID):
        lengths = in_control_run_lengths("binary", p, DRIFT_THRESHOLDS["binary"], runs=20_000, cap=201, seed=base + i)
        assert float(np.mean(lengths <= 200)) <= 0.0065, p
    for j, c in enumerate(C_GRID):
        lengths = in_control_run_lengths(
            "log_time", c, DRIFT_THRESHOLDS["log_time"], runs=20_000, cap=201, seed=base + 100 + j
        )
        assert float(np.mean(lengths <= 200)) <= 0.0065, c


def test_capped_average_run_length() -> None:
    binary = in_control_run_lengths("binary", 0.5, DRIFT_THRESHOLDS["binary"], runs=1000, cap=50_000, seed=20261017)
    timed = in_control_run_lengths(
        "log_time", math.inf, DRIFT_THRESHOLDS["log_time"], runs=1000, cap=50_000, seed=20261017
    )
    assert float(np.mean(binary)) >= 15_000
    assert float(np.mean(timed)) >= 15_000


def test_detection_power_for_a_shift_in_the_day() -> None:
    rng = np.random.Generator(np.random.PCG64(20261018))
    p0 = 0.10
    odds = 4 * p0 / (1 - p0)
    p1 = odds / (1 + odds)
    detected = 0
    for _ in range(2000):
        chart = BernoulliCusum()
        ys = np.concatenate([rng.random(100) < p0, rng.random(100) < p1])
        if any(chart.update(bool(y), p0) for y in ys):
            detected += 1
    assert detected / 2000 >= 0.90
    detected = 0
    for _ in range(2000):
        chart = GaussianCusum()
        zs = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 0.5])
        if any(chart.update(float(z), 0.0, 1.0) for z in zs):
            detected += 1
    assert detected / 2000 >= 0.85


def test_day_clock_and_constants() -> None:
    clock = DayClock()
    assert clock.advance(740_000) is None
    assert clock.advance(740_000) is None
    assert clock.advance(740_003) == 3
    assert clock.advance(740_100) == 30
    assert Q_SLOW == {"time": 0.03**2, "probit": 0.08**2, "ordinal": 0.08**2}
    assert Q_FAST == {"time": 0.01**2, "probit": 0.02**2, "ordinal": 0.02**2}


def _feed(model: CoAdaptationWorldModel, start: datetime, n: int) -> None:
    device = DeviceContext("mouse", ("mouse",), 1.0, 700.0, 500.0, False, False)
    layout = LayoutSummary("pointing", 2, 28.0, 28.0, 8.0, 3.0, 0, 350.0, 470.0)
    cands = (Rect(332.0, 200.0, 28.0, 28.0), Rect(368.0, 200.0, 28.0, 28.0))
    for i in range(n):
        ctx = TrialContext(start + timedelta(minutes=i), layout, cands, i % 2, "act", device, float(i), None, 0.25)
        # Every trial a slip: the ua chart climbs quickly.
        model.observe_l1_trial(ctx, TrialOutcome(False, True, True, False, math.log(10.0), True))


def test_detector_statistics_restart() -> None:
    device = DeviceContext("mouse", ("mouse",), 1.0, 700.0, 500.0, False, False)
    t0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
    model = CoAdaptationWorldModel()
    model.begin_session(day_index=740_000, device=device, learn_across_sessions=True)
    _feed(model, t0, 6)
    assert model.detectors[("l1", "ua")].chart.statistic > 0
    model.begin_session(day_index=740_001, device=device, learn_across_sessions=True)  # day boundary
    assert all(d.chart.statistic == 0.0 for d in model.detectors.values())
    _feed(model, t0 + timedelta(days=1), 6)
    model.reset()  # learning reset
    assert all(d.chart.statistic == 0.0 for d in model.detectors.values())
    model.begin_session(day_index=740_002, device=device, learn_across_sessions=True)
    _feed(model, t0 + timedelta(days=2), 30)
    alarms = model.drift_alarms()
    assert ("l1", "ua") in alarms
    other = model.detectors[("l1", "complete")].chart.statistic
    model.apply_drift("l1", "ua")  # after an alarm only that detector restarts
    assert model.detectors[("l1", "ua")].chart.statistic == 0.0
    assert model.detectors[("l1", "complete")].chart.statistic == other
    assert "l1:ua" in model.drift_flags


def test_detector_kinds() -> None:
    assert DriftDetector("l1", "ua").kind == "binary"
    timed = DriftDetector("l2", "log_time")
    assert timed.kind == "log_time"
    with pytest.raises(ValueError):
        timed.observe_binary(True, 0.5)
    timed.observe_time(3.0, 0.0, 1.0, censored=False)
    restored = DriftDetector("l2", "log_time")
    restored.from_array(timed.to_array())
    assert restored.chart.statistic == timed.chart.statistic

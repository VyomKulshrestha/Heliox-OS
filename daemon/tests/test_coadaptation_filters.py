"""Online filters (spec section 7.5): closed forms, assumed-density moments, the spread window and widening."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from pilot.coadaptation.core.filters import (
    ORDINAL_CUTPOINTS,
    SPREAD_TRUNCATION_FACTOR,
    SPREAD_WINDOW_C,
    GammaPoisson,
    GaussianFilter,
    OrdinalProbitFilter,
    ProbitFilter,
    SpreadInvGamma,
    in_spread_window,
    ordinal_expected,
    ordinal_expected_np,
    ordinal_probabilities,
)
from pilot.coadaptation.core.special import norm_cdf, norm_pdf


def _posterior_moments(m: float, p: float, likelihood) -> tuple[float, float]:
    """Mean and variance of N(theta; m, p) * likelihood(theta) by quadrature."""
    sd = math.sqrt(p)
    lo, hi = m - 12 * sd, m + 12 * sd

    def dens(t: float) -> float:
        return stats.norm.pdf(t, m, sd) * likelihood(t)

    z, _ = integrate.quad(dens, lo, hi, epsabs=0.0, epsrel=1e-11, limit=400)
    m1, _ = integrate.quad(lambda t: t * dens(t) / z, lo, hi, epsabs=0.0, epsrel=1e-11, limit=400)
    m2, _ = integrate.quad(lambda t: t * t * dens(t) / z, lo, hi, epsabs=0.0, epsrel=1e-11, limit=400)
    return m1, m2 - m1 * m1


def test_kalman_equals_the_conjugate_closed_form() -> None:
    rng = np.random.Generator(np.random.PCG64(3))
    p0 = np.diag([0.3, 0.5, 0.2, 0.4]) ** 2
    flt = GaussianFilter(p0, sigma=0.35)
    feats, ys, hs = [], [], []
    for _ in range(40):
        f = rng.normal(size=4)
        h = float(rng.normal())
        y = h + float(f @ np.array([0.2, -0.1, 0.3, 0.05])) + float(rng.normal(0, 0.35))
        flt.update(f, y, minutes=0.0, h=h)
        feats.append(f)
        ys.append(y - h)
        hs.append(h)
    big_f = np.array(feats)
    precision = np.linalg.inv(p0) + big_f.T @ big_f / 0.35**2
    cov = np.linalg.inv(precision)
    mean = cov @ (big_f.T @ np.array(ys) / 0.35**2)
    assert np.allclose(flt.m, mean, atol=1e-10)
    assert np.allclose(flt.P, cov, atol=1e-10)
    assert np.allclose(flt.P, flt.P.T, atol=0.0)


@pytest.mark.parametrize(("m", "p", "h", "y"), [(0.0, 0.64, -1.5, 1), (0.3, 0.25, 0.4, 0), (-0.5, 1.0, -2.5, 1)])
def test_probit_moments_match_quadrature(m: float, p: float, h: float, y: int) -> None:
    flt = ProbitFilter(np.array([[p]]), prior_mean=np.array([m]))
    flt.update(np.array([1.0]), y, h=h)
    sign = 1.0 if y else -1.0
    mean, var = _posterior_moments(m, p, lambda t: norm_cdf(sign * (h + t)))
    assert flt.m[0] == pytest.approx(mean, abs=1e-3)
    assert flt.P[0, 0] == pytest.approx(var, abs=1e-3)


@pytest.mark.parametrize(("m", "p", "h", "k"), [(0.0, 0.25, 0.0, 1), (0.2, 0.25, -0.3, 4), (-0.1, 0.5, 0.8, 7)])
def test_ordinal_moments_match_quadrature(m: float, p: float, h: float, k: int) -> None:
    flt = OrdinalProbitFilter(np.array([[p]]), prior_mean=np.array([m]))
    flt.update(np.array([1.0]), k, h=h)
    lo, hi = ORDINAL_CUTPOINTS[k - 1], ORDINAL_CUTPOINTS[k]
    mean, var = _posterior_moments(m, p, lambda t: norm_cdf(hi - h - t) - norm_cdf(lo - h - t))
    assert flt.m[0] == pytest.approx(mean, abs=1e-3)
    assert flt.P[0, 0] == pytest.approx(var, abs=1e-3)
    with pytest.raises(ValueError):
        flt.update(np.array([1.0]), 8)


@pytest.mark.parametrize(("m", "p", "h", "limit"), [(0.0, 0.09, 0.5, 0.8), (0.1, 0.2, 1.0, 1.2), (-0.2, 0.3, 2.0, 1.5)])
def test_censored_moments_match_quadrature(m: float, p: float, h: float, limit: float) -> None:
    sigma = 0.35
    flt = GaussianFilter(np.array([[p]]), sigma=sigma, prior_mean=np.array([m]))
    flt.update(np.array([1.0]), limit, h=h, censored=True)
    mean, var = _posterior_moments(m, p, lambda t: 1.0 - norm_cdf((limit - h - t) / sigma))
    assert flt.m[0] == pytest.approx(mean, abs=1e-3)
    assert flt.P[0, 0] == pytest.approx(var, abs=1e-3)


def test_ordinal_helpers() -> None:
    probs = ordinal_probabilities(0.2, 0.3)
    assert len(probs) == 7 and math.fsum(probs) == pytest.approx(1.0)
    etas = np.array([-2.0, 0.0, 0.7, 3.0])
    assert np.allclose(ordinal_expected_np(etas), [ordinal_expected(float(e)) for e in etas], atol=1e-12)
    assert ordinal_expected(0.0) == pytest.approx(4.0)


def test_truncation_factor() -> None:
    c = SPREAD_WINDOW_C
    assert c == 3.0
    expected = 1 - 2 * c * norm_pdf(c) / (2 * norm_cdf(c) - 1)
    assert expected == SPREAD_TRUNCATION_FACTOR
    assert round(SPREAD_TRUNCATION_FACTOR, 5) == 0.97334
    assert abs(SPREAD_TRUNCATION_FACTOR - stats.truncnorm(-3, 3).var()) < 1e-12


def test_spread_window_uses_r0_never_the_estimate() -> None:
    spread = SpreadInvGamma(1.0)
    sigma = math.sqrt((28 / 4.133) ** 2 + 1.0)
    inside, outside = 2.9 * sigma, 3.1 * sigma
    assert in_spread_window(inside, 0.0, sigma, sigma, 1.0)
    assert not in_spread_window(outside, 0.0, sigma, sigma, 1.0)
    for _ in range(40):  # learn a much larger spread; the window must not move
        spread.update(2.5, 2.5)
    assert spread.r_hat > 1.5
    before = spread.to_arrays()["ab"].copy()
    assert spread.observe(outside, 0.0, sigma, sigma) is False
    assert np.array_equal(spread.to_arrays()["ab"], before)  # no forgetting step, no update
    assert spread.observe(inside, 0.0, sigma, sigma) is True
    assert not np.array_equal(spread.to_arrays()["ab"], before)


def test_spread_forgetting_and_update() -> None:
    spread = SpreadInvGamma(1.6)
    assert (spread.a, spread.b) == (3.0, 2.0 * 1.6**2)
    assert spread.r_hat == pytest.approx(1.6)
    spread.update(1.0, 2.0)
    assert spread.a == pytest.approx(3.0 + 1.0)
    assert spread.b == pytest.approx(2 * 1.6**2 + 5.0 / (2 * SPREAD_TRUNCATION_FACTOR))
    a1, b1 = spread.a, spread.b
    spread.update(0.0, 0.0)
    assert spread.a == pytest.approx(3.0 + 0.98 * (a1 - 3.0) + 1.0)
    assert spread.b == pytest.approx(2 * 1.6**2 + 0.98 * (b1 - 2 * 1.6**2))


def _grid_centres(w: float = 28.0, gap: float = 8.0) -> np.ndarray:
    pitch = w + gap
    return np.array([(c * pitch, r * pitch) for r in range(3) for c in range(4)], dtype=np.float64)


def _simulate_spread(
    *,
    r_true: float,
    r0: float,
    w: float,
    floor: float,
    p_wrong: float,
    wrong: str = "uniform",
    correction: bool = True,
    seeds: int = 200,
    n: int = 500,
) -> float:
    """Mean r-hat over seeds after n first activations through the window and update of spec 7.5."""
    centres = _grid_centres(w)
    sigma = math.sqrt((w / 4.133) ** 2 + floor**2)
    shrink = 1.0 if correction else math.sqrt(SPREAD_TRUNCATION_FACTOR)
    pitch = w + 8.0
    estimates = []
    for seed in range(seeds):
        rng = np.random.Generator(np.random.PCG64(1000 + seed))
        spread = SpreadInvGamma(r0)
        intended = rng.integers(0, 12, size=n)
        mistakes = rng.random(n) < p_wrong
        offsets = rng.normal(0.0, r_true * sigma, size=(n, 2))
        picks = rng.random(n)
        for t in range(n):
            target = centres[intended[t]]
            aim = target
            if mistakes[t]:
                if wrong == "uniform":
                    others = [i for i in range(12) if i != intended[t]]
                else:
                    others = [i for i in range(12) if abs(np.hypot(*(centres[i] - target)) - pitch) < 1e-9]
                aim = centres[others[int(picks[t] * len(others))]]
            dx, dy = aim + offsets[t] - target
            if in_spread_window(dx, dy, sigma, sigma, r0):
                spread.update(dx / sigma * shrink, dy / sigma * shrink)
        estimates.append(spread.r_hat)
    return float(np.mean(estimates))


def test_spread_recovers_r_with_the_floor_counted_once() -> None:
    mean = _simulate_spread(r_true=1.2, r0=1.6, w=16.0, floor=6.0, p_wrong=0.0)
    assert abs(mean / 1.2 - 1.0) <= 0.045  # reference bias about +2.6%


def test_selection_errors_do_not_inflate_the_spread() -> None:
    with_errors = _simulate_spread(r_true=1.0, r0=1.0, w=28.0, floor=1.0, p_wrong=0.1)
    assert abs(with_errors - 1.0) <= 0.02  # reference about -0.7%
    clean = _simulate_spread(r_true=1.0, r0=1.0, w=28.0, floor=1.0, p_wrong=0.0)
    uncorrected = _simulate_spread(r_true=1.0, r0=1.0, w=28.0, floor=1.0, p_wrong=0.0, correction=False)
    assert abs(clean - 1.0) <= 0.02
    assert 0.008 <= clean - uncorrected <= 0.018  # about 1.3 points lower without the correction
    hand = _simulate_spread(r_true=1.6, r0=1.6, w=28.0, floor=6.0, p_wrong=0.1)
    assert abs(hand / 1.6 - 1.0) <= 0.05  # reference about +3%
    adjacent = _simulate_spread(r_true=1.6, r0=1.6, w=28.0, floor=6.0, p_wrong=0.1, wrong="adjacent")
    print(f"hand pointer, adjacent wrong selections: mean r-hat bias {adjacent / 1.6 - 1.0:+.2%} (limit of spec 7.5)")


def _worked_filter() -> tuple[ProbitFilter, np.ndarray]:
    p0 = np.diag([0.5, 0.8, 0.3, 0.35, 0.35]) ** 2
    flt = ProbitFilter(p0, gamma=slice(3, 5), q_fast=0.02**2)
    rng = np.random.Generator(np.random.PCG64(11))
    for _ in range(60):
        f = np.array([1.0, float(rng.integers(0, 2)), float(rng.normal()), 1.0, float(rng.integers(0, 2))])
        flt.update(f, int(rng.random() < 0.3), minutes=1.0)
    return flt, p0


def test_widen_beta_is_capped_at_the_prior_and_keeps_correlations() -> None:
    flt, p0 = _worked_filter()
    before = flt.P.copy()
    sd = np.sqrt(np.diag(before))
    corr_before = before / np.outer(sd, sd)
    flt.widen_beta(4.0, p0)
    after = flt.P
    for i in range(3):
        assert after[i, i] == pytest.approx(min(4 * before[i, i], p0[i, i]))
    for i in (3, 4):
        assert after[i, i] == pytest.approx(before[i, i])
    sd_after = np.sqrt(np.diag(after))
    assert np.allclose(after / np.outer(sd_after, sd_after), corr_before, atol=1e-12)
    m_before = flt.m.copy()
    for _ in range(10):
        flt.widen_beta(4.0, p0)
        assert np.all(np.diag(flt.P)[:3] <= np.diag(p0)[:3] + 1e-15)
    assert np.array_equal(flt.m, m_before)
    # A variance the day-boundary diffusion already put above the prior stays where it is.
    flt.inflate_beta(1.0)
    above = np.diag(flt.P)[:3].copy()
    flt.widen_beta(4.0, p0)
    assert np.allclose(np.diag(flt.P)[:3], above)


def test_within_day_diffusion_is_capped_and_gamma_reset() -> None:
    flt, p0 = _worked_filter()
    small = flt.P[3, 3]
    assert flt.diffused_cov(10.0)[3, 3] == pytest.approx(min(small + 10 * 0.02**2, p0[3, 3]))
    assert flt.diffused_cov(1e9)[3, 3] == pytest.approx(p0[3, 3])
    assert flt.diffused_cov(1e9)[0, 0] == flt.P[0, 0]
    flt.reset_gamma()
    assert np.array_equal(flt.P[3:, 3:], p0[3:, 3:])
    assert np.all(flt.P[:3, 3:] == 0.0) and np.all(flt.m[3:] == 0.0)


def test_predict_and_sample_never_mutate_and_arrays_round_trip() -> None:
    flt, _ = _worked_filter()
    snap = {k: v.copy() for k, v in flt.to_arrays().items()}
    f = np.array([1.0, 1.0, 0.3, 1.0, 0.0])
    for _ in range(5):
        flt.predict(f, minutes=30.0, h=-1.0)
        flt.probability(f, minutes=30.0)
        flt.sample(np.random.Generator(np.random.PCG64(5)), 64, minutes=30.0)
    after = flt.to_arrays()
    assert all(np.array_equal(snap[k], after[k]) for k in snap)
    clone = ProbitFilter(np.eye(5))
    clone.from_arrays(after)
    assert np.array_equal(clone.m, flt.m) and np.array_equal(clone.P, flt.P)
    with pytest.raises(ValueError):
        clone.from_arrays({"m": np.zeros(4), "P": np.eye(4)})
    spread = SpreadInvGamma(1.3)
    spread.update(0.5, -0.2)
    other = SpreadInvGamma(1.3)
    other.from_arrays(spread.to_arrays())
    assert (other.a, other.b) == (spread.a, spread.b)
    draws = flt.sample(np.random.Generator(np.random.PCG64(5)), 20_000)
    assert np.allclose(draws.mean(axis=0), flt.m, atol=0.02)
    assert np.allclose(np.cov(draws.T), flt.P, atol=0.02)


def test_gamma_poisson() -> None:
    rate = GammaPoisson(0.5, 10.0)
    assert rate.mean == pytest.approx(0.05)
    rate.update(3, minutes=0.5)
    assert (rate.alpha, rate.beta) == (3.5, 10.5)
    rate.day_decay()
    assert rate.alpha == pytest.approx(0.5 + 0.7 * 3.0) and rate.beta == pytest.approx(10.0 + 0.7 * 0.5)
    clone = GammaPoisson(0.5, 10.0)
    clone.from_arrays(rate.to_arrays())
    assert (clone.alpha, clone.beta) == (rate.alpha, rate.beta)

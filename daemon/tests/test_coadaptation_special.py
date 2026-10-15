"""Special functions of the co-adaptation core against scipy reference values (scipy only in tests)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, special, stats

from pilot.coadaptation.core.special import (
    binom_sf,
    erfc_np,
    log_norm_cdf,
    log_norm_cdf_np,
    mills_ratio,
    norm_cdf,
    norm_cdf_np,
    norm_pdf,
    norm_ppf,
    truncnorm_moments,
)


def test_norm_pdf_and_cdf_match_scipy() -> None:
    for x in np.linspace(-38.0, 9.0, 941):
        assert norm_pdf(float(x)) == pytest.approx(stats.norm.pdf(x), rel=1e-13, abs=1e-300)
        assert norm_cdf(float(x)) == pytest.approx(special.ndtr(x), rel=1e-12, abs=1e-300)
    xs = np.linspace(-6.0, 6.0, 101)
    assert np.allclose(norm_cdf_np(xs), special.ndtr(xs), rtol=1e-13, atol=0.0)


def test_erfc_np_matches_math_erfc_over_every_range() -> None:
    # Cody's three ranges, their edges, both signs and the far tails
    xs = np.concatenate([np.linspace(-30.0, 30.0, 12001), [-4.0, -0.46875, -0.0, 0.0, 0.46875, 4.0, 26.6, 27.4]])
    expected = np.array([math.erfc(float(x)) for x in xs])
    got = erfc_np(xs)
    normal = expected > 1e-300
    assert np.allclose(got[normal], expected[normal], rtol=4e-15, atol=0.0)
    assert np.all(np.abs(got[~normal] - expected[~normal]) < 1e-300)
    edges = erfc_np(np.array([np.inf, -np.inf, np.nan]))
    assert edges[0] == 0.0 and edges[1] == 2.0 and math.isnan(edges[2])
    assert float(norm_cdf_np(np.array(0.3))) == pytest.approx(norm_cdf(0.3), rel=1e-15)
    assert norm_cdf_np(np.zeros((2, 3))).shape == (2, 3)


def test_norm_ppf_inverts_the_cdf() -> None:
    ps = np.concatenate([np.logspace(-12, -1, 400), np.linspace(0.01, 0.99, 400), 1.0 - np.logspace(-12, -1, 400)])
    for p in ps:
        x = norm_ppf(float(p))
        assert abs(x - special.ndtri(p)) < 1e-12
    assert norm_ppf(0.5) == 0.0
    assert norm_ppf(0.0) == -math.inf and norm_ppf(1.0) == math.inf
    with pytest.raises(ValueError):
        norm_ppf(1.5)


def test_log_norm_cdf_and_mills_ratio() -> None:
    for x in np.linspace(-80.0, 30.0, 1101):
        ref = special.log_ndtr(x)
        assert log_norm_cdf(float(x)) == pytest.approx(ref, rel=1e-12, abs=1e-15)
    assert np.allclose(log_norm_cdf_np(np.array([-50.0, 0.0, 5.0])), special.log_ndtr([-50.0, 0.0, 5.0]))
    for x in (-5.0, -1.0, 0.0, 0.5, 2.0, 8.0, 30.0):
        expected = stats.norm.pdf(x) / stats.norm.sf(x)
        assert mills_ratio(x) == pytest.approx(expected, rel=1e-10)
    assert math.isfinite(mills_ratio(40.0)) and mills_ratio(40.0) == pytest.approx(40.0, rel=1e-3)


@pytest.mark.parametrize(
    ("a", "b"),
    [(-1.0, 2.0), (-math.inf, 0.3), (1.5, math.inf), (-0.2, 0.1), (-math.inf, -4.0), (3.0, 3.5), (-12.0, -11.0)],
)
def test_truncnorm_moments_match_numerical_integration(a: float, b: float) -> None:
    lo = max(a, -40.0)
    hi = min(b, 40.0)
    z, _ = integrate.quad(stats.norm.pdf, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)
    m1, _ = integrate.quad(lambda x: x * stats.norm.pdf(x) / z, lo, hi, epsabs=0.0, epsrel=1e-12, limit=200)
    m2, _ = integrate.quad(lambda x: x * x * stats.norm.pdf(x) / z, lo, hi, epsabs=0.0, epsrel=1e-12, limit=200)
    mean, var = truncnorm_moments(a, b)
    assert mean == pytest.approx(m1, rel=1e-8, abs=1e-10)
    assert var == pytest.approx(m2 - m1 * m1, rel=1e-6, abs=1e-10)
    assert mean == pytest.approx(stats.truncnorm.mean(a, b), rel=1e-8, abs=1e-10)


def test_truncnorm_moments_stay_finite_in_far_tails() -> None:
    mean, var = truncnorm_moments(30.0, 31.0)
    assert 30.0 < mean < 30.1 and 0.0 < var < 0.002
    with pytest.raises(ValueError):
        truncnorm_moments(1.0, 1.0)


@pytest.mark.parametrize(
    ("k", "n", "p"),
    [(0, 50, 0.02), (3, 100, 0.02), (9, 100, 0.05), (19, 100, 0.1), (49, 100, 0.5), (60, 100, 0.3), (5, 10, 0.9)],
)
def test_binom_sf_matches_tabulated_values(k: int, n: int, p: float) -> None:
    assert binom_sf(k, n, p) == pytest.approx(stats.binom.sf(k, n, p), rel=1e-12, abs=1e-300)


def test_binom_sf_edges() -> None:
    assert binom_sf(-1, 20, 0.1) == 1.0
    assert binom_sf(20, 20, 0.1) == 0.0
    assert binom_sf(3, 20, 0.0) == 0.0
    assert binom_sf(3, 20, 1.0) == 1.0
    with pytest.raises(ValueError):
        binom_sf(1, 10, 1.5)

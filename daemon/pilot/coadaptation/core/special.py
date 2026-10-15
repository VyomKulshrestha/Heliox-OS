"""Special functions for the co-adaptation core (standard library and numpy only).

Scalar functions take and return Python floats; the ``*_np`` variants apply them elementwise to numpy arrays.
``norm_cdf_np`` evaluates ``erfc`` with W. J. Cody's rational approximations as whole-array numpy expressions (Thompson
sampling calls it on millions of draws, where a per-element ``math.erfc`` dominated the run time); it agrees with
``norm_cdf`` to about 2e-15 relative error. All of them are deterministic, so live observation and replay compute
identical numbers.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# Acklam's rational approximation of the normal quantile (relative error about 1.15e-9 before the Newton step).
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01)
_A5 = 2.506628277459239e00
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
)
_C5 = 2.938163982698783e00
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425


def norm_pdf(x: float) -> float:
    """Standard normal density."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def norm_cdf(x: float) -> float:
    """Standard normal distribution function, via ``math.erfc`` (accurate in both tails)."""
    return 0.5 * math.erfc(-x / _SQRT2)


def log_norm_pdf(x: float) -> float:
    return -0.5 * x * x - _LOG_SQRT_2PI


def log_norm_cdf(x: float) -> float:
    """``ln Phi(x)``, finite for every finite x (asymptotic series below -37, where erfc would underflow)."""
    if math.isinf(x):
        return 0.0 if x > 0 else -math.inf
    if x > 0.0:
        return math.log1p(-0.5 * math.erfc(x / _SQRT2))
    if x > -37.0:
        return math.log(0.5 * math.erfc(-x / _SQRT2))
    inv2 = 1.0 / (x * x)
    series = 1.0 - inv2 + 3.0 * inv2**2 - 15.0 * inv2**3 + 105.0 * inv2**4
    return -0.5 * x * x - math.log(-x) - _LOG_SQRT_2PI + math.log(series)


def mills_ratio(x: float) -> float:
    """The normal hazard ``phi(x) / (1 - Phi(x))`` (the inverse Mills ratio), as ``exp(ln phi(x) - ln Phi(-x))``."""
    if math.isinf(x):
        return math.inf if x > 0 else 0.0
    return math.exp(log_norm_pdf(x) - log_norm_cdf(-x))


def _acklam(p: float) -> float:
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C5
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return num / den
    q = p - 0.5
    r = q * q
    num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A5) * q
    den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    return num / den


def norm_ppf(p: float) -> float:
    """Standard normal quantile: Acklam's approximation plus one Newton step.

    The upper half is computed as ``-norm_ppf(1 - p)`` (``1 - p`` is exact there), so both tails keep full
    relative accuracy; the absolute error is below 1e-12 on (1e-12, 1 - 1e-12).
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must lie in [0, 1]")
    if p > 0.5:
        return -norm_ppf(1.0 - p)
    x = _acklam(p)
    # Newton step on Phi(x) - p = 0; Phi(x) is computed with erfc, so the residual keeps relative accuracy.
    return x - (norm_cdf(x) - p) / norm_pdf(x)


def truncnorm_moments(a: float, b: float) -> tuple[float, float]:
    """Mean and variance of a standard normal truncated to ``[a, b]`` (``a`` may be -inf, ``b`` may be +inf).

    ``Z = Phi(b) - Phi(a)`` is formed in log space, and an interval in the upper tail is mirrored into the lower one,
    so the moments stay accurate where ``Z`` is tiny. ``a * phi(a)`` is 0 at ``a = -inf`` (likewise for ``b``).
    """
    if not a < b:
        raise ValueError("truncation interval must have a < b")
    if a > 0.0:  # mirror into the lower tail, where log Phi keeps precision
        mean, var = truncnorm_moments(-b, -a)
        return -mean, var
    log_phi_b = log_norm_cdf(b)
    log_phi_a = log_norm_cdf(a)
    log_z = log_phi_b + math.log1p(-math.exp(log_phi_a - log_phi_b)) if a > -math.inf else log_phi_b
    pdf_a = math.exp(log_norm_pdf(a) - log_z) if a > -math.inf else 0.0
    pdf_b = math.exp(log_norm_pdf(b) - log_z) if b < math.inf else 0.0
    mean = pdf_a - pdf_b
    a_term = a * pdf_a if a > -math.inf else 0.0
    b_term = b * pdf_b if b < math.inf else 0.0
    var = 1.0 + a_term - b_term - mean * mean
    return mean, max(var, 0.0)


def binom_sf(k: int, n: int, p: float) -> float:
    """``P(X > k)`` for ``X ~ Binomial(n, p)``, by exact summation of the upper tail in log space."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_n_fact = math.lgamma(n + 1)
    terms = [
        log_n_fact - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1, n + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(t - peak) for t in terms))


# --- vectorised variants ------------------------------------------------------------------------------------------

# W. J. Cody, "Rational Chebyshev approximations for the error function", Math. Comp. 23 (1969), with the
# coefficients of his SPECFUN routine CALERF: erf for |x| <= 0.46875, erfc for 0.46875 < |x| <= 4 and erfc for |x| > 4.
_ERFC_THRESH = 0.46875
_ERFC_A = (3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03)
_ERFC_A4 = 1.85777706184603153e-1
_ERFC_B = (2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03)
_ERFC_C = (
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
)
_ERFC_C8 = 2.15311535474403846e-8
_ERFC_D = (
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
)
_ERFC_P = (
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
)
_ERFC_P5 = 1.63153871373020978e-2
_ERFC_Q = (2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1, 6.05183413124413191e-2)
_ERFC_Q4 = 2.33520497626869185e-3
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
# erfc(y) is below the smallest subnormal double for y > 27.3.
_ERFC_UNDERFLOW = 27.5

_log_norm_cdf_np = np.vectorize(log_norm_cdf, otypes=[np.float64])


def _erfc_small(x: np.ndarray) -> np.ndarray:
    """``1 - erf(x)`` from Cody's first range (|x| <= 0.46875), signed ``x``."""
    xsq = x * x
    num = _ERFC_A4 * xsq
    den = xsq.copy()
    for a, b in zip(_ERFC_A[:3], _ERFC_B[:3]):
        num += a
        num *= xsq
        den += b
        den *= xsq
    num += _ERFC_A[3]
    den += _ERFC_B[3]
    num *= x
    num /= den
    return 1.0 - num


def _erfc_mid(y: np.ndarray) -> np.ndarray:
    """``erfc(y)`` from Cody's second range (0.46875 <= y <= 4). ``exp(-y^2)`` is taken directly: its rounding error
    is at most ``y^2`` ulps of the exponent, 2e-15 relative at y = 4."""
    num = _ERFC_C8 * y
    den = y.copy()
    for c, d in zip(_ERFC_C[:7], _ERFC_D[:7]):
        num += c
        num *= y
        den += d
        den *= y
    num += _ERFC_C[7]
    den += _ERFC_D[7]
    num /= den
    num *= np.exp(-(y * y))
    return num


def _erfc_large(y: np.ndarray) -> np.ndarray:
    """``erfc(y)`` from Cody's third range (y > 4, inf and nan included), with his split of ``exp(-y^2)``."""
    yc = np.minimum(y, _ERFC_UNDERFLOW)
    inv = 1.0 / (yc * yc)
    num = _ERFC_P5 * inv
    den = inv.copy()
    for p, q in zip(_ERFC_P[:4], _ERFC_Q):
        num += p
        num *= inv
        den += q
        den *= inv
    num += _ERFC_P[4]
    den += _ERFC_Q4
    result = (_INV_SQRT_PI - inv * num / den) / yc
    ysq = np.trunc(yc * 16.0) / 16.0
    result *= np.exp(-ysq * ysq) * np.exp(-(yc - ysq) * (yc + ysq))
    return np.where(y > _ERFC_UNDERFLOW, 0.0, result)


def erfc_np(x: np.ndarray) -> np.ndarray:
    """``erfc`` elementwise (Cody 1969; relative error below 2e-15 wherever the result is a normal double)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return erfc_np(x.reshape(1)).reshape(())
    y = np.abs(x)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out = _erfc_mid(np.clip(y, _ERFC_THRESH, 4.0))
        large = ~(y <= 4.0)  # nan included
        if large.any():
            out[large] = _erfc_large(y[large])
        out = np.where(x < 0.0, 2.0 - out, out)  # erfc(-y) = 2 - erfc(y)
        small = y <= _ERFC_THRESH
        if small.any():
            out = np.where(small, _erfc_small(x), out)  # this range takes the sign itself
    return out


def norm_pdf_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def norm_cdf_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * erfc_np(-x / _SQRT2)


def log_norm_cdf_np(x: np.ndarray) -> np.ndarray:
    return _log_norm_cdf_np(np.asarray(x, dtype=np.float64))

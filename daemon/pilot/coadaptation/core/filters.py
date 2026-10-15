"""Online filters of the outcome model (spec section 7.5; numpy only).

Each latent-Gaussian filter keeps ``theta = [beta; gamma] ~ N(m, P)`` over its head's feature vector. The linear
predictor of an outcome is ``eta = h + f'theta`` with the fixed physics prior ``h`` passed in by the caller.
Predictions, samples and diffused covariances are always computed on copies: ``predict`` and ``sample`` never change
``to_arrays()``. ``P`` is symmetrised after every update. Learning uses no random numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .special import mills_ratio, norm_cdf, norm_cdf_np, norm_pdf, truncnorm_moments

ORDINAL_CUTPOINTS: tuple[float, ...] = (-math.inf, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, math.inf)
ORDINAL_LEVELS = 7
SIGMA_TIME = 0.35

SPREAD_WINDOW_C = 3.0
# E[z^2 | |z| < c] / E[z^2] for a standard normal truncated at +-c: 1 - 2c phi(c) / (2 Phi(c) - 1); 0.97334 at c = 3.
SPREAD_TRUNCATION_FACTOR = 1.0 - 2.0 * SPREAD_WINDOW_C * norm_pdf(SPREAD_WINDOW_C) / (
    2.0 * norm_cdf(SPREAD_WINDOW_C) - 1.0
)
SPREAD_PRIOR_A = 3.0
SPREAD_FORGETTING = 0.98
IDLE_DAY_DECAY = 0.7

_JITTER = 1e-9
_CHOLESKY_RETRIES = 3


def _symmetrise(p: np.ndarray) -> np.ndarray:
    return (p + p.T) / 2.0


def cholesky_with_jitter(p: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor; on failure add ``1e-9 I`` and retry up to 3 times, then clip negative eigenvalues."""
    cov = _symmetrise(np.array(p, dtype=np.float64))
    eye = np.eye(cov.shape[0])
    for attempt in range(_CHOLESKY_RETRIES + 1):
        try:
            return np.linalg.cholesky(cov + attempt * _JITTER * eye)
        except np.linalg.LinAlgError:
            continue
    values, vectors = np.linalg.eigh(cov)
    return vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None)))


class LatentGaussianFilter:
    """State ``N(m, P)`` with a prior ``N(m0, P0)``; the base of the Gaussian, probit and ordinal filters.

    ``gamma`` is the index slice of the today offset (diffused within a day by ``q_fast`` per minute, each diagonal
    entry capped at its prior value); the other indices form the slow block ``beta``.
    """

    kind = "gaussian"

    def __init__(
        self,
        prior_cov: np.ndarray,
        *,
        gamma: slice | None = None,
        q_fast: float = 0.0,
        prior_mean: np.ndarray | None = None,
    ) -> None:
        p0 = np.array(prior_cov, dtype=np.float64)
        if p0.ndim != 2 or p0.shape[0] != p0.shape[1]:
            raise ValueError("prior covariance must be square")
        self.dim = p0.shape[0]
        self.prior_cov = p0
        self.prior_mean = (
            np.zeros(self.dim) if prior_mean is None else np.array(prior_mean, dtype=np.float64).reshape(self.dim)
        )
        self.gamma_idx = np.arange(self.dim)[gamma] if gamma is not None else np.zeros(0, dtype=np.int64)
        self.beta_idx = np.setdiff1d(np.arange(self.dim), self.gamma_idx)
        self.q_fast = float(q_fast)
        self.m = self.prior_mean.copy()
        self.P = self.prior_cov.copy()

    # --- dynamics -------------------------------------------------------------------------------------------------

    def diffused_cov(self, minutes: float) -> np.ndarray:
        """A copy of ``P`` after ``minutes`` of within-day diffusion of the gamma block (capped at the prior)."""
        cov = self.P.copy()
        if minutes > 0 and self.q_fast > 0 and self.gamma_idx.size:
            for i in self.gamma_idx:
                cap = self.prior_cov[i, i]
                cov[i, i] = max(cov[i, i], min(cov[i, i] + self.q_fast * minutes, cap))
        return cov

    def inflate_beta(self, q: float) -> None:
        """Day-boundary diffusion: add ``q`` to each beta variance (not capped, spec 8)."""
        for i in self.beta_idx:
            self.P[i, i] += q
        self.P = _symmetrise(self.P)

    def reset_gamma(self) -> None:
        """Today offset back to its prior: ``gamma <- m0_gamma``, ``P_gamma <- P0_gamma``, beta-gamma covariance 0."""
        if not self.gamma_idx.size:
            return
        g = self.gamma_idx
        self.m[g] = self.prior_mean[g]
        self.P[g, :] = 0.0
        self.P[:, g] = 0.0
        self.P[np.ix_(g, g)] = self.prior_cov[np.ix_(g, g)]
        self.P = _symmetrise(self.P)

    def widen_beta(self, factor: float, prior_cov: np.ndarray | None = None) -> None:
        """Drift alarm (spec 8): each beta variance becomes ``min(factor P_ii, P0_ii)``, or stays where it is when it is
        already above that; ``P_beta <- D P_beta D`` keeps every correlation; the mean is unchanged."""
        p0 = self.prior_cov if prior_cov is None else np.asarray(prior_cov, dtype=np.float64)
        d = np.ones(self.dim)
        for i in self.beta_idx:
            var = self.P[i, i]
            if var <= 0:
                continue
            target = max(var, min(factor * var, p0[i, i]))
            d[i] = math.sqrt(target / var)
        self.P = _symmetrise(self.P * np.outer(d, d))

    def reset(self) -> None:
        self.m = self.prior_mean.copy()
        self.P = self.prior_cov.copy()

    # --- prediction -----------------------------------------------------------------------------------------------

    def predict(self, f: np.ndarray, *, minutes: float = 0.0, h: float = 0.0) -> tuple[float, float]:
        """``(mu, s2)`` of the latent ``eta = h + f'theta`` with the covariance diffused on a copy."""
        f = np.asarray(f, dtype=np.float64)
        cov = self.diffused_cov(minutes)
        return float(h + f @ self.m), float(max(f @ cov @ f, 0.0))

    def sample(self, rng: np.random.Generator, n: int, *, minutes: float = 0.0) -> np.ndarray:
        """``n`` draws of theta from the diffused copy, shape ``(n, dim)``, via Cholesky."""
        chol = cholesky_with_jitter(self.diffused_cov(minutes))
        z = rng.standard_normal((n, self.dim))
        return self.m[None, :] + z @ chol.T

    # --- update helpers -------------------------------------------------------------------------------------------

    def _moment_update(self, f: np.ndarray, minutes: float, h: float, mean: float, var: float, v: float) -> None:
        """Linear update of theta after the observed latent's moments became ``(mean, var)`` (ADF)."""
        self.P = self.diffused_cov(minutes)
        t_hat = h + f @ self.m
        pf = self.P @ f
        self.m = self.m + pf * ((mean - t_hat) / v)
        self.P = _symmetrise(self.P - np.outer(pf, pf) * ((v - var) / (v * v)))

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"m": self.m.copy(), "P": self.P.copy()}

    def from_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        m = np.asarray(arrays["m"], dtype=np.float64)
        p = np.asarray(arrays["P"], dtype=np.float64)
        if m.shape != (self.dim,) or p.shape != (self.dim, self.dim):
            raise ValueError("filter arrays have the wrong shape")
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(p))):
            raise ValueError("filter arrays must be finite")
        self.m = m.copy()
        self.P = p.copy()


class GaussianFilter(LatentGaussianFilter):
    """``y = eta + eps``, ``eps ~ N(0, sigma^2)``; Kalman update, or assumed-density filtering when right-censored."""

    kind = "time"

    def __init__(
        self,
        prior_cov: np.ndarray,
        *,
        sigma: float = SIGMA_TIME,
        gamma: slice | None = None,
        q_fast: float = 0.0,
        prior_mean: np.ndarray | None = None,
    ) -> None:
        super().__init__(prior_cov, gamma=gamma, q_fast=q_fast, prior_mean=prior_mean)
        self.sigma = float(sigma)

    def predictive(self, f: np.ndarray, *, minutes: float = 0.0, h: float = 0.0) -> tuple[float, float]:
        """``(mu, sd)`` of the observation ``y``: ``sd = sqrt(s2 + sigma^2)``."""
        mu, s2 = self.predict(f, minutes=minutes, h=h)
        return mu, math.sqrt(s2 + self.sigma**2)

    def update(self, f: np.ndarray, y: float, *, minutes: float = 0.0, h: float = 0.0, censored: bool = False) -> None:
        """Observe ``y``; with ``censored`` the observation is only ``y_true >= y`` (right-censored at ``y``)."""
        f = np.asarray(f, dtype=np.float64)
        self.P = self.diffused_cov(minutes)
        t_hat = float(h + f @ self.m)
        pf = self.P @ f
        v = float(f @ pf) + self.sigma**2
        if not censored:
            gain = pf / v
            self.m = self.m + gain * (y - t_hat)
            self.P = _symmetrise(self.P - np.outer(gain, pf))
            return
        alpha = (y - t_hat) / math.sqrt(v)
        lam = mills_ratio(alpha)
        mean = t_hat + math.sqrt(v) * lam
        var = v * (1.0 + alpha * lam - lam * lam)
        self._moment_update(f, 0.0, h, mean, max(var, 0.0), v)


class ProbitFilter(LatentGaussianFilter):
    """``y ~ Bernoulli(Phi(eta))``; assumed-density update of Graepel et al. 2010."""

    kind = "probit"

    def probability(self, f: np.ndarray, *, minutes: float = 0.0, h: float = 0.0) -> float:
        """Predictive ``P(y = 1) = Phi(mu / sqrt(1 + s2))``."""
        mu, s2 = self.predict(f, minutes=minutes, h=h)
        return norm_cdf(mu / math.sqrt(1.0 + s2))

    def update(self, f: np.ndarray, y: int | bool, *, minutes: float = 0.0, h: float = 0.0) -> None:
        f = np.asarray(f, dtype=np.float64)
        self.P = self.diffused_cov(minutes)
        t_hat = float(h + f @ self.m)
        pf = self.P @ f
        v = 1.0 + float(f @ pf)
        sign = 1.0 if y else -1.0
        z = sign * t_hat / math.sqrt(v)
        lam = mills_ratio(-z)  # phi(z) / Phi(z)
        self.m = self.m + pf * (sign * lam / math.sqrt(v))
        self.P = _symmetrise(self.P - np.outer(pf, pf) * (lam * (lam + z) / v))


class OrdinalProbitFilter(LatentGaussianFilter):
    """Ratings 1..7 with fixed cutpoints ``ORDINAL_CUTPOINTS`` and unit latent noise."""

    kind = "ordinal"

    def update(self, f: np.ndarray, y: int, *, minutes: float = 0.0, h: float = 0.0) -> None:
        if isinstance(y, bool) or not 1 <= int(y) <= ORDINAL_LEVELS:
            raise ValueError("rating must be in 1..7")
        f = np.asarray(f, dtype=np.float64)
        self.P = self.diffused_cov(minutes)
        t_hat = float(h + f @ self.m)
        v = 1.0 + float(f @ self.P @ f)
        sd = math.sqrt(v)
        lo, hi = ORDINAL_CUTPOINTS[int(y) - 1], ORDINAL_CUTPOINTS[int(y)]
        mean_std, var_std = truncnorm_moments((lo - t_hat) / sd, (hi - t_hat) / sd)
        self._moment_update(f, 0.0, h, t_hat + sd * mean_std, v * var_std, v)


def ordinal_probabilities(mu: float, var: float = 0.0) -> tuple[float, ...]:
    """``P(k)``, k = 1..7, of a rating whose latent mean is ``N(mu, var)`` (unit latent noise added)."""
    sd = math.sqrt(1.0 + max(var, 0.0))
    cdf = [norm_cdf((c - mu) / sd) for c in ORDINAL_CUTPOINTS]
    return tuple(max(cdf[k + 1] - cdf[k], 0.0) for k in range(ORDINAL_LEVELS))


def ordinal_expected(mu: float, var: float = 0.0) -> float:
    """``E[y] = sum k P(k)``; with ``var = 0`` this is ``E[y | eta = mu]``."""
    return math.fsum((k + 1) * p for k, p in enumerate(ordinal_probabilities(mu, var)))


def ordinal_expected_np(eta: np.ndarray) -> np.ndarray:
    """``E[y | eta]`` elementwise (Thompson draws of the latent)."""
    eta = np.asarray(eta, dtype=np.float64)
    out = np.zeros_like(eta)
    inner = np.array(ORDINAL_CUTPOINTS[1:-1]).reshape((-1,) + (1,) * eta.ndim)
    # E[y | eta] = 1 + sum over the six inner cutpoints of P(y > k) = 1 + sum_k (1 - Phi(c_k - eta)).
    for cdf in norm_cdf_np(inner - eta):
        out += 1.0 - cdf
    return out + 1.0


class GammaPoisson:
    """Idle activation rate per minute: ``count ~ Poisson(lambda * minutes)``, ``lambda ~ Gamma(alpha, beta)``."""

    def __init__(self, alpha0: float, beta0: float) -> None:
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.alpha = self.alpha0
        self.beta = self.beta0

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    def predict(self, *, minutes: float) -> float:
        """Expected count in ``minutes``."""
        return self.mean * minutes

    def update(self, count: int, *, minutes: float) -> None:
        if count < 0 or minutes < 0:
            raise ValueError("count and minutes must be non-negative")
        self.alpha += count
        self.beta += minutes

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.gamma(self.alpha, 1.0 / self.beta, size=n)

    def day_decay(self, factor: float = IDLE_DAY_DECAY) -> None:
        """Statistics above the prior are multiplied by ``factor`` at a day boundary (spec 8)."""
        self.alpha = self.alpha0 + factor * (self.alpha - self.alpha0)
        self.beta = self.beta0 + factor * (self.beta - self.beta0)

    def reset(self) -> None:
        self.alpha, self.beta = self.alpha0, self.beta0

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"ab": np.array([self.alpha, self.beta])}

    def from_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        ab = np.asarray(arrays["ab"], dtype=np.float64)
        if ab.shape != (2,) or not np.all(np.isfinite(ab)) or ab[0] <= 0 or ab[1] <= 0:
            raise ValueError("Gamma statistics have the wrong shape or values")
        self.alpha, self.beta = float(ab[0]), float(ab[1])


def in_spread_window(dx: float, dy: float, sigma_x: float, sigma_y: float, r0: float) -> bool:
    """The selection-error window of spec 7.5: ``|dx| <= 3 r0 sigma_x`` and ``|dy| <= 3 r0 sigma_y``.

    ``r0`` is the engine's spread prior (``PRIOR_TABLE_V1`` or a ``PopulationPrior``), never the learned estimate, so
    live observation and replay include exactly the same activations.
    """
    return abs(dx) <= SPREAD_WINDOW_C * r0 * sigma_x and abs(dy) <= SPREAD_WINDOW_C * r0 * sigma_y


class SpreadInvGamma:
    """Endpoint spread factor: ``r^2 ~ IG(a, b)`` with prior ``a0 = 3``, ``b0 = 2 r0^2`` (mean ``r0^2``)."""

    def __init__(self, r0: float, *, a0: float = SPREAD_PRIOR_A, forgetting: float = SPREAD_FORGETTING) -> None:
        self.r0 = float(r0)
        self.a0 = float(a0)
        self.b0 = (self.a0 - 1.0) * self.r0**2
        self.forgetting = float(forgetting)
        self.a = self.a0
        self.b = self.b0

    @property
    def r_hat(self) -> float:
        """Point estimate ``sqrt(b / (a - 1))``."""
        return math.sqrt(self.b / (self.a - 1.0))

    def update(self, z_x: float, z_y: float) -> None:
        """Forgetting, then the conjugate update with the truncation correction of spec 7.5."""
        self.a = self.a0 + self.forgetting * (self.a - self.a0)
        self.b = self.b0 + self.forgetting * (self.b - self.b0)
        self.a += 1.0
        self.b += (z_x * z_x + z_y * z_y) / (2.0 * SPREAD_TRUNCATION_FACTOR)

    def observe(self, dx: float, dy: float, sigma_x: float, sigma_y: float) -> bool:
        """Apply one first activation's offset from the intended centre. Outside the window (a selection error) it
        changes nothing, not even the forgetting step, and returns False."""
        if not in_spread_window(dx, dy, sigma_x, sigma_y, self.r0):
            return False
        self.update(dx / sigma_x, dy / sigma_y)
        return True

    def reset(self) -> None:
        self.a, self.b = self.a0, self.b0

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"ab": np.array([self.a, self.b])}

    def from_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        ab = np.asarray(arrays["ab"], dtype=np.float64)
        if ab.shape != (2,) or not np.all(np.isfinite(ab)) or ab[0] <= 1 or ab[1] <= 0:
            raise ValueError("spread statistics have the wrong shape or values")
        self.a, self.b = float(ab[0]), float(ab[1])

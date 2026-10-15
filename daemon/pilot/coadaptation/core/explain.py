"""Explanations of decisions (section 9.8 of the co-adaptation spec).

An explanation is an i18n key with template values, the main predicted contrasts and the excluded variants with
their reasons. It reports predictions with 90% ranges and how many of the person's own selections they rest on;
it never claims certainty and never mentions stress, attention, load or health.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .outcome_model import InteractionPrediction

SUMMARY_KEYS = {
    "adaptive": "coadaptation.explain.adaptive",
    "randomized": "coadaptation.explain.randomized",
    "probe": "coadaptation.explain.probe",
    "comparator": "coadaptation.explain.comparator",
    "pinned": "coadaptation.explain.pinned",
    "baseline": "coadaptation.explain.fallback",
}
FALLBACK_KEY = "coadaptation.explain.fallback"
EVIDENCE_SOURCES = ("literature_prior", "your_history", "today")
_Z90 = 1.6448536269514722


@dataclass(frozen=True, slots=True)
class Contrast:
    metric: Literal["slips", "time", "corrections", "effort"]
    variant: str
    baseline: str
    variant_mean: float
    baseline_mean: float
    delta_ci90: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "variant": self.variant,
            "baseline": self.baseline,
            "variant_mean": self.variant_mean,
            "baseline_mean": self.baseline_mean,
            "delta_ci90": [self.delta_ci90[0], self.delta_ci90[1]],
        }


@dataclass(frozen=True, slots=True)
class Explanation:
    summary_key: str
    values: dict[str, str | float]
    contrasts: tuple[Contrast, ...]
    excluded: tuple[tuple[str, str], ...]
    evidence_source: Literal["literature_prior", "your_history", "today"]
    trials_person: int
    trials_today: int

    def to_dict(self) -> dict[str, Any]:
        """ExplanationV1 (section 14.3)."""
        return {
            "summary_key": self.summary_key,
            "values": dict(self.values),
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
            "excluded": [[variant, reason] for variant, reason in self.excluded],
            "evidence_source": self.evidence_source,
            "trials_person": self.trials_person,
            "trials_today": self.trials_today,
        }


def _sd(estimate: Any) -> float:
    """The SD of an estimate's mean, from its 90% credible range.

    A time estimate's (lo90, hi90) is a single-trial predictive range: it holds the trial noise sigma_T, which a
    difference of predicted medians does not have, so its ``credible90`` (parameter uncertainty only) is used."""
    credible = getattr(estimate, "credible90", None)
    lo, hi = credible if credible is not None else (estimate.lo90, estimate.hi90)
    return max(0.0, (float(hi) - float(lo)) / (2.0 * _Z90))


def _contrast(metric: str, variant: str, baseline: str, a: Any, b: Any) -> Contrast:
    """The difference of two predicted means with a 90% range from their credible SDs (taken as independent)."""
    delta = float(a.mean) - float(b.mean)
    half = _Z90 * math.sqrt(_sd(a) ** 2 + _sd(b) ** 2)
    return Contrast(
        metric=metric,  # type: ignore[arg-type]
        variant=variant,
        baseline=baseline,
        variant_mean=float(a.mean),
        baseline_mean=float(b.mean),
        delta_ci90=(delta - half, delta + half),
    )


def _relative(contrast: Contrast) -> float:
    scale = max(abs(contrast.baseline_mean), 1e-9)
    return abs(contrast.variant_mean - contrast.baseline_mean) / scale


def build_explanation(
    *,
    mode: str,
    chosen: str,
    predictions: Mapping[str, InteractionPrediction],
    excluded: Mapping[str, str],
    baseline: str | None,
    fallback: bool = False,
) -> Explanation:
    """The explanation of one decision.

    ``mode`` is the decision's ``PolicyMode`` value; ``fallback`` marks a decision that fell back (section 9.2 step
    4). In adaptive mode the contrasts are the one or two largest predicted differences (relative to the baseline's
    mean) between the chosen variant and ``baseline`` (``assistance_off`` at L1, ``act`` at L2) in slips, time and
    corrections. The evidence counts come from the chosen variant's prediction.
    """
    key = FALLBACK_KEY if fallback else SUMMARY_KEYS.get(str(mode), FALLBACK_KEY)
    chosen_prediction = predictions.get(chosen)
    contrasts: tuple[Contrast, ...] = ()
    values: dict[str, str | float] = {"variant": chosen}
    if chosen_prediction is not None:
        evidence = chosen_prediction.evidence
        source = str(evidence.source)
        trials_person = int(evidence.trials_person)
        trials_today = int(evidence.trials_today)
        values.update(
            {
                "slips_pct": float(round(100.0 * float(chosen_prediction.p_unintended.mean))),
                "slips_lo_pct": float(round(100.0 * float(chosen_prediction.p_unintended.lo90))),
                "slips_hi_pct": float(round(100.0 * float(chosen_prediction.p_unintended.hi90))),
                "time_s": round(float(chosen_prediction.time_ms.mean) / 1000.0, 1),
                "trials_today": float(trials_today),
                "trials_earlier": float(max(0, trials_person - trials_today)),
            }
        )
    else:
        source, trials_person, trials_today = "literature_prior", 0, 0
    if source not in EVIDENCE_SOURCES:
        source = "literature_prior"

    if str(mode) == "adaptive" and not fallback and chosen_prediction is not None and baseline is not None:
        base = predictions.get(baseline)
        if base is not None and baseline != chosen:
            candidates = [
                _contrast("slips", chosen, baseline, chosen_prediction.p_unintended, base.p_unintended),
                _contrast("time", chosen, baseline, chosen_prediction.time_ms, base.time_ms),
                _contrast("corrections", chosen, baseline, chosen_prediction.p_correction, base.p_correction),
            ]
            ranked = sorted(candidates, key=_relative, reverse=True)
            contrasts = tuple(contrast for contrast in ranked[:2] if contrast.variant_mean != contrast.baseline_mean)
            values["baseline"] = baseline
            values["baseline_slips_pct"] = float(round(100.0 * float(base.p_unintended.mean)))

    return Explanation(
        summary_key=key,
        values=values,
        contrasts=contrasts,
        excluded=tuple((variant, str(reason)) for variant, reason in excluded.items()),
        evidence_source=source,  # type: ignore[arg-type]
        trials_person=trials_person,
        trials_today=trials_today,
    )

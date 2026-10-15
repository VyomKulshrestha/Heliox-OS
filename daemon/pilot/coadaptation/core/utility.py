"""Utilities of the policy and the realized utility of the analysis (sections 9.1, 21.2 and 12.7).

The policy optimises the person's own priorities. The weight relations ``w_UA >= 2 w_T`` and ``w_UA >= 1`` hold for
every slider value, but they do not stop speed from offsetting slips: halving trial time gains ``w_T ln 2``, which
outweighs a rise in ``p_ua`` of up to ``w_T ln 2 / w_UA`` (5.2 to 20.8 percentage points). Limits on unintended
activations come from the person's rules, rule C2, rule C9 and the safety filter, not from these weights.
``realized_utility`` scores what happened: observed outcomes only, no prediction and no switching penalty.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SCALE_MIDPOINT = 4.0
INTRUSIVENESS_WITHOUT_HELP = 1.0  # "Did Heliox's help get in your way?" has no meaning without help (9.1, 12.8)
L2_INTERRUPTION_BURDEN = 0.3
ISO_EFFECTIVE_WIDTH_FACTOR = 4.133

_INTERRUPTING_L2 = frozenset({"confirm_selection:second_click", "ask_clarification:2", "ask_clarification:3"})


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    w_c: float
    w_ua: float
    w_k: float
    w_t: float
    w_b: float
    w_v: float

    @classmethod
    def from_priorities(cls, m: float, f: float, i: float) -> UtilityWeights:
        """Weights from the priorities (fewer mistakes ``m``, faster ``f``, fewer interruptions ``i``), each in [0, 1]."""
        for value in (m, f, i):
            if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("priorities must lie in [0, 1]")
        weights = cls(
            w_c=1.0,
            w_ua=2.0 + 2.0 * m,
            w_k=0.5 + 0.5 * m,
            w_t=0.3 + 0.3 * f,
            w_b=0.15 + 0.35 * i,
            w_v=0.5 + 0.5 * i,
        )
        if not (weights.w_ua >= 2.0 * weights.w_t and weights.w_ua >= 1.0):
            raise AssertionError("utility weight invariants violated")
        return weights

    @classmethod
    def from_mapping(cls, priorities: Mapping[str, float]) -> UtilityWeights:
        """From a logged ``{"fewer_mistakes", "faster", "fewer_interruptions"}`` object (RecommendationV1)."""
        return cls.from_priorities(
            float(priorities["fewer_mistakes"]), float(priorities["faster"]), float(priorities["fewer_interruptions"])
        )

    def speed_for_slips(self) -> float:
        """The rise in p_ua that halving trial time offsets: ``w_T ln 2 / w_UA``."""
        return self.w_t * math.log(2.0) / self.w_ua


PROTOCOL_WEIGHTS = UtilityWeights.from_priorities(0.5, 0.5, 0.5)


def burden_term(effort: Any, intrusiveness: Any) -> Any:
    """B = (effort - 4)/3 + (intrusiveness - 4)/3 (floats or numpy arrays)."""
    return (effort - SCALE_MIDPOINT) / 3.0 + (intrusiveness - SCALE_MIDPOINT) / 3.0


def l1_utility(
    variant: str,
    *,
    p_complete: Any,
    p_ua: Any,
    p_correction: Any,
    p_veto: Any,
    log_time_s: Any,
    effort: Any,
    intrusiveness: Any,
    weights: UtilityWeights,
    t_ref_s: float,
    previous_regime: str | None,
    switching_penalty: float,
) -> Any:
    """U1(v) of section 9.1 per trial; ``E[intrusiveness | assistance_off] = 1`` by definition.

    Arguments may be floats or numpy arrays of Thompson draws (broadcast together).
    """
    if variant == "assistance_off":
        intrusiveness = INTRUSIVENESS_WITHOUT_HELP  # broadcasts against arrays of draws
    switched = 1.0 if previous_regime is not None and variant != previous_regime else 0.0
    return (
        weights.w_c * p_complete
        - weights.w_ua * p_ua
        - weights.w_k * p_correction
        - weights.w_t * (log_time_s - math.log(t_ref_s))
        - weights.w_b * burden_term(effort, intrusiveness)
        - weights.w_v * p_veto
        - switching_penalty * switched
    )


def l2_utility(
    variant: str,
    *,
    p_complete: Any,
    p_ua: Any,
    p_correction: Any,
    p_veto: Any,
    log_time_s: Any,
    weights: UtilityWeights,
) -> Any:
    """U2(v) of section 9.1 for an L2 response in a cell (episode log time in seconds)."""
    interrupts = 1.0 if variant in _INTERRUPTING_L2 else 0.0
    return (
        weights.w_c * p_complete
        - weights.w_ua * p_ua
        - weights.w_k * p_correction
        - weights.w_t * log_time_s
        - weights.w_v * p_veto
        - weights.w_b * L2_INTERRUPTION_BURDEN * interrupts
    )


def l3_utility(variant: str, *, p_accept: Any, p_dismiss: Any, weights: UtilityWeights) -> Any:
    """U3(offer) = 0.3 P(accept) - 0.5 w_B P(dismiss); U3(no_offer) = 0."""
    if variant == "no_offer":
        return 0.0 * p_accept
    return 0.3 * p_accept - 0.5 * weights.w_b * p_dismiss


def _label(labels: Any, name: str) -> Any:
    if isinstance(labels, Mapping):
        return labels[name]
    return getattr(labels, name)


def realized_utility(
    labels: Any,
    *,
    time_s: float | None,
    t_ref_s: float,
    burden_term: float,
    weights: UtilityWeights,
) -> float:
    """U_obs of section 21.2 from observed outcomes only (no prediction, no switching penalty).

    ``labels`` holds the daemon's trial labels (``complete``, ``ua``, ``correction``, ``veto``; a ``TrialLabels`` or
    a mapping). A trial that was not completed counts at its limit: ``labels.limit_ms`` when the labels carry it,
    else ``time_s``, which the caller then sets to the limit. ``burden_term`` is the block's unweighted
    ``B_obs`` (``block_burden_term``); pass 0 for the utility without the burden term.
    """
    complete = bool(_label(labels, "complete"))
    if complete:
        if time_s is None:
            time_ms = _label(labels, "time_ms")
            if time_ms is None:
                raise ValueError("a completed trial needs its time")
            time_s = float(time_ms) / 1000.0
    else:
        limit_ms = _optional_label(labels, "limit_ms")
        if limit_ms is not None:
            time_s = float(limit_ms) / 1000.0
        if time_s is None:
            raise ValueError("an uncompleted trial counts at its limit; pass the limit as time_s")
    if time_s <= 0 or t_ref_s <= 0:
        raise ValueError("times must be positive")
    return (
        weights.w_c * float(complete)
        - weights.w_ua * float(bool(_label(labels, "ua")))
        - weights.w_k * float(bool(_label(labels, "correction")))
        - weights.w_t * (math.log(time_s) - math.log(t_ref_s))
        - weights.w_b * float(burden_term)
        - weights.w_v * float(bool(_label(labels, "veto")))
    )


def _optional_label(labels: Any, name: str) -> Any:
    if isinstance(labels, Mapping):
        return labels.get(name)
    return getattr(labels, name, None)


def block_burden_term(
    effort: float | None,
    intrusiveness: float | None,
    *,
    regime: str,
    session_means: Mapping[str, float | None] | None = None,
    all_means: Mapping[str, float | None] | None = None,
) -> float:
    """B_obs of the block a trial belongs to (section 21.2).

    Intrusiveness is 1 for ``assistance_off`` blocks by definition. A skipped item, or a block without a burden
    prompt, takes the participant's mean of that item for the same regime in the same session
    (``session_means``), else across sessions (``all_means``), else the scale midpoint 4.
    """

    def impute(value: float | None, item: str) -> float:
        if value is not None:
            return float(value)
        for means in (session_means, all_means):
            if means is not None and means.get(item) is not None:
                return float(means[item])  # type: ignore[arg-type]
        return SCALE_MIDPOINT

    effort_value = impute(effort, "effort")
    if regime == "assistance_off":
        intrusiveness_value = INTRUSIVENESS_WITHOUT_HELP
    else:
        intrusiveness_value = impute(intrusiveness, "intrusiveness")
    return float(burden_term(effort_value, intrusiveness_value))


# ------------------------------------------------------------------------------------------------- throughput


def _throughput_samples(trials: Sequence[Any]) -> list[tuple[float, float, float, float]]:
    """(o_i, W_i, D_i, MT_i) of each scored pointing trial (section 12.7)."""
    samples: list[tuple[float, float, float, float]] = []
    for item in trials:
        trial = getattr(item, "trial", item)
        if trial.task != "pointing" or trial.interrupted or trial.intended_index is None:
            continue
        if not trial.activations or trial.activations[0].modality == "keyboard":
            continue
        movement = trial.movement
        if movement is None or movement.mt_ms <= 0:
            continue
        target = trial.candidates[trial.intended_index]
        ux = target.cx - trial.layout.origin_x
        uy = target.cy - trial.layout.origin_y
        distance = math.hypot(ux, uy)
        if distance <= 0 or target.w <= 0:
            continue
        offset = (movement.endpoint_dx_norm * ux + movement.endpoint_dy_norm * uy) / distance
        samples.append((offset, float(target.w), distance, movement.mt_ms / 1000.0))
    return samples


def pointing_throughput(trials: Sequence[Any]) -> float | None:
    """Throughput (bits/s) from W-normalised offsets along the task axis (section 12.7).

    For each non-interrupted pointing trial whose first activation is a pointer or hand-pointer activation:
    ``o_i`` = the first activation's W-normalised endpoint offset projected on the unit vector from the layout
    origin to the intended candidate's cue-time centre; ``sigma_o`` = SD of the ``o_i``; ``W_e,i = 4.133 sigma_o
    W_i``; ``TP = mean_i log2(D_i / W_e,i + 1) / MT_i``. None with fewer than two scored trials or no scatter.
    """
    samples = _throughput_samples(trials)
    if len(samples) < 2:
        return None
    offsets = np.array([sample[0] for sample in samples], dtype=float)
    sigma = float(np.std(offsets, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    values = [
        math.log2(distance / (ISO_EFFECTIVE_WIDTH_FACTOR * sigma * width) + 1.0) / movement_s
        for _, width, distance, movement_s in samples
    ]
    return float(np.mean(values))

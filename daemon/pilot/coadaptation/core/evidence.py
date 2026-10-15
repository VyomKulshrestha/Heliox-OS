"""Evidence-source interface (section 11 of the co-adaptation spec).

An evidence source may reweight the prior over candidates (for listing in ``ask_clarification``). It never chooses
the expanded candidate, never lowers ``A_gate``, never commits and never changes whether ``act`` is applicable,
because applicability depends on the literal hit, which evidence cannot move. Interaction data and neural evidence
are never pooled across ``EvidenceKind`` or ``SignalProvenance``. ``NeuralEvidenceSource`` is defined for future work
and registered nowhere in v1.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from .state import Rect

GazeRegion = Literal["center", "left", "right", "up", "down"]

WEIGHT_MIN = 0.5
WEIGHT_MAX = 2.0
GAZE_KAPPA_MAX = 2.0


class EvidenceKind(StrEnum):
    """Where interaction data came from. Kinds are never pooled."""

    LIVE_HUMAN = "live_human"
    RECORDED_HUMAN = "recorded_human"
    SIMULATED = "simulated"


class SignalProvenance(StrEnum):
    """What produced an evidence frame."""

    POINTER = "pointer"
    KEYBOARD = "keyboard"
    GAZE_REGION = "gaze_region"
    SYNTHETIC_EEG = "synthetic_eeg"
    RECORDED_EEG = "recorded_eeg"
    LIVE_EEG = "live_eeg"


_EEG_PROVENANCES = frozenset({SignalProvenance.SYNTHETIC_EEG, SignalProvenance.RECORDED_EEG, SignalProvenance.LIVE_EEG})
_REJECTED_PROVENANCES = frozenset({SignalProvenance.SYNTHETIC_EEG, SignalProvenance.RECORDED_EEG})


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    source: str
    provenance: SignalProvenance
    candidate_weights: Mapping[int, float] | None  # multiplicative prior weights, each clipped to [0.5, 2.0]
    idle_signal: float | None  # [0, 1], reported only; never gates or commits
    t_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, SignalProvenance):
            raise ValueError("provenance must be a SignalProvenance")
        if self.candidate_weights is not None:
            for index, weight in self.candidate_weights.items():
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise ValueError("candidate_weights keys must be candidate indices")
                if not isinstance(weight, int | float) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("candidate_weights values must be positive finite numbers")
        if self.idle_signal is not None and not 0.0 <= self.idle_signal <= 1.0:
            raise ValueError("idle_signal must lie in [0, 1]")
        if not math.isfinite(self.t_ms):
            raise ValueError("t_ms must be finite")


class EvidenceSource(Protocol):
    name: str
    provenance: SignalProvenance
    consent_purpose: str

    def frame(self, candidates: Sequence[Rect], t_ms: float) -> EvidenceFrame | None: ...


class PointerEvidence:
    """The pointer endpoint enters the likelihood (section 6.1), never the prior, so it produces no frame."""

    name = "pointer"
    provenance = SignalProvenance.POINTER
    consent_purpose = "selection_study"

    def frame(self, candidates: Sequence[Rect], t_ms: float) -> EvidenceFrame | None:
        return None


class KeyboardEvidence:
    """Keyboard activations name the focused candidate (section 6.1); they produce no prior frame."""

    name = "keyboard"
    provenance = SignalProvenance.KEYBOARD
    consent_purpose = "selection_study"

    def frame(self, candidates: Sequence[Rect], t_ms: float) -> EvidenceFrame | None:
        return None


def gaze_kappa(accuracy: float, conf: float) -> float:
    """Prior weight for candidates in the gazed region: ``min(2, 1 + acc * conf)`` (section 6.2)."""
    return min(GAZE_KAPPA_MAX, 1.0 + float(accuracy) * float(conf))


def in_gaze_region(cx: float, cy: float, region: GazeRegion, surface_w: float, surface_h: float) -> bool:
    """Whether a candidate centre lies in a coarse gaze region of the surface (section 6.2).

    centre = the middle third in both axes; left/right = the left/right third; up/down = the top/bottom third. A
    candidate can lie in "up" and "left" at once.
    """
    third_w = surface_w / 3.0
    third_h = surface_h / 3.0
    if region == "center":
        return third_w <= cx <= 2.0 * third_w and third_h <= cy <= 2.0 * third_h
    if region == "left":
        return cx < third_w
    if region == "right":
        return cx > 2.0 * third_w
    if region == "up":
        return cy < third_h
    if region == "down":
        return cy > 2.0 * third_h
    raise ValueError("unknown gaze region")


class GazeRegionEvidence:
    """Reduce-only coarse gaze prior: weight kappa for candidates whose centre lies in the gazed region."""

    name = "gaze_region"
    provenance = SignalProvenance.GAZE_REGION
    consent_purpose = "gaze_prior"

    def __init__(self, *, region: GazeRegion, conf: float, accuracy: float, surface_w: float, surface_h: float) -> None:
        if region not in ("center", "left", "right", "up", "down"):
            raise ValueError("unknown gaze region")
        if not 0.0 <= conf <= 1.0 or not 0.0 <= accuracy <= 1.0:
            raise ValueError("gaze confidence and accuracy must lie in [0, 1]")
        if surface_w <= 0 or surface_h <= 0:
            raise ValueError("surface size must be positive")
        self.region: GazeRegion = region
        self.conf = float(conf)
        self.accuracy = float(accuracy)
        self.surface_w = float(surface_w)
        self.surface_h = float(surface_h)

    def frame(self, candidates: Sequence[Rect], t_ms: float) -> EvidenceFrame | None:
        kappa = gaze_kappa(self.accuracy, self.conf)
        weights = {
            index: kappa
            for index, rect in enumerate(candidates)
            if in_gaze_region(rect.cx, rect.cy, self.region, self.surface_w, self.surface_h)
        }
        return EvidenceFrame(
            source=self.name, provenance=self.provenance, candidate_weights=weights, idle_signal=None, t_ms=t_ms
        )


class NeuralEvidenceSource:
    """Future EEG evidence. Registered nowhere in v1.

    Synthetic and recorded provenance is rejected (``ValueError``). ``frame()`` returns ``None`` unless the provenance
    is ``LIVE_EEG``, the hardware is verified and a decoder is supplied; even then the frame only reweights the prior.
    """

    name = "neural"
    consent_purpose = "neural_evidence"

    def __init__(
        self,
        *,
        provenance: SignalProvenance,
        hardware_verified: bool = False,
        decoder: Callable[[Sequence[Rect], float], Mapping[int, float] | None] | None = None,
    ) -> None:
        if provenance not in _EEG_PROVENANCES:
            raise ValueError("neural evidence needs an EEG provenance")
        if provenance in _REJECTED_PROVENANCES:
            raise ValueError("synthetic and recorded EEG never enter the interaction model")
        self.provenance = provenance
        self.hardware_verified = hardware_verified is True
        self._decoder = decoder

    def frame(self, candidates: Sequence[Rect], t_ms: float) -> EvidenceFrame | None:
        if self.provenance in _REJECTED_PROVENANCES:
            raise ValueError("synthetic and recorded EEG never enter the interaction model")
        if self.provenance != SignalProvenance.LIVE_EEG or not self.hardware_verified or self._decoder is None:
            return None
        weights = self._decoder(candidates, t_ms)
        if weights is None:
            return None
        return EvidenceFrame(
            source=self.name,
            provenance=self.provenance,
            candidate_weights=dict(weights),
            idle_signal=None,
            t_ms=t_ms,
        )


# The evidence sources v1 can use. NeuralEvidenceSource is deliberately absent.
REGISTERED_EVIDENCE_SOURCES: tuple[type, ...] = (PointerEvidence, KeyboardEvidence, GazeRegionEvidence)


def apply_evidence(prior: np.ndarray, frames: Sequence[EvidenceFrame]) -> np.ndarray:
    """Reweight a candidate prior by each frame's weights (clipped to [0.5, 2.0]) and keep its total mass.

    ``prior`` holds the candidate priors without the null term (they sum to 0.98 in the study, section 6.1); the
    result sums to the same total. Synthetic or recorded EEG frames raise ``ValueError``. The input is not modified.
    """
    base = np.asarray(prior, dtype=float)
    if base.ndim != 1 or not np.all(np.isfinite(base)) or np.any(base < 0):
        raise ValueError("prior must be a finite non-negative vector")
    total = float(base.sum())
    weighted = base.copy()
    for frame in frames:
        if frame.provenance in _REJECTED_PROVENANCES:
            raise ValueError("synthetic and recorded EEG never enter the interaction model")
        if not frame.candidate_weights:
            continue
        for index, weight in frame.candidate_weights.items():
            if index >= weighted.size:
                raise ValueError("candidate_weights names a candidate outside the prior")
            weighted[index] *= min(WEIGHT_MAX, max(WEIGHT_MIN, float(weight)))
    mass = float(weighted.sum())
    if mass <= 0.0:
        return base.copy()
    return weighted * (total / mass)


def gate_ambiguity(a_without: float, a_with_each: Sequence[float]) -> float:
    """The gating ambiguity: the largest of the values with and without each evidence source (never lower)."""
    return max([float(a_without), *(float(value) for value in a_with_each)])

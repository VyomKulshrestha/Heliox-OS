"""Evidence-source interface: reduce-only, never commits, neural source unregistered (spec 11, 22.1)."""

from __future__ import annotations

import numpy as np
import pytest

from pilot.coadaptation.core.evidence import (
    REGISTERED_EVIDENCE_SOURCES,
    EvidenceFrame,
    EvidenceKind,
    GazeRegionEvidence,
    KeyboardEvidence,
    NeuralEvidenceSource,
    PointerEvidence,
    SignalProvenance,
    apply_evidence,
    gate_ambiguity,
    gaze_kappa,
)
from pilot.coadaptation.core.state import ActivationContext, LayoutSummary, Rect
from pilot.coadaptation.core.target_posterior import GazeInput, compute_posterior
from pilot.coadaptation.core.vocabulary import applicability_reason


def _grid(w: float = 16.0, gap: float = 2.0, centre: tuple[float, float] = (350.0, 250.0)) -> list[Rect]:
    pitch = w + gap
    x0, y0 = centre[0] - 1.5 * pitch, centre[1] - pitch
    return [Rect(cx=x0 + c * pitch, cy=y0 + r * pitch, w=w, h=w) for r in range(3) for c in range(4)]


def test_enums_keep_kinds_and_provenances_apart() -> None:
    assert {k.value for k in EvidenceKind} == {"live_human", "recorded_human", "simulated"}
    assert {p.value for p in SignalProvenance} == {
        "pointer",
        "keyboard",
        "gaze_region",
        "synthetic_eeg",
        "recorded_eeg",
        "live_eeg",
    }


def test_pointer_and_keyboard_sources_add_no_prior_frame() -> None:
    candidates = _grid()
    assert PointerEvidence().frame(candidates, 0.0) is None
    assert KeyboardEvidence().frame(candidates, 0.0) is None


def test_gaze_weights_are_capped_and_cover_only_the_gazed_region() -> None:
    assert gaze_kappa(1.0, 1.0) == 2.0
    assert gaze_kappa(0.6, 0.5) == pytest.approx(1.3)
    candidates = [Rect(100, 100, 16, 16), Rect(350, 250, 16, 16), Rect(600, 400, 16, 16)]
    frame = GazeRegionEvidence(region="left", conf=1.0, accuracy=1.0, surface_w=700, surface_h=500).frame(
        candidates, 5.0
    )
    assert frame is not None and frame.candidate_weights == {0: 2.0}
    centre = GazeRegionEvidence(region="center", conf=0.5, accuracy=0.8, surface_w=700, surface_h=500).frame(
        candidates, 5.0
    )
    assert centre is not None and centre.candidate_weights == {1: pytest.approx(1.4)}


def test_apply_evidence_keeps_the_prior_mass_and_clips_weights() -> None:
    prior = np.full(4, 0.98 / 4)
    frame = EvidenceFrame("gaze_region", SignalProvenance.GAZE_REGION, {0: 10.0, 1: 0.01}, None, 0.0)
    out = apply_evidence(prior, [frame])
    assert out.sum() == pytest.approx(0.98)
    # weights are clipped to [0.5, 2.0]: candidate 0 gets 2x and candidate 1 half of the untouched ones
    assert out[0] / out[2] == pytest.approx(2.0)
    assert out[1] / out[2] == pytest.approx(0.5)
    assert np.array_equal(prior, np.full(4, 0.98 / 4))  # the input is not modified


def test_evidence_never_lowers_the_gate_and_never_changes_act_eligibility() -> None:
    rng = np.random.default_rng(20261015)
    candidates = _grid()
    layout = LayoutSummary("pointing", 12, 16.0, 16.0, 2.0, 3.0, 0, 350.0, 460.0)
    for _ in range(300):
        target = candidates[int(rng.integers(0, len(candidates)))]
        x = int(np.floor(target.cx + rng.normal(0, 6) + 0.5))
        y = int(np.floor(target.cy + rng.normal(0, 6) + 0.5))
        region = str(rng.choice(["center", "left", "right", "up", "down"]))
        gaze = GazeInput(region=region, conf=float(rng.uniform(0, 1)), acc=float(rng.uniform(0.6, 1)))
        without = compute_posterior(candidates, (x, y), modality="mouse", r=1.0, floor=1.0, surface=(700.0, 500.0))
        with_gaze = compute_posterior(
            candidates, (x, y), modality="mouse", r=1.0, floor=1.0, surface=(700.0, 500.0), gaze=gaze
        )
        gate = gate_ambiguity(without.ambiguity, [with_gaze.ambiguity_gate])
        assert gate >= without.ambiguity
        assert gate == pytest.approx(with_gaze.ambiguity_gate)
        # act's applicability reads the literal hit only, which evidence cannot move
        assert with_gaze.has_hit == without.has_hit and with_gaze.hit_index == without.hit_index
        for result in (without, with_gaze):
            ctx = ActivationContext(
                ambiguity=result.ambiguity_gate,
                bin="b3",
                n_plausible=result.n_plausible,
                has_hit=result.has_hit,
                p_hit=result.p_hit,
                p_top=result.p_top,
                modality="mouse",
                layout=layout,
            )
            assert (applicability_reason("act", ctx) is None) == without.has_hit


def test_gate_is_the_max_of_all_values() -> None:
    assert gate_ambiguity(0.3, []) == 0.3
    assert gate_ambiguity(0.3, [0.1, 0.5]) == 0.5
    assert gate_ambiguity(0.7, [0.1]) == 0.7


def test_neural_source_rejects_synthetic_and_recorded_provenance_and_is_not_registered() -> None:
    for provenance in (SignalProvenance.SYNTHETIC_EEG, SignalProvenance.RECORDED_EEG):
        with pytest.raises(ValueError):
            NeuralEvidenceSource(provenance=provenance)
    with pytest.raises(ValueError):
        NeuralEvidenceSource(provenance=SignalProvenance.POINTER)
    candidates = _grid()
    unverified = NeuralEvidenceSource(provenance=SignalProvenance.LIVE_EEG, decoder=lambda c, t: {0: 2.0})
    assert unverified.frame(candidates, 0.0) is None
    verified = NeuralEvidenceSource(
        provenance=SignalProvenance.LIVE_EEG, hardware_verified=True, decoder=lambda c, t: {0: 2.0}
    )
    frame = verified.frame(candidates, 0.0)
    assert frame is not None and frame.candidate_weights == {0: 2.0}
    assert NeuralEvidenceSource not in REGISTERED_EVIDENCE_SOURCES
    with pytest.raises(ValueError):
        apply_evidence(
            np.full(3, 0.98 / 3),
            [EvidenceFrame("eeg", SignalProvenance.SYNTHETIC_EEG, {0: 2.0}, None, 0.0)],
        )


def test_frames_validate_their_values() -> None:
    with pytest.raises(ValueError):
        EvidenceFrame("x", SignalProvenance.GAZE_REGION, {0: -1.0}, None, 0.0)
    with pytest.raises(ValueError):
        EvidenceFrame("x", SignalProvenance.GAZE_REGION, None, 1.5, 0.0)

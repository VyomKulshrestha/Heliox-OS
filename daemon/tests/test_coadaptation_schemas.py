"""RPC payload schemas of the co-adaptation study (spec 14.4, tests of 22.2)."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from pilot.coadaptation import schemas
from pilot.coadaptation.core.preferences import PreferenceKind, preference_to_dict
from pilot.coadaptation.core.records import ActivationRecord, BurdenReport, TrialRecord
from pilot.coadaptation.core.state import ALLOWED_CONTEXT_KEYS, DENIED_FEATURE_SUBSTRINGS
from pilot.coadaptation.core.vocabulary import CAUTION_ORDER, L2_OPTIONS, L3_OPTIONS, VARIANTS, DecisionLevel
from pilot.coadaptation.schemas import (
    PAYLOAD_LIMIT_BYTES,
    TRIAL_PAYLOAD_LIMIT_BYTES,
    ActivationV1,
    AmbiguityTableContextV1,
    BlockRegimeContextV1,
    BurdenV1,
    ConsentGrantParams,
    DeviceV1,
    EmptyParams,
    FallbacksV1,
    L2AppliedV1,
    LayoutSummaryV1,
    OfferContextV1,
    OfferResponseV1,
    PayloadTooLarge,
    PreferenceSetParams,
    PreferenceV1,
    RecommendParams,
    RecordTrialParams,
    RectV1,
    RevertV1,
    SessionBeginParams,
    SpreadV1,
    TrialV1,
    check_trial_spread,
    rpc_level_to_core,
    validation_message,
    without_ineffective_fields,
)

NOW = datetime(2026, 10, 15, 10, 0, tzinfo=UTC).isoformat()
RECT = {"cx": 100.0, "cy": 120.0, "w": 44.0, "h": 44.0}
LAYOUT = {
    "task": "pointing",
    "n_candidates": 12,
    "w_px": 44.0,
    "h_px": 44.0,
    "gap_px": 8.0,
    "id_bar": 2.5,
    "menu_n": 0,
    "origin_x": 350.0,
    "origin_y": 470.0,
}


def _uuid() -> str:
    return str(uuid.uuid4())


def _activation(**overrides):
    data = {
        "index": 0,
        "t_ms": 640.5,
        "modality": "mouse",
        "x": 100,
        "y": 120,
        "hit_index": 0,
        "press_index": 0,
        "a_gate": 0.01,
        "n_plausible": 1,
        "expanded_index": None,
        "hit_rect": dict(RECT),
        "gaze": None,
        "l2": None,
        "irreversible_rule": False,
        "result": "committed",
        "outcome": "committed",
        "committed_index": 0,
    }
    data.update(overrides)
    return data


def _trial(**overrides):
    data = {
        "trial_id": _uuid(),
        "block_index": 2,
        "trial_index": 0,
        "task": "pointing",
        "regime": "act",
        "regime_decision_id": _uuid(),
        "table_decision_ids": [_uuid()],
        "applied_regime": "act",
        "l1_fallback_reason": None,
        "fallbacks_issued_at": NOW,
        "spread": {"r": 1.0, "floor": 1.0},
        "layout": dict(LAYOUT),
        "candidates": [dict(RECT), {"cx": 160.0, "cy": 120.0, "w": 44.0, "h": 44.0}],
        "surface_w": 700.0,
        "surface_h": 500.0,
        "dpr": 1.0,
        "reversible": [True, True],
        "cue_at": NOW,
        "intended_index": 0,
        "modality": "mouse",
        "activations": [_activation()],
        "revert": None,
        "committed_index": 0,
        "commit_t_ms": 640.5,
        "commit_via": "direct",
        "corrections": 0,
        "time_ms": 640.5,
        "limit_ms": 10000,
        "duration_ms": 640.5,
        "movement": None,
        "veto": None,
        "offers": [],
        "hand": None,
        "gaze": None,
        "interrupted": False,
        "interruption_reason": None,
        "explanation_views": 0,
    }
    data.update(overrides)
    return data


def _fallbacks(**overrides):
    data = {
        "l1": "confirm_selection:hold_700",
        "plain": "assistance_off",
        "refresh": "assistance_off",
        "practice": "assistance_off",
        "freezable": ["assistance_off", "act"],
        "freeze_conflict": None,
        "l2": ["ask_clarification:3", "ask_clarification:2", "wait_for_evidence"],
        "interruption_cap": None,
        "issued_at": NOW,
    }
    data.update(overrides)
    return data


def _block_context(**overrides):
    data = {
        "block_index": 1,
        "layouts": [dict(LAYOUT)] * 8,
        "previous_regime": None,
        "session_minute": 2.5,
        "trials_today": 8,
        "hand_quality_recent": None,
    }
    data.update(overrides)
    return data


def _all_models() -> list[type[BaseModel]]:
    found = []
    for value in vars(schemas).values():
        if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == schemas.__name__:
            found.append(value)
    return found


# ── bounds, enums, extra keys ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rect",
    [
        {**RECT, "w": 0.0},
        {**RECT, "h": 2000.5},
        {**RECT, "cx": -10.5},
        {**RECT, "cy": 10000.5},
        {**RECT, "cx": float("nan")},
        {**RECT, "cx": "100"},
    ],
)
def test_rect_bounds(rect):
    with pytest.raises(ValidationError):
        RectV1.parse(rect)


def test_layout_origin_bounds():
    assert LayoutSummaryV1.parse({**LAYOUT, "origin_x": -10.0, "origin_y": 10000.0}).origin_y == 10000.0
    for key, value in (("origin_x", -10.5), ("origin_y", 10000.5), ("id_bar", 12.5), ("n_candidates", 33)):
        with pytest.raises(ValidationError):
            LayoutSummaryV1.parse({**LAYOUT, key: value})


def test_layout_presses_mean_is_optional_and_reaches_the_core():
    plain = LayoutSummaryV1.parse(LAYOUT).to_core()
    assert plain.presses_mean is None and "presses_mean" not in plain.to_dict()
    assert LayoutSummaryV1.parse({**LAYOUT, "presses_mean": 3.5}).to_core().presses_mean == 3.5
    assert LayoutSummaryV1.parse({**LAYOUT, "presses_mean": None}).to_core().presses_mean is None
    for value in (0.5, 64.5):
        with pytest.raises(ValidationError):
            LayoutSummaryV1.parse({**LAYOUT, "presses_mean": value})


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (LayoutSummaryV1, {**LAYOUT, "task": "drag"}),
        (
            DeviceV1,
            {
                "modality": "eye",
                "pointer_types": ["mouse"],
                "dpr": 1.0,
                "surface_w": 700.0,
                "surface_h": 500.0,
                "gaze": False,
                "hand_pointer": False,
            },
        ),
        (
            DeviceV1,
            {
                "modality": "mouse",
                "pointer_types": ["mouse", "pen", "touch", "mouse", "pen"],
                "dpr": 1.0,
                "surface_w": 700.0,
                "surface_h": 500.0,
                "gaze": False,
                "hand_pointer": False,
            },
        ),
        (
            OfferResponseV1,
            {"decision_id": str(uuid.uuid4()), "variant": "offer_modality:voice", "response": "ignored"},
        ),
    ],
)
def test_closed_enums(model, data):
    with pytest.raises(ValidationError):
        model.parse(data)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (RectV1, {**RECT, "label": "Select: apple"}),
        (LayoutSummaryV1, {**LAYOUT, "target": 3}),
        (TrialV1, _trial(note="free text")),
        (ActivationV1, _activation(raw_path=[1, 2])),
        (FallbacksV1, _fallbacks(extra=1)),
        (SpreadV1, {"r": 1.0, "floor": 1.0, "window_title": "x"}),
    ],
)
def test_extra_keys_are_forbidden(model, data):
    with pytest.raises(ValidationError):
        model.parse(data)


def test_numbers_are_never_coerced_from_strings():
    with pytest.raises(ValidationError):
        BurdenV1.parse({"block_index": 1, "scope": "block", "effort": "5"})
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(x=100.5))


# ── contexts ─────────────────────────────────────────────────────────────────


def test_context_models_carry_exactly_the_allowed_keys():
    assert set(BlockRegimeContextV1.model_fields) == ALLOWED_CONTEXT_KEYS[DecisionLevel.BLOCK_REGIME]
    assert set(AmbiguityTableContextV1.model_fields) == ALLOWED_CONTEXT_KEYS[DecisionLevel.AMBIGUITY_RESPONSE]
    assert set(OfferContextV1.model_fields) == ALLOWED_CONTEXT_KEYS[DecisionLevel.OFFER]


@pytest.mark.parametrize("key", ["intended_index", "target", "cue", "is_idle", "nogo", "pointer_x", "transcript"])
@pytest.mark.parametrize("level", ["block_regime", "ambiguity_table", "offer"])
def test_ground_truth_keys_are_rejected_in_contexts(key, level):
    if level == "offer":
        context = {
            "block_index": 1,
            "trial_index": 2,
            "task": "pointing",
            "struggle_kind": "corrections",
            "keyboard_available": True,
        }
    elif level == "ambiguity_table":
        context = _block_context(block_decision_id=str(uuid.uuid4()), start_trial_index=0, trials_in_block=8)
    else:
        context = _block_context()
    request = RecommendParams.parse(
        {"session_id": str(uuid.uuid4()), "decision_id": str(uuid.uuid4()), "level": level, "context": context}
    )
    request.parsed_context()
    bad = RecommendParams.parse(
        {
            "session_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
            "level": level,
            "context": {**context, key: 3},
        }
    )
    with pytest.raises(ValidationError):
        bad.parsed_context()


def test_rpc_level_names_map_onto_decision_levels():
    assert rpc_level_to_core("ambiguity_table") == DecisionLevel.AMBIGUITY_RESPONSE
    assert rpc_level_to_core("block_regime") == DecisionLevel.BLOCK_REGIME
    assert rpc_level_to_core("offer") == DecisionLevel.OFFER
    with pytest.raises(ValueError):
        rpc_level_to_core("ambiguity_response")
    request = RecommendParams.parse(
        {
            "session_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
            "level": "ambiguity_table",
            "context": {},
        }
    )
    assert request.decision_level() == DecisionLevel.AMBIGUITY_RESPONSE


# ── payload limits ───────────────────────────────────────────────────────────


def _worst_case_trial() -> dict:
    """32 candidates and 32 activations with every optional field filled (spec 10.6: about 21 KiB)."""
    candidates = [
        {"cx": 1234.5678901234, "cy": 987.65432109876, "w": 43.999999999999, "h": 43.999999999999} for _ in range(32)
    ]
    activations = []
    for i in range(32):
        activations.append(
            _activation(
                index=i,
                t_ms=59999.123456789,
                x=19999,
                y=19999,
                hit_index=31,
                press_index=31,
                a_gate=0.123456789012345,
                n_plausible=3,
                expanded_index=31,
                hit_rect=dict(candidates[0]),
                gaze={"region": "center", "conf": 0.987654321098765},
                l2={
                    "decision_id": str(uuid.uuid4()),
                    "cell_id": "b3:p3:h1",
                    "draw_index": 4,
                    "variant": "ask_clarification:3",
                    "source": "fallback",
                    "fallback_reason": "stale_table",
                },
                irreversible_rule=True,
                result="confirming",
                outcome="open_at_end",
                committed_index=31,
            )
        )
    return _trial(
        table_decision_ids=[str(uuid.uuid4()) for _ in range(4)],
        l1_fallback_reason="rpc_error",
        spread={"r": 19.123456789012345, "floor": 49.123456789012345},
        layout={**LAYOUT, "n_candidates": 32, "id_bar": 11.123456789012345},
        candidates=candidates,
        reversible=[True] * 32,
        intended_index=31,
        activations=activations,
        revert={"reason": "refresh", "from_activation": 32, "regime": "assistance_off", "fallbacks_issued_at": NOW},
        committed_index=31,
        commit_t_ms=59999.123456789,
        commit_via="confirm",
        corrections=50,
        time_ms=59999.123456789,
        limit_ms=15000,
        duration_ms=59999.123456789,
        movement={
            "rt_ms": 59999.123456789,
            "mt_ms": 59999.123456789,
            "path_ratio": 49.123456789012345,
            "submovements": 50,
            "peak_speed_px_s": 49999.123456789,
            "reentries": 50,
            "endpoint_dx_norm": -19.123456789012345,
            "endpoint_dy_norm": 19.123456789012345,
        },
        veto={"decision_id": str(uuid.uuid4()), "level": "ambiguity_response"},
        offers=[
            {"decision_id": str(uuid.uuid4()), "variant": "no_offer", "response": "ignored", "fallback_reason": "late"},
            {
                "decision_id": str(uuid.uuid4()),
                "variant": "offer_modality:keyboard",
                "response": "dismissed",
                "fallback_reason": None,
            },
        ],
        hand={"quality_mean": 0.987654321098765, "quality_min": 0.123456789012345, "tracking_losses": 100},
        gaze={"agreed_with_intended": True},
        interrupted=True,
        interruption_reason="no_allowed_regime",
        explanation_views=20,
    )


def test_worst_case_trial_fits_the_trial_limit_only():
    params = {"session_id": str(uuid.uuid4()), "trial": _worst_case_trial()}
    size = len(json.dumps(params, separators=(",", ":")).encode())
    assert PAYLOAD_LIMIT_BYTES < size < TRIAL_PAYLOAD_LIMIT_BYTES
    parsed = RecordTrialParams.parse(params, limit=TRIAL_PAYLOAD_LIMIT_BYTES)
    assert len(parsed.trial.activations) == 32
    with pytest.raises(PayloadTooLarge):
        RecordTrialParams.parse(params)


def test_payloads_above_the_limits_are_rejected():
    with pytest.raises(PayloadTooLarge) as small:
        EmptyParams.parse({"padding": "x" * PAYLOAD_LIMIT_BYTES})
    assert small.value.reason == "payload_too_large"
    big = {"session_id": str(uuid.uuid4()), "trial": _worst_case_trial(), "padding": "x" * TRIAL_PAYLOAD_LIMIT_BYTES}
    with pytest.raises(PayloadTooLarge):
        RecordTrialParams.parse(big, limit=TRIAL_PAYLOAD_LIMIT_BYTES)


# ── activation, L2, offers, reverts ──────────────────────────────────────────


def test_press_index_accepts_minus_one_and_rejects_null():
    assert ActivationV1.parse(_activation(press_index=-1, hit_index=None, hit_rect=None)).press_index == -1
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(press_index=None))
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(press_index=-2))
    data = _activation()
    del data["press_index"]
    with pytest.raises(ValidationError):
        ActivationV1.parse(data)


@pytest.mark.parametrize("value", [1, 2, 3])
def test_n_plausible_accepts_one_to_three(value):
    assert ActivationV1.parse(_activation(n_plausible=value)).n_plausible == value


@pytest.mark.parametrize("value", [0, 4])
def test_n_plausible_rejects_values_outside_one_to_three(value):
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(n_plausible=value))


def test_activation_gaze_bounds():
    assert ActivationV1.parse(_activation(gaze={"region": "left", "conf": 1.0})).gaze is not None
    for gaze in ({"region": "north", "conf": 0.5}, {"region": "left", "conf": 1.5}, {"region": "left"}):
        with pytest.raises(ValidationError):
            ActivationV1.parse(_activation(gaze=gaze))


def test_hit_rect_is_given_exactly_with_a_hit():
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(hit_rect=None))
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(hit_index=None))


def test_result_is_the_immediate_response_and_outcome_is_required():
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(result="cancelled"))
    data = _activation()
    del data["outcome"]
    with pytest.raises(ValidationError):
        ActivationV1.parse(data)
    for outcome in ("committed", "cancelled", "aborted", "open_at_end"):
        assert ActivationV1.parse(_activation(result="confirming", outcome=outcome)).outcome == outcome
    assert ActivationV1.parse(_activation(result="none", outcome="none", committed_index=None)).outcome == "none"
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(result="none", outcome="committed"))
    with pytest.raises(ValidationError):
        ActivationV1.parse(_activation(result="committed", outcome="none"))


def _l2(**overrides):
    data = {
        "decision_id": str(uuid.uuid4()),
        "cell_id": "b2:p2:h1",
        "draw_index": 1,
        "variant": "wait_for_evidence",
        "source": "table",
        "fallback_reason": None,
    }
    data.update(overrides)
    return data


def test_l2_source_cap_and_fallback_reason():
    assert L2AppliedV1.parse(_l2(source="cap")).source == "cap"
    with pytest.raises(ValidationError):
        L2AppliedV1.parse(_l2(source="cap", variant="confirm_selection:second_click"))
    assert L2AppliedV1.parse(_l2(source="fallback", fallback_reason="stale_table")).fallback_reason == "stale_table"
    with pytest.raises(ValidationError):
        L2AppliedV1.parse(_l2(source="fallback"))
    with pytest.raises(ValidationError):
        L2AppliedV1.parse(_l2(source="table", fallback_reason="late"))
    with pytest.raises(ValidationError):
        L2AppliedV1.parse(_l2(cell_id="b3:p0:h0"))
    with pytest.raises(ValidationError):
        L2AppliedV1.parse(_l2(draw_index=5))


def test_offer_fallback_reason():
    offer = {"decision_id": str(uuid.uuid4()), "variant": "no_offer", "response": "ignored", "fallback_reason": "late"}
    assert OfferResponseV1.parse(offer).fallback_reason == "late"
    assert OfferResponseV1.parse({**offer, "fallback_reason": "rpc_error"}).fallback_reason == "rpc_error"
    with pytest.raises(ValidationError):
        OfferResponseV1.parse({**offer, "fallback_reason": "stale_table"})
    with pytest.raises(ValidationError):
        OfferResponseV1.parse({**offer, "response": "accepted"})
    with pytest.raises(ValidationError):
        OfferResponseV1.parse({**offer, "variant": "offer_modality:keyboard"})


def test_revert_has_regime_and_fallbacks_issued_at():
    revert = {"reason": "veto", "from_activation": 1, "regime": "assistance_off", "fallbacks_issued_at": NOW}
    assert RevertV1.parse(revert).regime == "assistance_off"
    with pytest.raises(ValidationError):
        RevertV1.parse({**{k: v for k, v in revert.items() if k != "regime"}, "plain_regime": "assistance_off"})
    with pytest.raises(ValidationError):
        RevertV1.parse({k: v for k, v in revert.items() if k != "fallbacks_issued_at"})
    with pytest.raises(ValidationError):
        RevertV1.parse({**revert, "fallbacks_issued_at": "2026-10-15T10:00:00"})
    with pytest.raises(ValidationError):
        RevertV1.parse({**revert, "from_activation": 33})


# ── trials ───────────────────────────────────────────────────────────────────


def test_valid_trial_parses():
    trial = TrialV1.parse(_trial(), limit=TRIAL_PAYLOAD_LIMIT_BYTES)
    assert trial.limit_ms == 10000.0


def test_trial_fallback_fields():
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(fallback_reason="late"))
    assert TrialV1.parse(_trial(l1_fallback_reason="late")).l1_fallback_reason == "late"
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(l1_fallback_reason="stale_table"))
    data = _trial()
    del data["fallbacks_issued_at"]
    with pytest.raises(ValidationError):
        TrialV1.parse(data)
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(fallbacks_issued_at="2026-10-15T10:00:00"))


def test_trial_spread_bounds_and_null():
    assert TrialV1.parse(_trial(spread=None)).spread is None
    for spread in ({"r": 0.0, "floor": 1.0}, {"r": 20.5, "floor": 1.0}, {"r": 1.0, "floor": -0.5}):
        with pytest.raises(ValidationError):
            TrialV1.parse(_trial(spread=spread))


def test_session_checks_spread_against_the_modality():
    keyboard_trial = TrialV1.parse(_trial(spread=None))
    check_trial_spread(keyboard_trial, "keyboard")
    with pytest.raises(ValueError):
        check_trial_spread(keyboard_trial, "mouse")
    mouse_trial = TrialV1.parse(_trial())
    check_trial_spread(mouse_trial, "mouse")
    with pytest.raises(ValueError):
        check_trial_spread(mouse_trial, "keyboard")


def test_trial_interruption_reason_accepts_no_allowed_regime():
    trial = TrialV1.parse(_trial(interrupted=True, interruption_reason="no_allowed_regime"))
    assert trial.interruption_reason == "no_allowed_regime"
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(interrupted=True, interruption_reason="bored"))


def test_intended_index_is_null_exactly_for_idle_epochs():
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(task="idle"))
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(intended_index=None))
    idle = TrialV1.parse(
        _trial(
            task="idle", intended_index=None, activations=[], committed_index=None, commit_via="none", limit_ms=30000
        )
    )
    assert idle.intended_index is None


def test_one_reversible_flag_per_candidate():
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(reversible=[True]))


def test_indices_stay_inside_the_candidate_list():
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(intended_index=5))
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(activations=[_activation(press_index=7)]))


def test_limit_ms_is_one_of_three_values():
    for limit in (10000, 15000, 30000):
        assert TrialV1.parse(_trial(limit_ms=limit)).limit_ms == float(limit)
    with pytest.raises(ValidationError):
        TrialV1.parse(_trial(limit_ms=12000))


def test_ineffective_gaze_and_hand_fields_are_dropped():
    trial = TrialV1.parse(
        _trial(
            activations=[_activation(gaze={"region": "center", "conf": 0.4})],
            gaze={"agreed_with_intended": True},
            hand={"quality_mean": 0.9, "quality_min": 0.5, "tracking_losses": 1},
        )
    )
    stripped = without_ineffective_fields(trial, gaze_effective=False, hand_effective=False)
    assert stripped.gaze is None
    assert stripped.hand is None
    assert all(activation.gaze is None for activation in stripped.activations)
    kept = without_ineffective_fields(trial, gaze_effective=True, hand_effective=True)
    assert kept.gaze is not None and kept.hand is not None and kept.activations[0].gaze is not None


# ── fallbacks ────────────────────────────────────────────────────────────────


def test_fallbacks_required_fields():
    assert FallbacksV1.parse(_fallbacks()).refresh == "assistance_off"
    for key in ("refresh", "freezable", "freeze_conflict", "interruption_cap", "issued_at"):
        data = _fallbacks()
        del data[key]
        with pytest.raises(ValidationError):
            FallbacksV1.parse(data)


def test_fallbacks_freezable_and_freeze_conflict():
    assert FallbacksV1.parse(_fallbacks(freezable=[])).freezable == []
    with pytest.raises(ValidationError):
        FallbacksV1.parse(_fallbacks(freezable=["act"] * 7))
    conflict = {"message": "Your rule 'At most 2 interruptions per block' blocks this.", "conflicting": ["cap-1"]}
    assert FallbacksV1.parse(_fallbacks(freeze_conflict=conflict)).freeze_conflict.conflicting == ["cap-1"]
    with pytest.raises(ValidationError):
        FallbacksV1.parse(_fallbacks(freeze_conflict={**conflict, "extra": 1}))
    with pytest.raises(ValidationError):
        FallbacksV1.parse(_fallbacks(freeze_conflict={"message": "x" * 2001, "conflicting": []}))
    with pytest.raises(ValidationError):
        FallbacksV1.parse(_fallbacks(freeze_conflict={"message": "x", "conflicting": ["a"] * 33}))


def test_fallbacks_cap_and_issued_at_bounds():
    assert FallbacksV1.parse(_fallbacks(interruption_cap=8)).interruption_cap == 8
    for cap in (9, -1):
        with pytest.raises(ValidationError):
            FallbacksV1.parse(_fallbacks(interruption_cap=cap))
    with pytest.raises(ValidationError):
        FallbacksV1.parse(_fallbacks(issued_at="2026-10-15T10:00:00"))
    with pytest.raises(ValidationError):
        FallbacksV1.parse(_fallbacks(l2=["act", "act", "act", "wait_for_evidence"]))


# ── session begin, device, consent grant ─────────────────────────────────────


def _begin(**overrides):
    data = {
        "participant_key": "p-3fa29c1e",
        "context": "study",
        "consent_text_sha256": "a" * 64,
        "device": {
            "modality": "mouse",
            "pointer_types": ["mouse"],
            "dpr": 1.0,
            "surface_w": 700.0,
            "surface_h": 500.0,
            "gaze": True,
            "hand_pointer": False,
        },
        "protocol_id": "coadapt-study-v1",
        "gaze_check": {"accuracy": 0.8},
    }
    data.update(overrides)
    return data


def test_session_begin_gaze_check_bounds():
    assert SessionBeginParams.parse(_begin()).gaze_check.accuracy == 0.8
    assert SessionBeginParams.parse(_begin(gaze_check=None)).gaze_check is None
    for accuracy in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            SessionBeginParams.parse(_begin(gaze_check={"accuracy": accuracy}))
    with pytest.raises(ValidationError):
        SessionBeginParams.parse(_begin(context="desktop"))


def test_no_schema_field_names_a_denied_feature():
    for model in _all_models():
        for name in model.model_fields:
            assert not any(denied in name for denied in DENIED_FEATURE_SUBSTRINGS), (model.__name__, name)
    for name in DeviceV1.model_fields:
        assert not any(denied in name for denied in DENIED_FEATURE_SUBSTRINGS)


def _grant(**overrides):
    data = {
        "participant_key": None,
        "label": None,
        "purposes": ["selection_study"],
        "retention_days": 90,
        "consent_text_version": "coadapt-consent-v1",
        "consent_text_sha256": "a" * 64,
        "protocol_id": "coadapt-study-v1",
        "protocol_sha256": "b" * 64,
        "authorized": True,
    }
    data.update(overrides)
    return data


def test_consent_grant_params():
    assert ConsentGrantParams.parse(_grant()).enable is True
    for overrides in (
        {"authorized": False},
        {"authorized": "true"},
        {"purposes": ["gaze_prior"]},
        {"purposes": ["selection_study", "approval_observation"]},
        {"retention_days": 400},
        {"consent_text_sha256": "ABC"},
        {"participant_key": "Lab A"},
        {"label": "x" * 33},
        {"label": "line\nbreak"},
    ):
        with pytest.raises(ValidationError):
            ConsentGrantParams.parse(_grant(**overrides))


# ── preferences ──────────────────────────────────────────────────────────────


def test_schema_vocabularies_match_the_core():
    assert set(VARIANTS) == schemas.ALL_VARIANT_IDS
    assert set(CAUTION_ORDER) == schemas.CAUTION_ORDER_IDS
    assert set(L2_OPTIONS) == schemas.L2_VARIANT_IDS
    assert set(L3_OPTIONS) == schemas.L3_VARIANT_IDS
    assert set(schemas.PreferenceKindValue.__args__) == {kind.value for kind in PreferenceKind}


@pytest.mark.parametrize(
    ("kind", "value", "ok"),
    [
        ("forbid_variant", "ask_clarification:3", True),
        ("forbid_variant", "confirm_selection", True),
        ("forbid_variant", "approve", False),
        ("pin_regime", "enlarge_target:1.5", True),
        ("pin_regime", "assistance_off", False),
        ("pin_act", True, True),
        ("pin_act", False, False),
        ("min_caution", "confirm_selection:second_click", True),
        ("min_caution", "no_offer", False),
        ("max_interruptions_per_block", 8, True),
        ("max_interruptions_per_block", 9, False),
        ("max_interruptions_per_block", True, False),
        ("priorities", {"fewer_mistakes": 0.5, "faster": 1, "fewer_interruptions": 0.0}, True),
        ("priorities", {"fewer_mistakes": 0.5, "faster": 1.5, "fewer_interruptions": 0.0}, False),
        ("priorities", {"fewer_mistakes": 0.5}, False),
        ("freeze_adaptation", {"regime": "assistance_off"}, True),
        ("freeze_adaptation", {"regime": "wait_for_evidence"}, False),
    ],
)
def test_preference_values_per_kind(kind, value, ok):
    params = {"kind": kind, "scope": {"context": "study", "modality": None, "participant_key": None}, "value": value}
    if ok:
        assert PreferenceSetParams.parse(params).kind == kind
    else:
        with pytest.raises(ValidationError):
            PreferenceSetParams.parse(params)


def test_preference_scope_never_takes_a_label():
    params = {"kind": "pin_act", "scope": {"context": "study", "participant_key": "Lab A"}, "value": True}
    with pytest.raises(ValidationError):
        PreferenceSetParams.parse(params)


# ── to_core ──────────────────────────────────────────────────────────────────


def test_to_core_round_trips():
    trial = TrialV1.parse(_worst_case_trial(), limit=TRIAL_PAYLOAD_LIMIT_BYTES)
    recorded_at = datetime(2026, 10, 15, 10, 5, tzinfo=UTC)
    record = trial.to_core(session_id="s-1", recorded_at=recorded_at, seq=7)
    assert isinstance(record, TrialRecord)
    assert record.seq == 7 and record.recorded_at == recorded_at and record.session_id == "s-1"
    assert TrialRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    expected = trial.to_json_dict()
    if expected["layout"]["presses_mean"] is None:
        del expected["layout"]["presses_mean"]  # an optional field the core record omits while None
    produced = record.to_dict()
    for key, value in expected.items():
        if key in ("fallbacks_issued_at", "cue_at", "revert"):
            continue
        assert produced[key] == value, key

    activation = ActivationV1.parse(_activation()).to_core()
    assert isinstance(activation, ActivationRecord)
    assert activation.to_dict() == ActivationV1.parse(_activation()).to_json_dict()

    burden = BurdenV1.parse(
        {
            "block_index": 3,
            "scope": "block",
            "effort": 2,
            "intrusiveness": None,
            "control": 6,
            "tlx": None,
            "predictability": None,
            "preferred_regime": None,
            "skipped": ["intrusiveness"],
        }
    ).to_core(report_id="r-1", session_id="s-1", recorded_at=recorded_at, seq=8)
    assert isinstance(burden, BurdenReport)
    assert BurdenReport.from_dict(burden.to_dict()) == burden

    fallbacks = FallbacksV1.parse(_fallbacks())
    assert FallbacksV1.parse(fallbacks.to_core().to_dict()) == fallbacks

    pref = PreferenceV1.parse(
        {
            "pref_id": "rule-1",
            "kind": "freeze_adaptation",
            "scope": {"context": "study", "modality": "mouse", "participant_key": "p-3fa29c1e"},
            "value": {"regime": "act"},
            "origin": "explicit_user",
            "created_at": NOW,
            "retracted_at": None,
        }
    )
    assert PreferenceV1.parse(preference_to_dict(pref.to_core())) == pref
    assert RectV1.parse(RECT).to_core().w == 44.0
    assert LayoutSummaryV1.parse(LAYOUT).to_core().origin_y == 470.0


# ── error messages ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (ConsentGrantParams, _grant(label="SECRET-NAME-0123456789-0123456789-xyz")),
        (ConsentGrantParams, _grant(consent_text_sha256="SECRET-NOT-A-HASH")),
        (ActivationV1, _activation(modality="SECRET-MODALITY")),
        (BurdenV1, {"block_index": 1, "scope": "block", "effort": "SECRET-RATING"}),
        (RectV1, {**RECT, "w": "SECRET-WIDTH"}),
    ],
)
def test_validation_errors_never_echo_values(model, data):
    with pytest.raises(ValidationError) as caught:
        model.parse(copy.deepcopy(data))
    assert "SECRET" not in str(caught.value)
    assert "SECRET" not in validation_message(caught.value)

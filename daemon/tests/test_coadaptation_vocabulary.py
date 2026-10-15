"""Strategy vocabulary (spec section 4): the variant table, arms per level and applicability."""

from __future__ import annotations

from pilot.actions import ActionType
from pilot.coadaptation.core.state import ActivationContext, DeviceContext, LayoutSummary, OfferContext
from pilot.coadaptation.core.vocabulary import (
    ASSISTANCE_OFF,
    CAUTION_ORDER,
    L2_OPTIONS,
    L3_OPTIONS,
    NO_OFFER,
    STRATEGY_VERSION,
    VARIANTS,
    AssistanceStrategy,
    DecisionLevel,
    VariantRole,
    acting,
    applicability_reason,
    family_of,
    l1_arms,
)

L1 = DecisionLevel.BLOCK_REGIME
L2 = DecisionLevel.AMBIGUITY_RESPONSE
L3 = DecisionLevel.OFFER
S = AssistanceStrategy

# (id, strategy, params, role, acting, interrupts, caution, levels): the table of spec 4.2, in order.
EXPECTED = [
    (ASSISTANCE_OFF, None, (), VariantRole.REGIME_OFF, True, False, 0, {L1}),
    ("act", S.ACT, (), VariantRole.COMMIT_RULE, True, False, 1, {L1, L2}),
    ("enlarge_target:1.5", S.ENLARGE_TARGET, (("scale", 1.5),), VariantRole.PRESENTATION, True, False, 2, {L1}),
    ("enlarge_target:2.0", S.ENLARGE_TARGET, (("scale", 2.0),), VariantRole.PRESENTATION, True, False, 3, {L1}),
    (
        "confirm_selection:second_click",
        S.CONFIRM_SELECTION,
        (("method", "second_click"),),
        VariantRole.COMMIT_RULE,
        False,
        True,
        4,
        {L1, L2},
    ),
    (
        "confirm_selection:hold_400",
        S.CONFIRM_SELECTION,
        (("method", "hold"), ("hold_ms", 400)),
        VariantRole.COMMIT_RULE,
        False,
        True,
        5,
        {L1},
    ),
    (
        "confirm_selection:hold_700",
        S.CONFIRM_SELECTION,
        (("method", "hold"), ("hold_ms", 700)),
        VariantRole.COMMIT_RULE,
        False,
        True,
        6,
        {L1},
    ),
    ("wait_for_evidence", S.WAIT_FOR_EVIDENCE, (), VariantRole.COMMIT_RULE, False, False, 7, {L2}),
    ("ask_clarification:2", S.ASK_CLARIFICATION, (("k", 2),), VariantRole.COMMIT_RULE, False, True, 8, {L2}),
    ("ask_clarification:3", S.ASK_CLARIFICATION, (("k", 3),), VariantRole.COMMIT_RULE, False, True, 9, {L2}),
    (NO_OFFER, None, (), VariantRole.OFFER, False, False, -1, {L3}),
    (
        "offer_modality:keyboard",
        S.OFFER_MODALITY,
        (("target", "keyboard"),),
        VariantRole.OFFER,
        False,
        False,
        -1,
        {L3},
    ),
    ("offer_modality:pointer", S.OFFER_MODALITY, (("target", "pointer"),), VariantRole.OFFER, False, False, -1, {L3}),
]


def _layout() -> LayoutSummary:
    return LayoutSummary("pointing", 12, 16.0, 16.0, 2.0, 3.0, 0, 350.0, 470.0)


def _act_ctx(*, has_hit: bool, n_plausible: int, modality: str = "mouse") -> ActivationContext:
    return ActivationContext(0.5, "b2", n_plausible, has_hit, 0.5 if has_hit else 0.0, 0.5, modality, _layout())


def _offer(modality: str, keyboard_available: bool) -> OfferContext:
    return OfferContext("s", 1, 2, modality, "menu", "corrections", keyboard_available)


def test_variant_table_is_exact_and_ordered() -> None:
    assert STRATEGY_VERSION == "coadapt-strategies-v1"
    assert list(VARIANTS) == [row[0] for row in EXPECTED]
    for vid, strategy, params, role, is_acting, interrupts, caution, levels in EXPECTED:
        v = VARIANTS[vid]
        assert (v.id, v.strategy, v.params, v.role) == (vid, strategy, params, role)
        assert (v.acting, v.interrupts, v.caution, v.levels) == (is_acting, interrupts, caution, frozenset(levels))


def test_acting_flags_and_families() -> None:
    assert {vid for vid in VARIANTS if acting(vid)} == {
        ASSISTANCE_OFF,
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
    }
    assert family_of("enlarge_target:2.0") == "enlarge_target"
    assert family_of("confirm_selection:hold_400") == "confirm_selection"
    assert family_of(ASSISTANCE_OFF) == ASSISTANCE_OFF
    assert family_of(NO_OFFER) == NO_OFFER


def test_caution_order() -> None:
    assert CAUTION_ORDER == (
        ASSISTANCE_OFF,
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
        "confirm_selection:second_click",
        "confirm_selection:hold_400",
        "confirm_selection:hold_700",
        "wait_for_evidence",
        "ask_clarification:2",
        "ask_clarification:3",
    )


def test_no_strategy_or_variant_is_an_action_type() -> None:
    assert len(ActionType) == 162
    action_values = {a.value for a in ActionType}
    assert not {s.value for s in AssistanceStrategy} & action_values
    assert not set(VARIANTS) & action_values
    assert ASSISTANCE_OFF not in {s.value for s in AssistanceStrategy}
    assert NO_OFFER not in {s.value for s in AssistanceStrategy}


def test_no_timed_dwell_gaze_or_voice_commit_exists() -> None:
    forbidden = ("timeout", "dwell", "gaze", "voice")
    for v in VARIANTS.values():
        for key, value in v.params:
            text = f"{key}={value}".lower()
            assert not any(word in text for word in forbidden), v.id
        assert not any(word in v.id for word in forbidden)


def test_arms_per_level() -> None:
    pointer = (
        ASSISTANCE_OFF,
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
        "confirm_selection:second_click",
        "confirm_selection:hold_700",
    )
    for modality in ("mouse", "touchpad", "pen", "touch"):
        assert l1_arms(modality) == pointer
    assert l1_arms("hand_pointer") == (
        ASSISTANCE_OFF,
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
        "confirm_selection:hold_400",
        "confirm_selection:hold_700",
    )
    assert l1_arms("keyboard") == (
        ASSISTANCE_OFF,
        "act",
        "confirm_selection:second_click",
        "confirm_selection:hold_700",
    )
    assert L2_OPTIONS == (
        "act",
        "confirm_selection:second_click",
        "wait_for_evidence",
        "ask_clarification:2",
        "ask_clarification:3",
    )
    assert L3_OPTIONS == (NO_OFFER, "offer_modality:keyboard", "offer_modality:pointer")
    for vid in L2_OPTIONS:
        assert L2 in VARIANTS[vid].levels
    for modality in ("mouse", "keyboard", "hand_pointer"):
        assert all(L1 in VARIANTS[vid].levels for vid in l1_arms(modality))


def test_applicability_l1() -> None:
    keyboard = DeviceContext("keyboard", (), 1.0, 700.0, 500.0, False, False)
    mouse = DeviceContext("mouse", ("mouse",), 1.0, 700.0, 500.0, False, False)
    for vid in ("enlarge_target:1.5", "enlarge_target:2.0"):
        assert applicability_reason(vid, keyboard) == "not_applicable"
        assert applicability_reason(vid, mouse) is None
        assert applicability_reason(vid, "keyboard") == "not_applicable"
    assert applicability_reason("act", keyboard) is None
    assert applicability_reason("wait_for_evidence", mouse) == "not_applicable"


def test_applicability_l2() -> None:
    no_hit = _act_ctx(has_hit=False, n_plausible=3)
    one = _act_ctx(has_hit=True, n_plausible=1)
    two = _act_ctx(has_hit=True, n_plausible=2)
    three = _act_ctx(has_hit=True, n_plausible=3)
    for vid in ("act", "confirm_selection:second_click"):
        assert applicability_reason(vid, no_hit) == "not_applicable"
        assert applicability_reason(vid, one) is None
    assert applicability_reason("wait_for_evidence", no_hit) is None
    assert applicability_reason("ask_clarification:2", one) == "not_applicable"
    assert applicability_reason("ask_clarification:2", two) is None
    assert applicability_reason("ask_clarification:3", two) == "not_applicable"
    assert applicability_reason("ask_clarification:3", three) is None
    assert applicability_reason("ask_clarification:3", no_hit) is None
    assert applicability_reason("enlarge_target:1.5", three) == "not_applicable"


def test_applicability_l3() -> None:
    assert applicability_reason(NO_OFFER, _offer("keyboard", False)) is None
    assert applicability_reason("offer_modality:keyboard", _offer("mouse", True)) is None
    assert applicability_reason("offer_modality:keyboard", _offer("mouse", False)) == "not_applicable"
    assert applicability_reason("offer_modality:keyboard", _offer("keyboard", True)) == "not_applicable"
    assert applicability_reason("offer_modality:pointer", _offer("hand_pointer", True)) is None
    assert applicability_reason("offer_modality:pointer", _offer("mouse", True)) == "not_applicable"

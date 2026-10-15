"""Explicit preferences: combination, precedence, conflicts, freezes and pins (spec 9.6, 22.1)."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pilot.coadaptation.core.constraints import ExclusionReason, fallbacks
from pilot.coadaptation.core.preferences import (
    Preference,
    PreferenceConflict,
    PreferenceKind,
    PreferenceScope,
    PreferenceSet,
    preference_from_dict,
    preference_to_dict,
)
from pilot.coadaptation.core.vocabulary import CAUTION_ORDER, DecisionLevel, l1_arms

T0 = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)
PK = "p-0000abcd"
OTHER = "p-0000ffff"
MODALITIES = ("mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer")
_counter = itertools.count()


def rule(
    kind: PreferenceKind,
    value: Any,
    *,
    context: str = "study",
    modality: str | None = None,
    participant: str | None = None,
    minutes: int = 0,
    pref_id: str | None = None,
) -> Preference:
    return Preference(
        pref_id=pref_id or f"r{next(_counter)}",
        kind=kind,
        scope=PreferenceScope(context=context, modality=modality, participant_key=participant),  # type: ignore[arg-type]
        value=value,
        origin="explicit_user",
        created_at=T0 + timedelta(minutes=minutes),
    )


def save(prefs: PreferenceSet, pref: Preference) -> tuple[PreferenceSet, tuple[str, ...]]:
    notes = prefs.validate_new(pref)
    return PreferenceSet([*prefs.all, pref]), notes


def build(*rules: Preference) -> PreferenceSet:
    prefs = PreferenceSet([])
    for item in rules:
        prefs, _ = save(prefs, item)
    return prefs


def l1_allowed(prefs: PreferenceSet, modality: str = "mouse", participant: str | None = PK) -> list[str]:
    allowed, _ = prefs.apply(
        DecisionLevel.BLOCK_REGIME,
        l1_arms(modality),
        context="study",
        modality=modality,  # type: ignore[arg-type]
        participant_key=participant,
        interruptions_used=0,
    )
    return allowed


def assert_rejected_both_orders(first: Preference, second: Preference, fragment: str | None = None) -> None:
    for existing, new in ((first, second), (second, first)):
        prefs = build(existing)
        with pytest.raises(PreferenceConflict) as error:
            prefs.validate_new(new)
        assert error.value.conflicting_pref_ids == (existing.pref_id,)
        assert existing.summary() in error.value.message
        assert "Remove that rule" in error.value.message
        if fragment is not None:
            assert fragment in error.value.message
        # nothing is rewritten: the existing rule is unchanged and still active
        assert prefs.get(existing.pref_id) == existing and existing.retracted_at is None


# ------------------------------------------------------------------------------------------- combination rules


def test_priorities_use_precedence_participant_modality_newest() -> None:
    device = rule(PreferenceKind.PRIORITIES, {"fewer_mistakes": 0.1, "faster": 0.1, "fewer_interruptions": 0.1})
    newer_device = rule(
        PreferenceKind.PRIORITIES, {"fewer_mistakes": 0.2, "faster": 0.2, "fewer_interruptions": 0.2}, minutes=5
    )
    modality = rule(
        PreferenceKind.PRIORITIES,
        {"fewer_mistakes": 0.3, "faster": 0.3, "fewer_interruptions": 0.3},
        modality="mouse",
    )
    participant = rule(
        PreferenceKind.PRIORITIES,
        {"fewer_mistakes": 0.9, "faster": 0.4, "fewer_interruptions": 0.0},
        participant=PK,
        minutes=-30,
    )
    assert PreferenceSet([]).priorities(modality="mouse", participant_key=PK) == (0.5, 0.5, 0.5)
    assert PreferenceSet([device, newer_device]).priorities(modality="mouse", participant_key=PK) == (0.2, 0.2, 0.2)
    prefs = PreferenceSet([device, newer_device, modality])
    assert prefs.priorities(modality="mouse", participant_key=PK) == (0.3, 0.3, 0.3)
    assert prefs.priorities(modality="touch", participant_key=PK) == (0.2, 0.2, 0.2)
    prefs = PreferenceSet([device, newer_device, modality, participant])
    assert prefs.priorities(modality="mouse", participant_key=PK) == (0.9, 0.4, 0.0)
    assert prefs.priorities(modality="mouse", participant_key=OTHER) == (0.3, 0.3, 0.3)


@pytest.mark.parametrize("participant_first", [True, False])
def test_overlapping_min_caution_rules_combine_to_the_strictest(participant_first: bool) -> None:
    mine = rule(PreferenceKind.MIN_CAUTION, "act", participant=PK, minutes=10 if not participant_first else 0)
    device = rule(
        PreferenceKind.MIN_CAUTION, "confirm_selection:second_click", minutes=0 if not participant_first else 10
    )
    ordered = (mine, device) if participant_first else (device, mine)
    prefs = build(*ordered)
    allowed = l1_allowed(prefs)
    # precedence would have kept the participant's act minimum; combination keeps the strictest
    assert allowed == ["confirm_selection:second_click", "confirm_selection:hold_700"]
    _, excluded = prefs.apply(
        DecisionLevel.BLOCK_REGIME,
        l1_arms("mouse"),
        context="study",
        modality="mouse",
        participant_key=PK,
        interruptions_used=0,
    )
    assert excluded["act"] is ExclusionReason.BELOW_YOUR_MIN_CAUTION
    stricter_retracted = PreferenceSet([mine, _retract(device, minutes=120)])
    assert l1_allowed(stricter_retracted)[0] == "act"


def _retract(pref: Preference, minutes: int = 60) -> Preference:
    return replace(pref, retracted_at=T0 + timedelta(minutes=minutes))


def test_retracting_the_stricter_minimum_restores_the_other() -> None:
    mine = rule(PreferenceKind.MIN_CAUTION, "act", participant=PK)
    device = rule(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click", minutes=1)
    assert l1_allowed(PreferenceSet([mine, _retract(device)])) == [
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
        "confirm_selection:second_click",
        "confirm_selection:hold_700",
    ]


def test_overlapping_caps_combine_to_the_smallest() -> None:
    for first_value, second_value in ((4, 2), (2, 4)):
        older = rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, first_value, participant=PK)
        newer = rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, second_value, modality="mouse", minutes=5)
        prefs = build(older, newer)
        assert prefs.interruption_cap(modality="mouse", participant_key=PK) == 2
        fb = fallbacks(prefs, "mouse", participant_key=PK, calibration_suspended=False, now=T0 + timedelta(hours=1))
        assert fb.interruption_cap == 2
        allowed, excluded = prefs.apply(
            DecisionLevel.AMBIGUITY_RESPONSE,
            ["confirm_selection:second_click", "wait_for_evidence", "ask_clarification:2"],
            context="study",
            modality="mouse",
            participant_key=PK,
            interruptions_used=2,
        )
        assert allowed == ["wait_for_evidence"]
        assert excluded["ask_clarification:2"] is ExclusionReason.INTERRUPTION_LIMIT


# ---------------------------------------------------------------------------------------------------- conflicts


def test_wait_for_evidence_cannot_be_forbidden() -> None:
    with pytest.raises(PreferenceConflict) as error:
        PreferenceSet([]).validate_new(rule(PreferenceKind.FORBID_VARIANT, "wait_for_evidence"))
    assert "Always select exactly what I click" in error.value.message


def test_the_l1_set_never_empties_for_any_modality_in_scope() -> None:
    # keyboard has no enlarge arms: forbidding act and the confirm family empties keyboard only
    prefs = build(rule(PreferenceKind.FORBID_VARIANT, "act"))
    with pytest.raises(PreferenceConflict) as error:
        prefs.validate_new(rule(PreferenceKind.FORBID_VARIANT, "confirm_selection"))
    assert "keyboard" in error.value.message
    # the same rule scoped to the mouse passes (enlarge arms remain)
    prefs.validate_new(rule(PreferenceKind.FORBID_VARIANT, "confirm_selection", modality="mouse"))
    # only the hand pointer has hold_400 and no second_click: emptying it alone is rejected
    prefs = build(
        rule(PreferenceKind.FORBID_VARIANT, "act"),
        rule(PreferenceKind.FORBID_VARIANT, "enlarge_target"),
        rule(PreferenceKind.FORBID_VARIANT, "confirm_selection:hold_400"),
    )
    with pytest.raises(PreferenceConflict) as error:
        prefs.validate_new(rule(PreferenceKind.FORBID_VARIANT, "confirm_selection:hold_700"))
    assert "hand pointer" in error.value.message
    # a device-wide forbid that empties one participant's set (with their own minimum) is rejected too
    prefs = build(rule(PreferenceKind.MIN_CAUTION, "confirm_selection:hold_700", participant=PK))
    with pytest.raises(PreferenceConflict):
        prefs.validate_new(rule(PreferenceKind.FORBID_VARIANT, "confirm_selection:hold_700"))


def test_pin_and_freeze_against_forbid_in_both_orders() -> None:
    assert_rejected_both_orders(
        rule(PreferenceKind.FORBID_VARIANT, "enlarge_target"), rule(PreferenceKind.PIN_REGIME, "enlarge_target:1.5")
    )
    assert_rejected_both_orders(
        rule(PreferenceKind.FORBID_VARIANT, "act"), rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"})
    )


def test_pin_act_contradictions_are_rejected_with_their_messages() -> None:
    pin_act = rule(PreferenceKind.PIN_ACT, True)
    for other, message in (
        (rule(PreferenceKind.FORBID_VARIANT, "act"), "cannot be turned off"),
        (rule(PreferenceKind.MIN_CAUTION, "enlarge_target:1.5"), "A minimum caution above direct selection"),
        (rule(PreferenceKind.PIN_REGIME, "confirm_selection:second_click"), "Always select exactly what I click"),
        (rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "confirm_selection:hold_700"}), "confirm step"),
    ):
        assert_rejected_both_orders(pin_act, other, message)
    # a minimum caution at act itself is compatible with pin_act
    build(pin_act, rule(PreferenceKind.MIN_CAUTION, "act"))


def test_cap_with_a_confirm_level_min_caution_is_accepted_with_a_note() -> None:
    prefs = build(rule(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click", participant=PK))
    prefs, notes = save(prefs, rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2))
    assert notes and "does not remove them" in notes[0]
    # the cap then does not remove the confirm regimes (9.2 C1)
    assert l1_allowed(prefs) == ["confirm_selection:second_click", "confirm_selection:hold_700"]
    # and in the other order
    prefs = build(rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2))
    _, notes = save(prefs, rule(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click", participant=PK))
    assert notes


@pytest.mark.parametrize("hold_kind", [PreferenceKind.FREEZE_ADAPTATION, PreferenceKind.PIN_REGIME])
def test_freezes_and_pins_against_the_other_rules_in_both_orders(hold_kind: PreferenceKind) -> None:
    def hold(regime: str, **kwargs: Any) -> Preference:
        value = {"regime": regime} if hold_kind is PreferenceKind.FREEZE_ADAPTATION else regime
        return rule(hold_kind, value, **kwargs)

    # below an active minimum caution, and a minimum caution above an active freeze or pin
    assert_rejected_both_orders(
        rule(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click"), hold("enlarge_target:2.0"), "below"
    )
    # a cap of 2 against a frozen or pinned hold_700, and a hold of second_click against a cap of 2
    assert_rejected_both_orders(
        rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2), hold("confirm_selection:hold_700"), "step in every"
    )
    assert_rejected_both_orders(
        rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2), hold("confirm_selection:second_click")
    )
    # a cap of 8 is accepted in both orders
    build(rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 8), hold("confirm_selection:hold_700"))
    build(hold("confirm_selection:second_click"), rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 8))


def test_a_freeze_and_a_pin_must_name_the_same_regime() -> None:
    assert_rejected_both_orders(
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}), rule(PreferenceKind.PIN_REGIME, "enlarge_target:1.5")
    )
    build(rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}), rule(PreferenceKind.PIN_REGIME, "act"))


def test_same_kind_rules_naming_different_regimes_are_rejected() -> None:
    assert_rejected_both_orders(
        rule(PreferenceKind.PIN_REGIME, "act"), rule(PreferenceKind.PIN_REGIME, "enlarge_target:2.0"), "already pinned"
    )
    assert_rejected_both_orders(
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, participant=PK),
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "assistance_off"}),
        "already paused adapting",
    )
    build(rule(PreferenceKind.PIN_REGIME, "act"), rule(PreferenceKind.PIN_REGIME, "act", participant=PK))
    # a keyboard-scoped freeze and a hand-pointer-scoped freeze do not overlap
    build(
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, modality="keyboard"),
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "enlarge_target:1.5"}, modality="hand_pointer"),
    )


def test_scope_overlap_rules() -> None:
    # device-wide minimum against a participant-scoped freeze, and the reverse
    assert_rejected_both_orders(
        rule(PreferenceKind.MIN_CAUTION, "enlarge_target:2.0"),
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, participant=PK, modality="mouse"),
    )
    # a keyboard-scoped rule is not checked against a hand-pointer freeze
    build(
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, modality="hand_pointer"),
        rule(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click", modality="keyboard"),
    )
    # a context "all" rule is checked against a study freeze
    assert_rejected_both_orders(
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}),
        rule(PreferenceKind.MIN_CAUTION, "enlarge_target:1.5", context="all"),
    )
    # rules of two different participants do not overlap
    build(
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, participant=PK),
        rule(PreferenceKind.MIN_CAUTION, "enlarge_target:1.5", participant=OTHER),
    )


# ------------------------------------------------------------------------------------- freezable, freeze_conflict


def test_freezable_lists_the_arms_whose_freeze_passes_in_caution_order() -> None:
    prefs = PreferenceSet([])
    assert prefs.freezable(modality="mouse", participant_key=PK) == tuple(
        v for v in CAUTION_ORDER if v in l1_arms("mouse")
    )
    assert prefs.freezable(modality="mouse", participant_key=PK)[0] == "assistance_off"
    prefs = build(rule(PreferenceKind.MIN_CAUTION, "enlarge_target:2.0"))
    assert prefs.freezable(modality="mouse", participant_key=PK) == (
        "enlarge_target:2.0",
        "confirm_selection:second_click",
        "confirm_selection:hold_700",
    )
    # empty under a confirm-level minimum caution with a cap of 2
    prefs = build(
        rule(PreferenceKind.MIN_CAUTION, "confirm_selection:second_click"),
        rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 2),
    )
    assert prefs.freezable(modality="mouse", participant_key=PK) == ()
    for regime in l1_arms("mouse"):
        assert prefs.freeze_conflict(regime, modality="mouse", participant_key=PK) is not None


def test_freezable_leaves_out_only_the_same_scope_freeze() -> None:
    mine = rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, modality="mouse", participant=PK)
    prefs = build(mine)
    full = tuple(v for v in CAUTION_ORDER if v in l1_arms("mouse"))
    assert prefs.freezable(modality="mouse", participant_key=PK) == full
    assert prefs.freeze_conflict("confirm_selection:hold_700", modality="mouse", participant_key=PK) is None
    device = build(rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}))
    assert device.freezable(modality="mouse", participant_key=PK) == ("act",)
    conflict = device.freeze_conflict("assistance_off", modality="mouse", participant_key=PK)
    assert conflict is not None and len(conflict.conflicting_pref_ids) == 1


def test_freeze_conflict_is_none_exactly_for_freezable_regimes() -> None:
    sets = [
        PreferenceSet([]),
        build(rule(PreferenceKind.MIN_CAUTION, "enlarge_target:1.5")),
        build(rule(PreferenceKind.PIN_REGIME, "enlarge_target:2.0")),
        build(rule(PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 3)),
        build(rule(PreferenceKind.FORBID_VARIANT, "assistance_off"), rule(PreferenceKind.PIN_ACT, True)),
    ]
    for prefs in sets:
        for modality in MODALITIES:
            freezable = prefs.freezable(modality=modality, participant_key=PK)  # type: ignore[arg-type]
            for regime in l1_arms(modality):
                conflict = prefs.freeze_conflict(regime, modality=modality, participant_key=PK)  # type: ignore[arg-type]
                assert (conflict is None) == (regime in freezable)


def test_freezable_ignores_calibration_suspension() -> None:
    prefs = PreferenceSet([])
    suspended = fallbacks(prefs, "mouse", participant_key=PK, calibration_suspended=True, now=T0)
    assert suspended.freezable == prefs.freezable(modality="mouse", participant_key=PK)
    assert "assistance_off" in suspended.freezable


# ------------------------------------------------------------------------------------------------ lookups, I/O


def test_active_at_honours_created_and_retracted_times() -> None:
    pref = _retract(rule(PreferenceKind.FORBID_VARIANT, "act", minutes=10), minutes=20)
    prefs = PreferenceSet([pref])
    kwargs = {"context": "study", "modality": "mouse", "participant_key": PK}
    assert prefs.active_at(T0 + timedelta(minutes=9), **kwargs) == ()
    assert prefs.active_at(T0 + timedelta(minutes=10), **kwargs) == (pref,)
    assert prefs.active_at(T0 + timedelta(minutes=19), **kwargs) == (pref,)
    assert prefs.active_at(T0 + timedelta(minutes=20), **kwargs) == ()
    assert prefs.active(**kwargs) == ()
    assert prefs.as_of(T0 + timedelta(minutes=15)).active(**kwargs)[0].retracted_at is None


def test_scope_participant_keys_must_match_the_pattern() -> None:
    for bad in ("Alice", "p-123", "p-3FA29C1E", "p-3fa29c1e0"):
        with pytest.raises(ValueError):
            PreferenceScope(context="study", participant_key=bad)
    with pytest.raises(ValueError):
        PreferenceScope(context="approval")  # type: ignore[arg-type]
    PreferenceScope(context="all", modality="pen", participant_key="p-3fa29c1e")


def test_values_are_validated_per_kind() -> None:
    bad = [
        (PreferenceKind.PIN_REGIME, "assistance_off"),
        (PreferenceKind.PIN_REGIME, "wait_for_evidence"),
        (PreferenceKind.PIN_ACT, False),
        (PreferenceKind.MIN_CAUTION, "offer_modality:keyboard"),
        (PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, 9),
        (PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, True),
        (PreferenceKind.PRIORITIES, {"fewer_mistakes": 2.0, "faster": 0.5, "fewer_interruptions": 0.5}),
        (PreferenceKind.PRIORITIES, {"faster": 0.5}),
        (PreferenceKind.FREEZE_ADAPTATION, {"regime": "ask_clarification:2"}),
        (PreferenceKind.FORBID_VARIANT, "click"),
    ]
    for kind, value in bad:
        with pytest.raises(ValueError):
            rule(kind, value)
    with pytest.raises(ValueError):
        Preference("x", PreferenceKind.PIN_ACT, PreferenceScope("study"), True, "learned", T0)  # type: ignore[arg-type]


def test_round_trip_and_stable_hash() -> None:
    prefs = build(
        rule(PreferenceKind.PRIORITIES, {"fewer_mistakes": 1, "faster": 0.25, "fewer_interruptions": 0.0}),
        rule(PreferenceKind.FREEZE_ADAPTATION, {"regime": "act"}, participant=PK, modality="mouse"),
        rule(PreferenceKind.FORBID_VARIANT, "ask_clarification"),
    )
    for pref in prefs.all:
        again = preference_from_dict(json.loads(json.dumps(preference_to_dict(pref))))
        assert again == pref
    reordered = PreferenceSet(reversed(prefs.all))
    assert prefs.hash() == reordered.hash() == PreferenceSet(list(prefs.all)).hash()
    assert len(prefs.hash()) == 64
    assert build(*prefs.all[:2]).hash() != prefs.hash()
    with pytest.raises(ValueError):
        preference_from_dict({**preference_to_dict(prefs.all[0]), "extra": 1})
    with pytest.raises(ValueError):
        preference_from_dict({key: v for key, v in preference_to_dict(prefs.all[0]).items() if key != "kind"})


def test_the_engine_exposes_no_preference_writing_method() -> None:
    from pilot.coadaptation.core.engine import CoAdaptationEngine

    names = [name for name in dir(CoAdaptationEngine) if not name.startswith("__")]
    writers = [n for n in names if any(w in n.lower() for w in ("add_pref", "write_pref", "save_pref", "forbid"))]
    assert writers == []
    assert "set_preferences" in names  # receives a whole set built from the person's file, never creates rules

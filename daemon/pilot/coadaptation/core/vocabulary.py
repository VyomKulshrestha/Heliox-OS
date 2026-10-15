"""Assistance strategy vocabulary (spec section 4).

Strategies change how a Heliox-rendered selection surface presents candidates and when the person's own input is
accepted as a selection. They are not planner action types: the planner, executor, gateway, verifier and capability
catalog never import this module, and no strategy or variant id equals an action type value (tested).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

STRATEGY_VERSION = "coadapt-strategies-v1"


class AssistanceStrategy(StrEnum):
    ACT = "act"
    ENLARGE_TARGET = "enlarge_target"
    CONFIRM_SELECTION = "confirm_selection"
    WAIT_FOR_EVIDENCE = "wait_for_evidence"
    ASK_CLARIFICATION = "ask_clarification"
    OFFER_MODALITY = "offer_modality"


ASSISTANCE_OFF = "assistance_off"  # L1 regime id: the plain interface (today's Heliox behaviour); not a strategy
NO_OFFER = "no_offer"  # L3 baseline id


class DecisionLevel(StrEnum):
    BLOCK_REGIME = "block_regime"  # L1, once per block
    AMBIGUITY_RESPONSE = "ambiguity_response"  # L2, per ambiguous activation (served as a per-block table)
    OFFER = "offer"  # L3, when a struggle signal fires


class VariantRole(StrEnum):
    REGIME_OFF = "regime_off"
    PRESENTATION = "presentation"
    COMMIT_RULE = "commit_rule"
    OFFER = "offer"


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    strategy: AssistanceStrategy | None
    params: tuple[tuple[str, str | float], ...]
    role: VariantRole
    acting: bool  # commits the literal hit on a single activation
    interrupts: bool  # adds an explicit step (counts toward max_interruptions_per_block; an L1 regime's steps
    #   at most once per trial, 9.3)
    caution: int  # CAUTION_ORDER rank; -1 = outside the order
    levels: frozenset[DecisionLevel]

    def param(self, name: str) -> str | float | None:
        """The value of one parameter, or None when the variant has no such parameter."""
        for key, value in self.params:
            if key == name:
                return value
        return None


_L1 = frozenset({DecisionLevel.BLOCK_REGIME})
_L2 = frozenset({DecisionLevel.AMBIGUITY_RESPONSE})
_L3 = frozenset({DecisionLevel.OFFER})
_L1_L2 = frozenset({DecisionLevel.BLOCK_REGIME, DecisionLevel.AMBIGUITY_RESPONSE})

_S = AssistanceStrategy
_R = VariantRole

# The table of spec 4.2, in its order.
VARIANTS: Mapping[str, Variant] = MappingProxyType(
    {
        v.id: v
        for v in (
            Variant(ASSISTANCE_OFF, None, (), _R.REGIME_OFF, True, False, 0, _L1),
            Variant("act", _S.ACT, (), _R.COMMIT_RULE, True, False, 1, _L1_L2),
            Variant("enlarge_target:1.5", _S.ENLARGE_TARGET, (("scale", 1.5),), _R.PRESENTATION, True, False, 2, _L1),
            Variant("enlarge_target:2.0", _S.ENLARGE_TARGET, (("scale", 2.0),), _R.PRESENTATION, True, False, 3, _L1),
            Variant(
                "confirm_selection:second_click",
                _S.CONFIRM_SELECTION,
                (("method", "second_click"),),
                _R.COMMIT_RULE,
                False,
                True,
                4,
                _L1_L2,
            ),
            Variant(
                "confirm_selection:hold_400",
                _S.CONFIRM_SELECTION,
                (("method", "hold"), ("hold_ms", 400)),
                _R.COMMIT_RULE,
                False,
                True,
                5,
                _L1,
            ),
            Variant(
                "confirm_selection:hold_700",
                _S.CONFIRM_SELECTION,
                (("method", "hold"), ("hold_ms", 700)),
                _R.COMMIT_RULE,
                False,
                True,
                6,
                _L1,
            ),
            Variant("wait_for_evidence", _S.WAIT_FOR_EVIDENCE, (), _R.COMMIT_RULE, False, False, 7, _L2),
            Variant("ask_clarification:2", _S.ASK_CLARIFICATION, (("k", 2),), _R.COMMIT_RULE, False, True, 8, _L2),
            Variant("ask_clarification:3", _S.ASK_CLARIFICATION, (("k", 3),), _R.COMMIT_RULE, False, True, 9, _L2),
            Variant(NO_OFFER, None, (), _R.OFFER, False, False, -1, _L3),
            Variant(
                "offer_modality:keyboard",
                _S.OFFER_MODALITY,
                (("target", "keyboard"),),
                _R.OFFER,
                False,
                False,
                -1,
                _L3,
            ),
            Variant(
                "offer_modality:pointer",
                _S.OFFER_MODALITY,
                (("target", "pointer"),),
                _R.OFFER,
                False,
                False,
                -1,
                _L3,
            ),
        )
    }
)

# Variants inside the caution order, least cautious first (spec 4.2).
CAUTION_ORDER: tuple[str, ...] = tuple(
    v.id for v in sorted((v for v in VARIANTS.values() if v.caution >= 0), key=lambda v: v.caution)
)

L2_OPTIONS: tuple[str, ...] = (
    "act",
    "confirm_selection:second_click",
    "wait_for_evidence",
    "ask_clarification:2",
    "ask_clarification:3",
)
L3_OPTIONS: tuple[str, ...] = (NO_OFFER, "offer_modality:keyboard", "offer_modality:pointer")

NOT_APPLICABLE = "not_applicable"  # the ExclusionReason value applicability_reason returns (spec 4.5, 9.8)


def l1_arms(modality: str) -> tuple[str, ...]:
    """The L1 regimes of a modality (spec 4.4). Keyboard has no enlarge arms."""
    if modality == "keyboard":
        return (ASSISTANCE_OFF, "act", "confirm_selection:second_click", "confirm_selection:hold_700")
    fast_confirm = "confirm_selection:hold_400" if modality == "hand_pointer" else "confirm_selection:second_click"
    return (
        ASSISTANCE_OFF,
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
        fast_confirm,
        "confirm_selection:hold_700",
    )


def variant(variant_id: str) -> Variant:
    """The Variant for an id; raises ValueError for an unknown id."""
    try:
        return VARIANTS[variant_id]
    except KeyError:
        raise ValueError("unknown variant id") from None


def acting(variant_id: str) -> bool:
    """True when the variant commits the literal hit on a single activation."""
    return variant(variant_id).acting


def family_of(variant_id: str) -> str:
    """The family of a variant: its strategy value, or the id itself for assistance_off and no_offer."""
    v = variant(variant_id)
    return v.strategy.value if v.strategy is not None else v.id


def _modality_of(ctx: Any) -> str | None:
    if isinstance(ctx, str):
        return ctx
    device = getattr(ctx, "device", None)
    if device is not None:
        return getattr(device, "modality", None)
    return getattr(ctx, "modality", None)


def applicability_reason(variant_id: str, ctx: Any) -> str | None:
    """None when the variant applies in this context, else ``"not_applicable"`` (spec 4.5).

    The level is read from the context: an ``ActivationContext`` (it has ``has_hit``) applies the L2 rules, an
    ``OfferContext`` (it has ``keyboard_available``) the L3 rules, and anything else (a ``BlockContext``, a
    ``DeviceContext`` or a modality string) the L1 rules. A variant that is not an option of that level is not
    applicable there. The returned string is exactly the ``ExclusionReason.NOT_APPLICABLE`` value, so a caller can use
    it as the logged exclusion reason.
    """
    v = variant(variant_id)
    modality = _modality_of(ctx)
    if hasattr(ctx, "has_hit"):
        if variant_id not in L2_OPTIONS:
            return NOT_APPLICABLE
        if variant_id in ("act", "confirm_selection:second_click") and not ctx.has_hit:
            return NOT_APPLICABLE
        k = v.param("k")
        if k is not None and int(ctx.n_plausible) < int(k):
            return NOT_APPLICABLE
        return None
    if hasattr(ctx, "keyboard_available"):
        if variant_id not in L3_OPTIONS:
            return NOT_APPLICABLE
        if variant_id == "offer_modality:keyboard":
            return None if ctx.keyboard_available and modality != "keyboard" else NOT_APPLICABLE
        if variant_id == "offer_modality:pointer":
            return None if modality == "hand_pointer" else NOT_APPLICABLE
        return None
    if modality is None or variant_id not in l1_arms(modality):
        return NOT_APPLICABLE
    return None

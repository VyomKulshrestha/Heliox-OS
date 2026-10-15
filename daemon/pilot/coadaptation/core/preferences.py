"""Explicit preferences: the person's own rules (section 9.6 of the co-adaptation spec).

Preferences are hard constraints with origin ``explicit_user``. They outrank learning, are retractable, and are never
written by the engine or the model: the only way to build a ``PreferenceSet`` is from rules the person saved.
``validate_new`` rejects a rule that conflicts with an active rule whose scope overlaps it (freezes and pins
included) and names the conflicting rule; an existing rule is never rewritten, ignored or outranked.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from .vocabulary import VARIANTS, DecisionLevel, l1_arms

if TYPE_CHECKING:
    from .constraints import ExclusionReason
    from .state import Modality

MODALITIES: tuple[str, ...] = ("mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer")
FAMILIES: tuple[str, ...] = ("confirm_selection", "ask_clarification", "enlarge_target", "offer_modality")
PRIORITY_KEYS: tuple[str, ...] = ("fewer_mistakes", "faster", "fewer_interruptions")
DEFAULT_PRIORITIES: tuple[float, float, float] = (0.5, 0.5, 0.5)
BLOCK_TRIALS = 8  # trials per main block; a confirm or hold regime adds one interruption per trial (9.3)
MAX_CAP = 8

_PARTICIPANT_KEY_RE = re.compile(r"^p-[0-9a-f]{8}$")
_RETRACT_OFFER = "Remove that rule and save this one, or keep it."
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

VARIANT_NAMES: dict[str, str] = {
    "assistance_off": "Standard selection",
    "act": "Select what I click",
    "enlarge_target:1.5": "Larger targets (1.5x)",
    "enlarge_target:2.0": "Larger targets (2x)",
    "confirm_selection:second_click": "Confirm each selection",
    "confirm_selection:hold_400": "Hold to select (0.4 s)",
    "confirm_selection:hold_700": "Hold to select (0.7 s)",
    "wait_for_evidence": "Wait and ask me to aim again",
    "ask_clarification:2": "Ask which of two I meant",
    "ask_clarification:3": "Ask which of three I meant",
    "no_offer": "No input suggestions",
    "offer_modality:keyboard": "Suggest the keyboard",
    "offer_modality:pointer": "Suggest the mouse",
}
FAMILY_NAMES: dict[str, str] = {
    "confirm_selection": "Confirm and hold steps",
    "ask_clarification": "Clarifying questions",
    "enlarge_target": "Larger targets",
    "offer_modality": "Input suggestions",
}


class PreferenceKind(StrEnum):
    FORBID_VARIANT = "forbid_variant"
    PIN_REGIME = "pin_regime"
    PIN_ACT = "pin_act"
    MIN_CAUTION = "min_caution"
    MAX_INTERRUPTIONS_PER_BLOCK = "max_interruptions_per_block"
    PRIORITIES = "priorities"
    FREEZE_ADAPTATION = "freeze_adaptation"


def _l1_ids() -> frozenset[str]:
    return frozenset(arm for modality in MODALITIES for arm in l1_arms(modality))


def caution_of(variant_id: str) -> int:
    """CAUTION_ORDER rank of a variant (-1 outside the order)."""
    return VARIANTS[variant_id].caution


def family_name(variant_id: str) -> str:
    return variant_id.split(":", 1)[0]


def is_confirm_regime(variant_id: str) -> bool:
    """Confirm and hold regimes: one explicit step per trial (``confirm_selection:*``)."""
    return family_name(variant_id) == "confirm_selection"


def variant_name(variant_id: str) -> str:
    return VARIANT_NAMES.get(variant_id, FAMILY_NAMES.get(variant_id, variant_id))


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class PreferenceScope:
    context: Literal["study", "all"]  # "all" also covers SDK host surfaces; there is no approval context
    modality: Modality | None = None
    participant_key: str | None = None  # None = everyone on this device; else an existing ^p-[0-9a-f]{8}$ key

    def __post_init__(self) -> None:
        if self.context not in ("study", "all"):
            raise ValueError("scope context must be 'study' or 'all'")
        if self.modality is not None and self.modality not in MODALITIES:
            raise ValueError("scope modality is not a known modality")
        if self.participant_key is not None and (
            not isinstance(self.participant_key, str) or _PARTICIPANT_KEY_RE.fullmatch(self.participant_key) is None
        ):
            raise ValueError("scope participant_key must be null or match ^p-[0-9a-f]{8}$")

    def overlaps(self, other: PreferenceScope) -> bool:
        """Contexts equal or one is 'all'; modalities equal or one null; participants equal or one null."""
        return (
            (self.context == other.context or "all" in (self.context, other.context))
            and (self.modality is None or other.modality is None or self.modality == other.modality)
            and (
                self.participant_key is None
                or other.participant_key is None
                or self.participant_key == other.participant_key
            )
        )

    def covers(self, *, context: str, modality: str, participant_key: str | None) -> bool:
        """Whether a rule with this scope applies at one point (a context, a modality and a participant)."""
        return (
            (self.context == "all" or self.context == context)
            and (self.modality is None or self.modality == modality)
            and (self.participant_key is None or self.participant_key == participant_key)
        )

    def modalities(self) -> tuple[str, ...]:
        return MODALITIES if self.modality is None else (self.modality,)

    def describe(self) -> str:
        who = "Everyone on this device" if self.participant_key is None else f"Participant {self.participant_key}"
        return who if self.modality is None else f"{who}, {self.modality.replace('_', ' ')} only"


def _validate_value(kind: PreferenceKind, value: Any) -> None:
    if kind is PreferenceKind.FORBID_VARIANT:
        if not isinstance(value, str) or (value not in VARIANTS and value not in FAMILIES):
            raise ValueError("forbid_variant needs a variant id or a family name")
    elif kind is PreferenceKind.PIN_REGIME:
        if not isinstance(value, str) or value not in _l1_ids() or value == "assistance_off":
            raise ValueError("pin_regime needs an L1 regime other than assistance_off")
    elif kind is PreferenceKind.PIN_ACT:
        if value is not True:
            raise ValueError("pin_act needs the value true")
    elif kind is PreferenceKind.MIN_CAUTION:
        if not isinstance(value, str) or value not in VARIANTS or VARIANTS[value].caution < 0:
            raise ValueError("min_caution needs a variant in CAUTION_ORDER")
    elif kind is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK:
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_CAP:
            raise ValueError("max_interruptions_per_block needs an integer in [0, 8]")
    elif kind is PreferenceKind.PRIORITIES:
        if not isinstance(value, Mapping) or set(value) != set(PRIORITY_KEYS):
            raise ValueError("priorities needs fewer_mistakes, faster and fewer_interruptions")
        for key in PRIORITY_KEYS:
            item = value[key]
            if not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(item):
                raise ValueError("priorities values must be numbers")
            if not 0.0 <= float(item) <= 1.0:
                raise ValueError("priorities values must lie in [0, 1]")
    elif kind is PreferenceKind.FREEZE_ADAPTATION:
        if not isinstance(value, Mapping) or set(value) != {"regime"}:
            raise ValueError("freeze_adaptation needs {'regime': L1 id}")
        if not isinstance(value["regime"], str) or value["regime"] not in _l1_ids():
            raise ValueError("freeze_adaptation needs an L1 regime")


@dataclass(frozen=True, slots=True)
class Preference:
    pref_id: str
    kind: PreferenceKind
    scope: PreferenceScope
    value: Any
    origin: Literal["explicit_user"]
    created_at: datetime
    retracted_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pref_id, str) or not 1 <= len(self.pref_id) <= 128:
            raise ValueError("pref_id must be a non-empty string")
        if not isinstance(self.kind, PreferenceKind):
            raise ValueError("kind must be a PreferenceKind")
        if not isinstance(self.scope, PreferenceScope):
            raise ValueError("scope must be a PreferenceScope")
        if self.origin != "explicit_user":
            raise ValueError("origin must be explicit_user")
        if not _is_aware(self.created_at):
            raise ValueError("created_at must carry a UTC offset")
        if self.retracted_at is not None and (not _is_aware(self.retracted_at) or self.retracted_at < self.created_at):
            raise ValueError("retracted_at must carry a UTC offset and not precede created_at")
        _validate_value(self.kind, self.value)
        if self.kind is PreferenceKind.PRIORITIES:
            object.__setattr__(self, "value", {key: float(self.value[key]) for key in PRIORITY_KEYS})
        elif self.kind is PreferenceKind.FREEZE_ADAPTATION:
            object.__setattr__(self, "value", {"regime": self.value["regime"]})

    def __hash__(self) -> int:
        return hash((self.pref_id, self.kind, self.created_at, self.retracted_at))

    @property
    def regime(self) -> str | None:
        """The regime a pin or freeze names; None for other kinds."""
        if self.kind is PreferenceKind.PIN_REGIME:
            return self.value
        if self.kind is PreferenceKind.FREEZE_ADAPTATION:
            return self.value["regime"]
        return None

    def active_at(self, when: datetime) -> bool:
        return self.created_at <= when and (self.retracted_at is None or when < self.retracted_at)

    def forbids(self, variant_id: str) -> bool:
        return self.kind is PreferenceKind.FORBID_VARIANT and self.value in (variant_id, family_name(variant_id))

    def summary(self) -> str:
        """The rule in words, as the rules editor shows it."""
        kind = self.kind
        if kind is PreferenceKind.FORBID_VARIANT:
            return f"Never use: {variant_name(self.value)}"
        if kind is PreferenceKind.PIN_REGIME:
            return f"Always use: {variant_name(self.value)}"
        if kind is PreferenceKind.PIN_ACT:
            return "Always select exactly what I click"
        if kind is PreferenceKind.MIN_CAUTION:
            return f"Minimum caution: {variant_name(self.value)}"
        if kind is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK:
            return f"At most {self.value} interruptions per block"
        if kind is PreferenceKind.PRIORITIES:
            return "Priorities"
        return f"Pause adapting at {variant_name(self.value['regime'])}"


class PreferenceConflict(ValueError):
    """A new rule conflicts with active rules. Nothing is written and no existing rule changes."""

    def __init__(self, message: str, conflicting_pref_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.conflicting_pref_ids: tuple[str, ...] = tuple(conflicting_pref_ids)


def _format_datetime(value: datetime) -> str:
    return value.isoformat()


def _parse_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from None
    if not _is_aware(parsed):
        raise ValueError(f"{name} must carry a UTC offset")
    return parsed


def preference_to_dict(pref: Preference) -> dict[str, Any]:
    value = pref.value
    if isinstance(value, Mapping):
        value = dict(value)
    return {
        "pref_id": pref.pref_id,
        "kind": pref.kind.value,
        "scope": {
            "context": pref.scope.context,
            "modality": pref.scope.modality,
            "participant_key": pref.scope.participant_key,
        },
        "value": value,
        "origin": pref.origin,
        "created_at": _format_datetime(pref.created_at),
        "retracted_at": None if pref.retracted_at is None else _format_datetime(pref.retracted_at),
    }


def preference_from_dict(d: Mapping[str, Any]) -> Preference:
    """Parse one stored rule. Raises ``ValueError`` on unknown, missing or wrongly typed keys (values not echoed)."""
    if not isinstance(d, Mapping):
        raise ValueError("Preference must be an object")
    keys = {"pref_id", "kind", "scope", "value", "origin", "created_at", "retracted_at"}
    unknown = sorted(str(key) for key in d if key not in keys)
    if unknown:
        raise ValueError(f"Preference has unknown keys: {unknown}")
    missing = sorted(keys - {"retracted_at"} - set(d))
    if missing:
        raise ValueError(f"Preference is missing keys: {missing}")
    scope = d["scope"]
    if not isinstance(scope, Mapping):
        raise ValueError("Preference scope must be an object")
    scope_keys = {"context", "modality", "participant_key"}
    if set(scope) - scope_keys:
        raise ValueError("Preference scope has unknown keys")
    if "context" not in scope:
        raise ValueError("Preference scope is missing keys: ['context']")
    for name in ("modality", "participant_key"):
        if scope.get(name) is not None and not isinstance(scope[name], str):
            raise ValueError(f"Preference scope {name} must be a string or null")
    if not isinstance(d["pref_id"], str) or not isinstance(d["kind"], str) or not isinstance(d["origin"], str):
        raise ValueError("pref_id, kind and origin must be strings")
    try:
        kind = PreferenceKind(d["kind"])
    except ValueError:
        raise ValueError("Preference kind is unknown") from None
    retracted = d.get("retracted_at")
    return Preference(
        pref_id=d["pref_id"],
        kind=kind,
        scope=PreferenceScope(
            context=scope["context"], modality=scope.get("modality"), participant_key=scope.get("participant_key")
        ),
        value=d["value"],
        origin=d["origin"],
        created_at=_parse_datetime(d["created_at"], "created_at"),
        retracted_at=None if retracted is None else _parse_datetime(retracted, "retracted_at"),
    )


@dataclass(frozen=True, slots=True)
class _Issue:
    message: str
    pref_ids: tuple[str, ...]


def _in_effect(pref: Preference, modality: str) -> bool:
    """A pin or freeze is in effect for a modality only when it names one of that modality's L1 arms."""
    regime = pref.regime
    return regime is not None and regime in l1_arms(modality)


def _held(pref: Preference) -> str:
    """'paused adapting at X' or 'pinned X' for a freeze or a pin."""
    regime = variant_name(pref.regime or "")
    return f"paused adapting at {regime}" if pref.kind is PreferenceKind.FREEZE_ADAPTATION else f"pinned {regime}"


def _pair_issue(new: Preference, old: Preference, modality: str) -> str | None:
    """The conflict message between a new and an existing rule at one modality, or None."""
    k_new, k_old = new.kind, old.kind
    holds = (PreferenceKind.PIN_REGIME, PreferenceKind.FREEZE_ADAPTATION)

    # Pins and freezes against forbids, in either order.
    if k_new in holds and k_old is PreferenceKind.FORBID_VARIANT and _in_effect(new, modality):
        if old.forbids(new.regime or ""):
            return f"You turned off {variant_name(old.value)}, so {variant_name(new.regime or '')} cannot be kept."
    if k_new is PreferenceKind.FORBID_VARIANT and k_old in holds and _in_effect(old, modality):
        if new.forbids(old.regime or ""):
            return f"You {_held(old)}; turning it off would contradict that."

    # Pins and freezes against a minimum caution, in either order.
    if k_new in holds and k_old is PreferenceKind.MIN_CAUTION and _in_effect(new, modality):
        if caution_of(new.regime or "") < caution_of(old.value):
            return f"Your minimum caution is {variant_name(old.value)}; {variant_name(new.regime or '')} is below it."
    if k_new is PreferenceKind.MIN_CAUTION and k_old in holds and _in_effect(old, modality):
        if caution_of(old.regime or "") < caution_of(new.value):
            return f"You {_held(old)}, which is below this minimum."

    # Pins and freezes of a confirm or hold regime against an interruption cap below the block's trials.
    if k_new in holds and k_old is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK and _in_effect(new, modality):
        if is_confirm_regime(new.regime or "") and old.value < BLOCK_TRIALS:
            return (
                f"{variant_name(new.regime or '')} asks for a step in every selection, more than your limit of "
                f"{old.value} interruptions per block allows."
            )
    if k_new is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK and k_old in holds and _in_effect(old, modality):
        if is_confirm_regime(old.regime or "") and new.value < BLOCK_TRIALS:
            return f"You {_held(old)}, which asks for a step in every selection, more than this limit allows."

    # Pins and freezes against each other: they may only name the same regime.
    if k_new in holds and k_old in holds and _in_effect(new, modality) and _in_effect(old, modality):
        if new.regime != old.regime:
            if k_new is PreferenceKind.FREEZE_ADAPTATION and k_old is PreferenceKind.PIN_REGIME:
                return f"You already pinned {variant_name(old.regime or '')}; pausing at {variant_name(new.regime or '')} would contradict it."
            if k_new is PreferenceKind.PIN_REGIME and k_old is PreferenceKind.FREEZE_ADAPTATION:
                return f"You already paused adapting at {variant_name(old.regime or '')}; pinning {variant_name(new.regime or '')} would contradict it."
            if k_new is PreferenceKind.PIN_REGIME:
                return f"You already pinned {variant_name(old.regime or '')}; pinning {variant_name(new.regime or '')} would contradict it."
            return f"You already paused adapting at {variant_name(old.regime or '')}; pausing at {variant_name(new.regime or '')} would contradict it."

    # "Always select exactly what I click" contradictions, in either order.
    pin_act_message = None
    if k_new is PreferenceKind.PIN_ACT:
        pin_act_message = _pin_act_issue(old, modality)
    elif k_old is PreferenceKind.PIN_ACT:
        pin_act_message = _pin_act_issue(new, modality)
    return pin_act_message


def _pin_act_issue(other: Preference, modality: str) -> str | None:
    if other.kind is PreferenceKind.FORBID_VARIANT and other.forbids("act"):
        return (
            "You chose to have every click select what it hits, so 'act' cannot be turned off. "
            "Remove one of the two rules."
        )
    if other.kind is PreferenceKind.MIN_CAUTION and caution_of(other.value) > caution_of("act"):
        return "A minimum caution above direct selection contradicts 'Always select exactly what I click'."
    if (
        other.kind in (PreferenceKind.PIN_REGIME, PreferenceKind.FREEZE_ADAPTATION)
        and _in_effect(other, modality)
        and is_confirm_regime(other.regime or "")
    ):
        return f"'Always select exactly what I click' contradicts a rule that {_held(other)}, which asks for a confirm step."
    return None


@dataclass(frozen=True, slots=True)
class _Point:
    context: str
    modality: str
    participant_key: str | None


class PreferenceSet:
    """The person's rules, with the lookups the policy, the fallbacks and the audit need."""

    def __init__(self, prefs: Iterable[Preference]) -> None:
        items = tuple(prefs)
        for pref in items:
            if not isinstance(pref, Preference):
                raise ValueError("PreferenceSet holds Preference objects only")
            if pref.origin != "explicit_user":
                raise ValueError("preferences must come from the person")
        ids = [pref.pref_id for pref in items]
        if len(ids) != len(set(ids)):
            raise ValueError("pref_id values must be unique")
        self._prefs: tuple[Preference, ...] = tuple(sorted(items, key=lambda p: (p.created_at, p.pref_id)))

    # ----------------------------------------------------------------- lookups

    @property
    def all(self) -> tuple[Preference, ...]:
        """Every rule held, retracted ones included, oldest first."""
        return self._prefs

    def __len__(self) -> int:
        return len(self._prefs)

    def __iter__(self) -> Iterator[Preference]:
        return iter(self._prefs)

    def get(self, pref_id: str) -> Preference | None:
        for pref in self._prefs:
            if pref.pref_id == pref_id:
                return pref
        return None

    def active_rules(self) -> tuple[Preference, ...]:
        """Every rule not retracted, whatever its scope."""
        return tuple(pref for pref in self._prefs if pref.retracted_at is None)

    def active(self, *, context: str, modality: Modality, participant_key: str | None) -> tuple[Preference, ...]:
        return tuple(
            pref
            for pref in self._prefs
            if pref.retracted_at is None
            and pref.scope.covers(context=context, modality=modality, participant_key=participant_key)
        )

    def active_at(
        self, when: datetime, *, context: str, modality: Modality, participant_key: str | None
    ) -> tuple[Preference, ...]:
        """Rules in force at ``when``: ``created_at <= when < retracted_at`` (audit A1, section 9.7)."""
        return tuple(
            pref
            for pref in self._prefs
            if pref.active_at(when)
            and pref.scope.covers(context=context, modality=modality, participant_key=participant_key)
        )

    def as_of(self, when: datetime) -> PreferenceSet:
        """The rules as they stood at ``when``: rules created later are left out, later retractions undone."""
        kept = []
        for pref in self._prefs:
            if pref.created_at > when:
                continue
            if pref.retracted_at is not None and pref.retracted_at > when:
                pref = replace(pref, retracted_at=None)
            kept.append(pref)
        return PreferenceSet(kept)

    def _active_point(self, point: _Point) -> tuple[Preference, ...]:
        return self.active(context=point.context, modality=point.modality, participant_key=point.participant_key)

    def pin_act(self, *, context: str = "study", modality: Modality, participant_key: str | None) -> bool:
        return any(
            pref.kind is PreferenceKind.PIN_ACT
            for pref in self.active(context=context, modality=modality, participant_key=participant_key)
        )

    def min_caution(self, *, context: str = "study", modality: Modality, participant_key: str | None) -> str | None:
        """The strictest active ``min_caution`` (overlapping rules combine to the highest rank)."""
        values = [
            pref.value
            for pref in self.active(context=context, modality=modality, participant_key=participant_key)
            if pref.kind is PreferenceKind.MIN_CAUTION
        ]
        return max(values, key=caution_of) if values else None

    def interruption_cap(
        self, *, context: str = "study", modality: Modality, participant_key: str | None
    ) -> int | None:
        """The smallest active ``max_interruptions_per_block`` (overlapping caps combine to the smallest)."""
        values = [
            pref.value
            for pref in self.active(context=context, modality=modality, participant_key=participant_key)
            if pref.kind is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK
        ]
        return min(values) if values else None

    def forbidden(
        self, variant_id: str, *, context: str = "study", modality: Modality, participant_key: str | None
    ) -> bool:
        return any(
            pref.forbids(variant_id)
            for pref in self.active(context=context, modality=modality, participant_key=participant_key)
        )

    def priorities(
        self, *, context: str = "study", modality: Modality, participant_key: str | None
    ) -> tuple[float, float, float]:
        """(fewer_mistakes, faster, fewer_interruptions); precedence: participant over device, modality over
        unscoped, newest first. Priorities are the only kind with precedence; they cannot be combined."""
        candidates = [
            pref
            for pref in self.active(context=context, modality=modality, participant_key=participant_key)
            if pref.kind is PreferenceKind.PRIORITIES
        ]
        if not candidates:
            return DEFAULT_PRIORITIES
        best = max(
            candidates,
            key=lambda p: (p.scope.participant_key is not None, p.scope.modality is not None, p.created_at, p.pref_id),
        )
        return tuple(float(best.value[key]) for key in PRIORITY_KEYS)  # type: ignore[return-value]

    def _held_regime(
        self, kind: PreferenceKind, *, context: str, modality: Modality, participant_key: str | None
    ) -> str | None:
        held = [
            pref
            for pref in self.active(context=context, modality=modality, participant_key=participant_key)
            if pref.kind is kind and _in_effect(pref, modality)
        ]
        if not held:
            return None
        return max(held, key=lambda p: (p.created_at, p.pref_id)).regime

    def frozen_regime(self, *, context: str = "study", modality: Modality, participant_key: str | None) -> str | None:
        """The regime of the active ``freeze_adaptation`` for this modality (one of its L1 arms), or None."""
        return self._held_regime(
            PreferenceKind.FREEZE_ADAPTATION, context=context, modality=modality, participant_key=participant_key
        )

    def pinned_regime(self, *, context: str = "study", modality: Modality, participant_key: str | None) -> str | None:
        """The regime of the active ``pin_regime`` for this modality (one of its L1 arms), or None."""
        return self._held_regime(
            PreferenceKind.PIN_REGIME, context=context, modality=modality, participant_key=participant_key
        )

    def held_regime(self, *, context: str = "study", modality: Modality, participant_key: str | None) -> str | None:
        """The frozen or pinned regime in force for this modality (a freeze and a pin always agree), or None."""
        frozen = self.frozen_regime(context=context, modality=modality, participant_key=participant_key)
        if frozen is not None:
            return frozen
        return self.pinned_regime(context=context, modality=modality, participant_key=participant_key)

    # ------------------------------------------------------------------ C1

    def apply(
        self,
        level: DecisionLevel,
        candidates: Sequence[str],
        *,
        context: str,
        modality: Modality,
        participant_key: str | None,
        interruptions_used: int,
        hold: bool = True,
    ) -> tuple[list[str], dict[str, ExclusionReason]]:
        """Rule C1 at one decision level (section 9.2 step 2).

        Forbids remove variants and families; ``min_caution`` removes ranked variants below the strictest minimum
        (``wait_for_evidence``, which cannot be forbidden, stays); at L1 ``pin_act`` removes the confirm regimes, a
        freeze or pin collapses the set to its regime, and a cap below the block's 8 trials removes the confirm
        regimes unless they are all that is left; at L2 a cap already reached removes the interrupting responses.
        The ``pin_act`` collapse of L2 cells is applied by the policy (``constraints.pin_act_collapse``). With
        ``hold=False`` the freeze or pin collapse is skipped (the fallbacks' "without the freeze or pin" set).
        """
        from .constraints import ExclusionReason

        rules = self.active(context=context, modality=modality, participant_key=participant_key)
        minimum = self.min_caution(context=context, modality=modality, participant_key=participant_key)
        cap = self.interruption_cap(context=context, modality=modality, participant_key=participant_key)
        pin_act = any(pref.kind is PreferenceKind.PIN_ACT for pref in rules)
        allowed: list[str] = []
        excluded: dict[str, ExclusionReason] = {}
        for variant in candidates:
            if variant != "wait_for_evidence" and any(pref.forbids(variant) for pref in rules):
                excluded[variant] = ExclusionReason.FORBIDDEN_BY_YOU
            elif (
                minimum is not None
                and variant != "wait_for_evidence"
                and 0 <= caution_of(variant) < caution_of(minimum)
            ):
                excluded[variant] = ExclusionReason.BELOW_YOUR_MIN_CAUTION
            elif level == DecisionLevel.BLOCK_REGIME and pin_act and is_confirm_regime(variant):
                excluded[variant] = ExclusionReason.DIRECT_SELECTION_PINNED
            else:
                allowed.append(variant)
        if level == DecisionLevel.BLOCK_REGIME:
            held = (
                self.held_regime(context=context, modality=modality, participant_key=participant_key) if hold else None
            )
            if held is not None and held in allowed:
                for variant in allowed:
                    if variant != held:
                        excluded[variant] = ExclusionReason.PINNED_OTHER
                allowed = [held]
            elif cap is not None and cap < BLOCK_TRIALS:
                remaining = [variant for variant in allowed if not is_confirm_regime(variant)]
                if remaining:
                    for variant in allowed:
                        if is_confirm_regime(variant):
                            excluded[variant] = ExclusionReason.INTERRUPTION_LIMIT
                    allowed = remaining
        elif (
            level == DecisionLevel.AMBIGUITY_RESPONSE and cap is not None and interruptions_used >= cap and not pin_act
        ):
            kept = []
            for variant in allowed:
                if VARIANTS[variant].interrupts:
                    excluded[variant] = ExclusionReason.INTERRUPTION_LIMIT
                else:
                    kept.append(variant)
            allowed = kept
        return allowed, excluded

    # ------------------------------------------------------------- validation

    def _points(self, scope: PreferenceScope, rules: Sequence[Preference]) -> list[_Point]:
        """Every (modality, participant) at which a rule with ``scope`` combines with other rules (study context)."""
        if scope.participant_key is not None:
            participants: list[str | None] = [scope.participant_key]
        else:
            participants = [None]
            participants.extend(
                sorted({pref.scope.participant_key for pref in rules if pref.scope.participant_key is not None})
            )
        return [_Point("study", modality, pk) for modality in scope.modalities() for pk in participants]

    def _allowed_l1(self, rules: Sequence[Preference], modality: str) -> list[str]:
        """L1 arms (assistance_off left out) after forbids, the strictest min_caution and pin_act."""
        minimum = max((p.value for p in rules if p.kind is PreferenceKind.MIN_CAUTION), key=caution_of, default=None)
        pin_act = any(p.kind is PreferenceKind.PIN_ACT for p in rules)
        allowed = []
        for arm in l1_arms(modality):
            if arm == "assistance_off":
                continue
            if any(p.forbids(arm) for p in rules):
                continue
            if minimum is not None and caution_of(arm) < caution_of(minimum):
                continue
            if pin_act and is_confirm_regime(arm):
                continue
            allowed.append(arm)
        return allowed

    def validate_new(self, pref: Preference) -> tuple[str, ...]:
        """Check a new rule against every active rule whose scope overlaps it.

        Raises ``PreferenceConflict`` (message naming the conflicting rule, with the retract offer, and its id in
        ``conflicting_pref_ids``) and returns explanatory notes for the one accepted combination (a cap with a
        minimum caution that leaves only confirm or hold regimes).
        """
        if pref.retracted_at is not None:
            raise PreferenceConflict("A retracted rule cannot be saved.")
        if pref.kind is PreferenceKind.FORBID_VARIANT and pref.value == "wait_for_evidence":
            raise PreferenceConflict(
                "Waiting only shows a hint and never selects for you. To have every click select immediately, "
                "choose 'Always select exactly what I click'."
            )
        if pref.kind is PreferenceKind.FORBID_VARIANT and pref.value == "no_offer":
            raise PreferenceConflict(
                "Making no suggestion only leaves the screen as it is, so it cannot be turned off."
            )
        existing = [p for p in self.active_rules() if p.pref_id != pref.pref_id]
        overlapping = [p for p in existing if p.scope.overlaps(pref.scope)]

        # Pairwise conflicts, judged at every modality both rules cover.
        issues: list[_Issue] = []
        for old in overlapping:
            modalities = [m for m in pref.scope.modalities() if old.scope.modality in (None, m)]
            for modality in modalities:
                message = _pair_issue(pref, old, modality)
                if message is not None:
                    issues.append(_Issue(message, (old.pref_id,)))
                    break
        if issues:
            first = issues[0]
            names = "; ".join(
                f"'{self._rule(issue.pref_ids[0]).summary()}' ({self._rule(issue.pref_ids[0]).scope.describe()})"
                for issue in issues
            )
            noun = "rule" if len(issues) == 1 else "rules"
            offer = _RETRACT_OFFER if len(issues) == 1 else "Remove those rules and save this one, or keep them."
            raise PreferenceConflict(
                f"{first.message} This conflicts with your {noun} {names}. {offer}",
                tuple(issue.pref_ids[0] for issue in issues),
            )

        # At least one L1 regime besides assistance_off must stay allowed at every point the rule reaches.
        notes: list[str] = []
        for point in self._points(pref.scope, existing):
            rules = [
                p
                for p in existing
                if p.scope.covers(context=point.context, modality=point.modality, participant_key=point.participant_key)
            ]
            rules.append(pref)
            allowed = self._allowed_l1(rules, point.modality)
            if not allowed:
                raise PreferenceConflict(
                    f"At least one way of presenting targets must stay allowed for "
                    f"{point.modality.replace('_', ' ')}. To turn assistance off, use Pause adapting."
                )
            caps = [p for p in rules if p.kind is PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK and p.value < BLOCK_TRIALS]
            if (
                pref.kind in (PreferenceKind.MAX_INTERRUPTIONS_PER_BLOCK, PreferenceKind.MIN_CAUTION)
                and caps
                and any(p.kind is PreferenceKind.MIN_CAUTION for p in rules)
                and all(is_confirm_regime(arm) for arm in allowed)
            ):
                note = (
                    "Your minimum caution leaves only confirm or hold steps, so the interruption limit does not "
                    "remove them: each selection still asks for one step. With both rules active, Pause adapting "
                    "is not available until you remove one of them."
                )
                if note not in notes:
                    notes.append(note)
        return tuple(notes)

    def _rule(self, pref_id: str) -> Preference:
        rule = self.get(pref_id)
        if rule is None:  # pragma: no cover - ids come from this set
            raise KeyError(pref_id)
        return rule

    def _without_same_scope_freeze(self, scope: PreferenceScope) -> PreferenceSet:
        return PreferenceSet(
            p
            for p in self._prefs
            if not (p.kind is PreferenceKind.FREEZE_ADAPTATION and p.retracted_at is None and p.scope == scope)
        )

    def freeze_conflict(
        self, regime: str, *, modality: Modality, participant_key: str | None
    ) -> PreferenceConflict | None:
        """The conflict ``validate_new`` raises for a study-scoped freeze of ``regime`` (the active freeze of exactly
        that scope left out, because a new pause replaces it), or None when that freeze passes."""
        scope = PreferenceScope(context="study", modality=modality, participant_key=participant_key)
        base = self._without_same_scope_freeze(scope)
        latest = max((p.created_at for p in self._prefs), default=_EPOCH)
        probe = Preference(
            pref_id="__freeze_probe__",
            kind=PreferenceKind.FREEZE_ADAPTATION,
            scope=scope,
            value={"regime": regime},
            origin="explicit_user",
            created_at=latest,
        )
        try:
            base.validate_new(probe)
        except PreferenceConflict as conflict:
            return conflict
        return None

    def freezable(self, *, modality: Modality, participant_key: str | None) -> tuple[str, ...]:
        """``l1_arms(modality)`` in CAUTION_ORDER whose study-scoped freeze for this participant and modality passes
        ``validate_new``, judged without the active freeze of exactly that scope (C1 only; C7 does not apply)."""
        arms = sorted(l1_arms(modality), key=caution_of)
        return tuple(
            arm for arm in arms if self.freeze_conflict(arm, modality=modality, participant_key=participant_key) is None
        )

    def hash(self) -> str:
        """sha256 of the canonical JSON of the active rules."""
        payload = [preference_to_dict(p) for p in sorted(self.active_rules(), key=lambda p: p.pref_id)]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

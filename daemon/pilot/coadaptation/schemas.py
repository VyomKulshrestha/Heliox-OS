"""Pydantic models for the co-adaptation RPC payloads (spec section 14.4).

Every model forbids extra keys, rejects NaN and infinity, bounds every number and closes every enum. Requests are
validated from their JSON encoding in strict mode, so a string never turns into a number and a number never turns
into a string. Error messages name the field and the rule that failed and never echo a payload value, and payloads
are never logged. Each payload model has ``to_core()`` returning the core record of ``pilot.coadaptation.core``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from pilot.coadaptation.consent import ConsentTextVersion, Identifier, ParticipantKey, Sha256Hex
from pilot.coadaptation.core.consent import CoAdaptationPurpose
from pilot.coadaptation.core.constraints import Fallbacks, FreezeConflict
from pilot.coadaptation.core.preferences import Preference, preference_from_dict
from pilot.coadaptation.core.records import (
    ActivationGaze,
    ActivationRecord,
    BurdenReport,
    GazeSummary,
    HandSummary,
    L2Applied,
    MovementSummary,
    OfferResponse,
    RevertRecord,
    SpreadUsed,
    TrialRecord,
    VetoRecord,
)
from pilot.coadaptation.core.state import BlockContext, ClockContext, DeviceContext, LayoutSummary, OfferContext, Rect
from pilot.coadaptation.core.vocabulary import DecisionLevel

PAYLOAD_LIMIT_BYTES = 8 * 1024
TRIAL_PAYLOAD_LIMIT_BYTES = 32 * 1024

Modality = Literal["mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer"]
TaskKind = Literal["pointing", "menu", "idle"]
L1VariantId = Literal[
    "assistance_off",
    "act",
    "enlarge_target:1.5",
    "enlarge_target:2.0",
    "confirm_selection:second_click",
    "confirm_selection:hold_400",
    "confirm_selection:hold_700",
]
L2VariantId = Literal[
    "act", "confirm_selection:second_click", "wait_for_evidence", "ask_clarification:2", "ask_clarification:3"
]
L3VariantId = Literal["no_offer", "offer_modality:keyboard", "offer_modality:pointer"]
RpcLevel = Literal["block_regime", "ambiguity_table", "offer"]
PreferenceKindValue = Literal[
    "forbid_variant",
    "pin_regime",
    "pin_act",
    "min_caution",
    "max_interruptions_per_block",
    "priorities",
    "freeze_adaptation",
]

UuidStr = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
]
CandidateIndex = Annotated[int, Field(ge=0, le=31)]
ParticipantLabel = Annotated[str, StringConstraints(min_length=1, max_length=32)]

L1_VARIANT_IDS: frozenset[str] = frozenset(L1VariantId.__args__)
L2_VARIANT_IDS: frozenset[str] = frozenset(L2VariantId.__args__)
L3_VARIANT_IDS: frozenset[str] = frozenset(L3VariantId.__args__)
ALL_VARIANT_IDS: frozenset[str] = L1_VARIANT_IDS | L2_VARIANT_IDS | L3_VARIANT_IDS
VARIANT_FAMILIES: frozenset[str] = frozenset(
    {"confirm_selection", "ask_clarification", "enlarge_target", "offer_modality"}
)
CAUTION_ORDER_IDS: frozenset[str] = frozenset(
    {
        "assistance_off",
        "act",
        "enlarge_target:1.5",
        "enlarge_target:2.0",
        "confirm_selection:second_click",
        "confirm_selection:hold_400",
        "confirm_selection:hold_700",
        "wait_for_evidence",
        "ask_clarification:2",
        "ask_clarification:3",
    }
)
PRIORITY_KEYS = ("fewer_mistakes", "faster", "fewer_interruptions")
BurdenItem = Literal[
    "effort",
    "intrusiveness",
    "control",
    "tlx",
    "mental",
    "physical",
    "temporal",
    "performance",
    "frustration",
    "predictability",
    "preferred_regime",
]
BURDEN_ITEMS: tuple[str, ...] = BurdenItem.__args__

RPC_LEVELS: Mapping[str, DecisionLevel] = {
    "block_regime": DecisionLevel.BLOCK_REGIME,
    "ambiguity_table": DecisionLevel.AMBIGUITY_RESPONSE,
    "offer": DecisionLevel.OFFER,
}

_MODEL_CONFIG = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True, strict=True)


class PayloadTooLarge(ValueError):
    """The JSON encoding of a request is above its size limit."""

    reason = "payload_too_large"


def encode_payload(data: Any, *, limit: int = PAYLOAD_LIMIT_BYTES) -> bytes:
    """JSON-encode ``data`` and enforce the payload limit (8 KiB, 32 KiB for a trial)."""
    try:
        raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("The request parameters must be plain JSON.")
    if len(raw) > limit:
        raise PayloadTooLarge(f"The request is larger than the {limit // 1024} KiB this call accepts.")
    return raw


def validation_message(exc: ValidationError) -> str:
    """Describe what failed by field path and rule only; never include the offending value."""
    parts: list[str] = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False)[:3]:
        location = ".".join(str(part)[:64] for part in error.get("loc", ())[:8]) or "request"
        if error.get("type") == "extra_forbidden":
            parts.append(f"{location}: this field is not allowed")
        else:
            message = str(error.get("msg", "invalid value"))
            parts.append(f"{location}: {message[:200]}")
    return "Invalid request: " + "; ".join(parts) if parts else "Invalid request."


def _jsonable(value: Any) -> Any:
    """Convert a python-mode dump into the JSON shapes that the core ``from_dict`` methods read."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class _Payload(BaseModel):
    model_config = _MODEL_CONFIG

    @classmethod
    def parse(cls, data: Any, *, limit: int = PAYLOAD_LIMIT_BYTES) -> Self:
        """Validate a raw RPC payload: size limit first, then the strict schema."""
        return cls.model_validate_json(encode_payload(data, limit=limit))

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(self.model_dump(mode="python"))


def _check_candidate_index(value: int | None, n: int, *, allow_none_press: bool = False) -> None:
    if value is None:
        return
    if allow_none_press and value == -1:
        return
    if not 0 <= value < n:
        raise ValueError("a candidate index is outside the trial's candidate list")


# ── leaf payloads ────────────────────────────────────────────────────────────


class RectV1(_Payload):
    cx: float = Field(ge=-10, le=10000)
    cy: float = Field(ge=-10, le=10000)
    w: float = Field(gt=0, le=2000)
    h: float = Field(gt=0, le=2000)

    def to_core(self) -> Rect:
        return Rect(cx=self.cx, cy=self.cy, w=self.w, h=self.h)


class LayoutSummaryV1(_Payload):
    task: TaskKind
    n_candidates: int = Field(ge=1, le=32)
    w_px: float = Field(gt=0, le=400)
    h_px: float = Field(gt=0, le=400)
    gap_px: float = Field(ge=0, le=200)
    id_bar: float = Field(ge=0, le=12)
    menu_n: int = Field(ge=0, le=32)
    origin_x: float = Field(ge=-10, le=10000)
    origin_y: float = Field(ge=-10, le=10000)
    # Pointing grids: the mean keyboard presses over the candidates (spec 7.3, 12.6). Optional; without it the prior
    # approximates the grid.
    presses_mean: float | None = Field(default=None, ge=1, le=64)

    def to_core(self) -> LayoutSummary:
        return LayoutSummary(
            task=self.task,
            n_candidates=self.n_candidates,
            w_px=self.w_px,
            h_px=self.h_px,
            gap_px=self.gap_px,
            id_bar=self.id_bar,
            menu_n=self.menu_n,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            presses_mean=self.presses_mean,
        )


class DeviceV1(_Payload):
    modality: Modality
    pointer_types: list[Literal["mouse", "pen", "touch"]] = Field(max_length=4)
    dpr: float = Field(gt=0, le=8)
    surface_w: float = Field(gt=0, le=20000)
    surface_h: float = Field(gt=0, le=20000)
    gaze: bool
    hand_pointer: bool

    def to_core(self) -> DeviceContext:
        """The device as the UI reported it; the service replaces both flags with their effective values."""
        return DeviceContext(
            modality=self.modality,
            pointer_types=tuple(self.pointer_types),
            dpr=self.dpr,
            surface_w=self.surface_w,
            surface_h=self.surface_h,
            gaze_available=self.gaze,
            hand_pointer=self.hand_pointer,
        )


class SpreadV1(_Payload):
    r: float = Field(gt=0, le=20)
    floor: float = Field(ge=0, le=50)

    def to_core(self) -> SpreadUsed:
        return SpreadUsed.from_dict(self.to_json_dict())


class GazeCheckV1(_Payload):
    accuracy: float = Field(ge=0, le=1)

    def to_core(self) -> float:
        return self.accuracy


class ActivationGazeV1(_Payload):
    region: Literal["center", "left", "right", "up", "down"]
    conf: float = Field(ge=0, le=1)

    def to_core(self) -> ActivationGaze:
        return ActivationGaze.from_dict(self.to_json_dict())


class L2AppliedV1(_Payload):
    decision_id: UuidStr
    cell_id: Annotated[str, StringConstraints(pattern=r"^b[123]:p[123]:h[01]$")]
    draw_index: int = Field(ge=0, le=4)
    variant: L2VariantId
    source: Literal["table", "fixed_rule", "fallback", "cap"]
    fallback_reason: Literal["late", "rpc_error", "stale_table"] | None = None

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if (self.source == "fallback") != (self.fallback_reason is not None):
            raise ValueError("fallback_reason is required for source 'fallback' and not allowed otherwise")
        if self.source == "cap" and self.variant != "wait_for_evidence":
            raise ValueError("a 'cap' response is always wait_for_evidence")
        return self

    def to_core(self) -> L2Applied:
        return L2Applied.from_dict(self.to_json_dict())


class ActivationV1(_Payload):
    index: int = Field(ge=0, le=31)
    t_ms: float = Field(ge=0, le=60000)
    modality: Modality
    x: int = Field(ge=-100, le=20000)
    y: int = Field(ge=-100, le=20000)
    hit_index: CandidateIndex | None = None
    press_index: int = Field(ge=-1, le=31)
    a_gate: float = Field(ge=0, le=1)
    n_plausible: int = Field(ge=1, le=3)
    expanded_index: CandidateIndex | None = None
    hit_rect: RectV1 | None = None
    gaze: ActivationGazeV1 | None = None
    l2: L2AppliedV1 | None = None
    irreversible_rule: bool
    result: Literal["committed", "confirming", "held", "asked", "none"]
    outcome: Literal["committed", "cancelled", "aborted", "open_at_end", "none"]
    committed_index: CandidateIndex | None = None

    @model_validator(mode="after")
    def _check_activation(self) -> Self:
        if (self.hit_rect is None) != (self.hit_index is None):
            raise ValueError("hit_rect is given exactly when hit_index is given")
        if (self.outcome == "none") != (self.result == "none"):
            raise ValueError("outcome is 'none' exactly when result is 'none'")
        return self

    def to_core(self) -> ActivationRecord:
        return ActivationRecord.from_dict(self.to_json_dict())


class RevertV1(_Payload):
    reason: Literal["veto", "refresh"]
    from_activation: int = Field(ge=0, le=32)
    regime: L1VariantId
    fallbacks_issued_at: AwareDatetime

    def to_core(self) -> RevertRecord:
        return RevertRecord.from_dict(self.to_json_dict())


class MovementSummaryV1(_Payload):
    rt_ms: float = Field(ge=0, le=60000)
    mt_ms: float = Field(ge=0, le=60000)
    path_ratio: float = Field(ge=1, le=50)
    submovements: int = Field(ge=0, le=50)
    peak_speed_px_s: float = Field(ge=0, le=50000)
    reentries: int = Field(ge=0, le=50)
    endpoint_dx_norm: float = Field(ge=-20, le=20)
    endpoint_dy_norm: float = Field(ge=-20, le=20)

    def to_core(self) -> MovementSummary:
        return MovementSummary.from_dict(self.to_json_dict())


class OfferResponseV1(_Payload):
    decision_id: UuidStr
    variant: L3VariantId
    response: Literal["accepted", "dismissed", "ignored"]
    fallback_reason: Literal["late", "rpc_error"] | None = None

    @model_validator(mode="after")
    def _check_fallback(self) -> Self:
        if self.fallback_reason is not None and (self.variant != "no_offer" or self.response != "ignored"):
            raise ValueError("an offer fallback shows no offer: variant no_offer, response ignored")
        return self

    def to_core(self) -> OfferResponse:
        return OfferResponse.from_dict(self.to_json_dict())


class VetoV1(_Payload):
    decision_id: UuidStr
    level: Literal["block_regime", "ambiguity_response"]

    def to_core(self) -> VetoRecord:
        return VetoRecord.from_dict(self.to_json_dict())


class HandSummaryV1(_Payload):
    quality_mean: float = Field(ge=0, le=1)
    quality_min: float = Field(ge=0, le=1)
    tracking_losses: int = Field(ge=0, le=100)

    def to_core(self) -> HandSummary:
        return HandSummary.from_dict(self.to_json_dict())


class GazeSummaryV1(_Payload):
    agreed_with_intended: bool | None = None

    def to_core(self) -> GazeSummary:
        return GazeSummary.from_dict(self.to_json_dict())


class TrialV1(_Payload):
    trial_id: UuidStr
    block_index: int = Field(ge=0, le=64)
    trial_index: int = Field(ge=0, le=64)
    task: TaskKind
    regime: L1VariantId
    regime_decision_id: UuidStr
    table_decision_ids: list[UuidStr] = Field(max_length=4)
    applied_regime: L1VariantId
    l1_fallback_reason: Literal["late", "rpc_error"] | None = None
    fallbacks_issued_at: AwareDatetime
    spread: SpreadV1 | None = None
    layout: LayoutSummaryV1
    candidates: list[RectV1] = Field(min_length=1, max_length=32)
    surface_w: float = Field(gt=0, le=20000)
    surface_h: float = Field(gt=0, le=20000)
    dpr: float = Field(gt=0, le=8)
    reversible: list[bool] = Field(min_length=1, max_length=32)
    cue_at: AwareDatetime
    intended_index: CandidateIndex | None = None
    modality: Modality
    activations: list[ActivationV1] = Field(max_length=32)
    revert: RevertV1 | None = None
    committed_index: CandidateIndex | None = None
    commit_t_ms: float | None = Field(default=None, ge=0, le=60000)
    commit_via: Literal["direct", "confirm", "hold", "ask", "none"]
    corrections: int = Field(ge=0, le=50)
    time_ms: float = Field(ge=0, le=60000)
    limit_ms: float
    duration_ms: float = Field(ge=0, le=60000)
    movement: MovementSummaryV1 | None = None
    veto: VetoV1 | None = None
    offers: list[OfferResponseV1] = Field(default_factory=list, max_length=2)
    hand: HandSummaryV1 | None = None
    gaze: GazeSummaryV1 | None = None
    interrupted: bool
    interruption_reason: (
        Literal[
            "modal",
            "hidden",
            "blur",
            "resize",
            "palm_exit",
            "paused",
            "daemon",
            "refused",
            "refresh",
            "no_allowed_regime",
        ]
        | None
    ) = None
    explanation_views: int = Field(ge=0, le=20)

    @model_validator(mode="after")
    def _check_trial(self) -> Self:
        if self.limit_ms not in (10000.0, 15000.0, 30000.0):
            raise ValueError("limit_ms must be 10000, 15000 or 30000")
        n = len(self.candidates)
        if len(self.reversible) != n:
            raise ValueError("reversible needs exactly one flag per candidate")
        if (self.intended_index is None) != (self.task == "idle"):
            raise ValueError("intended_index is null exactly for idle epochs")
        _check_candidate_index(self.intended_index, n)
        _check_candidate_index(self.committed_index, n)
        for activation in self.activations:
            _check_candidate_index(activation.hit_index, n)
            _check_candidate_index(activation.press_index, n, allow_none_press=True)
            _check_candidate_index(activation.expanded_index, n)
            _check_candidate_index(activation.committed_index, n)
        return self

    def to_core(self, *, session_id: str, recorded_at: datetime, seq: int) -> TrialRecord:
        """The core trial record; ``recorded_at`` and ``seq`` are set by the service, never by the UI."""
        data = self.to_json_dict()
        data["session_id"] = session_id
        data["recorded_at"] = recorded_at.isoformat()
        data["seq"] = seq
        return TrialRecord.from_dict(data)


class TlxV1(_Payload):
    mental: int | None = Field(ge=0, le=100)
    physical: int | None = Field(ge=0, le=100)
    temporal: int | None = Field(ge=0, le=100)
    performance: int | None = Field(ge=0, le=100)
    effort: int | None = Field(ge=0, le=100)
    frustration: int | None = Field(ge=0, le=100)


class BurdenV1(_Payload):
    block_index: int | None = Field(default=None, ge=0, le=64)
    scope: Literal["block", "session"]
    effort: int | None = Field(default=None, ge=1, le=7)
    intrusiveness: int | None = Field(default=None, ge=1, le=7)
    control: int | None = Field(default=None, ge=1, le=7)
    tlx: TlxV1 | None = None
    predictability: int | None = Field(default=None, ge=1, le=7)
    preferred_regime: L1VariantId | None = None
    skipped: list[BurdenItem] = Field(default_factory=list, max_length=len(BURDEN_ITEMS))

    @model_validator(mode="after")
    def _check_scope(self) -> Self:
        if self.scope == "block" and self.block_index is None:
            raise ValueError("a block report names its block_index")
        return self

    def to_core(self, *, report_id: str, session_id: str, recorded_at: datetime, seq: int) -> BurdenReport:
        data = self.to_json_dict()
        data["report_id"] = report_id
        data["session_id"] = session_id
        data["recorded_at"] = recorded_at.isoformat()
        data["seq"] = seq
        return BurdenReport.from_dict(data)


class FreezeConflictV1(_Payload):
    message: Annotated[str, StringConstraints(max_length=2000)]
    conflicting: list[Identifier] = Field(max_length=32)

    def to_core(self) -> FreezeConflict:
        return FreezeConflict(message=self.message, conflicting=tuple(self.conflicting))


class FallbacksV1(_Payload):
    l1: L1VariantId | None
    plain: L1VariantId | None
    refresh: L1VariantId | None
    practice: L1VariantId | None
    freezable: list[L1VariantId] = Field(max_length=6)
    freeze_conflict: FreezeConflictV1 | None
    l2: list[L2VariantId] = Field(max_length=3)
    interruption_cap: int | None = Field(ge=0, le=8)
    issued_at: AwareDatetime

    def to_core(self) -> Fallbacks:
        return Fallbacks(
            l1=self.l1,
            plain=self.plain,
            refresh=self.refresh,
            practice=self.practice,
            freezable=tuple(self.freezable),
            freeze_conflict=None if self.freeze_conflict is None else self.freeze_conflict.to_core(),
            l2=tuple(self.l2),
            interruption_cap=self.interruption_cap,
            issued_at=self.issued_at,
        )


# ── preferences ──────────────────────────────────────────────────────────────


def validate_preference_value(kind: str, value: Any) -> Any:
    """Check a rule value against its kind (spec section 9.6) and return it in its stored JSON form."""
    if kind == "forbid_variant":
        if isinstance(value, str) and (value in ALL_VARIANT_IDS or value in VARIANT_FAMILIES):
            return value
        raise ValueError("forbid_variant names a variant id or a variant family")
    if kind == "pin_regime":
        if isinstance(value, str) and value in L1_VARIANT_IDS and value != "assistance_off":
            return value
        raise ValueError("pin_regime names a block regime other than assistance_off (use freeze_adaptation)")
    if kind == "pin_act":
        if value is True:
            return True
        raise ValueError("pin_act takes the value true")
    if kind == "min_caution":
        if isinstance(value, str) and value in CAUTION_ORDER_IDS:
            return value
        raise ValueError("min_caution names a variant in the caution order")
    if kind == "max_interruptions_per_block":
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 8:
            return value
        raise ValueError("max_interruptions_per_block takes an integer from 0 to 8")
    if kind == "priorities":
        if not isinstance(value, Mapping) or set(value) != set(PRIORITY_KEYS):
            raise ValueError("priorities takes fewer_mistakes, faster and fewer_interruptions")
        weights: dict[str, float] = {}
        for key in PRIORITY_KEYS:
            weight = value[key]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise ValueError("each priority is a number from 0 to 1")
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError("each priority is a number from 0 to 1")
            weights[key] = float(weight)
        return weights
    if kind == "freeze_adaptation":
        if (
            isinstance(value, Mapping)
            and set(value) == {"regime"}
            and isinstance(value["regime"], str)
            and value["regime"] in L1_VARIANT_IDS
        ):
            return {"regime": value["regime"]}
        raise ValueError("freeze_adaptation takes {'regime': <block regime>}")
    raise ValueError("unknown rule kind")


class PreferenceScopeV1(_Payload):
    context: Literal["study", "all"]
    modality: Modality | None = None
    participant_key: ParticipantKey | None = None


class PreferenceV1(_Payload):
    pref_id: Identifier
    kind: PreferenceKindValue
    scope: PreferenceScopeV1
    value: Any
    origin: Literal["explicit_user"]
    created_at: AwareDatetime
    retracted_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        self.value = validate_preference_value(self.kind, self.value)
        if self.retracted_at is not None and self.retracted_at < self.created_at:
            raise ValueError("a rule cannot be retracted before it was created")
        return self

    def to_core(self) -> Preference:
        return preference_from_dict(self.to_json_dict())


# ── decision contexts (spec 14.3; keys equal ALLOWED_CONTEXT_KEYS) ────────────


class BlockRegimeContextV1(_Payload):
    block_index: int = Field(ge=0, le=64)
    layouts: list[LayoutSummaryV1] = Field(min_length=1, max_length=16)
    previous_regime: L1VariantId | None = None
    session_minute: float = Field(ge=0, le=1440)
    trials_today: int = Field(ge=0, le=100000)
    hand_quality_recent: float | None = Field(default=None, ge=0, le=1)

    def to_core(
        self,
        *,
        session_id: str,
        phase: str,
        task: str,
        device: DeviceContext,
        day_index: int,
        days_since_last_session: float | None,
    ) -> BlockContext:
        return BlockContext(
            session_id=session_id,
            block_index=self.block_index,
            phase=phase,
            task=task,
            layouts=tuple(layout.to_core() for layout in self.layouts),
            device=device,
            clock=ClockContext(
                day_index=day_index,
                session_minute=self.session_minute,
                trials_today=self.trials_today,
                days_since_last_session=days_since_last_session,
            ),
            previous_regime=self.previous_regime,
            hand_quality_recent=self.hand_quality_recent,
        )


class AmbiguityTableContextV1(BlockRegimeContextV1):
    block_decision_id: UuidStr
    start_trial_index: int = Field(ge=0, le=64)
    trials_in_block: int = Field(ge=1, le=64)


class OfferContextV1(_Payload):
    block_index: int = Field(ge=0, le=64)
    trial_index: int = Field(ge=0, le=64)
    task: TaskKind
    struggle_kind: Literal["corrections", "tracking_loss"]
    keyboard_available: bool

    def to_core(self, *, session_id: str, modality: str) -> OfferContext:
        return OfferContext(
            session_id=session_id,
            block_index=self.block_index,
            trial_index=self.trial_index,
            modality=modality,
            task=self.task,
            struggle_kind=self.struggle_kind,
            keyboard_available=self.keyboard_available,
        )


CONTEXT_MODELS: Mapping[DecisionLevel, type[_Payload]] = {
    DecisionLevel.BLOCK_REGIME: BlockRegimeContextV1,
    DecisionLevel.AMBIGUITY_RESPONSE: AmbiguityTableContextV1,
    DecisionLevel.OFFER: OfferContextV1,
}


def rpc_level_to_core(level: str) -> DecisionLevel:
    """Map the RPC level name onto ``DecisionLevel`` (``ambiguity_table`` -> ``AMBIGUITY_RESPONSE``, spec 5.4)."""
    try:
        return RPC_LEVELS[level]
    except KeyError:
        raise ValueError("unknown decision level")


# ── RPC parameters ───────────────────────────────────────────────────────────


class EmptyParams(_Payload):
    """``co_adaptation_status`` takes no parameters."""


class ConsentGrantParams(_Payload):
    participant_key: ParticipantKey | None = None
    label: ParticipantLabel | None = None
    purposes: list[CoAdaptationPurpose] = Field(min_length=1, max_length=6)
    retention_days: int = Field(ge=1, le=365)
    consent_text_version: ConsentTextVersion
    consent_text_sha256: Sha256Hex
    protocol_id: Identifier | None = None
    protocol_sha256: Sha256Hex | None = None
    authorized: bool
    enable: bool = True

    @model_validator(mode="after")
    def _check_grant(self) -> Self:
        if self.authorized is not True:
            raise ValueError("consent must be authorized explicitly")
        if CoAdaptationPurpose.SELECTION_STUDY not in self.purposes:
            raise ValueError("the selection_study purpose is required")
        if len(set(self.purposes)) != len(self.purposes):
            raise ValueError("each purpose may be listed once")
        if (self.protocol_id is None) != (self.protocol_sha256 is None):
            raise ValueError("protocol_sha256 must be given exactly when protocol_id is given")
        if self.label is not None and any(ord(ch) < 32 or 0x7F <= ord(ch) <= 0x9F for ch in self.label):
            raise ValueError("the label may not contain control characters")
        return self


class ParticipantParams(_Payload):
    participant_key: ParticipantKey


class PreferenceSetParams(_Payload):
    kind: PreferenceKindValue
    scope: PreferenceScopeV1
    value: Any

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        self.value = validate_preference_value(self.kind, self.value)
        return self


class PreferenceRetractParams(_Payload):
    pref_id: Identifier


class EraseParams(_Payload):
    participant_key: ParticipantKey
    scope: Literal["today"]


class ResultsParams(_Payload):
    participant_key: ParticipantKey
    scope: Literal["today", "all"]


class SessionBeginParams(_Payload):
    participant_key: ParticipantKey
    context: Literal["study"]
    consent_text_sha256: Sha256Hex
    device: DeviceV1
    protocol_id: Identifier
    gaze_check: GazeCheckV1 | None = None


class SessionRefParams(_Payload):
    session_id: UuidStr


class RecommendParams(_Payload):
    session_id: UuidStr
    decision_id: UuidStr
    level: RpcLevel
    context: dict[str, Any]

    def decision_level(self) -> DecisionLevel:
        return rpc_level_to_core(self.level)

    def parsed_context(self) -> BlockRegimeContextV1 | AmbiguityTableContextV1 | OfferContextV1:
        """Validate the context against the level's allow-list; any other key fails (spec 5.4)."""
        model = CONTEXT_MODELS[self.decision_level()]
        return model.model_validate_json(encode_payload(self.context))  # type: ignore[return-value]


class RecordTrialParams(_Payload):
    session_id: UuidStr
    trial: TrialV1


class ReportBurdenParams(_Payload):
    session_id: UuidStr
    report: BurdenV1


class SessionEndParams(_Payload):
    session_id: UuidStr
    reason: Literal["completed", "stopped", "abandoned", "interrupted"]


# ── checks that need the session (spec 14.4 "additional checks") ─────────────


def without_ineffective_fields(trial: TrialV1, *, gaze_effective: bool, hand_effective: bool) -> TrialV1:
    """Drop the gaze fields (trial summary and every activation) and the hand summary the session may not keep."""
    update: dict[str, Any] = {}
    if not gaze_effective:
        update["gaze"] = None
        if any(activation.gaze is not None for activation in trial.activations):
            update["activations"] = [activation.model_copy(update={"gaze": None}) for activation in trial.activations]
    if not hand_effective:
        update["hand"] = None
    return trial.model_copy(update=update) if update else trial


def check_trial_spread(trial: TrialV1, session_modality: str) -> None:
    """A keyboard session has no spread; every other session echoes one (spec 6.1)."""
    if session_modality == "keyboard" and trial.spread is not None:
        raise ValueError("A keyboard session's trials carry no spread.")
    if session_modality != "keyboard" and trial.spread is None:
        raise ValueError("Trials of a pointer or hand-pointer session must carry the spread they used.")

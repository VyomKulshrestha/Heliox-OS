"""Stored records of the co-adaptation study and their only serialization (section 13.1.2 of the spec).

Every record is a frozen dataclass with ``to_dict()`` and ``from_dict()``. The store writes ``to_dict()`` into its
encrypted payload columns, the export bundle is a list of ``to_dict()`` outputs, and the rebuild and the analysis
script read with ``from_dict()``. Rules: keys are the field names; nested records become nested objects; tuples
become lists; enums become their values; datetimes become ISO-8601 strings with an explicit UTC offset; floats are
never rounded; ``None`` becomes ``null``. ``from_dict`` rejects unknown, missing and wrongly typed keys with
``ValueError`` and never echoes a value; ``from_dict(x.to_dict()) == x``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Literal, Self

from .evidence import EvidenceKind
from .policy import PolicyMode
from .state import ActivationContext, DeviceContext, LayoutSummary, Rect
from .study_plan import SessionPlan
from .vocabulary import L2_OPTIONS, L3_OPTIONS, DecisionLevel

RECORD_SCHEMA_VERSION = "coadapt-records-v1"

MODALITIES = ("mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer")
TASKS = ("pointing", "menu", "idle")
GAZE_REGIONS = ("center", "left", "right", "up", "down")
L2_SOURCES = ("table", "fixed_rule", "fallback", "cap")
L2_FALLBACK_REASONS = ("late", "rpc_error", "stale_table")
OFFER_FALLBACK_REASONS = ("late", "rpc_error")
ACTIVATION_RESULTS = ("committed", "confirming", "held", "asked", "none")
ACTIVATION_OUTCOMES = ("committed", "cancelled", "aborted", "open_at_end", "none")
VERIFICATIONS = ("ok", "draw_mismatch", "posterior_mismatch", "interrupted", "fallback")
COMMIT_VIA = ("direct", "confirm", "hold", "ask", "none")
OFFER_RESPONSES = ("accepted", "dismissed", "ignored")
REVERT_REASONS = ("veto", "refresh")
VETO_LEVELS = (DecisionLevel.BLOCK_REGIME.value, DecisionLevel.AMBIGUITY_RESPONSE.value)
TIME_FLOOR_MS = 150.0  # floor of trial and episode times before the log (section 7.1)
_CELL_ID_RE = re.compile(r"^b[123]:p[123]:h[01]$")
_SALT_RE = re.compile(r"^[0-9a-f]{32}$")


# ------------------------------------------------------------------------------------------------------ codecs


class _Codec:
    """Encodes one field to JSON and decodes it back with strict type checks (no value is ever echoed)."""

    name = "any"

    def encode(self, value: Any) -> Any:
        return value

    def decode(self, value: Any, where: str) -> Any:
        return value


def _fail(where: str, expected: str) -> ValueError:
    return ValueError(f"{where}: expected {expected}")


class _Str(_Codec):
    name = "str"

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, str):
            raise _fail(where, self.name)
        return value


class _Int(_Codec):
    name = "int"

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, int) or isinstance(value, bool):
            raise _fail(where, self.name)
        return value


class _Float(_Codec):
    name = "float"

    def encode(self, value: Any) -> Any:
        return float(value)

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            raise _fail(where, "finite number")
        return float(value)


class _Bool(_Codec):
    name = "bool"

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, bool):
            raise _fail(where, self.name)
        return value


class _DateTime(_Codec):
    name = "datetime"

    def encode(self, value: Any) -> Any:
        return value.isoformat()

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, str):
            raise _fail(where, "ISO-8601 string")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise _fail(where, "ISO-8601 string") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise _fail(where, "ISO-8601 string with a UTC offset")
        return parsed


class _Literal(_Codec):
    def __init__(self, values: Sequence[str]) -> None:
        self.values = tuple(values)
        self.name = "|".join(self.values)

    def encode(self, value: Any) -> Any:
        return str(value.value) if hasattr(value, "value") else value

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, str) or value not in self.values:
            raise _fail(where, f"one of {list(self.values)}")
        return value


class _Enum(_Codec):
    def __init__(self, enum: type) -> None:
        self.enum = enum
        self.name = enum.__name__

    def encode(self, value: Any) -> Any:
        return self.enum(value).value

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, str):
            raise _fail(where, self.name)
        try:
            return self.enum(value)
        except ValueError:
            raise _fail(where, self.name) from None


class _Optional(_Codec):
    def __init__(self, inner: _Codec) -> None:
        self.inner = inner
        self.name = f"{inner.name}|null"

    def encode(self, value: Any) -> Any:
        return None if value is None else self.inner.encode(value)

    def decode(self, value: Any, where: str) -> Any:
        return None if value is None else self.inner.decode(value, where)


class _Omittable(_Optional):
    """An optional field added after the record format: left out of the encoding while None and read as None when
    absent, so rows written before it, and their hashes, stay unchanged."""


class _List(_Codec):
    def __init__(self, inner: _Codec) -> None:
        self.inner = inner
        self.name = f"list[{inner.name}]"

    def encode(self, value: Any) -> Any:
        return [self.inner.encode(item) for item in value]

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, list | tuple):
            raise _fail(where, self.name)
        return tuple(self.inner.decode(item, f"{where}[{i}]") for i, item in enumerate(value))


class _Dict(_Codec):
    """A string-keyed object whose values all use one codec."""

    def __init__(self, inner: _Codec) -> None:
        self.inner = inner
        self.name = f"dict[str, {inner.name}]"

    def encode(self, value: Any) -> Any:
        return {str(key): self.inner.encode(item) for key, item in value.items()}

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise _fail(where, self.name)
        return {key: self.inner.decode(item, f"{where}.{key}") for key, item in value.items()}


class _Scalar(_Codec):
    """A JSON scalar setting value: bool, int or finite float."""

    name = "bool|int|float"

    def decode(self, value: Any, where: str) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise _fail(where, self.name)


class _Json(_Codec):
    """An arbitrary JSON object (decision payloads); checked for JSON types only."""

    name = "object"

    def encode(self, value: Any) -> Any:
        return _json_copy(value)

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, Mapping):
            raise _fail(where, self.name)
        _check_json(value, where)
        return _json_copy(value)


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_copy(item) for item in value]
    return value


def _check_json(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail(where, "string keys")
            _check_json(item, where)
    elif isinstance(value, list | tuple):
        for item in value:
            _check_json(item, where)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(where, "finite numbers")
    elif value is not None and not isinstance(value, str | int | bool):
        raise _fail(where, "JSON values")


class _Record(_Codec):
    """A nested record class with its own field codecs (records of this module and the state twins)."""

    def __init__(self, cls: type) -> None:
        self.cls = cls
        self.name = cls.__name__

    def encode(self, value: Any) -> Any:
        return _encode(value, _schema(self.cls))

    def decode(self, value: Any, where: str) -> Any:
        return _decode(self.cls, value, _schema(self.cls), where)


class _Plan(_Codec):
    name = "SessionPlan"

    def encode(self, value: Any) -> Any:
        return value.to_dict()

    def decode(self, value: Any, where: str) -> Any:
        if not isinstance(value, Mapping):
            raise _fail(where, self.name)
        return SessionPlan.from_dict(value)


STR, INT, FLOAT, BOOL, DATETIME = _Str(), _Int(), _Float(), _Bool(), _DateTime()


def _opt(codec: _Codec) -> _Codec:
    return _Optional(codec)


_SCHEMAS: dict[type, dict[str, _Codec]] = {}


def _schema(cls: type) -> dict[str, _Codec]:
    try:
        return _SCHEMAS[cls]
    except KeyError:  # pragma: no cover - every record registers its schema below
        raise TypeError(f"no schema for {cls.__name__}") from None


def _encode(obj: Any, schema: Mapping[str, _Codec]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, codec in schema.items():
        value = getattr(obj, name)
        if value is None and isinstance(codec, _Omittable):
            continue
        out[name] = codec.encode(value)
    return out


def _decode(cls: type, d: Any, schema: Mapping[str, _Codec], where: str | None = None) -> Any:
    label = where or cls.__name__
    if not isinstance(d, Mapping):
        raise ValueError(f"{label}: expected an object")
    unknown = sorted(str(key) for key in d if key not in schema)
    if unknown:
        raise ValueError(f"{label}: unknown keys {unknown}")
    missing = sorted(name for name, codec in schema.items() if name not in d and not isinstance(codec, _Omittable))
    if missing:
        raise ValueError(f"{label}: missing keys {missing}")
    values = {name: codec.decode(d[name], f"{label}.{name}") if name in d else None for name, codec in schema.items()}
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"{label}: invalid fields") from None


class _Serializable:
    """Mixin: ``to_dict()`` and ``from_dict()`` from the class's registered field codecs."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return _encode(self, _schema(type(self)))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return _decode(cls, d, _schema(cls))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ------------------------------------------------------------------------------------------------------ records


@dataclass(frozen=True, slots=True)
class SpreadUsed(_Serializable):
    """The posterior parameters of a block (section 6.1)."""

    r: float
    floor: float


@dataclass(frozen=True, slots=True)
class ActivationGaze(_Serializable):
    """The gaze region and the exact confidence used for an activation's posterior (gaze effective only)."""

    region: Literal["center", "left", "right", "up", "down"]
    conf: float

    def __post_init__(self) -> None:
        _require(self.region in GAZE_REGIONS, "gaze region is unknown")
        _require(0.0 <= self.conf <= 1.0, "gaze confidence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class L2Applied(_Serializable):
    """The L2 response an ambiguous activation received and where it came from."""

    decision_id: str
    cell_id: str
    draw_index: int
    variant: str
    source: Literal["table", "fixed_rule", "fallback", "cap"]
    fallback_reason: Literal["late", "rpc_error", "stale_table"] | None

    def __post_init__(self) -> None:
        _require(_CELL_ID_RE.fullmatch(self.cell_id) is not None, "cell_id must look like b1:p2:h1")
        _require(_is_int(self.draw_index) and 0 <= self.draw_index <= 4, "draw_index must lie in [0, 4]")
        _require(self.variant in L2_OPTIONS, "variant must be an L2 option")
        _require(self.source in L2_SOURCES, "source is unknown")
        if self.source == "fallback":
            _require(self.fallback_reason in L2_FALLBACK_REASONS, "a fallback response needs its reason")
        else:
            _require(self.fallback_reason is None, "fallback_reason is set only for source 'fallback'")
        if self.source == "cap":
            _require(self.variant == "wait_for_evidence", "a cap response is wait_for_evidence")


@dataclass(frozen=True, slots=True)
class ActivationRecord(_Serializable):
    index: int
    t_ms: float
    modality: str
    x: int
    y: int
    hit_index: int | None
    a_gate: float
    n_plausible: int
    press_index: int  # candidate under the press point, -1 if none (keyboard: the focused candidate)
    expanded_index: int | None
    hit_rect: Rect | None
    gaze: ActivationGaze | None
    l2: L2Applied | None
    irreversible_rule: bool
    result: Literal["committed", "confirming", "held", "asked", "none"]  # immediate response, set once
    outcome: Literal["committed", "cancelled", "aborted", "open_at_end", "none"]  # final outcome; labels read this
    committed_index: int | None

    def __post_init__(self) -> None:
        _require(self.modality in MODALITIES, "modality is unknown")
        _require(_is_int(self.press_index) and self.press_index >= -1, "press_index is an integer >= -1")
        _require(_is_int(self.x) and _is_int(self.y), "endpoints are integers")
        _require(1 <= self.n_plausible <= 3, "n_plausible lies in [1, 3]")
        _require(0.0 <= self.a_gate <= 1.0, "a_gate lies in [0, 1]")
        _require(self.result in ACTIVATION_RESULTS, "result is unknown")
        _require(self.outcome in ACTIVATION_OUTCOMES, "outcome is unknown")
        _require((self.outcome == "none") == (self.result == "none"), "outcome is none exactly when result is none")
        _require((self.hit_rect is None) == (self.hit_index is None), "hit_rect is null exactly without a hit")


@dataclass(frozen=True, slots=True)
class MovementSummary(_Serializable):
    rt_ms: float
    mt_ms: float
    path_ratio: float
    submovements: int
    peak_speed_px_s: float
    reentries: int
    endpoint_dx_norm: float
    endpoint_dy_norm: float


@dataclass(frozen=True, slots=True)
class OfferResponse(_Serializable):
    decision_id: str
    variant: str
    response: Literal["accepted", "dismissed", "ignored"]
    fallback_reason: Literal["late", "rpc_error"] | None

    def __post_init__(self) -> None:
        _require(self.variant in L3_OPTIONS, "variant must be an L3 option")
        _require(self.response in OFFER_RESPONSES, "response is unknown")
        if self.fallback_reason is not None:
            _require(self.fallback_reason in OFFER_FALLBACK_REASONS, "fallback_reason is unknown")
            _require(
                self.variant == "no_offer" and self.response == "ignored",
                "a fallback offer shows nothing (no_offer, ignored)",
            )


@dataclass(frozen=True, slots=True)
class RevertRecord(_Serializable):
    reason: Literal["veto", "refresh"]
    from_activation: int
    regime: str  # fallbacks.plain for a veto, fallbacks.refresh for a refresh
    fallbacks_issued_at: datetime

    def __post_init__(self) -> None:
        _require(self.reason in REVERT_REASONS, "revert reason is unknown")
        _require(_is_int(self.from_activation) and 0 <= self.from_activation <= 32, "from_activation lies in [0, 32]")


@dataclass(frozen=True, slots=True)
class VetoRecord(_Serializable):
    decision_id: str
    level: DecisionLevel

    def __post_init__(self) -> None:
        _require(DecisionLevel(self.level).value in VETO_LEVELS, "a veto names an L1 or L2 decision")


@dataclass(frozen=True, slots=True)
class HandSummary(_Serializable):
    quality_mean: float
    quality_min: float
    tracking_losses: int


@dataclass(frozen=True, slots=True)
class GazeSummary(_Serializable):
    agreed_with_intended: bool | None


@dataclass(frozen=True, slots=True)
class TrialRecord(_Serializable):
    trial_id: str
    session_id: str
    block_index: int
    trial_index: int
    task: Literal["pointing", "menu", "idle"]
    regime: str
    regime_decision_id: str
    table_decision_ids: tuple[str, ...]
    applied_regime: str
    l1_fallback_reason: Literal["late", "rpc_error"] | None
    fallbacks_issued_at: datetime
    spread: SpreadUsed | None
    layout: LayoutSummary
    candidates: tuple[Rect, ...]
    surface_w: float
    surface_h: float
    dpr: float
    reversible: tuple[bool, ...]
    cue_at: datetime
    intended_index: int | None
    modality: str
    activations: tuple[ActivationRecord, ...]
    revert: RevertRecord | None
    committed_index: int | None
    commit_t_ms: float | None
    commit_via: Literal["direct", "confirm", "hold", "ask", "none"]
    corrections: int
    time_ms: float
    limit_ms: float
    duration_ms: float
    movement: MovementSummary | None
    veto: VetoRecord | None
    offers: tuple[OfferResponse, ...]
    hand: HandSummary | None
    gaze: GazeSummary | None
    interrupted: bool
    interruption_reason: str | None
    explanation_views: int
    recorded_at: datetime
    seq: int

    def __post_init__(self) -> None:
        _require(self.task in TASKS, "task is unknown")
        _require(self.modality in MODALITIES, "modality is unknown")
        _require(self.commit_via in COMMIT_VIA, "commit_via is unknown")
        _require(len(self.reversible) == len(self.candidates), "reversible has one entry per candidate")
        _require((self.intended_index is None) == (self.task == "idle"), "intended_index is null exactly for idle")
        if self.l1_fallback_reason is not None:
            _require(self.l1_fallback_reason in OFFER_FALLBACK_REASONS, "l1_fallback_reason is unknown")


@dataclass(frozen=True, slots=True)
class TrialLabels(_Serializable):
    """The daemon's trial labels (section 7.1), derived from the trial record, never from the UI's claims."""

    complete: bool  # the committed candidate equals the intended one within the limit
    ua: bool  # a commit of a non-intended candidate; in an idle epoch any commit
    correction: bool  # undo, confirm cancel, early hold release, "None of these" or re-selection
    veto: bool  # "Not this time" during the trial
    time_ms: float  # cue to correct commit (floor 150 ms), or the limit when censored
    censored: bool  # not completed: log time right-censored at the limit
    limit_ms: float
    idle: bool
    idle_commits: int  # commits during an idle epoch (0 otherwise)
    idle_minutes: float  # idle epoch duration in minutes (0 otherwise)

    def reply(self) -> dict[str, Any]:
        """The ``labels`` object of the ``co_adaptation_record_trial`` reply."""
        return {
            "complete": self.complete,
            "ua": self.ua,
            "correction": self.correction,
            "veto": self.veto,
            "time_ms": self.time_ms,
        }


@dataclass(frozen=True, slots=True)
class EpisodeLabels(_Serializable):
    """One L2 episode: an ambiguous activation's response and what happened from it to the trial's end."""

    activation_index: int
    decision_id: str
    variant: str
    cell_id: str
    source: Literal["table", "fixed_rule", "fallback", "cap"]
    context: ActivationContext
    complete: bool
    ua: bool
    correction: bool
    veto: bool
    time_ms: float  # activation to commit (floor 150 ms), or to the trial limit when censored
    censored: bool


@dataclass(frozen=True, slots=True)
class StoredTrial(_Serializable):
    """One ``trials`` row and one export entry; the replay input of section 10.8."""

    trial: TrialRecord
    verification: Literal["ok", "draw_mismatch", "posterior_mismatch", "interrupted", "fallback"]
    day_index: int
    regime_ucb_unintended: float | None
    labels: TrialLabels | None
    episodes: tuple[EpisodeLabels, ...]
    preference_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(self.verification in VERIFICATIONS, "verification is unknown")


@dataclass(frozen=True, slots=True)
class BurdenReport(_Serializable):
    report_id: str
    session_id: str
    seq: int
    block_index: int | None
    scope: Literal["block", "session"]
    effort: int | None
    intrusiveness: int | None
    control: int | None
    tlx: dict[str, int | None] | None
    predictability: int | None
    preferred_regime: str | None
    skipped: tuple[str, ...]
    recorded_at: datetime

    def __post_init__(self) -> None:
        _require(self.scope in ("block", "session"), "scope is unknown")
        for value in (self.effort, self.intrusiveness, self.control, self.predictability):
            _require(value is None or (_is_int(value) and 1 <= value <= 7), "ratings lie in 1..7")


@dataclass(frozen=True, slots=True)
class SessionRow(_Serializable):
    session_id: str
    participant_key: str
    context: Literal["study"]
    day_index: int
    session_number: int
    protocol_id: str | None
    protocol_sha256: str | None
    evidence_kind: EvidenceKind
    started_at: datetime
    ended_at: datetime | None
    end_reason: str | None
    device: DeviceContext
    plan: SessionPlan
    layout_seed: int
    seed_salt: str  # 32 hex characters; never sent to the UI
    gaze_check_accuracy: float | None
    comparator_ranking: tuple[str, ...]
    settings: dict[str, float | int | bool]  # every EngineSettings field; the session runs and replays with these
    protocol_deviations: tuple[str, ...]
    versions: dict[str, str]

    def __post_init__(self) -> None:
        _require(self.context == "study", "context must be 'study'")
        _require(isinstance(self.seed_salt, str) and _SALT_RE.fullmatch(self.seed_salt) is not None, "bad seed_salt")
        if self.gaze_check_accuracy is not None:
            _require(0.0 <= self.gaze_check_accuracy <= 1.0, "gaze_check_accuracy lies in [0, 1]")


@dataclass(frozen=True, slots=True)
class DecisionRecord(_Serializable):
    decision_id: str
    session_id: str
    participant_key: str
    seq: int
    level: DecisionLevel
    mode: PolicyMode
    behavior: Literal["policy", "baseline"]
    created_at: datetime
    superseded_by: str | None
    payload: dict[str, Any]  # exactly the RecommendationV1 (L1, L3) or AmbiguityTableV1 (L2) returned to the UI
    applied: str | None
    fallback_reason: str | None
    versions: dict[str, str]

    def __post_init__(self) -> None:
        _require(self.behavior in ("policy", "baseline"), "behavior is unknown")
        schema = (
            "AmbiguityTableV1" if DecisionLevel(self.level) == DecisionLevel.AMBIGUITY_RESPONSE else "RecommendationV1"
        )
        missing = [key for key in RECORD_SCHEMAS[schema] if key not in self.payload]
        _require(not missing, f"payload is missing {schema} keys")

    @classmethod
    def from_decision(
        cls, decision: Any, *, session_id: str, seq: int, behavior: Literal["policy", "baseline"], created_at: datetime
    ) -> DecisionRecord:
        """The stored record of a ``Decision`` or ``AmbiguityTable`` (payload = its ``to_dict()``)."""
        return cls(
            decision_id=decision.decision_id,
            session_id=session_id,
            participant_key=decision.participant_key,
            seq=seq,
            level=DecisionLevel(decision.level),
            mode=PolicyMode(decision.mode),
            behavior=behavior,
            created_at=created_at,
            superseded_by=None,
            payload=decision.to_dict(),
            applied=None,
            fallback_reason=None,
            versions=dict(decision.versions),
        )


# --------------------------------------------------------------------------------------------- field schemas

_RECT = _Record(Rect)
_SCHEMAS[Rect] = {"cx": FLOAT, "cy": FLOAT, "w": FLOAT, "h": FLOAT}
_SCHEMAS[LayoutSummary] = {
    "task": _Literal(TASKS),
    "n_candidates": INT,
    "w_px": FLOAT,
    "h_px": FLOAT,
    "gap_px": FLOAT,
    "id_bar": FLOAT,
    "menu_n": INT,
    "origin_x": FLOAT,
    "origin_y": FLOAT,
    # A pointing grid's mean keyboard presses (decision-time prior, 7.3); kept so a logged block rebuilt from its
    # trials (OPE) predicts as the live decision did.
    "presses_mean": _Omittable(FLOAT),
}
_SCHEMAS[DeviceContext] = {
    "modality": _Literal(MODALITIES),
    "pointer_types": _List(STR),
    "dpr": FLOAT,
    "surface_w": FLOAT,
    "surface_h": FLOAT,
    "gaze_available": BOOL,
    "hand_pointer": BOOL,
}
_SCHEMAS[ActivationContext] = {
    "ambiguity": FLOAT,
    "bin": _Literal(("b1", "b2", "b3")),
    "n_plausible": INT,
    "has_hit": BOOL,
    "p_hit": FLOAT,
    "p_top": FLOAT,
    "modality": _Literal(MODALITIES),
    "layout": _Record(LayoutSummary),
}
_SCHEMAS[SpreadUsed] = {"r": FLOAT, "floor": FLOAT}
_SCHEMAS[ActivationGaze] = {"region": _Literal(GAZE_REGIONS), "conf": FLOAT}
_SCHEMAS[L2Applied] = {
    "decision_id": STR,
    "cell_id": STR,
    "draw_index": INT,
    "variant": _Literal(L2_OPTIONS),
    "source": _Literal(L2_SOURCES),
    "fallback_reason": _opt(_Literal(L2_FALLBACK_REASONS)),
}
_SCHEMAS[ActivationRecord] = {
    "index": INT,
    "t_ms": FLOAT,
    "modality": _Literal(MODALITIES),
    "x": INT,
    "y": INT,
    "hit_index": _opt(INT),
    "a_gate": FLOAT,
    "n_plausible": INT,
    "press_index": INT,
    "expanded_index": _opt(INT),
    "hit_rect": _opt(_RECT),
    "gaze": _opt(_Record(ActivationGaze)),
    "l2": _opt(_Record(L2Applied)),
    "irreversible_rule": BOOL,
    "result": _Literal(ACTIVATION_RESULTS),
    "outcome": _Literal(ACTIVATION_OUTCOMES),
    "committed_index": _opt(INT),
}
_SCHEMAS[MovementSummary] = {
    "rt_ms": FLOAT,
    "mt_ms": FLOAT,
    "path_ratio": FLOAT,
    "submovements": INT,
    "peak_speed_px_s": FLOAT,
    "reentries": INT,
    "endpoint_dx_norm": FLOAT,
    "endpoint_dy_norm": FLOAT,
}
_SCHEMAS[OfferResponse] = {
    "decision_id": STR,
    "variant": _Literal(L3_OPTIONS),
    "response": _Literal(OFFER_RESPONSES),
    "fallback_reason": _opt(_Literal(OFFER_FALLBACK_REASONS)),
}
_SCHEMAS[RevertRecord] = {
    "reason": _Literal(REVERT_REASONS),
    "from_activation": INT,
    "regime": STR,
    "fallbacks_issued_at": DATETIME,
}
_SCHEMAS[VetoRecord] = {"decision_id": STR, "level": _Enum(DecisionLevel)}
_SCHEMAS[HandSummary] = {"quality_mean": FLOAT, "quality_min": FLOAT, "tracking_losses": INT}
_SCHEMAS[GazeSummary] = {"agreed_with_intended": _opt(BOOL)}
_SCHEMAS[TrialRecord] = {
    "trial_id": STR,
    "session_id": STR,
    "block_index": INT,
    "trial_index": INT,
    "task": _Literal(TASKS),
    "regime": STR,
    "regime_decision_id": STR,
    "table_decision_ids": _List(STR),
    "applied_regime": STR,
    "l1_fallback_reason": _opt(_Literal(OFFER_FALLBACK_REASONS)),
    "fallbacks_issued_at": DATETIME,
    "spread": _opt(_Record(SpreadUsed)),
    "layout": _Record(LayoutSummary),
    "candidates": _List(_RECT),
    "surface_w": FLOAT,
    "surface_h": FLOAT,
    "dpr": FLOAT,
    "reversible": _List(BOOL),
    "cue_at": DATETIME,
    "intended_index": _opt(INT),
    "modality": _Literal(MODALITIES),
    "activations": _List(_Record(ActivationRecord)),
    "revert": _opt(_Record(RevertRecord)),
    "committed_index": _opt(INT),
    "commit_t_ms": _opt(FLOAT),
    "commit_via": _Literal(COMMIT_VIA),
    "corrections": INT,
    "time_ms": FLOAT,
    "limit_ms": FLOAT,
    "duration_ms": FLOAT,
    "movement": _opt(_Record(MovementSummary)),
    "veto": _opt(_Record(VetoRecord)),
    "offers": _List(_Record(OfferResponse)),
    "hand": _opt(_Record(HandSummary)),
    "gaze": _opt(_Record(GazeSummary)),
    "interrupted": BOOL,
    "interruption_reason": _opt(STR),
    "explanation_views": INT,
    "recorded_at": DATETIME,
    "seq": INT,
}
_SCHEMAS[TrialLabels] = {
    "complete": BOOL,
    "ua": BOOL,
    "correction": BOOL,
    "veto": BOOL,
    "time_ms": FLOAT,
    "censored": BOOL,
    "limit_ms": FLOAT,
    "idle": BOOL,
    "idle_commits": INT,
    "idle_minutes": FLOAT,
}
_SCHEMAS[EpisodeLabels] = {
    "activation_index": INT,
    "decision_id": STR,
    "variant": _Literal(L2_OPTIONS),
    "cell_id": STR,
    "source": _Literal(L2_SOURCES),
    "context": _Record(ActivationContext),
    "complete": BOOL,
    "ua": BOOL,
    "correction": BOOL,
    "veto": BOOL,
    "time_ms": FLOAT,
    "censored": BOOL,
}
_SCHEMAS[StoredTrial] = {
    "trial": _Record(TrialRecord),
    "verification": _Literal(VERIFICATIONS),
    "day_index": INT,
    "regime_ucb_unintended": _opt(FLOAT),
    "labels": _opt(_Record(TrialLabels)),
    "episodes": _List(_Record(EpisodeLabels)),
    "preference_violations": _List(STR),
}
_SCHEMAS[BurdenReport] = {
    "report_id": STR,
    "session_id": STR,
    "seq": INT,
    "block_index": _opt(INT),
    "scope": _Literal(("block", "session")),
    "effort": _opt(INT),
    "intrusiveness": _opt(INT),
    "control": _opt(INT),
    "tlx": _opt(_Dict(_opt(INT))),
    "predictability": _opt(INT),
    "preferred_regime": _opt(STR),
    "skipped": _List(STR),
    "recorded_at": DATETIME,
}
_SCHEMAS[SessionRow] = {
    "session_id": STR,
    "participant_key": STR,
    "context": _Literal(("study",)),
    "day_index": INT,
    "session_number": INT,
    "protocol_id": _opt(STR),
    "protocol_sha256": _opt(STR),
    "evidence_kind": _Enum(EvidenceKind),
    "started_at": DATETIME,
    "ended_at": _opt(DATETIME),
    "end_reason": _opt(STR),
    "device": _Record(DeviceContext),
    "plan": _Plan(),
    "layout_seed": INT,
    "seed_salt": STR,
    "gaze_check_accuracy": _opt(FLOAT),
    "comparator_ranking": _List(STR),
    "settings": _Dict(_Scalar()),
    "protocol_deviations": _List(STR),
    "versions": _Dict(STR),
}
_SCHEMAS[DecisionRecord] = {
    "decision_id": STR,
    "session_id": STR,
    "participant_key": STR,
    "seq": INT,
    "level": _Enum(DecisionLevel),
    "mode": _Enum(PolicyMode),
    "behavior": _Literal(("policy", "baseline")),
    "created_at": DATETIME,
    "superseded_by": _opt(STR),
    "payload": _Json(),
    "applied": _opt(STR),
    "fallback_reason": _opt(STR),
    "versions": _Dict(STR),
}

_RECORD_CLASSES: tuple[type, ...] = (
    SpreadUsed,
    ActivationGaze,
    L2Applied,
    ActivationRecord,
    MovementSummary,
    OfferResponse,
    RevertRecord,
    VetoRecord,
    HandSummary,
    GazeSummary,
    TrialRecord,
    TrialLabels,
    EpisodeLabels,
    StoredTrial,
    BurdenReport,
    SessionRow,
    DecisionRecord,
)

# Required keys and types per record, for SDK hosts. The RPC payload objects a DecisionRecord stores are listed too.
RECORD_SCHEMAS: dict[str, dict[str, str]] = {
    cls.__name__: {name: codec.name for name, codec in _SCHEMAS[cls].items()} for cls in _RECORD_CLASSES
}
RECORD_SCHEMAS["Rect"] = {name: codec.name for name, codec in _SCHEMAS[Rect].items()}
RECORD_SCHEMAS["LayoutSummary"] = {name: codec.name for name, codec in _SCHEMAS[LayoutSummary].items()}
RECORD_SCHEMAS["DeviceContext"] = {name: codec.name for name, codec in _SCHEMAS[DeviceContext].items()}
RECORD_SCHEMAS["ActivationContext"] = {name: codec.name for name, codec in _SCHEMAS[ActivationContext].items()}
RECORD_SCHEMAS["RecommendationV1"] = {
    "decision_id": "str",
    "level": "block_regime|offer",
    "mode": "PolicyMode",
    "variant": "str",
    "params": "object",
    "eligible": "list[str]",
    "excluded": "dict[str, ExclusionReason]",
    "propensities": "dict[str, float]",
    "p_thompson": "dict[str, float]|null",
    "exploration": "float",
    "fallback": "bool",
    "protocol_deviation": "bool",
    "round_index": "int|null",
    "spread": "SpreadUsed|null",
    "priorities": "dict[str, float]|null",
    "predictions": "dict[str, PredictionV1]",
    "explanation": "ExplanationV1",
    "seed": "str",
    "versions": "dict[str, str]",
    "authority": "presentation_only",
    "execution_authority": "false",
}
RECORD_SCHEMAS["AmbiguityTableV1"] = {
    "decision_id": "str",
    "level": "ambiguity_table",
    "block_decision_id": "str",
    "mode": "PolicyMode",
    "ambiguity_threshold": "float",
    "bins": "list[{id, lo, hi}]",
    "cells": "list[{cell_id, bin, n_plausible, has_hit, order, propensities, eligible, excluded}]",
    "start_trial_index": "int",
    "draws": "list[list[float]]",
    "max_draws_per_trial": "int",
    "after_max": "wait_for_evidence",
    "explanation": "ExplanationV1",
    "seed": "str",
    "versions": "dict[str, str]",
    "authority": "presentation_only",
    "execution_authority": "false",
}
RECORD_SCHEMAS["PredictionV1"] = {
    "variant": "str",
    "p_complete": "{mean, lo90, hi90}",
    "p_unintended": "{mean, lo90, hi90}",
    "ucb_unintended": "float",
    "p_correction": "{mean, lo90, hi90}",
    "p_veto": "{mean, lo90, hi90}",
    "time_ms": "{mean, lo90, hi90}",
    "burden": "dict[str, float]|null",
    "evidence": "{trials_person, trials_today, source}",
    "authority": "presentation_only",
}


def record_fields(cls: type) -> tuple[str, ...]:
    """Dataclass field names of a record (RECORD_SCHEMAS keys equal them; tested)."""
    return tuple(f.name for f in fields(cls))


def vetoed_variant(trial: TrialRecord) -> str | None:
    """The variant a trial's "Not this time" was pressed on (C5, section 9.7), or None without a veto.

    The study shows the regime chip for the regime in force, which is the revert regime once the trial is reverted,
    and the response chip only for the open L2 step of the latest activation (never for ``wait_for_evidence``). A
    trial holds at most one revert and one veto, which fixes their order:

    - a revert for a veto was made by this veto, which named the cue regime (L1) or the latest response before the
      revert (L2);
    - a revert for a refresh came before the veto, because a veto before a refresh reverts the trial for the veto and
      the refresh then interrupts it: an L1 veto named the refresh regime, and an L2 veto a response of the
      activation whose step was open when the refresh arrived (``from_activation - 1``) or of a later one;
    - without a revert the veto had nothing to revert to: the cue regime, or the latest response.

    An L2 veto names the decision of the response it was pressed on (a table response carries the table's id, a
    fixed-rule or fallback response the block's L1 decision id); a response with that id is preferred.
    """
    veto = trial.veto
    if veto is None:
        return None
    revert = trial.revert
    after_refresh = revert is not None and revert.reason == "refresh"
    if DecisionLevel(veto.level) == DecisionLevel.BLOCK_REGIME:
        return revert.regime if after_refresh and revert is not None else trial.applied_regime

    def in_window(index: int) -> bool:
        if revert is None:
            return True
        if revert.reason == "veto":
            return index < revert.from_activation
        return index >= revert.from_activation - 1

    responses = [
        activation.l2
        for activation in trial.activations
        if activation.l2 is not None and activation.l2.variant != "wait_for_evidence" and in_window(activation.index)
    ]
    matching = [response for response in responses if response.decision_id == veto.decision_id]
    chosen = matching or responses
    return chosen[-1].variant if chosen else None


# ------------------------------------------------------------------------------------------------------ labels


def _wrong_commit(activation: ActivationRecord, intended: int | None) -> bool:
    return (
        activation.outcome == "committed"
        and activation.committed_index is not None
        and (intended is None or activation.committed_index != intended)
    )


def derive_labels(trial: TrialRecord, tables: Mapping[str, Any] | None = None) -> TrialLabels:
    """Trial labels of section 7.1 from the record alone (the UI's claims are never trusted as labels).

    ``complete``: the final committed candidate is the intended one, committed within the limit. ``ua``: any
    activation committed a non-intended candidate (in an idle epoch, any commit). ``correction``: a correction was
    counted or an activation's step ended ``cancelled``. ``veto``: "Not this time" was pressed. A trial that was not
    completed is censored at its limit. Activation outcomes are read, never the immediate ``result``. ``tables`` is
    accepted for the stored-table context and not needed for the trial labels.
    """
    del tables
    cancelled = any(activation.outcome == "cancelled" for activation in trial.activations)
    correction = trial.corrections > 0 or cancelled
    veto = trial.veto is not None
    if trial.task == "idle":
        commits = sum(1 for activation in trial.activations if activation.outcome == "committed")
        if trial.committed_index is not None and commits == 0:
            commits = 1
        return TrialLabels(
            complete=False,
            ua=commits > 0,
            correction=correction,
            veto=veto,
            time_ms=float(trial.duration_ms),
            censored=False,
            limit_ms=float(trial.limit_ms),
            idle=True,
            idle_commits=commits,
            idle_minutes=float(trial.duration_ms) / 60000.0,
        )
    intended = trial.intended_index
    complete = (
        trial.committed_index is not None
        and trial.committed_index == intended
        and trial.commit_t_ms is not None
        and trial.commit_t_ms <= trial.limit_ms
    )
    ua = any(_wrong_commit(activation, intended) for activation in trial.activations) or (
        trial.committed_index is not None and trial.committed_index != intended
    )
    if complete:
        time_ms = max(float(trial.commit_t_ms or 0.0), TIME_FLOOR_MS)
    else:
        time_ms = float(trial.limit_ms)
    return TrialLabels(
        complete=complete,
        ua=ua,
        correction=correction,
        veto=veto,
        time_ms=time_ms,
        censored=not complete,
        limit_ms=float(trial.limit_ms),
        idle=False,
        idle_commits=0,
        idle_minutes=0.0,
    )


def derive_episodes(
    trial: TrialRecord, contexts: Mapping[int, ActivationContext], labels: TrialLabels | None = None
) -> tuple[EpisodeLabels, ...]:
    """L2 episodes (section 7.1): every activation with an L2 response, with what happened from it to the end.

    ``contexts`` maps activation indices to the ``ActivationContext`` the daemon recomputed. An episode completes
    when the trial's correct commit came at or after the activation; its time runs from the activation to that
    commit (floor 150 ms), else it is censored at the trial limit.
    """
    if trial.task == "idle":
        return ()
    trial_labels = labels or derive_labels(trial)
    intended = trial.intended_index
    cancelled_total = sum(1 for activation in trial.activations if activation.outcome == "cancelled")
    undos = max(0, trial.corrections - cancelled_total)
    last_l2_before_revert = None
    if trial.revert is not None:
        before = [a.index for a in trial.activations if a.l2 is not None and a.index < trial.revert.from_activation]
        last_l2_before_revert = max(before) if before else None
    episodes: list[EpisodeLabels] = []
    for activation in trial.activations:
        applied = activation.l2
        if applied is None or activation.index not in contexts:
            continue
        later = [a for a in trial.activations if a.index >= activation.index]
        wrong_after = any(_wrong_commit(a, intended) for a in later)
        complete = trial_labels.complete and trial.commit_t_ms is not None and trial.commit_t_ms >= activation.t_ms
        correction = any(a.outcome == "cancelled" for a in later) or (undos > 0 and wrong_after)
        veto = (
            trial.veto is not None
            and DecisionLevel(trial.veto.level) == DecisionLevel.AMBIGUITY_RESPONSE
            and activation.index == last_l2_before_revert
        )
        if complete:
            time_ms = max(float(trial.commit_t_ms or 0.0) - activation.t_ms, TIME_FLOOR_MS)
        else:
            time_ms = max(float(trial.limit_ms) - activation.t_ms, TIME_FLOOR_MS)
        episodes.append(
            EpisodeLabels(
                activation_index=activation.index,
                decision_id=applied.decision_id,
                variant=applied.variant,
                cell_id=applied.cell_id,
                source=applied.source,
                context=contexts[activation.index],
                complete=complete,
                ua=wrong_after,
                correction=correction,
                veto=veto,
                time_ms=time_ms,
                censored=not complete,
            )
        )
    return tuple(episodes)

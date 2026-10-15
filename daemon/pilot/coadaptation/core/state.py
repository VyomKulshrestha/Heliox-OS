"""Interaction state types and feature vectors (spec section 5).

Every type here is built from the study surface's own measurements (layout geometry, device, clock, practice) and
never from the cued target, cognitive or stress estimates, screen content, transcripts or window titles. Decision-time
features use layout-level geometry (``id_bar``) only; the realized target enters the physics prior on the update side
(``priors.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol, get_args

import numpy as np

from .vocabulary import ASSISTANCE_OFF, VARIANTS, DecisionLevel

FEATURE_SCHEMA = "coadapt-features-v1"

Modality = Literal["mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer"]
TaskKind = Literal["pointing", "menu", "idle"]
Phase = Literal["practice", "A", "A_lite", "B", "probe", "comparator"]
# Phase B consists of blocks with phase "B" (adaptive), "probe" (assistance_off) and "comparator" (the participant's
# best fixed assisted regime from session 1), section 12.3
AmbiguityBin = Literal["b1", "b2", "b3"]
StruggleKind = Literal["corrections", "tracking_loss"]

MODALITIES: tuple[str, ...] = get_args(Modality)
TASK_KINDS: tuple[str, ...] = get_args(TaskKind)
PHASES: tuple[str, ...] = get_args(Phase)
AMBIGUITY_BINS: tuple[str, ...] = get_args(AmbiguityBin)
STRUGGLE_KINDS: tuple[str, ...] = get_args(StruggleKind)

L1_DIM = 18
L2_DIM = 12
BURDEN_DIM = 10

L1_FEATURE_NAMES: tuple[str, ...] = (
    "intercept",
    "g_off",
    "g_enl",
    "g_conf",
    "enl_x_hand",
    "conf_x_hand",
    "id_c",
    "small_gap",
    "enl_x_small_gap",
    "practice_variant",
    "practice_total",
    "switch_transient",
    "is_menu",
    "fatigue",  # session time, not a fatigue estimate (spec 5.4)
    "quality_deficit",
    "psi_one",
    "psi_is_hand",
    "psi_nonacting",
)
L2_FEATURE_NAMES: tuple[str, ...] = (
    "intercept",
    "g_conf",
    "g_wait",
    "g_ask",
    "ambiguity_c",
    "nonacting_x_hand",
    "practice_variant",
    "is_menu",
    "k3",
    "psi_one",
    "psi_is_hand",
    "psi_nonacting",
)
BURDEN_FEATURE_NAMES: tuple[str, ...] = (
    "intercept",
    "g_off",
    "g_enl",
    "g_conf",
    "ipt",
    "practice_variant",
    "is_hand",
    "psi_one",
    "psi_is_hand",
    "psi_nonacting",
)

# Index sets used by the model: the today offset gamma, and the practice and switch features (NO_PRACTICE_SWITCH).
L1_GAMMA = slice(15, 18)
L2_GAMMA = slice(9, 12)
BURDEN_GAMMA = slice(7, 10)
L1_PRACTICE_SWITCH: tuple[int, ...] = (9, 10, 11)
L2_PRACTICE_SWITCH: tuple[int, ...] = (6,)
BURDEN_PRACTICE_SWITCH: tuple[int, ...] = (5,)

# RPC context keys per level (spec 5.4 and 14.3). Anything else fails validation.
_BLOCK_KEYS = frozenset(
    {"block_index", "layouts", "previous_regime", "session_minute", "trials_today", "hand_quality_recent"}
)
ALLOWED_CONTEXT_KEYS: dict[DecisionLevel, frozenset[str]] = {
    DecisionLevel.BLOCK_REGIME: _BLOCK_KEYS,
    DecisionLevel.AMBIGUITY_RESPONSE: _BLOCK_KEYS | {"block_decision_id", "start_trial_index", "trials_in_block"},
    DecisionLevel.OFFER: frozenset({"block_index", "trial_index", "task", "struggle_kind", "keyboard_available"}),
}
DENIED_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "stress",
    "attention",
    "load",
    "cognitive",
    "mood",
    "emotion",
    "fatigue_score",
    "ocr",
    "window_title",
    "transcript",
    "keystroke",
    "clipboard",
    "screen",
    "latent",
)


# --- serialization helpers (spec 13.1.2 rules; errors never echo values) -------------------------------------------


def _fields(cls_name: str, d: Any, names: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(d, Mapping):
        raise ValueError(f"{cls_name}: expected an object")
    expected = set(names)
    for key in d:
        if key not in expected:
            raise ValueError(f"{cls_name}: unknown key")
    for name in names:
        if name not in d:
            raise ValueError(f"{cls_name}: missing key {name}")
    return d


def _float(cls_name: str, d: Mapping[str, Any], key: str) -> float:
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{cls_name}: {key} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{cls_name}: {key} must be finite")
    return out


def _opt_float(cls_name: str, d: Mapping[str, Any], key: str) -> float | None:
    return None if d[key] is None else _float(cls_name, d, key)


def _int(cls_name: str, d: Mapping[str, Any], key: str) -> int:
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{cls_name}: {key} must be an integer")
    return int(value)


def _bool(cls_name: str, d: Mapping[str, Any], key: str) -> bool:
    value = d[key]
    if not isinstance(value, bool):
        raise ValueError(f"{cls_name}: {key} must be a boolean")
    return value


def _str(cls_name: str, d: Mapping[str, Any], key: str) -> str:
    value = d[key]
    if not isinstance(value, str):
        raise ValueError(f"{cls_name}: {key} must be a string")
    return value


def _opt_str(cls_name: str, d: Mapping[str, Any], key: str) -> str | None:
    return None if d[key] is None else _str(cls_name, d, key)


def _choice(cls_name: str, d: Mapping[str, Any], key: str, allowed: Sequence[str]) -> Any:
    value = _str(cls_name, d, key)
    if value not in allowed:
        raise ValueError(f"{cls_name}: {key} has an unknown value")
    return value


def _list(cls_name: str, d: Mapping[str, Any], key: str) -> list[Any]:
    value = d[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{cls_name}: {key} must be a list")
    return list(value)


# --- state types (spec 5.1) ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rect:  # surface CSS px, measured with getBoundingClientRect at cue time
    cx: float
    cy: float
    w: float
    h: float

    def to_dict(self) -> dict[str, Any]:
        return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Rect:
        f = _fields("Rect", d, ("cx", "cy", "w", "h"))
        return cls(_float("Rect", f, "cx"), _float("Rect", f, "cy"), _float("Rect", f, "w"), _float("Rect", f, "h"))


@dataclass(frozen=True, slots=True)
class LayoutSummary:  # INTERFACE state; never contains the cued target
    task: TaskKind
    n_candidates: int  # 2..32
    w_px: float  # candidate width (uniform within a study layout)
    h_px: float  # candidate height
    gap_px: float  # gap to nearest neighbour along the dominant axis
    id_bar: float  # mean over candidates i of log2(D_i / W_i + 1), D_i from the origin point; bits
    menu_n: int  # items in the open menu (0 for pointing)
    origin_x: float  # surface CSS px: centre of the home button (pointing, idle) or of the menu button
    origin_y: float  #   (menu); every target distance D is measured from here (section 12.4)
    # Pointing grids: the mean keyboard presses over the layout's candidates, |dcolumn| + |drow| + 1 from the start
    # candidate (the one nearest the origin, the lower index on ties), as keyboard focus moves (section 12.6). Grid
    # geometry, not the cue; prediction mode uses it (7.3). Optional: None (and absent from the dict) when the
    # surface did not send it, and the prior then approximates the grid. Menus never need it: (n + 1) / 2.
    presses_mean: float | None = None

    def __post_init__(self) -> None:
        if self.presses_mean is not None and (
            isinstance(self.presses_mean, bool)
            or not isinstance(self.presses_mean, int | float)
            or not math.isfinite(self.presses_mean)
            or not 1.0 <= float(self.presses_mean) <= 64.0
        ):
            raise ValueError("LayoutSummary: presses_mean must be a number in [1, 64]")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "task": self.task,
            "n_candidates": self.n_candidates,
            "w_px": self.w_px,
            "h_px": self.h_px,
            "gap_px": self.gap_px,
            "id_bar": self.id_bar,
            "menu_n": self.menu_n,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
        }
        if self.presses_mean is not None:
            out["presses_mean"] = self.presses_mean
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LayoutSummary:
        n = "LayoutSummary"
        names = ("task", "n_candidates", "w_px", "h_px", "gap_px", "id_bar", "menu_n", "origin_x", "origin_y")
        if not isinstance(d, Mapping):
            raise ValueError(f"{n}: expected an object")
        has_presses = "presses_mean" in d
        f = _fields(n, {key: value for key, value in d.items() if key != "presses_mean"}, names)
        return cls(
            task=_choice(n, f, "task", TASK_KINDS),
            n_candidates=_int(n, f, "n_candidates"),
            w_px=_float(n, f, "w_px"),
            h_px=_float(n, f, "h_px"),
            gap_px=_float(n, f, "gap_px"),
            id_bar=_float(n, f, "id_bar"),
            menu_n=_int(n, f, "menu_n"),
            origin_x=_float(n, f, "origin_x"),
            origin_y=_float(n, f, "origin_y"),
            presses_mean=_opt_float(n, d, "presses_mean") if has_presses else None,
        )


@dataclass(frozen=True, slots=True)
class DeviceContext:
    modality: Modality
    pointer_types: tuple[str, ...]  # PointerEvent.pointerType values seen at session start
    dpr: float
    surface_w: float
    surface_h: float
    gaze_available: bool  # consent purpose + config + vision.gaze_tracking_enabled + gaze check passed
    hand_pointer: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "pointer_types": list(self.pointer_types),
            "dpr": self.dpr,
            "surface_w": self.surface_w,
            "surface_h": self.surface_h,
            "gaze_available": self.gaze_available,
            "hand_pointer": self.hand_pointer,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DeviceContext:
        n = "DeviceContext"
        f = _fields(
            n, d, ("modality", "pointer_types", "dpr", "surface_w", "surface_h", "gaze_available", "hand_pointer")
        )
        pointer_types = _list(n, f, "pointer_types")
        if not all(isinstance(p, str) for p in pointer_types):
            raise ValueError(f"{n}: pointer_types must hold strings")
        return cls(
            modality=_choice(n, f, "modality", MODALITIES),
            pointer_types=tuple(pointer_types),
            dpr=_float(n, f, "dpr"),
            surface_w=_float(n, f, "surface_w"),
            surface_h=_float(n, f, "surface_h"),
            gaze_available=_bool(n, f, "gaze_available"),
            hand_pointer=_bool(n, f, "hand_pointer"),
        )


@dataclass(frozen=True, slots=True)
class ClockContext:
    day_index: int  # the session's stored day_index (computed once at session_begin, section 8)
    session_minute: float
    trials_today: int
    days_since_last_session: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_index": self.day_index,
            "session_minute": self.session_minute,
            "trials_today": self.trials_today,
            "days_since_last_session": self.days_since_last_session,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClockContext:
        n = "ClockContext"
        f = _fields(n, d, ("day_index", "session_minute", "trials_today", "days_since_last_session"))
        return cls(
            day_index=_int(n, f, "day_index"),
            session_minute=_float(n, f, "session_minute"),
            trials_today=_int(n, f, "trials_today"),
            days_since_last_session=_opt_float(n, f, "days_since_last_session"),
        )


@dataclass(frozen=True, slots=True)
class BlockContext:  # input to an L1 decision and to an L2 table
    session_id: str
    block_index: int
    phase: Phase
    task: TaskKind
    layouts: tuple[LayoutSummary, ...]  # one per planned trial in the block (8)
    device: DeviceContext
    clock: ClockContext
    previous_regime: str | None
    hand_quality_recent: float | None  # mean hand quality over the previous block (hand pointer only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "block_index": self.block_index,
            "phase": self.phase,
            "task": self.task,
            "layouts": [layout.to_dict() for layout in self.layouts],
            "device": self.device.to_dict(),
            "clock": self.clock.to_dict(),
            "previous_regime": self.previous_regime,
            "hand_quality_recent": self.hand_quality_recent,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BlockContext:
        n = "BlockContext"
        f = _fields(
            n,
            d,
            (
                "session_id",
                "block_index",
                "phase",
                "task",
                "layouts",
                "device",
                "clock",
                "previous_regime",
                "hand_quality_recent",
            ),
        )
        previous = _opt_str(n, f, "previous_regime")
        if previous is not None and previous not in VARIANTS:
            raise ValueError(f"{n}: previous_regime has an unknown value")
        return cls(
            session_id=_str(n, f, "session_id"),
            block_index=_int(n, f, "block_index"),
            phase=_choice(n, f, "phase", PHASES),
            task=_choice(n, f, "task", TASK_KINDS),
            layouts=tuple(LayoutSummary.from_dict(x) for x in _list(n, f, "layouts")),
            device=DeviceContext.from_dict(f["device"]),
            clock=ClockContext.from_dict(f["clock"]),
            previous_regime=previous,
            hand_quality_recent=_opt_float(n, f, "hand_quality_recent"),
        )


@dataclass(frozen=True, slots=True)
class ActivationContext:  # input to L2 predictions; computed from the endpoint, never from the cue
    ambiguity: float  # A_gate
    bin: AmbiguityBin
    n_plausible: int  # 1, 2 or 3: plausible candidates clipped to [1, 3] (1 also covers "none", 6.3)
    has_hit: bool
    p_hit: float  # posterior of the literal hit (0 when no hit)
    p_top: float  # max posterior
    modality: Modality
    layout: LayoutSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguity": self.ambiguity,
            "bin": self.bin,
            "n_plausible": self.n_plausible,
            "has_hit": self.has_hit,
            "p_hit": self.p_hit,
            "p_top": self.p_top,
            "modality": self.modality,
            "layout": self.layout.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ActivationContext:
        n = "ActivationContext"
        f = _fields(n, d, ("ambiguity", "bin", "n_plausible", "has_hit", "p_hit", "p_top", "modality", "layout"))
        return cls(
            ambiguity=_float(n, f, "ambiguity"),
            bin=_choice(n, f, "bin", AMBIGUITY_BINS),
            n_plausible=_int(n, f, "n_plausible"),
            has_hit=_bool(n, f, "has_hit"),
            p_hit=_float(n, f, "p_hit"),
            p_top=_float(n, f, "p_top"),
            modality=_choice(n, f, "modality", MODALITIES),
            layout=LayoutSummary.from_dict(f["layout"]),
        )


@dataclass(frozen=True, slots=True)
class OfferContext:
    session_id: str
    block_index: int
    trial_index: int
    modality: Modality
    task: TaskKind
    struggle_kind: StruggleKind
    keyboard_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "block_index": self.block_index,
            "trial_index": self.trial_index,
            "modality": self.modality,
            "task": self.task,
            "struggle_kind": self.struggle_kind,
            "keyboard_available": self.keyboard_available,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OfferContext:
        n = "OfferContext"
        f = _fields(
            n,
            d,
            ("session_id", "block_index", "trial_index", "modality", "task", "struggle_kind", "keyboard_available"),
        )
        return cls(
            session_id=_str(n, f, "session_id"),
            block_index=_int(n, f, "block_index"),
            trial_index=_int(n, f, "trial_index"),
            modality=_choice(n, f, "modality", MODALITIES),
            task=_choice(n, f, "task", TASK_KINDS),
            struggle_kind=_choice(n, f, "struggle_kind", STRUGGLE_KINDS),
            keyboard_available=_bool(n, f, "keyboard_available"),
        )


class PracticeView(Protocol):
    """Exposure counts read by the feature builders (``outcome_model.PracticeCounters`` implements it)."""

    @property
    def n_total(self) -> int: ...

    @property
    def n_variant(self) -> Mapping[str, int]: ...  # L1 trials per regime

    @property
    def n_episode(self) -> Mapping[str, int]: ...  # L2 episodes per response


# --- feature vectors (spec 5.3) -----------------------------------------------------------------------------------


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _groups(variant_id: str) -> tuple[float, float, float, float, float]:
    """(g_off, g_enl, g_conf, g_wait, g_ask) for a variant id."""
    v = VARIANTS[variant_id]
    family = v.strategy.value if v.strategy is not None else v.id
    return (
        1.0 if variant_id == ASSISTANCE_OFF else 0.0,
        1.0 if family == "enlarge_target" else 0.0,
        1.0 if family == "confirm_selection" else 0.0,
        1.0 if family == "wait_for_evidence" else 0.0,
        1.0 if family == "ask_clarification" else 0.0,
    )


def _practice_feature(count: int, scale: float) -> float:
    return _clip(math.log1p(max(count, 0)) / math.log1p(scale), 0.0, 1.5)


def build_l1_features(
    layout: LayoutSummary,
    variant: str,
    device: DeviceContext,
    clock: ClockContext,
    practice: PracticeView,
    switch_age: int | None,
    quality: float | None,
) -> np.ndarray:
    """L1 trial-head features f1 in R^18, in the order of spec 5.3.

    ``switch_age`` is k, the trials since the regime last changed in this session (0 on the first trial after a
    switch), or None when the regime has not switched in this session. ``quality`` is the hand-quality mean (hand
    pointer only; None counts as no deficit).
    """
    g_off, g_enl, g_conf, _, _ = _groups(variant)
    is_hand = 1.0 if device.modality == "hand_pointer" else 0.0
    small_gap = 1.0 if layout.gap_px <= 4 else 0.0
    switch = math.exp(-max(switch_age, 0) / 3.0) if switch_age is not None else 0.0
    deficit = _clip(1.0 - quality, 0.0, 1.0) if (is_hand and quality is not None) else 0.0
    f = np.array(
        [
            1.0,
            g_off,
            g_enl,
            g_conf,
            g_enl * is_hand,
            g_conf * is_hand,
            _clip((layout.id_bar - 3.0) / 2.0, -1.5, 1.5),
            small_gap,
            g_enl * small_gap,
            _practice_feature(practice.n_variant.get(variant, 0), 200.0),
            _practice_feature(practice.n_total, 1000.0),
            switch,
            1.0 if layout.task == "menu" else 0.0,
            min(max(clock.session_minute, 0.0), 60.0) / 30.0,
            deficit,
            1.0,
            is_hand,
            g_conf,
        ],
        dtype=np.float64,
    )
    return f


def build_l2_features(act_ctx: ActivationContext, variant: str, practice: PracticeView) -> np.ndarray:
    """L2 episode-head features f2 in R^12 (spec 5.3; reference variant ``act``)."""
    _, _, g_conf, g_wait, g_ask = _groups(variant)
    is_hand = 1.0 if act_ctx.modality == "hand_pointer" else 0.0
    nonacting = 0.0 if VARIANTS[variant].acting else 1.0
    return np.array(
        [
            1.0,
            g_conf,
            g_wait,
            g_ask,
            _clip((act_ctx.ambiguity - 0.5) / 0.5, -1.0, 1.0),
            nonacting * is_hand,
            _practice_feature(practice.n_episode.get(variant, 0), 200.0),
            1.0 if act_ctx.layout.task == "menu" else 0.0,
            1.0 if variant == "ask_clarification:3" else 0.0,
            1.0,
            is_hand,
            nonacting,
        ],
        dtype=np.float64,
    )


def build_burden_features(
    variant: str, interruptions_per_trial: float, practice: PracticeView, modality: str
) -> np.ndarray:
    """Burden-head features fb in R^10 (spec 5.3) for the block regime ``variant``."""
    g_off, g_enl, g_conf, _, _ = _groups(variant)
    is_hand = 1.0 if modality == "hand_pointer" else 0.0
    nonacting = 0.0 if VARIANTS[variant].acting else 1.0
    return np.array(
        [
            1.0,
            g_off,
            g_enl,
            g_conf,
            _clip(interruptions_per_trial, 0.0, 3.0) / 3.0,
            _practice_feature(practice.n_variant.get(variant, 0), 200.0),
            is_hand,
            1.0,
            is_hand,
            nonacting,
        ],
        dtype=np.float64,
    )


def device_class(device: DeviceContext) -> tuple[str, float, int, int]:
    """(modality, dpr, floor(w / 200), floor(h / 150)): a change of any part starts a new today context (spec 8)."""
    return (device.modality, float(device.dpr), int(device.surface_w // 200), int(device.surface_h // 150))


def day_index_for(local_dt: datetime, boundary_hour: int) -> int:
    """The study day of a local time: ``(local_dt - boundary_hour).date().toordinal()`` (spec 8).

    ``local_dt`` must already carry the local UTC offset; this function reads only its wall-clock fields and never
    converts it, so the machine's time zone plays no part. Only the service calls it, once per session.
    """
    if local_dt.tzinfo is None or local_dt.utcoffset() is None:
        raise ValueError("local_dt must carry its UTC offset")
    if isinstance(boundary_hour, bool) or not isinstance(boundary_hour, int) or not 0 <= boundary_hour <= 23:
        raise ValueError("boundary_hour must be an integer in [0, 23]")
    return (local_dt - timedelta(hours=boundary_hour)).date().toordinal()

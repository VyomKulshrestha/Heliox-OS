"""Physics priors of the outcome model (spec section 7.3).

Every constant lives in ``PRIOR_TABLE_V1`` with its standard deviation (where the literature gives one) and its
reference. Changing any number bumps ``PRIOR_VERSION``. The priors are refined only by the pilot study, never by
simulated users.

Prediction mode evaluates a layout at its mean index of difficulty ``id_bar``, the mean ``n_item_eff`` over the menu's
items and the mean keyboard presses over its candidates. Update mode evaluates the realized target: its distance from
the layout's ``(origin_x, origin_y)``, its item's ``n_item_eff`` and, for keyboard trials, its realized presses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

from .special import norm_cdf, norm_ppf
from .state import LayoutSummary, Rect
from .vocabulary import ASSISTANCE_OFF, VARIANTS

PRIOR_VERSION = "literature-v1"

ProbitOutcome = Literal["complete", "ua", "correction", "veto"]


@dataclass(frozen=True, slots=True)
class PriorConstant:
    value: float
    sd: float | None  # None where the constant is a design choice rather than an estimate
    unit: str
    ref: str


_FITTS = "MacKenzie 1992; ISO 9241-411 (Shannon form)"
_KEYS = "Card, Moran and Newell 1983 (keystroke-level model), section 7.3"
_MENU = "Cockburn, Gutwin and Greenberg 2007 (search-decision-pointing)"
_ENDPOINT = "effective-width convention, MacKenzie 1992; section 7.3"
_DESIGN = "Heliox design value, section 7.3"


@dataclass(frozen=True, slots=True)
class PriorTable:
    fitts_a: PriorConstant
    fitts_b_mouse: PriorConstant
    fitts_b_touchpad: PriorConstant
    fitts_b_pen: PriorConstant
    fitts_b_touch: PriorConstant
    fitts_b_hand_pointer: PriorConstant
    reaction: PriorConstant
    key_base: PriorConstant
    key_press: PriorConstant
    key_search: PriorConstant
    menu_novice_base: PriorConstant
    menu_novice_per_item: PriorConstant
    menu_expert_base: PriorConstant
    menu_expert_per_bit: PriorConstant
    menu_expertise_scale: PriorConstant
    item_half_life: PriorConstant
    practice_amplitude: PriorConstant
    practice_exponent: PriorConstant
    effective_width: PriorConstant
    floor_mouse: PriorConstant
    floor_touchpad: PriorConstant
    floor_pen: PriorConstant
    floor_touch: PriorConstant
    floor_hand_pointer: PriorConstant
    r0_mouse: PriorConstant
    r0_touchpad: PriorConstant
    r0_pen: PriorConstant
    r0_touch: PriorConstant
    r0_hand_pointer: PriorConstant
    enlarge_right_pointer: PriorConstant
    enlarge_right_hand: PriorConstant
    enlarge_bad_min_fraction: PriorConstant
    l2_overhead: PriorConstant
    confirm_base: PriorConstant
    confirm_id: PriorConstant
    wait_id: PriorConstant
    hold_overhead: PriorConstant
    ask_search: PriorConstant
    act_episode: PriorConstant
    episode_time_floor: PriorConstant
    sigma_time: PriorConstant
    p_complete_plain: PriorConstant
    p_complete_confirm: PriorConstant
    p_complete_wait: PriorConstant
    p_veto_off: PriorConstant
    p_veto_act: PriorConstant
    veto_confirm_shift: PriorConstant
    correction_plain_add: PriorConstant
    ua_ambiguous_fraction: PriorConstant
    correction_ambiguous_fraction: PriorConstant
    confirm_ua_fraction: PriorConstant
    confirm_correction_fraction: PriorConstant
    l2_confirm_correction_add: PriorConstant
    wait_ua: PriorConstant
    wait_correction: PriorConstant
    ask_ua: PriorConstant
    keyboard_wrong: PriorConstant
    burden_effort_ipt: PriorConstant
    burden_effort_conf: PriorConstant
    burden_effort_enl: PriorConstant
    burden_intrusive_ipt: PriorConstant
    burden_intrusive_conf: PriorConstant
    burden_intrusive_enl: PriorConstant
    burden_control_base: PriorConstant
    burden_control_ipt: PriorConstant
    burden_control_conf: PriorConstant
    q_int_default: PriorConstant
    idle_alpha_pointer: PriorConstant
    idle_beta_pointer: PriorConstant
    idle_alpha_hand: PriorConstant
    idle_beta_hand: PriorConstant
    offer_alpha: PriorConstant
    offer_beta: PriorConstant
    null_prior: PriorConstant

    def entries(self) -> dict[str, PriorConstant]:
        """Every constant by field name."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _c(value: float, sd: float | None, unit: str, ref: str) -> PriorConstant:
    return PriorConstant(float(value), None if sd is None else float(sd), unit, ref)


PRIOR_TABLE_V1 = PriorTable(
    fitts_a=_c(100, 100, "ms", _FITTS),
    fitts_b_mouse=_c(230, 60, "ms/bit", _FITTS + "; about 4.3 bits/s, inside Soukoreff and MacKenzie 2004"),
    fitts_b_touchpad=_c(330, 100, "ms/bit", _FITTS),
    fitts_b_pen=_c(250, 80, "ms/bit", _FITTS),
    fitts_b_touch=_c(250, 80, "ms/bit", _FITTS),
    fitts_b_hand_pointer=_c(600, 250, "ms/bit", _FITTS + "; wide on purpose: no Heliox data exist"),
    reaction=_c(250, None, "ms", _KEYS),
    key_base=_c(300, None, "ms", _KEYS),
    key_press=_c(180, None, "ms/press", _KEYS),
    key_search=_c(150, None, "ms/bit", _KEYS + " (visual search the cue does not remove)"),
    menu_novice_base=_c(0.2, None, "s", _MENU),
    menu_novice_per_item=_c(0.1, None, "s/item", _MENU),
    menu_expert_base=_c(0.2, None, "s", _MENU),
    menu_expert_per_bit=_c(0.15, None, "s/bit", _MENU),
    menu_expertise_scale=_c(5, None, "exposures", _MENU),
    item_half_life=_c(7, None, "days", _MENU + "; section 8"),
    practice_amplitude=_c(0.3, None, "fraction", "Newell and Rosenbloom 1981 (power law of practice)"),
    practice_exponent=_c(0.4, None, "exponent", "Newell and Rosenbloom 1981 (power law of practice)"),
    effective_width=_c(4.133, None, "SD per width", _ENDPOINT),
    floor_mouse=_c(1.0, None, "px", _ENDPOINT),
    floor_touchpad=_c(1.5, None, "px", _ENDPOINT),
    floor_pen=_c(1.5, None, "px", _ENDPOINT),
    floor_touch=_c(3.0, None, "px", _ENDPOINT),
    floor_hand_pointer=_c(6.0, None, "px", _ENDPOINT),
    r0_mouse=_c(1.0, None, "factor", _ENDPOINT),
    r0_touchpad=_c(1.15, None, "factor", _ENDPOINT),
    r0_pen=_c(1.1, None, "factor", _ENDPOINT),
    r0_touch=_c(1.3, None, "factor", _ENDPOINT),
    r0_hand_pointer=_c(1.6, None, "factor", _ENDPOINT),
    enlarge_right_pointer=_c(0.85, None, "probability", "McGuffin and Balakrishnan 2002; " + _DESIGN),
    enlarge_right_hand=_c(0.70, None, "probability", "McGuffin and Balakrishnan 2002; " + _DESIGN),
    enlarge_bad_min_fraction=_c(0.25, None, "fraction of W", _DESIGN),
    l2_overhead=_c(700, None, "ms", _DESIGN),
    confirm_base=_c(250, None, "ms", _DESIGN + " (confirm chip delay)"),
    confirm_id=_c(2.0, None, "bits", _DESIGN),
    wait_id=_c(1.5, None, "bits", _DESIGN),
    hold_overhead=_c(150, None, "ms", _DESIGN),
    ask_search=_c(150, None, "ms/bit", _DESIGN),
    act_episode=_c(150, None, "ms", _DESIGN),
    episode_time_floor=_c(0.15, None, "s", "section 7.1"),
    sigma_time=_c(0.35, None, "log s", "section 7.1"),
    p_complete_plain=_c(0.97, None, "probability", _DESIGN),
    p_complete_confirm=_c(0.965, None, "probability", _DESIGN),
    p_complete_wait=_c(0.95, None, "probability", _DESIGN),
    p_veto_off=_c(0.01, None, "probability", _DESIGN),
    p_veto_act=_c(0.02, None, "probability", _DESIGN),
    veto_confirm_shift=_c(0.3, None, "probit units", _DESIGN),
    correction_plain_add=_c(0.02, None, "probability", _DESIGN),
    ua_ambiguous_fraction=_c(0.02, None, "fraction of P_amb", _DESIGN),
    correction_ambiguous_fraction=_c(0.3, None, "fraction of P_amb", _DESIGN),
    confirm_ua_fraction=_c(0.1, None, "fraction", _DESIGN),
    confirm_correction_fraction=_c(0.9, None, "fraction", _DESIGN),
    l2_confirm_correction_add=_c(0.05, None, "probability", _DESIGN),
    wait_ua=_c(0.02, None, "probability", _DESIGN),
    wait_correction=_c(0.05, None, "probability", _DESIGN),
    ask_ua=_c(0.01, None, "probability", _DESIGN),
    keyboard_wrong=_c(0.01, None, "probability", _DESIGN),
    burden_effort_ipt=_c(0.3, None, "latent/interruption", _DESIGN),
    burden_effort_conf=_c(0.2, None, "latent", _DESIGN),
    burden_effort_enl=_c(-0.1, None, "latent", _DESIGN),
    burden_intrusive_ipt=_c(0.5, None, "latent/interruption", _DESIGN),
    burden_intrusive_conf=_c(0.3, None, "latent", _DESIGN),
    burden_intrusive_enl=_c(0.1, None, "latent", _DESIGN),
    burden_control_base=_c(0.3, None, "latent", _DESIGN),
    burden_control_ipt=_c(-0.3, None, "latent/interruption", _DESIGN),
    burden_control_conf=_c(-0.1, None, "latent", _DESIGN),
    q_int_default=_c(0.5, None, "probability", "section 7.4"),
    idle_alpha_pointer=_c(0.5, None, "Gamma shape", _DESIGN),
    idle_beta_pointer=_c(10.0, None, "Gamma rate (min)", _DESIGN),
    idle_alpha_hand=_c(1.0, None, "Gamma shape", _DESIGN),
    idle_beta_hand=_c(4.0, None, "Gamma rate (min)", _DESIGN),
    offer_alpha=_c(1.0, None, "Beta a", _DESIGN),
    offer_beta=_c(4.0, None, "Beta b", _DESIGN),
    null_prior=_c(0.02, None, "probability", "section 6.1"),
)

POINTER_MODALITIES: tuple[str, ...] = ("mouse", "touchpad", "pen", "touch", "hand_pointer")
PROBIT_OUTCOMES: tuple[str, ...] = ("complete", "ua", "correction", "veto")
TRIAL_OUTCOMES: tuple[str, ...] = ("complete", "ua", "correction", "veto", "log_time")
BURDEN_OUTCOMES: tuple[str, ...] = ("effort", "intrusiveness", "control")
_P_CLIP = 1e-4


def _require_pointer(modality: str) -> str:
    if modality not in POINTER_MODALITIES:
        raise ValueError("modality has no endpoint scatter")
    return modality


def fitts_b(modality: str, table: PriorTable = PRIOR_TABLE_V1) -> float:
    return getattr(table, f"fitts_b_{_require_pointer(modality)}").value


def floor_px(modality: str, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """The endpoint floor of a pointer modality (spec 6.1); keyboard has none."""
    return getattr(table, f"floor_{_require_pointer(modality)}").value


def spread_r0(modality: str, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """The prior-mean spread factor of a pointer modality (spec 7.3); keyboard has none."""
    return getattr(table, f"r0_{_require_pointer(modality)}").value


def idle_prior(modality: str, table: PriorTable = PRIOR_TABLE_V1) -> tuple[float, float]:
    """(alpha0, beta0 per minute) of the idle activation rate; keyboard uses the pointer values."""
    if modality == "hand_pointer":
        return table.idle_alpha_hand.value, table.idle_beta_hand.value
    return table.idle_alpha_pointer.value, table.idle_beta_pointer.value


def probit(p: float) -> float:
    """``Phi^-1(clip(p, 1e-4, 1 - 1e-4))``: the probit latent of a prior probability."""
    return norm_ppf(min(max(p, _P_CLIP), 1.0 - _P_CLIP))


def fitts_time_ms(id_bits: float, modality: str, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """``T_point(ID, m) = a + b_m * ID`` in ms. Keyboard has no pointing movement: one key press (180 ms)."""
    if modality == "keyboard":
        return table.key_press.value
    return table.fitts_a.value + fitts_b(modality, table) * max(id_bits, 0.0)


def keyboard_menu_time_ms(presses: float, n: int, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """``T_key = 300 + 180 * presses + 150 * log2(n + 1)`` ms for a menu of n items."""
    return table.key_base.value + table.key_press.value * presses + table.key_search.value * math.log2(n + 1)


def keyboard_grid_time_ms(presses: float, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """``T_key = 300 + 180 * presses`` ms on a pointing grid (the cue marks the target: no search term)."""
    return table.key_base.value + table.key_press.value * presses


def _grid_cells(candidates: Sequence[Rect]) -> list[tuple[int, int]]:
    xs = sorted({round(c.cx, 6) for c in candidates})
    ys = sorted({round(c.cy, 6) for c in candidates})
    return [(xs.index(round(c.cx, 6)), ys.index(round(c.cy, 6))) for c in candidates]


def keyboard_presses(
    layout: LayoutSummary, target_index: int | None, *, candidates: Sequence[Rect] | None = None
) -> float:
    """Keyboard presses to select a target: ``moves + 1`` (spec 7.3).

    Menus: ``moves`` is the item index (focus starts on the first item), so the mean over a menu of n items is
    ``(n + 1) / 2``. Pointing grids: ``moves`` is the grid distance ``|dcolumn| + |drow|`` from the start candidate
    (the candidate nearest the layout's origin, the lower index on ties) to the target, read from ``candidates``.
    ``target_index`` None gives the mean over the layout's candidates. Without ``candidates`` (decision time, where
    only the layout summary is known) that mean is the summary's ``presses_mean``, computed by the surface from the
    same candidates and start rule; only a summary without it falls back to a ``ceil(sqrt(n))``-column grid filled row
    by row with the start candidate in the middle of the row nearest the origin.
    """
    if layout.task == "menu":
        n = layout.menu_n if layout.menu_n > 0 else layout.n_candidates
        if target_index is None:
            return (n + 1) / 2.0
        if not 0 <= target_index < n:
            raise ValueError("target_index outside the menu")
        return float(target_index + 1)
    if candidates:
        cells = _grid_cells(candidates)
        dists = [math.hypot(c.cx - layout.origin_x, c.cy - layout.origin_y) for c in candidates]
        start = min(range(len(candidates)), key=lambda i: (dists[i], i))
        sx, sy = cells[start]
        moves = [abs(cx - sx) + abs(cy - sy) for cx, cy in cells]
        if target_index is None:
            return sum(moves) / len(moves) + 1.0
        if not 0 <= target_index < len(candidates):
            raise ValueError("target_index outside the layout")
        return float(moves[target_index] + 1)
    if target_index is not None:
        raise ValueError("realized grid presses need the candidates")
    if layout.presses_mean is not None:
        return float(layout.presses_mean)
    n = max(layout.n_candidates, 1)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    sx, sy = (cols - 1) // 2, rows - 1
    moves = [abs(i % cols - sx) + abs(i // cols - sy) for i in range(n)]
    return sum(moves) / n + 1.0


def menu_sdp_time_s(n: int, n_item_eff: float, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """Search-decision time ``T_sd`` in seconds for a menu of n items at exposure ``n_item_eff``."""
    e = 1.0 - math.exp(-max(n_item_eff, 0.0) / table.menu_expertise_scale.value)
    novice = table.menu_novice_base.value + table.menu_novice_per_item.value * n
    expert = table.menu_expert_base.value + table.menu_expert_per_bit.value * math.log2(n + 1)
    return (1.0 - e) * novice + e * expert


def practice_multiplier(n_variant: float, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """``M_p = 1 + 0.3 * (n_variant + 1)^(-0.4)`` (Newell and Rosenbloom 1981)."""
    return 1.0 + table.practice_amplitude.value * (max(n_variant, 0.0) + 1.0) ** (-table.practice_exponent.value)


def endpoint_sd_px(w: float, r: float, floor: float, table: PriorTable = PRIOR_TABLE_V1) -> float:
    """Per-axis endpoint SD ``s = r * sqrt((W / 4.133)^2 + floor^2)`` (same form as the posterior, spec 6.1)."""
    return r * math.sqrt((w / table.effective_width.value) ** 2 + floor**2)


def p_miss(w: float, s: float) -> float:
    """One-axis miss probability ``2 * Phi(-W / (2 s))``."""
    return 2.0 * norm_cdf(-w / (2.0 * s))


@dataclass(frozen=True, slots=True)
class AmbiguityBand:
    x_star: float  # end of the confident-hit zone (px from the intended centre)
    p_amb: float  # A_gate > tau: no hit (gap) or a hit with P(hit | e) < 1 - tau
    p_unamb_wrg: float  # lands deep in a neighbour: confidently wrong hit
    p_wrong0: float  # assistance_off: any miss that lands on a neighbour


def ambiguity_band(w: float, gap: float, s: float, tau: float, *, d: float | None = None) -> AmbiguityBand:
    """Two-candidate Gaussian approximation along the approach axis (spec 7.3).

    ``w`` is the intended target's visible width, ``d`` the distance between the two candidate centres (default
    ``w + gap``) and ``s`` the endpoint SD. ``x* = clamp(d/2 + s^2 ln(tau / (1 - tau)) / d, 0, w/2)``.
    """
    if d is None:
        d = w + gap
    x_star = min(max(d / 2.0 + s * s * math.log(tau / (1.0 - tau)) / d, 0.0), w / 2.0)
    p_amb = 2.0 * (norm_cdf((d - x_star) / s) - norm_cdf(x_star / s))
    p_unamb = 2.0 * norm_cdf(-(d - x_star) / s)
    return AmbiguityBand(x_star=x_star, p_amb=max(p_amb, 0.0), p_unamb_wrg=p_unamb, p_wrong0=p_miss(w, s) * w / d)


def interruptions_per_trial(variant: str, *, p_amb: float, q_int: float | None = None) -> float:
    """Predicted interruptions per trial ``ipt(v)`` (spec 7.4); ``q_int`` defaults to 0.5 (no table yet)."""
    q = PRIOR_TABLE_V1.q_int_default.value if q_int is None else q_int
    family = _family(variant)
    if variant == ASSISTANCE_OFF:
        return 0.0
    if family in ("act", "enlarge_target"):
        return p_amb * q
    if family == "confirm_selection":
        return 1.0
    raise ValueError("not an L1 regime")


@dataclass(frozen=True, slots=True)
class PhysicsPrior:
    """Prior probabilities and time of one head for one variant and context, with the latent means ``h``."""

    p_complete: float
    p_ua: float
    p_correction: float
    p_veto: float
    time_ms: float  # includes the practice multiplier for the L1 head
    p_amb: float  # expected ambiguous share of activations (0 for L2 priors and keyboard)

    def h(self, outcome: str) -> float:
        """The fixed latent mean of an outcome: a probit for the binary ones, ``log seconds`` for ``log_time``."""
        if outcome == "log_time":
            return math.log(self.time_ms / 1000.0)
        if outcome == "complete":
            return probit(self.p_complete)
        if outcome == "ua":
            return probit(self.p_ua)
        if outcome == "correction":
            return probit(self.p_correction)
        if outcome == "veto":
            return probit(self.p_veto)
        raise ValueError("unknown outcome")


def _family(variant_id: str) -> str:
    v = VARIANTS[variant_id]
    return v.strategy.value if v.strategy is not None else v.id


def _confirm_veto(table: PriorTable) -> float:
    return norm_cdf(norm_ppf(table.p_veto_act.value) + table.veto_confirm_shift.value)


def enlarge_effective_width(
    w: float, scale: float, modality: str, table: PriorTable = PRIOR_TABLE_V1
) -> tuple[float, float, float]:
    """``(p_x, W_bad, W_eff)`` of an enlarge regime: the expanded candidate is the intended one with probability
    ``p_x``; when it is not, the intended target's visible width is ``W_bad = max(0.25 W, W (1 - (k - 1) / 2))``;
    ``W_eff`` is the ``p_x``-weighted mixture of ``k W`` and ``W_bad`` (spec 7.3)."""
    p_x = (table.enlarge_right_hand if modality == "hand_pointer" else table.enlarge_right_pointer).value
    w_bad = max(table.enlarge_bad_min_fraction.value * w, w * (1.0 - (scale - 1.0) / 2.0))
    return p_x, w_bad, p_x * scale * w + (1.0 - p_x) * w_bad


def _target_width(w_px: float, h_px: float) -> float:
    # The constraining dimension along the approach axis: W = H for pointing targets, the item height for menus.
    return min(w_px, h_px)


def l1_physics(
    variant: str,
    *,
    modality: str,
    task: str,
    w_px: float,
    h_px: float,
    gap_px: float,
    id_bits: float,
    menu_n: int,
    n_item_eff: float,
    presses: float | None,
    n_variant: float,
    r: float | None,
    tau: float,
    table: PriorTable = PRIOR_TABLE_V1,
) -> PhysicsPrior:
    """L1 trial-head prior for one layout (spec 7.3).

    ``id_bits`` is the realized target's ID on update and the layout's ``id_bar`` on prediction; ``n_item_eff`` the
    item's (update) or the mean over the menu's items (prediction); ``presses`` the realized or mean keyboard presses
    (keyboard only); ``r`` the spread factor of a pointer modality (None means the table's ``r0``). Enlarge regimes
    mix the right-expansion case (width ``k W``) and the wrong one (visible width ``W_bad``) with weight ``p_x``; the
    endpoint scatter stays planned for the original ``W`` and the centre distance ``d = W + gap`` is unchanged.
    """
    family = _family(variant)
    if family not in ("assistance_off", "act", "enlarge_target", "confirm_selection"):
        raise ValueError("not an L1 regime")
    w = _target_width(w_px, h_px)
    m_p = practice_multiplier(n_variant, table)
    react = table.reaction.value
    p_veto_confirm = _confirm_veto(table)
    hold_ms = VARIANTS[variant].param("hold_ms")

    if modality == "keyboard":
        if presses is None:
            raise ValueError("keyboard priors need presses")
        t_key = (
            keyboard_menu_time_ms(presses, menu_n, table) if task == "menu" else keyboard_grid_time_ms(presses, table)
        )
        wrong = table.keyboard_wrong.value
        band = AmbiguityBand(x_star=0.0, p_amb=0.0, p_unamb_wrg=wrong, p_wrong0=wrong)
        if hold_ms is not None:
            strategy = float(hold_ms) + table.hold_overhead.value
        elif family == "confirm_selection":
            strategy = table.confirm_base.value + table.key_press.value
        else:
            strategy = 0.0
        t = react + t_key + strategy
        return _l1_from_band(variant, family, band, t * m_p, table, p_veto_confirm)

    r_m = spread_r0(modality, table) if r is None else r
    s = endpoint_sd_px(w, r_m, floor_px(modality, table), table)
    d = w + gap_px
    if family == "enlarge_target":
        k = float(VARIANTS[variant].param("scale"))  # type: ignore[arg-type]
        p_x, w_bad, w_eff = enlarge_effective_width(w, k, modality, table)
        right = ambiguity_band(k * w, gap_px, s, tau, d=d)
        wrong = ambiguity_band(w_bad, gap_px, s, tau, d=d)
        band = AmbiguityBand(
            x_star=p_x * right.x_star + (1.0 - p_x) * wrong.x_star,
            p_amb=p_x * right.p_amb + (1.0 - p_x) * wrong.p_amb,
            p_unamb_wrg=p_x * right.p_unamb_wrg + (1.0 - p_x) * wrong.p_unamb_wrg,
            p_wrong0=p_x * right.p_wrong0 + (1.0 - p_x) * wrong.p_wrong0,
        )
    else:
        band = ambiguity_band(w, gap_px, s, tau, d=d)
        w_eff = w
    distance = w * (2.0 ** max(id_bits, 0.0) - 1.0)
    id_eff = math.log2(distance / w_eff + 1.0)
    t_point = fitts_time_ms(id_eff, modality, table)
    t_sd = menu_sdp_time_s(menu_n, n_item_eff, table) * 1000.0 if task == "menu" else 0.0
    if family in ("act", "enlarge_target"):
        strategy = band.p_amb * table.l2_overhead.value
    elif hold_ms is not None:
        strategy = float(hold_ms) + table.hold_overhead.value
    elif family == "confirm_selection":
        strategy = table.confirm_base.value + fitts_time_ms(table.confirm_id.value, modality, table)
    else:
        strategy = 0.0
    t = react + t_sd + t_point + strategy
    return _l1_from_band(variant, family, band, t * m_p, table, p_veto_confirm)


def _l1_from_band(
    variant: str, family: str, band: AmbiguityBand, time_ms: float, table: PriorTable, p_veto_confirm: float
) -> PhysicsPrior:
    if variant == ASSISTANCE_OFF:
        p_ua = band.p_wrong0
        return PhysicsPrior(
            p_complete=table.p_complete_plain.value,
            p_ua=p_ua,
            p_correction=p_ua + table.correction_plain_add.value,
            p_veto=table.p_veto_off.value,
            time_ms=time_ms,
            p_amb=0.0,
        )
    if family in ("act", "enlarge_target"):
        p_ua = band.p_unamb_wrg + table.ua_ambiguous_fraction.value * band.p_amb
        return PhysicsPrior(
            p_complete=table.p_complete_plain.value,
            p_ua=p_ua,
            p_correction=p_ua + table.correction_ambiguous_fraction.value * band.p_amb,
            p_veto=table.p_veto_act.value,
            time_ms=time_ms,
            p_amb=band.p_amb,
        )
    return PhysicsPrior(
        p_complete=table.p_complete_confirm.value,
        p_ua=table.confirm_ua_fraction.value * band.p_wrong0,
        p_correction=table.confirm_correction_fraction.value * band.p_wrong0 + table.correction_plain_add.value,
        p_veto=p_veto_confirm,
        time_ms=time_ms,
        p_amb=0.0,
    )


def target_id_bits(layout: LayoutSummary, target: Rect) -> float:
    """``log2(D / W + 1)`` of a realized target, D measured from the layout's origin to its cue-time centre."""
    distance = math.hypot(target.cx - layout.origin_x, target.cy - layout.origin_y)
    return math.log2(distance / _target_width(target.w, target.h) + 1.0)


def l1_physics_prediction(
    layout: LayoutSummary,
    variant: str,
    *,
    modality: str,
    n_item_eff_mean: float,
    n_variant: float,
    r: float | None,
    tau: float,
    table: PriorTable = PRIOR_TABLE_V1,
) -> PhysicsPrior:
    """Prediction mode for one planned layout: ``id_bar``, the mean ``n_item_eff`` and the mean presses."""
    return l1_physics(
        variant,
        modality=modality,
        task=layout.task,
        w_px=layout.w_px,
        h_px=layout.h_px,
        gap_px=layout.gap_px,
        id_bits=layout.id_bar,
        menu_n=layout.menu_n,
        n_item_eff=n_item_eff_mean,
        presses=keyboard_presses(layout, None) if modality == "keyboard" else None,
        n_variant=n_variant,
        r=r,
        tau=tau,
        table=table,
    )


def l1_physics_update(
    layout: LayoutSummary,
    candidates: Sequence[Rect],
    target_index: int,
    variant: str,
    *,
    modality: str,
    n_item_eff: float,
    n_variant: float,
    r: float | None,
    tau: float,
    table: PriorTable = PRIOR_TABLE_V1,
) -> PhysicsPrior:
    """Update mode: the realized target (its distance from the layout origin, its item exposure, its presses)."""
    target = candidates[target_index]
    return l1_physics(
        variant,
        modality=modality,
        task=layout.task,
        w_px=target.w,
        h_px=target.h,
        gap_px=layout.gap_px,
        id_bits=target_id_bits(layout, target),
        menu_n=layout.menu_n,
        n_item_eff=n_item_eff,
        presses=keyboard_presses(layout, target_index, candidates=candidates) if modality == "keyboard" else None,
        n_variant=n_variant,
        r=r,
        tau=tau,
        table=table,
    )


def topk_mass(p_top: float, n_plausible: int, k: int) -> float:
    """Approximate posterior mass of the top-k candidates from an activation context alone.

    The other plausible candidates share ``1 - p_top`` with the implausible ones and the null term:
    each gets ``min(p_top, (1 - p_top) / n_plausible)``.
    """
    n = max(int(n_plausible), 1)
    other = min(p_top, (1.0 - p_top) / n)
    return min(1.0, p_top + (min(k, n) - 1) * other)


def l2_physics(
    variant: str,
    *,
    modality: str,
    p_hit: float,
    p_top: float,
    n_plausible: int,
    table: PriorTable = PRIOR_TABLE_V1,
) -> PhysicsPrior:
    """L2 episode-head prior for a response in an activation context (spec 7.3)."""
    family = _family(variant)
    p_veto_confirm = _confirm_veto(table)
    base = table.confirm_base.value
    t_confirm = fitts_time_ms(table.confirm_id.value, modality, table)
    if variant == "act":
        return PhysicsPrior(
            p_complete=table.p_complete_plain.value,
            p_ua=1.0 - p_hit,
            p_correction=1.0 - p_hit,
            p_veto=table.p_veto_act.value,
            time_ms=table.act_episode.value,
            p_amb=0.0,
        )
    if variant == "confirm_selection:second_click":
        return PhysicsPrior(
            p_complete=table.p_complete_confirm.value,
            p_ua=table.confirm_ua_fraction.value * (1.0 - p_hit),
            p_correction=table.confirm_correction_fraction.value * (1.0 - p_hit)
            + table.l2_confirm_correction_add.value,
            p_veto=p_veto_confirm,
            time_ms=base + t_confirm,
            p_amb=0.0,
        )
    if family == "wait_for_evidence":
        return PhysicsPrior(
            p_complete=table.p_complete_wait.value,
            p_ua=table.wait_ua.value,
            p_correction=table.wait_correction.value,
            p_veto=p_veto_confirm,
            time_ms=base + fitts_time_ms(table.wait_id.value, modality, table),
            p_amb=0.0,
        )
    if family == "ask_clarification":
        k = int(VARIANTS[variant].param("k"))  # type: ignore[arg-type]
        return PhysicsPrior(
            p_complete=table.p_complete_plain.value,
            p_ua=table.ask_ua.value,
            p_correction=1.0 - topk_mass(p_top, n_plausible, k),
            p_veto=p_veto_confirm,
            time_ms=base + table.ask_search.value * math.log2(k + 2) + t_confirm,
            p_amb=0.0,
        )
    raise ValueError("not an L2 response")


def burden_physics(variant: str, *, ipt: float, table: PriorTable = PRIOR_TABLE_V1) -> dict[str, float]:
    """Burden-head latent means (ordinal probit) for a block regime at ``ipt`` interruptions per trial."""
    family = _family(variant)
    g_conf = 1.0 if family == "confirm_selection" else 0.0
    g_enl = 1.0 if family == "enlarge_target" else 0.0
    i = min(max(ipt, 0.0), 3.0)
    return {
        "effort": table.burden_effort_ipt.value * i
        + table.burden_effort_conf.value * g_conf
        + table.burden_effort_enl.value * g_enl,
        "intrusiveness": table.burden_intrusive_ipt.value * i
        + table.burden_intrusive_conf.value * g_conf
        + table.burden_intrusive_enl.value * g_enl,
        "control": table.burden_control_base.value
        + table.burden_control_ipt.value * i
        + table.burden_control_conf.value * g_conf,
    }

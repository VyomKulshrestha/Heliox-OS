"""Physics priors of the outcome model (spec section 7.3)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pilot.coadaptation.core.priors import (
    PRIOR_TABLE_V1,
    PRIOR_VERSION,
    ambiguity_band,
    burden_physics,
    endpoint_sd_px,
    enlarge_effective_width,
    fitts_time_ms,
    floor_px,
    keyboard_grid_time_ms,
    keyboard_menu_time_ms,
    keyboard_presses,
    l1_physics,
    l1_physics_prediction,
    l1_physics_update,
    l2_physics,
    menu_sdp_time_s,
    p_miss,
    practice_multiplier,
    spread_r0,
)
from pilot.coadaptation.core.special import norm_cdf, norm_ppf
from pilot.coadaptation.core.state import LayoutSummary, Rect
from pilot.coadaptation.core.study_plan import LAYOUT_GRID


def _pointing(w: float = 28.0, gap: float = 8.0, id_bits: float = 3.0) -> LayoutSummary:
    return LayoutSummary("pointing", 12, w, w, gap, id_bits, 0, 350.0, 470.0)


def _menu(n: int) -> LayoutSummary:
    return LayoutSummary("menu", n, 220.0, 24.0, 0.0, 2.5, n, 350.0, 40.0)


def _l1(variant: str, **kw: object) -> object:
    args: dict[str, object] = {
        "modality": "mouse",
        "task": "pointing",
        "w_px": 28.0,
        "h_px": 28.0,
        "gap_px": 8.0,
        "id_bits": 3.0,
        "menu_n": 0,
        "n_item_eff": 0.0,
        "presses": None,
        "n_variant": 0,
        "r": None,
        "tau": 0.25,
    }
    args.update(kw)
    return l1_physics(variant, **args)  # type: ignore[arg-type]


def test_table_entries_have_values_and_references() -> None:
    assert PRIOR_VERSION == "literature-v1"
    entries = PRIOR_TABLE_V1.entries()
    assert entries["fitts_a"].value == 100 and entries["fitts_a"].sd == 100
    assert entries["fitts_b_mouse"].value == 230 and entries["fitts_b_mouse"].sd == 60
    assert entries["fitts_b_hand_pointer"].value == 600 and entries["fitts_b_hand_pointer"].sd == 250
    for name, constant in entries.items():
        assert math.isfinite(constant.value), name
        assert constant.ref, name
        assert constant.unit, name


def test_no_menu_size_constant_outside_the_layout_grid() -> None:
    menu_sizes = set(LAYOUT_GRID["menu"]["n_items"])
    assert menu_sizes == {6, 12, 16}
    size_constants = [c for c in PRIOR_TABLE_V1.entries().values() if c.unit in ("items", "menu items")]
    assert all(c.value in menu_sizes for c in size_constants)
    assert size_constants == []  # the menu terms are evaluated at the realized or planned N


def test_fitts_time_rises_with_distance_and_falls_with_width() -> None:
    times_d = [_l1("assistance_off", id_bits=math.log2(d / 28.0 + 1)).time_ms for d in (120, 240, 360)]
    assert times_d == sorted(times_d) and times_d[0] < times_d[-1]
    times_w = [
        _l1("assistance_off", w_px=w, h_px=w, id_bits=math.log2(240 / w + 1)).time_ms for w in (16.0, 28.0, 44.0)
    ]
    assert times_w == sorted(times_w, reverse=True) and times_w[0] > times_w[-1]
    assert fitts_time_ms(4.0, "mouse") == pytest.approx(100 + 230 * 4.0)
    assert fitts_time_ms(4.0, "hand_pointer") > fitts_time_ms(4.0, "touchpad") > fitts_time_ms(4.0, "mouse")


def test_miss_probability_at_nominal_width() -> None:
    for w in (16.0, 28.0, 44.0):
        s = endpoint_sd_px(w, 1.0, 0.0)
        assert p_miss(w, s) == pytest.approx(0.0388, abs=0.001)
    assert floor_px("hand_pointer") == 6.0 and spread_r0("hand_pointer") == 1.6
    with pytest.raises(ValueError):
        floor_px("keyboard")


def test_enlarge_lowers_predicted_time_and_miss() -> None:
    for modality in ("mouse", "hand_pointer"):
        for w in (16.0, 28.0, 44.0):
            s = endpoint_sd_px(w, spread_r0(modality), floor_px(modality))
            for scale, variant in ((1.5, "enlarge_target:1.5"), (2.0, "enlarge_target:2.0")):
                _, _, w_eff = enlarge_effective_width(w, scale, modality)
                assert w_eff > w
                assert p_miss(w_eff, s) < p_miss(w, s)
                act = _l1("act", modality=modality, w_px=w, h_px=w)
                enl = _l1(variant, modality=modality, w_px=w, h_px=w)
                assert enl.time_ms < act.time_ms


def test_menu_hick_term_and_expertise() -> None:
    novice = [menu_sdp_time_s(n, 0.0) for n in (6, 12, 16)]
    expert = [menu_sdp_time_s(n, 1e6) for n in (6, 12, 16)]
    assert novice == sorted(novice) and expert == sorted(expert)
    assert novice == pytest.approx([0.8, 1.4, 1.8], abs=0.005)
    assert expert == pytest.approx([0.62, 0.76, 0.81], abs=0.005)
    for n in (6, 12, 16):
        series = [menu_sdp_time_s(n, k) for k in (0.0, 1.0, 3.0, 10.0, 30.0)]
        assert series == sorted(series, reverse=True)
        expert_n = 0.2 + 0.15 * math.log2(n + 1)
        assert abs(series[-1] - expert_n) < abs(series[0] - expert_n)


def test_practice_multiplier_decreases() -> None:
    values = [practice_multiplier(n) for n in (0, 1, 5, 20, 100, 1000)]
    assert values[0] == pytest.approx(1.3)
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] > 1.0


@pytest.mark.parametrize(
    ("w", "gap", "r", "floor", "tau"),
    [
        (16.0, 2.0, 1.0, 1.0, 0.25),
        (28.0, 8.0, 1.0, 1.0, 0.25),
        (28.0, 2.0, 1.2, 1.5, 0.15),
        (44.0, 8.0, 1.6, 6.0, 0.25),
    ],
)
def test_ambiguity_band_matches_two_candidate_monte_carlo(
    w: float, gap: float, r: float, floor: float, tau: float
) -> None:
    s = endpoint_sd_px(w, r, floor)
    d = w + gap
    band = ambiguity_band(w, gap, s, tau)
    rng = np.random.Generator(np.random.PCG64(20261015))
    x = rng.normal(0.0, s, 400_000)
    side = np.where(x >= 0, 1.0, -1.0)
    neighbour = side * d
    log_ratio = (x * x - (x - neighbour) ** 2) / (2 * s * s)  # log L_n - log L_i
    p_intended = 1.0 / (1.0 + np.exp(log_ratio))
    hit_intended = np.abs(x) <= w / 2
    hit_neighbour = np.abs(x - neighbour) <= w / 2
    a = np.where(hit_intended, 1.0 - p_intended, np.where(hit_neighbour, p_intended, 1.0))
    ambiguous = a > tau
    confident_wrong = hit_neighbour & ~ambiguous
    assert band.p_amb == pytest.approx(float(ambiguous.mean()), abs=0.01)
    assert band.p_unamb_wrg == pytest.approx(float(confident_wrong.mean()), abs=0.01)
    assert band.p_wrong0 == pytest.approx(p_miss(w, s) * w / d)


def test_l1_probability_table() -> None:
    off = _l1("assistance_off")
    act = _l1("act")
    conf = _l1("confirm_selection:second_click")
    s = endpoint_sd_px(28.0, 1.0, 1.0)
    band = ambiguity_band(28.0, 8.0, s, 0.25)
    assert off.p_ua == pytest.approx(band.p_wrong0)
    assert off.p_correction == pytest.approx(band.p_wrong0 + 0.02)
    assert (off.p_complete, off.p_veto) == (0.97, 0.01)
    assert act.p_ua == pytest.approx(band.p_unamb_wrg + 0.02 * band.p_amb)
    assert act.p_correction == pytest.approx(act.p_ua + 0.3 * band.p_amb)
    assert (act.p_complete, act.p_veto) == (0.97, 0.02)
    assert conf.p_ua == pytest.approx(0.1 * band.p_wrong0)
    assert conf.p_correction == pytest.approx(0.9 * band.p_wrong0 + 0.02)
    assert conf.p_complete == 0.965
    assert conf.p_veto == pytest.approx(norm_cdf(norm_ppf(0.02) + 0.3))
    assert act.h("ua") < 0 and off.h("log_time") == pytest.approx(math.log(off.time_ms / 1000.0))
    assert act.time_ms == pytest.approx(off.time_ms + band.p_amb * 700 * 1.3)


def test_keyboard_presses() -> None:
    menu = _menu(12)
    assert [keyboard_presses(menu, i) for i in range(3)] == [1.0, 2.0, 3.0]
    assert keyboard_presses(menu, None) == pytest.approx(6.5)
    # A 4 x 3 grid whose origin is below the bottom row: the start candidate is the bottom-row candidate nearest it.
    grid = [Rect(cx=300.0 + 30 * (i % 4), cy=200.0 + 30 * (i // 4), w=28.0, h=28.0) for i in range(12)]
    layout = LayoutSummary("pointing", 12, 28.0, 28.0, 2.0, 3.0, 0, 340.0, 400.0)
    start = 9  # (330, 260) and (360, 260) tie at 10 px horizontally from x = 340: the lower index wins
    assert keyboard_presses(layout, start, candidates=grid) == 1.0
    assert keyboard_presses(layout, 0, candidates=grid) == 1.0 + 1 + 2  # one column left, two rows up
    assert keyboard_presses(layout, 11, candidates=grid) == 1.0 + 2
    distances = [abs(i % 4 - 1) + abs(i // 4 - 2) for i in range(12)]
    assert keyboard_presses(layout, None, candidates=grid) == pytest.approx(np.mean(distances) + 1.0)
    with pytest.raises(ValueError):
        keyboard_presses(layout, 3)  # realized grid presses need the candidates


def test_decision_time_grid_presses_come_from_the_layout_summary() -> None:
    """Prediction mode averages over the layout's own candidates (spec 7.3). A 6 x 2 grid left of and above the home
    button starts at its bottom-right candidate; a centred square grid would put the start in the middle and predict
    a full press fewer than update mode averages over the same targets."""
    grid = [Rect(cx=100.0 + 30 * (i % 6), cy=300.0 + 30 * (i // 6), w=28.0, h=28.0) for i in range(12)]
    bare = LayoutSummary("pointing", 12, 28.0, 28.0, 2.0, 3.0, 0, 400.0, 400.0)
    assert [keyboard_presses(bare, i, candidates=grid) for i in (11, 0)] == [1.0, 7.0]  # start: bottom-right
    real = keyboard_presses(bare, None, candidates=grid)
    assert real == pytest.approx(4.0)
    assert keyboard_presses(bare, None) == pytest.approx(3.0)  # the approximation, only without the summary's mean
    summary = replace(bare, presses_mean=real)
    assert keyboard_presses(summary, None) == real
    kw = {"modality": "keyboard", "n_variant": 0, "r": None, "tau": 0.25}
    for variant in ("assistance_off", "act", "confirm_selection:second_click", "confirm_selection:hold_700"):
        predicted = l1_physics_prediction(summary, variant, n_item_eff_mean=0.0, **kw)  # type: ignore[arg-type]
        updates = [
            l1_physics_update(summary, grid, i, variant, n_item_eff=0.0, **kw).time_ms  # type: ignore[arg-type]
            for i in range(12)
        ]
        assert predicted.time_ms == pytest.approx(float(np.mean(updates)))
    # the mean travels with the summary; a value outside [1, 64] is refused
    assert LayoutSummary.from_dict(summary.to_dict()) == summary
    assert "presses_mean" not in bare.to_dict() and LayoutSummary.from_dict(bare.to_dict()) == bare
    for bad in (0.5, 65.0, float("nan"), True):
        with pytest.raises(ValueError):
            replace(bare, presses_mean=bad)


def test_keyboard_trials_use_key_time() -> None:
    menu = _menu(12)
    prior = l1_physics_prediction(
        menu, "assistance_off", modality="keyboard", n_item_eff_mean=0.0, n_variant=0, r=None, tau=0.25
    )
    expected = (250 + keyboard_menu_time_ms(6.5, 12)) * 1.3
    assert prior.time_ms == pytest.approx(expected)
    grid = [Rect(cx=300.0 + 30 * (i % 4), cy=200.0 + 30 * (i // 4), w=28.0, h=28.0) for i in range(12)]
    layout = LayoutSummary("pointing", 12, 28.0, 28.0, 2.0, 3.0, 0, 340.0, 400.0)
    upd = l1_physics_update(layout, grid, 0, "act", modality="keyboard", n_item_eff=0.0, n_variant=0, r=None, tau=0.25)
    assert upd.time_ms == pytest.approx((250 + keyboard_grid_time_ms(4.0)) * 1.3)
    assert upd.p_amb == 0.0 and upd.p_ua == pytest.approx(0.01)
    off = l1_physics_update(
        layout, grid, 0, "assistance_off", modality="keyboard", n_item_eff=0.0, n_variant=0, r=None, tau=0.25
    )
    second = l1_physics_update(
        layout,
        grid,
        0,
        "confirm_selection:second_click",
        modality="keyboard",
        n_item_eff=0.0,
        n_variant=0,
        r=None,
        tau=0.25,
    )
    assert second.time_ms - off.time_ms == pytest.approx((250 + 180) * 1.3)
    assert off.p_ua == pytest.approx(0.01)


def test_mean_keyboard_menu_time_at_study_sizes() -> None:
    times = [keyboard_menu_time_ms((n + 1) / 2, n) / 1000.0 for n in (6, 12, 16)]
    assert times == pytest.approx([1.35, 2.03, 2.44], abs=0.005)


def test_update_mode_uses_the_realized_target() -> None:
    layout = _pointing(28.0, 8.0, id_bits=3.0)
    near = Rect(cx=350.0, cy=350.0, w=28.0, h=28.0)  # 120 px from the origin
    far = Rect(cx=350.0, cy=110.0, w=28.0, h=28.0)  # 360 px
    kw = {"modality": "mouse", "n_item_eff": 0.0, "n_variant": 0, "r": None, "tau": 0.25}
    t_near = l1_physics_update(layout, [near, far], 0, "assistance_off", **kw).time_ms  # type: ignore[arg-type]
    t_far = l1_physics_update(layout, [near, far], 1, "assistance_off", **kw).time_ms  # type: ignore[arg-type]
    assert t_near < t_far
    expected = (250 + 100 + 230 * math.log2(120 / 28 + 1)) * 1.3
    assert t_near == pytest.approx(expected)


def test_l2_priors() -> None:
    kw = {"modality": "mouse", "p_hit": 0.6, "p_top": 0.6, "n_plausible": 2}
    act = l2_physics("act", **kw)  # type: ignore[arg-type]
    assert act.p_ua == pytest.approx(0.4) and act.time_ms == 150
    second = l2_physics("confirm_selection:second_click", **kw)  # type: ignore[arg-type]
    assert second.p_ua == pytest.approx(0.04) and second.p_correction == pytest.approx(0.41)
    assert second.time_ms == pytest.approx(250 + 100 + 230 * 2.0)
    wait = l2_physics("wait_for_evidence", **kw)  # type: ignore[arg-type]
    assert (wait.p_ua, wait.p_correction, wait.p_complete) == (0.02, 0.05, 0.95)
    assert wait.time_ms == pytest.approx(250 + 100 + 230 * 1.5)
    ask2 = l2_physics("ask_clarification:2", **kw)  # type: ignore[arg-type]
    ask3 = l2_physics("ask_clarification:3", modality="mouse", p_hit=0.6, p_top=0.6, n_plausible=3)
    assert ask2.p_ua == 0.01 and 0.0 <= ask2.p_correction < 1.0 - 0.6
    assert ask3.p_correction <= ask2.p_correction
    assert ask2.time_ms == pytest.approx(250 + 150 * 2.0 + 100 + 230 * 2.0)


def test_burden_priors() -> None:
    off = burden_physics("assistance_off", ipt=0.0)
    conf = burden_physics("confirm_selection:hold_700", ipt=1.0)
    enl = burden_physics("enlarge_target:1.5", ipt=0.2)
    assert off == {"effort": 0.0, "intrusiveness": 0.0, "control": pytest.approx(0.3)}
    assert conf["effort"] == pytest.approx(0.5) and conf["intrusiveness"] == pytest.approx(0.8)
    assert conf["control"] == pytest.approx(-0.1)
    assert enl["effort"] == pytest.approx(0.3 * 0.2 - 0.1)

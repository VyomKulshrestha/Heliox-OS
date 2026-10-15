"""Target posterior and ambiguity (spec section 6) and the parity fixture shared with the TypeScript twin."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot.coadaptation.core.state import Rect
from pilot.coadaptation.core.target_posterior import (
    FIXTURE_CASES,
    GazeInput,
    ambiguity_bin,
    cell_id,
    compute_posterior,
    keyboard_endpoint,
    rendered_rects,
)

DAEMON = Path(__file__).resolve().parents[1]
FIXTURE = DAEMON / "tests" / "fixtures" / "coadaptation_target_posterior.json"
SURFACE = (700.0, 500.0)


def _grid(cols: int = 4, rows: int = 3, w: float = 16.0, gap: float = 2.0, centre=(350.0, 200.0)) -> list[Rect]:
    pitch = w + gap
    x0 = centre[0] - (cols - 1) * pitch / 2
    y0 = centre[1] - (rows - 1) * pitch / 2
    return [Rect(x0 + c * pitch, y0 + r * pitch, w, w) for r in range(rows) for c in range(cols)]


def _post(cands, endpoint, **kw):
    args = {"modality": "mouse", "r": 1.0, "floor": 1.0, "surface": SURFACE}
    args.update(kw)
    return compute_posterior(cands, endpoint, **args)


def test_posterior_sums_to_one_with_the_null_term() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    for _ in range(200):
        cands = _grid(w=float(rng.choice([16.0, 28.0, 44.0])))
        x, y = int(rng.integers(0, 700)), int(rng.integers(0, 500))
        res = _post(cands, (x, y), r=float(rng.uniform(0.8, 2.0)))
        assert math.fsum(res.posterior) + res.p_null == pytest.approx(1.0, abs=1e-12)
        assert 1 <= res.n_plausible <= 3
        assert res.ambiguity_gate >= res.ambiguity


def test_null_dominates_far_endpoints_and_far_clicks_get_a_valid_cell() -> None:
    cands = _grid()
    res = _post(cands, (60, 470))
    assert res.p_null > 0.99
    assert all(p < 0.15 for p in res.posterior)
    assert res.n_plausible == 1
    assert not res.has_hit and res.ambiguity == 1.0 and res.ambiguity_gate == 1.0
    assert cell_id(ambiguity_bin(res.ambiguity_gate, 0.25), res.n_plausible, res.has_hit) == "b3:p1:h0"
    assert math.isfinite(res.p_top) and res.p_hit == 0.0


def test_a_hit_rises_with_distance_from_the_hit_centre() -> None:
    cands = _grid(w=28.0, gap=2.0)
    target = cands[5]
    values = [_post(cands, (int(target.cx + dx), int(target.cy))).ambiguity for dx in range(0, 15)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[0] < 0.01 < values[-1]


def test_no_literal_hit_gives_ambiguity_one() -> None:
    cands = _grid(w=16.0, gap=8.0)
    gap_point = (int(cands[0].cx + 12), int(cands[0].cy))  # inside the 8 px gap
    assert _post(cands, gap_point).ambiguity == 1.0
    release = (int(cands[1].cx), int(cands[1].cy))
    free = _post(cands, release)
    assert free.has_hit and free.ambiguity < 0.01
    other = _post(cands, release, press_index=0)  # pressed on candidate 0, released on candidate 1
    assert not other.has_hit and other.ambiguity == 1.0 and other.ambiguity_gate == 1.0
    on_gap = _post(cands, release, press_index=-1)  # pressed on a gap, released on candidate 1
    assert not on_gap.has_hit and on_gap.hit_index is None and on_gap.ambiguity == 1.0
    assert on_gap.posterior == free.posterior  # the posterior itself does not depend on the press
    same = _post(cands, release, press_index=1)
    assert same.has_hit and same.ambiguity == free.ambiguity


def test_wrongly_expanded_neighbour_is_ambiguous() -> None:
    row = [Rect(305.0 + 30 * i, 250.0, 28.0, 28.0) for i in range(4)]
    rendered = rendered_rects(row, 1, 1.5)
    res = _post(rendered, (355, 250))
    assert res.hit_index == 1
    assert res.ambiguity == pytest.approx(0.84, abs=0.01)
    assert res.ambiguity > 0.25
    assert 1.0 - max(res.posterior) <= 0.25  # the max-posterior form would have called it unambiguous


def test_gaze_weight_is_capped_and_never_lowers_the_gate() -> None:
    # Two candidates equidistant from the endpoint, one in the left third and one in the centre third.
    cands = [Rect(220.0, 250.0, 28.0, 28.0), Rect(250.0, 250.0, 28.0, 28.0)]
    endpoint = (235, 250)
    for acc, conf in ((1.0, 1.0), (1.5, 1.0), (0.8, 0.5)):
        res = _post(cands, endpoint, gaze=GazeInput("left", conf, acc), press_index=None)
        kappa = min(2.0, 1.0 + acc * conf)
        assert res.posterior[0] / res.posterior[1] == pytest.approx(kappa, rel=1e-12)
        assert res.posterior_no_gaze[0] == pytest.approx(res.posterior_no_gaze[1])
    rng = np.random.Generator(np.random.PCG64(7))
    regions = ("center", "left", "right", "up", "down")
    for _ in range(300):
        cands = _grid(w=float(rng.choice([16.0, 28.0])), centre=(float(rng.integers(150, 550)), 250.0))
        target = cands[int(rng.integers(0, 12))]
        endpoint = (int(target.cx + rng.normal(0, 6)), int(target.cy + rng.normal(0, 6)))
        gaze = GazeInput(str(rng.choice(regions)), float(rng.uniform(0, 1)), float(rng.uniform(0.6, 1)))
        with_gaze = _post(cands, endpoint, gaze=gaze)
        without = _post(cands, endpoint)
        assert with_gaze.ambiguity == without.ambiguity
        assert with_gaze.ambiguity_gate >= without.ambiguity_gate


def test_keyboard_result_is_fixed() -> None:
    cands = _grid()
    for endpoint in ((0, 0), (999, 999), keyboard_endpoint(cands[3])):
        res = compute_posterior(
            cands,
            endpoint,
            modality="keyboard",
            r=None,
            floor=None,
            surface=SURFACE,
            gaze=GazeInput("left", 1.0, 1.0),
            press_index=3,
        )
        assert res.posterior == tuple(1.0 if i == 3 else 0.0 for i in range(12))
        assert (res.ambiguity, res.ambiguity_gate, res.n_plausible) == (0.0, 0.0, 1)
        assert res.has_hit and res.hit_index == 3 and res.p_hit == 1.0 and res.p_top == 1.0 and res.p_null == 0.0
    for bad in (None, -1, 12):
        with pytest.raises(ValueError):
            compute_posterior(cands, (0, 0), modality="keyboard", r=None, floor=None, surface=SURFACE, press_index=bad)
    with pytest.raises(ValueError):
        compute_posterior(cands, (0, 0), modality="mouse", r=None, floor=None, surface=SURFACE)


def test_keyboard_endpoint_rounds_half_up() -> None:
    assert keyboard_endpoint(Rect(100.5, 200.5, 28.0, 28.0)) == (101, 201)
    assert keyboard_endpoint(Rect(100.49, 200.0, 28.0, 28.0)) == (100, 200)
    assert keyboard_endpoint(Rect(-0.5, 3.5, 28.0, 28.0)) == (0, 4)


def test_rendered_rects_scale_only_the_expanded_candidate() -> None:
    cands = _grid()
    out = rendered_rects(cands, 4, 2.0)
    for i, (a, b) in enumerate(zip(cands, out)):
        if i == 4:
            assert (b.cx, b.cy, b.w, b.h) == (a.cx, a.cy, a.w * 2.0, a.h * 2.0)
        else:
            assert b == a
    assert rendered_rects(cands, None, 1.5) == tuple(cands)
    with pytest.raises(ValueError):
        rendered_rects(cands, 12, 1.5)


def test_bins_and_cells() -> None:
    assert ambiguity_bin(0.25, 0.25) is None
    assert ambiguity_bin(0.3, 0.25) == "b1"
    assert ambiguity_bin(0.4, 0.25) == "b1"
    assert ambiguity_bin(0.41, 0.25) == "b2"
    assert ambiguity_bin(0.61, 0.25) == "b3"
    assert ambiguity_bin(0.12, 0.1) == "b1"
    assert cell_id("b2", 3, True) == "b2:p3:h1"
    with pytest.raises(ValueError):
        cell_id("b2", 0, True)


def test_fixture_cli_reproduces_the_committed_fixture(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "posterior.json"
    subprocess.run(
        [sys.executable, "-m", "pilot.coadaptation.core.target_posterior", "--write-fixture", str(out)],
        cwd=DAEMON,
        check=True,
        capture_output=True,
    )
    # Compared as parsed JSON: line endings change with core.autocrlf, and the last float digits change with the
    # platform's math library. Same structure and values to 1e-9 relative is the parity contract.
    _assert_json_close(json.loads(out.read_text(encoding="utf-8")), json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_fixture_contents() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert data["seed"] == 20261015 and len(cases) == FIXTURE_CASES == 44
    keyboard = [c for c in cases if c["modality"] == "keyboard"]
    assert len(keyboard) == 4
    assert sum(1 for c in keyboard if any(r["cx"] % 1 == 0.5 for r in c["candidates"])) >= 2
    for case in keyboard:
        assert case["r"] is None and case["floor"] is None and case["gaze"] is None
        focus = case["press_index"]
        assert (case["endpoint"]["x"], case["endpoint"]["y"]) == keyboard_endpoint(Rect(**case["candidates"][focus]))
        assert case["expected"]["ambiguity_gate"] == 0.0 and case["expected"]["n_plausible"] == 1
    assert any(c["expanded_index"] is not None and c["scale"] in (1.5, 2.0) for c in cases)
    assert any(not c["expected"]["has_hit"] and c["expected"]["ambiguity"] == 1.0 for c in cases)
    assert any(c["press_index"] == -1 and not c["expected"]["has_hit"] for c in cases)
    assert any(all(p < 0.15 for p in c["expected"]["posterior"]) and c["expected"]["n_plausible"] == 1 for c in cases)
    assert any(c["gaze"] is not None for c in cases)
    b37 = cases[0]
    assert b37["expanded_index"] == 1 and b37["scale"] == 1.5 and b37["r"] == 1.0
    assert b37["expected"]["ambiguity"] == pytest.approx(0.84, abs=0.01)
    for case in cases:
        assert isinstance(case["endpoint"]["x"], int) and isinstance(case["endpoint"]["y"], int)
        cands = [Rect(**r) for r in case["candidates"]]
        rendered = rendered_rects(cands, case["expanded_index"], case["scale"])
        assert [r.to_dict() for r in rendered] == case["expected"]["rendered"]
        gaze = case["gaze"]
        res = compute_posterior(
            rendered,
            (case["endpoint"]["x"], case["endpoint"]["y"]),
            modality=case["modality"],
            r=case["r"],
            floor=case["floor"],
            surface=(case["surface"]["w"], case["surface"]["h"]),
            gaze=None if gaze is None else GazeInput(gaze["region"], gaze["conf"], gaze["acc"]),
            press_index=case["press_index"],
        )
        exp = case["expected"]
        for a, b in zip(res.posterior, exp["posterior"]):
            assert abs(a - b) <= 1e-12 + 1e-9 * abs(b)
        assert (res.n_plausible, res.has_hit, res.hit_index) == (exp["n_plausible"], exp["has_hit"], exp["hit_index"])
        assert abs(res.ambiguity_gate - exp["ambiguity_gate"]) <= 1e-12


def _assert_json_close(actual: object, expected: object, path: str = "$") -> None:
    """Equal JSON, with floats compared to 1e-9 relative: the last digits differ between platforms' math libraries."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (path, actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_close(a, e, f"{path}[{index}]")
    else:
        assert actual == expected, (path, actual, expected)

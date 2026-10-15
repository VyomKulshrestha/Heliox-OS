"""Target posterior and ambiguity (spec section 6).

The posterior over candidates given an activation endpoint uses a Gaussian endpoint model per candidate and a null
hypothesis "no intended candidate". Ambiguity is ``A_hit = 1 - P(hit | e)`` for the literal hit (1 without one); the
max posterior is never used for gating. Gaze can only raise ambiguity: ``A_gate`` is the larger of ``A_hit`` with and
without gaze. ``lib/coadaptation/targetPosterior.ts`` is the TypeScript twin; both read the parity fixture written by
``python -m pilot.coadaptation.core.target_posterior --write-fixture PATH``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .state import Rect

GazeRegion = Literal["center", "left", "right", "up", "down"]
GAZE_REGIONS: tuple[str, ...] = ("center", "left", "right", "up", "down")

NULL_PRIOR = 0.02  # pi_0
CANDIDATE_PRIOR_MASS = 0.98  # sum of pi_i over candidates
EFFECTIVE_WIDTH = 4.133
PLAUSIBLE_POSTERIOR = 0.15
FIXTURE_SEED = 20261015
FIXTURE_CASES = 44

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, slots=True)
class GazeInput:
    """The gaze used for one posterior: stabilized region, its confidence and the session's gaze-check accuracy."""

    region: GazeRegion
    conf: float
    acc: float


@dataclass(frozen=True, slots=True)
class PosteriorResult:
    posterior: tuple[float, ...]  # P(i | e) per candidate, with gaze when gaze was given (null mass = p_null)
    posterior_no_gaze: tuple[float, ...]
    p_null: float  # null-hypothesis mass of `posterior`
    ambiguity: float  # A_hit without gaze
    ambiguity_gate: float  # A_gate = max(A_hit with gaze, A_hit without gaze)
    n_plausible: int  # max(1, min(3, #{i: P(i | e) > 0.15})), with-gaze posterior (spec 6.3)
    has_hit: bool
    hit_index: int | None
    p_hit: float  # 1 - A_gate for the literal hit, 0 without one
    p_top: float  # max of `posterior`

    def top_k(self, k: int) -> tuple[int, ...]:
        """Indices of the k candidates with the highest with-gaze posterior (lower index first on ties)."""
        order = sorted(range(len(self.posterior)), key=lambda i: (-self.posterior[i], i))
        return tuple(order[:k])


def keyboard_endpoint(rect: Rect) -> tuple[int, int]:
    """The recorded endpoint of a keyboard activation: ``floor(c + 0.5)`` per axis of the focused candidate."""
    return math.floor(rect.cx + 0.5), math.floor(rect.cy + 0.5)


def rendered_rects(candidates: Sequence[Rect], expanded_index: int | None, scale: float) -> tuple[Rect, ...]:
    """The cue-time rects with the expanded candidate, if any, scaled by ``scale`` about its own centre."""
    if expanded_index is None:
        return tuple(candidates)
    if not 0 <= expanded_index < len(candidates):
        raise ValueError("expanded_index outside the candidates")
    out = list(candidates)
    c = out[expanded_index]
    out[expanded_index] = Rect(cx=c.cx, cy=c.cy, w=c.w * scale, h=c.h * scale)
    return tuple(out)


def _contains(rect: Rect, x: float, y: float) -> bool:
    return abs(x - rect.cx) <= rect.w / 2.0 and abs(y - rect.cy) <= rect.h / 2.0


def literal_hit(rects: Sequence[Rect], x: float, y: float) -> int | None:
    """The topmost rendered rect containing the point (inclusive bounds), or None.

    Draw order: an expanded candidate is drawn on top of its neighbours, and otherwise later candidates are drawn
    over earlier ones. Read from the rects alone, the containing rect with the largest area wins, and the higher index
    wins a tie (for example the shared edge of two adjacent menu items).
    """
    best: int | None = None
    best_area = -1.0
    for i, rect in enumerate(rects):
        if _contains(rect, x, y):
            area = rect.w * rect.h
            if area >= best_area:
                best, best_area = i, area
    return best


def in_gaze_region(rect: Rect, region: str, surface_w: float, surface_h: float) -> bool:
    """Whether a candidate centre lies in a gaze region of the surface (spec 6.2).

    left: ``cx < w/3``; right: ``cx > 2w/3``; up: ``cy < h/3``; down: ``cy > 2h/3``; center: the middle third of both
    axes, bounds included.
    """
    lo_x, hi_x = surface_w / 3.0, 2.0 * surface_w / 3.0
    lo_y, hi_y = surface_h / 3.0, 2.0 * surface_h / 3.0
    if region == "left":
        return rect.cx < lo_x
    if region == "right":
        return rect.cx > hi_x
    if region == "up":
        return rect.cy < lo_y
    if region == "down":
        return rect.cy > hi_y
    if region == "center":
        return lo_x <= rect.cx <= hi_x and lo_y <= rect.cy <= hi_y
    raise ValueError("unknown gaze region")


def _likelihoods(rects: Sequence[Rect], x: float, y: float, r: float, floor: float) -> list[float]:
    out = []
    for rect in rects:
        sx = r * math.sqrt((rect.w / EFFECTIVE_WIDTH) ** 2 + floor * floor)
        sy = r * math.sqrt((rect.h / EFFECTIVE_WIDTH) ** 2 + floor * floor)
        dx = x - rect.cx
        dy = y - rect.cy
        out.append(math.exp(-0.5 * (dx * dx / (sx * sx) + dy * dy / (sy * sy))) / (_TWO_PI * sx * sy))
    return out


def _normalise(priors: Sequence[float], likes: Sequence[float], null_term: float) -> tuple[tuple[float, ...], float]:
    weighted = [p * lk for p, lk in zip(priors, likes)]
    total = null_term + math.fsum(weighted)
    return tuple(w / total for w in weighted), null_term / total


def compute_posterior(
    candidates: Sequence[Rect],
    endpoint: tuple[float, float],
    *,
    modality: str,
    r: float | None,
    floor: float | None,
    surface: tuple[float, float],
    prior: Sequence[float] | None = None,
    gaze: GazeInput | None = None,
    press_index: int | None = None,
) -> PosteriorResult:
    """Posterior over the rendered ``candidates`` for an activation endpoint (spec 6.1-6.3).

    ``press_index``: the candidate under the press point, ``-1`` when the press hit no candidate (then there is no
    literal hit), or None for no press constraint (an argument value only; recorded activations always carry an int).
    With a press index the literal hit is the topmost candidate under the endpoint only when it is the pressed one.
    Keyboard activations return the fixed result of spec 6.1 (one-hot on ``press_index``, ``A_hit = A_gate = 0``,
    ``n_plausible = 1``) without reading ``r``, ``floor``, the endpoint or gaze; ``press_index`` must name a candidate.
    ``prior`` (default uniform) holds ``pi_i`` summing to 0.98; the caller passes ``gaze`` only when gaze is effective.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("no candidates")
    if modality == "keyboard":
        if press_index is None or not 0 <= press_index < n:
            raise ValueError("a keyboard activation needs press_index naming a candidate")
        one_hot = tuple(1.0 if i == press_index else 0.0 for i in range(n))
        return PosteriorResult(
            posterior=one_hot,
            posterior_no_gaze=one_hot,
            p_null=0.0,
            ambiguity=0.0,
            ambiguity_gate=0.0,
            n_plausible=1,
            has_hit=True,
            hit_index=press_index,
            p_hit=1.0,
            p_top=1.0,
        )
    if r is None or floor is None:
        raise ValueError("pointer posteriors need r and floor")
    if press_index is not None and not -1 <= press_index < n:
        raise ValueError("press_index outside the candidates")
    x, y = float(endpoint[0]), float(endpoint[1])
    surface_w, surface_h = float(surface[0]), float(surface[1])
    base = [CANDIDATE_PRIOR_MASS / n] * n if prior is None else [float(p) for p in prior]
    if len(base) != n:
        raise ValueError("prior must have one entry per candidate")
    likes = _likelihoods(candidates, x, y, float(r), float(floor))
    null_term = NULL_PRIOR / (surface_w * surface_h)
    post_plain, null_plain = _normalise(base, likes, null_term)
    if gaze is not None:
        kappa = min(2.0, 1.0 + gaze.acc * gaze.conf)
        weighted = [
            p * kappa if in_gaze_region(rect, gaze.region, surface_w, surface_h) else p
            for p, rect in zip(base, candidates)
        ]
        scale = CANDIDATE_PRIOR_MASS / math.fsum(weighted)
        post_gaze, null_gaze = _normalise([p * scale for p in weighted], likes, null_term)
    else:
        post_gaze, null_gaze = post_plain, null_plain

    hit = literal_hit(candidates, x, y)
    if press_index is not None and hit != press_index:
        hit = None
    if hit is None:
        a_plain = 1.0
        a_gaze = 1.0
    else:
        a_plain = 1.0 - post_plain[hit]
        a_gaze = 1.0 - post_gaze[hit]
    a_gate = max(a_plain, a_gaze)
    plausible = sum(1 for p in post_gaze if p > PLAUSIBLE_POSTERIOR)
    return PosteriorResult(
        posterior=post_gaze,
        posterior_no_gaze=post_plain,
        p_null=null_gaze,
        ambiguity=a_plain,
        ambiguity_gate=a_gate,
        n_plausible=max(1, min(3, plausible)),
        has_hit=hit is not None,
        hit_index=hit,
        p_hit=(1.0 - a_gate) if hit is not None else 0.0,
        p_top=max(post_gaze),
    )


def ambiguity_bin(a: float, tau: float) -> str | None:
    """b1, b2 or b3 for an ambiguous activation (``a > tau``), else None (spec 6.3)."""
    if a <= tau:
        return None
    if a <= max(tau, 0.4):
        return "b1"
    if a <= max(tau, 0.6):
        return "b2"
    return "b3"


def bin_bounds(tau: float) -> dict[str, tuple[float, float]]:
    """``{bin: (lo, hi)}`` with ``lo < A_gate <= hi`` (the session's bins, spec 6.3 and 14.3)."""
    return {"b1": (tau, max(tau, 0.4)), "b2": (max(tau, 0.4), max(tau, 0.6)), "b3": (max(tau, 0.6), 1.0)}


def cell_id(bin: str, n_plausible: int, has_hit: bool) -> str:
    """``f"{bin}:p{n_plausible}:h{1 if has_hit else 0}"`` (spec 6.3)."""
    if bin not in ("b1", "b2", "b3") or n_plausible not in (1, 2, 3):
        raise ValueError("invalid cell")
    return f"{bin}:p{n_plausible}:h{1 if has_hit else 0}"


# --- parity fixture (spec 6.4) ------------------------------------------------------------------------------------

_FLOORS = {"mouse": 1.0, "touchpad": 1.5, "pen": 1.5, "touch": 3.0, "hand_pointer": 6.0}
_R0 = {"mouse": 1.0, "touchpad": 1.15, "pen": 1.1, "touch": 1.3, "hand_pointer": 1.6}
_SURFACE = (700.0, 500.0)


def _grid(cols: int, rows: int, w: float, gap: float, centre: tuple[float, float]) -> list[Rect]:
    pitch = w + gap
    x0 = centre[0] - (cols - 1) * pitch / 2.0
    y0 = centre[1] - (rows - 1) * pitch / 2.0
    return [Rect(cx=x0 + c * pitch, cy=y0 + row * pitch, w=w, h=w) for row in range(rows) for c in range(cols)]


def _menu(n_items: int, origin: tuple[float, float]) -> list[Rect]:
    # 220 x 24 px items directly below a 32 px menu button centred at `origin`.
    top = origin[1] + 16.0
    return [Rect(cx=origin[0], cy=top + 12.0 + 24.0 * i, w=220.0, h=24.0) for i in range(n_items)]


def _case(
    modality: str,
    candidates: Sequence[Rect],
    endpoint: tuple[int, int],
    *,
    expanded_index: int | None = None,
    scale: float = 1.0,
    press_index: int | None = None,
    r: float | None = None,
    gaze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keyboard = modality == "keyboard"
    floor = None if keyboard else _FLOORS[modality]
    r_val = None if keyboard else (_R0[modality] if r is None else r)
    rendered = rendered_rects(candidates, expanded_index, scale)
    gaze_in = GazeInput(region=gaze["region"], conf=gaze["conf"], acc=gaze["acc"]) if gaze is not None else None
    result = compute_posterior(
        rendered,
        endpoint,
        modality=modality,
        r=r_val,
        floor=floor,
        surface=_SURFACE,
        gaze=gaze_in,
        press_index=press_index,
    )
    return {
        "modality": modality,
        "candidates": [c.to_dict() for c in candidates],
        "expanded_index": expanded_index,
        "scale": scale,
        "press_index": press_index,
        "endpoint": {"x": int(endpoint[0]), "y": int(endpoint[1])},
        "r": r_val,
        "floor": floor,
        "surface": {"w": _SURFACE[0], "h": _SURFACE[1]},
        "gaze": gaze,
        "expected": {
            "rendered": [c.to_dict() for c in rendered],
            "posterior": list(result.posterior),
            "ambiguity": result.ambiguity,
            "ambiguity_gate": result.ambiguity_gate,
            "n_plausible": result.n_plausible,
            "has_hit": result.has_hit,
            "hit_index": result.hit_index,
        },
    }


def _fixed_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Click inside a wrongly expanded neighbour (B3.7): W = 28, gap 2, scale 1.5, endpoint 20 px from the expanded
    # centre, r = 1, mouse floor. A_hit is about 0.84; the max-posterior form would give about 0.16.
    row = _grid(4, 1, 28.0, 2.0, (350.0, 250.0))
    ex = row[1]
    cases.append(_case("mouse", row, (int(ex.cx + 20), int(ex.cy)), expanded_index=1, scale=1.5, r=1.0))
    # Same click with the press constraint satisfied (press and release on the expanded candidate).
    cases.append(_case("mouse", row, (int(ex.cx + 20), int(ex.cy)), expanded_index=1, scale=1.5, r=1.0, press_index=1))
    # Correct expansion: endpoint near the expanded candidate's centre.
    cases.append(_case("mouse", row, (int(ex.cx + 3), int(ex.cy - 2)), expanded_index=1, scale=2.0, r=1.0))
    grid = _grid(4, 3, 16.0, 2.0, (350.0, 200.0))
    # Press on a gap, release on a candidate: press_index -1, no literal hit, A = 1 (and the same endpoint without a
    # press constraint has a hit).
    gap_x = int(grid[0].cx + 9)  # the 2 px gap between candidates 0 and 1 spans cx + 8 .. cx + 10
    cases.append(_case("mouse", grid, (int(grid[1].cx), int(grid[1].cy)), press_index=-1))
    cases.append(_case("mouse", grid, (int(grid[1].cx), int(grid[1].cy)), press_index=None))
    # Endpoint in the gap (no literal hit).
    cases.append(_case("mouse", grid, (gap_x, int(grid[0].cy))))
    # Press on one candidate, release on its neighbour: no literal hit.
    cases.append(_case("touchpad", grid, (int(grid[5].cx + 1), int(grid[5].cy)), press_index=4))
    # Endpoint far from every candidate (every posterior below 0.15): n_plausible = 1.
    cases.append(_case("mouse", grid, (120, 460)))
    cases.append(_case("hand_pointer", _grid(6, 2, 44.0, 8.0, (350.0, 220.0)), (640, 40)))
    # Adjacent menu items share an edge: the later item wins the tie.
    menu = _menu(12, (350.0, 40.0))
    cases.append(_case("mouse", menu, (350, int(menu[3].cy + 12))))
    # Gaze: reduce-only weighting (A_gate is never below A_hit without gaze), including a capped kappa.
    hand_grid = _grid(6, 2, 28.0, 8.0, (350.0, 250.0))
    cases.append(
        _case(
            "hand_pointer",
            hand_grid,
            (int(hand_grid[2].cx + 12), int(hand_grid[2].cy + 5)),
            gaze={"region": "center", "conf": 0.9, "acc": 0.8},
        )
    )
    cases.append(
        _case(
            "mouse",
            _grid(4, 3, 16.0, 2.0, (150.0, 100.0)),
            (151, 101),
            gaze={"region": "left", "conf": 1.0, "acc": 1.0},
        )
    )
    # Keyboard: endpoint = the focused candidate's rounded centre (two centres at .5); one-hot posterior.
    kb_grid = [Rect(cx=100.5 + 30.0 * i, cy=200.5, w=28.0, h=28.0) for i in range(4)]
    kb_menu = _menu(6, (300.0, 60.0))
    for cands, focus in ((kb_grid, 1), (kb_grid, 3), (kb_menu, 0), (kb_menu, 4)):
        cases.append(_case("keyboard", cands, keyboard_endpoint(cands[focus]), press_index=focus))
    return cases


def _random_case(rng: np.random.Generator) -> dict[str, Any]:
    modality = str(rng.choice(["mouse", "touchpad", "pen", "touch", "hand_pointer"]))
    if rng.random() < 0.25:
        n_items = int(rng.choice([6, 12, 16]))
        candidates = _menu(n_items, (float(rng.integers(200, 480)), float(rng.integers(20, 60))))
        expanded = None
        scale = 1.0
    else:
        w = float(rng.choice([16.0, 28.0, 44.0]))
        gap = float(rng.choice([2.0, 8.0]))
        cols, rows = (4, 3) if rng.random() < 0.5 else (6, 2)
        centre = (float(rng.integers(250, 451)), float(rng.integers(150, 301)))
        candidates = _grid(cols, rows, w, gap, centre)
        if rng.random() < 0.4:
            expanded = int(rng.integers(0, len(candidates)))
            scale = float(rng.choice([1.5, 2.0]))
        else:
            expanded = None
            scale = 1.0
    rendered = rendered_rects(candidates, expanded, scale)
    target = int(rng.integers(0, len(candidates)))
    r = round(float(_R0[modality] * rng.uniform(0.8, 1.4)), 4)
    floor = _FLOORS[modality]
    sd = r * math.sqrt((candidates[target].w / EFFECTIVE_WIDTH) ** 2 + floor**2)
    sd_y = r * math.sqrt((candidates[target].h / EFFECTIVE_WIDTH) ** 2 + floor**2)
    x = math.floor(candidates[target].cx + rng.normal(0.0, sd) + 0.5)
    y = math.floor(candidates[target].cy + rng.normal(0.0, sd_y) + 0.5)
    u = rng.random()
    if u < 0.35:
        press_index: int | None = literal_hit(rendered, x, y)
        press_index = -1 if press_index is None else press_index
    elif u < 0.5:
        press_index = int(rng.integers(0, len(candidates)))
    elif u < 0.6:
        press_index = -1
    else:
        press_index = None
    gaze = None
    if rng.random() < 0.3:
        gaze = {
            "region": str(rng.choice(list(GAZE_REGIONS))),
            "conf": round(float(rng.uniform(0.3, 1.0)), 3),
            "acc": round(float(rng.uniform(0.6, 1.0)), 3),
        }
    return _case(
        modality, candidates, (x, y), expanded_index=expanded, scale=scale, press_index=press_index, r=r, gaze=gaze
    )


def fixture_cases(seed: int = FIXTURE_SEED) -> list[dict[str, Any]]:
    """The 44 parity cases of spec 6.4: fixed edge cases first, then seeded random layouts and endpoints."""
    rng = np.random.Generator(np.random.PCG64(seed))
    cases = _fixed_cases()
    while len(cases) < FIXTURE_CASES:
        cases.append(_random_case(rng))
    return cases


def fixture_json(seed: int = FIXTURE_SEED) -> str:
    payload = {"seed": seed, "cases": fixture_cases(seed)}
    return json.dumps(payload, indent=1) + "\n"


def write_fixture(path: Path, seed: int = FIXTURE_SEED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture_json(seed).encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the target-posterior parity fixture (spec 6.4).")
    parser.add_argument("--write-fixture", metavar="PATH", required=True)
    args = parser.parse_args(argv)
    write_fixture(Path(args.write_fixture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

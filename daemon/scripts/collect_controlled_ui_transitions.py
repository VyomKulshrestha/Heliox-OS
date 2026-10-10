"""Collect causal GUI transitions from an isolated, generated browser page.

The script executes only clicks or mouse moves inside a headless local page.
It verifies the page's expected state and retains no screenshot bytes; the
ignored output contains 16x16 grayscale features for research evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pilot.agents.screen_vision import _world_embedding_from_frame

WIDTH = 640
HEIGHT = 480
FAMILIES = (
    ("#f4f4ef", "#232323", "#484848", "#c0c0c0", "0px"),
    ("#22252a", "#ededed", "#adadad", "#f4f4f4", "18px"),
    ("#ece4dd", "#302a28", "#5b514c", "#a9a09b", "4px"),
    ("#1e2730", "#d9e1e7", "#8091a0", "#e0e6eb", "26px"),
    ("#e1e7dc", "#283328", "#405c40", "#a0b7a0", "8px"),
    ("#292429", "#f0e3ef", "#ac8aaa", "#e6cce3", "0px"),
)


def _page_html(family: int, positions: tuple[tuple[int, int], tuple[int, int]]) -> str:
    background, foreground, button, panel, radius = FAMILIES[family]
    controls = []
    for index, (x, y) in enumerate(positions):
        controls.append(
            f'<button id="button-{index}" style="left:{x}px;top:{y}px" '
            f"onclick=\"document.getElementById('panel-{index}').hidden=false\">"
            f"Select {index + 1}</button>"
            f'<div class="panel" id="panel-{index}" hidden '
            f'style="left:{x}px;top:{y + 58}px"></div>'
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        f"body{{margin:0;background:{background};color:{foreground};font:18px Arial}}"
        f"button{{position:absolute;width:115px;height:42px;background:{button};"
        f"color:{background};border:2px solid {foreground};border-radius:{radius};cursor:pointer}}"
        f".panel{{position:absolute;width:115px;height:72px;background:{panel};"
        f"border:3px solid {foreground};border-radius:{radius}}}"
        "[hidden]{display:none!important}"
        "</style></head><body>" + "".join(controls) + "</body></html>"
    )


def _features(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as image:
        frame = np.asarray(image.convert("RGB").resize((64, 64)).convert("L"), dtype=np.float32)
    return np.asarray(_world_embedding_from_frame(frame), dtype=np.float32)


def _positions(family: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    """Keep validation and evaluation geometry distinct from fit layouts."""
    if not 0 <= family < len(FAMILIES):
        raise ValueError("Unknown controlled layout family")

    def left() -> int:
        return rng.randint(25, 185)

    def right() -> int:
        return rng.randint(335, 485)

    if family == 0:
        return ((left(), rng.randint(35, 300)), (right(), rng.randint(35, 300)))
    if family == 1:
        return ((left(), rng.randint(35, 90)), (right(), rng.randint(35, 90)))
    if family == 2:
        return ((left(), rng.randint(230, 300)), (right(), rng.randint(230, 300)))
    if family == 3:
        return ((left(), rng.randint(35, 90)), (left(), rng.randint(230, 300)))
    if family == 4:
        return ((right(), rng.randint(35, 90)), (right(), rng.randint(230, 300)))
    return ((left(), rng.randint(35, 90)), (right(), rng.randint(230, 300)))


def _collect(page, family: int, sample_index: int, rng: random.Random) -> dict:
    positions = _positions(family, rng)
    page.set_content(_page_html(family, positions), wait_until="load")
    chosen = rng.randrange(2)
    x, y = positions[chosen]
    center_x, center_y = x + 57, y + 21
    action = (sample_index + family) % 2  # balanced click versus move per family
    before = _features(page.screenshot())
    if action:
        page.mouse.click(center_x, center_y)
    else:
        page.mouse.move(center_x, center_y)
    observed = page.locator(".panel:not([hidden])").count()
    if observed != action:
        raise ValueError("Browser state did not match the executed action")
    after = _features(page.screenshot())
    group = hashlib.sha256(f"{family}:{sample_index}".encode("ascii")).hexdigest()
    return {
        "state": before,
        "target": after,
        "action": action,
        "coords": (center_x / WIDTH, center_y / HEIGHT),
        "group": group,
        "split": "train" if family < 4 else "validation" if family == 4 else "eval",
    }


def _collect_counterfactual_pair(page, family: int, sample_index: int, rng: random.Random) -> list[dict]:
    """Execute both clicks from the same reset page, never from each other's result."""
    positions = _positions(family, rng)
    html = _page_html(family, positions)
    group = hashlib.sha256(f"{family}:{sample_index}".encode("ascii")).hexdigest()
    split = "train" if family < 4 else "validation" if family == 4 else "eval"
    rows = []
    for chosen, (x, y) in enumerate(positions):
        page.set_content(html, wait_until="load")
        before = _features(page.screenshot())
        center_x, center_y = x + 57, y + 21
        page.mouse.click(center_x, center_y)
        if page.locator(".panel:not([hidden])").count() != 1 or not page.locator(f"#panel-{chosen}").is_visible():
            raise ValueError("Browser state did not match the chosen counterfactual action")
        rows.append(
            {
                "state": before,
                "target": _features(page.screenshot()),
                "action": 1,
                "coords": (center_x / WIDTH, center_y / HEIGHT),
                "group": group,
                "split": split,
            }
        )
    if not np.array_equal(rows[0]["state"], rows[1]["state"]):
        raise ValueError("Counterfactual clicks did not start from the same observed state")
    if np.array_equal(rows[0]["target"], rows[1]["target"]):
        raise ValueError("Counterfactual clicks did not produce distinct observed states")
    return rows


def main() -> None:
    # Optional dependency (pip install -e ".[browser]"); imported here so the helpers stay importable.
    from playwright.sync_api import sync_playwright

    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-family", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--paired-counterfactuals", action="store_true")
    parser.add_argument("--features", type=Path, default=state / "controlled_ui_transitions_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "controlled_ui_transitions_v1.json")
    args = parser.parse_args()
    if args.samples_per_family < 2 or args.features.exists() or args.report.exists():
        parser.error("Use at least two samples per family and fresh output paths")
    rng = random.Random(args.seed)
    rows: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
            for family in range(len(FAMILIES)):
                for index in range(args.samples_per_family):
                    if args.paired_counterfactuals:
                        rows.extend(_collect_counterfactual_pair(page, family, index, rng))
                    else:
                        rows.append(_collect(page, family, index, rng))
                print(json.dumps({"family": family, "completed": args.samples_per_family}), flush=True)
        finally:
            browser.close()
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        state=np.stack([row["state"] for row in rows]),
        target=np.stack([row["target"] for row in rows]),
        action=np.array([row["action"] for row in rows], dtype=np.int16),
        coords=np.array([row["coords"] for row in rows], dtype=np.float32),
        group=np.array([row["group"] for row in rows]),
        split=np.array([row["split"] for row in rows]),
    )
    report = {
        "purpose": "research_only_controlled_browser_visual_transitions",
        "seed": args.seed,
        "samples_per_family": args.samples_per_family,
        "counterfactual_pairs": args.paired_counterfactuals,
        "split_basis": "layout_family",
        "fit_layouts": [0, 1, 2, 3],
        "validation_layout": 4,
        "eval_layout": 5,
        "rows": dict(Counter(row["split"] for row in rows)),
        "groups": {
            split: len({row["group"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "eval")
        },
        "browser_state_verified": True,
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "limitation": "Generated local pages with button toggles only; no user screen capture, "
        "real desktop transfer, safety, or runtime-promotion claim.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

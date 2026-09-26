"""Collect paired navigation transitions from an isolated local Heliox UI.

The browser has fresh storage and no desktop/daemon control. Screenshots remain
in memory; only 16x16 grayscale research features are saved to ignored files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from pilot.agents.screen_vision import _world_embedding_from_frame

VIEWPORTS = ((1100, 700), (1200, 700), (1280, 720),
             (1360, 768), (1440, 800), (1536, 864))
TAB_TITLES = ("Open Dashboard", "Open Settings")
STABLE_STYLE = (
    "canvas, .titlebar-right { visibility: hidden !important } "
    "* { animation: none !important; transition: none !important }"
)
MIN_FREE_BYTES = 512 * 1024 * 1024


def validate_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Heliox UI collector accepts only a local HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Heliox UI collector URL must not contain credentials or query data")
    return url


def require_free_space(path: Path) -> None:
    if shutil.disk_usage(path).free < MIN_FREE_BYTES:
        raise RuntimeError("At least 512 MiB of free disk space is required for collection")


def _features(page) -> np.ndarray:
    with Image.open(io.BytesIO(page.screenshot(animations="disabled"))) as image:
        frame = np.asarray(image.convert("RGB").resize((64, 64)).convert("L"), dtype=np.float32)
    return np.asarray(_world_embedding_from_frame(frame), dtype=np.float32)


def _load_baseline(page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    page.locator('nav.tabs button[title="Open Command Panel"]').wait_for()
    page.add_style_tag(content=STABLE_STYLE)
    page.wait_for_timeout(500)
    if page.locator("nav.tabs button.active").get_attribute("title") != "Open Command Panel":
        raise RuntimeError("Heliox did not reset to the command view")


def _pair(page, url: str, family: int, index: int, width: int, height: int) -> list[dict]:
    page.set_viewport_size({"width": width, "height": height})
    group = hashlib.sha256(f"heliox-ui:{family}:{index}".encode("ascii")).hexdigest()
    split = "train" if family < 4 else "validation" if family == 4 else "eval"
    rows = []
    for title in TAB_TITLES:
        _load_baseline(page, url)
        before = _features(page)
        button = page.locator(f'nav.tabs button[title="{title}"]')
        box = button.bounding_box()
        if box is None:
            raise RuntimeError(f"Heliox navigation button is not visible: {title}")
        coords = ((box["x"] + box["width"] / 2) / width,
                  (box["y"] + box["height"] / 2) / height)
        button.click()
        page.locator(f'nav.tabs button.active[title="{title}"]').wait_for()
        page.wait_for_timeout(500)
        rows.append({"state": before, "target": _features(page), "action": 1,
                     "coords": coords, "group": group, "split": split})
    if not np.array_equal(rows[0]["state"], rows[1]["state"]):
        raise RuntimeError("Heliox navigation alternatives did not start from the same screen")
    if np.array_equal(rows[0]["target"], rows[1]["target"]):
        raise RuntimeError("Heliox navigation alternatives did not show distinct screens")
    return rows


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:1420/")
    parser.add_argument("--samples-per-family", type=int, default=12)
    parser.add_argument("--eval-samples-per-family", type=int,
                        help="Override only the held-out viewport family's sample count")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--training-theme", choices=("light", "dark"), default="light",
                        help="Browser color scheme for fit and validation viewport families")
    parser.add_argument("--eval-dark", action="store_true",
                        help="Use dark browser color scheme for the evaluation family")
    parser.add_argument("--features", type=Path, default=state / "heliox_ui_transitions_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "heliox_ui_transitions_v1.json")
    args = parser.parse_args()
    try:
        validate_url(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.samples_per_family < 2
            or (args.eval_samples_per_family is not None and args.eval_samples_per_family < 2)
            or args.features.exists() or args.report.exists()):
        parser.error("Use at least two samples per family and fresh output paths")
    args.features.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for family, (base_width, base_height) in enumerate(VIEWPORTS):
                theme = "dark" if args.eval_dark and family == 5 else args.training_theme
                sample_count = (args.eval_samples_per_family
                                if family == 5 and args.eval_samples_per_family is not None
                                else args.samples_per_family)
                context = browser.new_context(device_scale_factor=1, reduced_motion="reduce",
                                              color_scheme=theme, service_workers="block")
                try:
                    context.route("**/*", lambda route: route.continue_()
                                  if urlsplit(route.request.url).netloc == urlsplit(args.url).netloc
                                  else route.abort())
                    context.add_init_script('''
                        localStorage.setItem("heliox_first_run_complete", "true");
                        const NativeWebSocket = window.WebSocket;
                        window.WebSocket = class extends NativeWebSocket {
                            constructor(url, protocols) {
                                if (new URL(url, location.href).host !== location.host) {
                                    throw new Error("External WebSocket disabled for isolated UI collection");
                                }
                                if (protocols === undefined) super(url);
                                else super(url, protocols);
                            }
                        };
                    ''')
                    page = context.new_page()
                    for index in range(sample_count):
                        require_free_space(args.features.parent)
                        width = base_width + rng.randint(-16, 16)
                        height = base_height + rng.randint(-16, 16)
                        rows.extend(_pair(page, args.url, family, index, width, height))
                finally:
                    context.close()
                print(json.dumps({"viewport_family": family,
                                  "theme": theme, "completed": sample_count}), flush=True)
        finally:
            browser.close()

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
        "purpose": "research_only_local_heliox_browser_navigation",
        "feature_shape": [1, 16, 16],
        "seed": args.seed,
        "samples_per_family": args.samples_per_family,
        "eval_samples_per_family": args.eval_samples_per_family or args.samples_per_family,
        "counterfactual_pairs": True,
        "split_basis": ("viewport_family_and_theme"
                        if args.eval_dark and args.training_theme != "dark"
                        else "viewport_family"),
        "training_theme": args.training_theme,
        "eval_dark": args.eval_dark,
        "fit_layouts": [0, 1, 2, 3], "validation_layout": 4, "eval_layout": 5,
        "rows": dict(Counter(row["split"] for row in rows)),
        "groups": {split: len({row["group"] for row in rows if row["split"] == split})
                   for split in ("train", "validation", "eval")},
        "same_screen_and_distinct_outcome_verified": True,
        "network_scope": "same-origin HTTP only; external WebSocket and service workers blocked",
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "limitation": "Fresh browser storage; decorative canvases and live header telemetry hidden "
                      "during capture. Two fixed Heliox navigation actions and held-out viewport "
                      "layouts only; color scheme is recorded for fit and evaluation. "
                      "No personal desktop recording, real OS action transfer, "
                      "safety, autonomy, or Jev parity claim.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Collect reproducible, task-disjoint MiniWoB click branches for research.

Task assets are fetched from a pinned MIT-licensed source revision into memory
and served to an isolated headless browser. No screenshot bytes or task source
are persisted; the output contains only 16x16 visual features and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import re
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from pilot.agents.screen_vision import _world_embedding_from_frame

SOURCE_REPO = "Farama-Foundation/MiniWoB-plusplus"
SOURCE_COMMIT = "33c3b4ddef8c6eb67c57a29663d844b1eda7e614"
VIRTUAL_ORIGIN = "https://miniwob.local"
MAX_ASSET_BYTES = 2 * 1024 * 1024
VIEWPORT = {"width": 320, "height": 360}
CONTROLS = '#wrap button, #wrap input[type="checkbox"], #wrap input[type="radio"], #wrap select'
STABLE_STYLE = "#timer-countdown,#reward-avg,#episode-id,#click-track-canvas{visibility:hidden!important}"


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in value.split(",") if part.strip())
    if not tasks or len(set(tasks)) != len(tasks) or any(
        re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", task) is None for task in tasks
    ):
        raise ValueError("Tasks must be unique comma-separated MiniWoB page names")
    return tasks


def asset_path(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "miniwob.local"
            or parsed.query or parsed.fragment or "%" in parsed.path):
        raise ValueError("Only pinned MiniWoB virtual-origin assets are allowed")
    path = parsed.path.removeprefix("/")
    if (not path.startswith("miniwob/html/") or ".." in path.split("/")
            or re.fullmatch(r"[A-Za-z0-9_./-]+", path) is None):
        raise ValueError("Invalid MiniWoB asset path")
    return path


class PinnedAssets:
    def __init__(self, commit: str) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("Source commit must be a full SHA-1")
        self._base = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{commit}/"
        self._cache: dict[str, bytes] = {}

    def fetch(self, path: str) -> bytes:
        if path not in self._cache:
            with urllib.request.urlopen(self._base + path, timeout=20) as response:
                data = response.read(MAX_ASSET_BYTES + 1)
            if len(data) > MAX_ASSET_BYTES:
                raise ValueError("MiniWoB asset exceeds bounded read limit")
            self._cache[path] = data
        return self._cache[path]

    def route(self, route) -> None:
        try:
            path = asset_path(route.request.url)
            data = self.fetch(path)
        except (OSError, ValueError):
            route.abort()
            return
        route.fulfill(status=200, body=data,
                      content_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

    def hashes(self) -> dict[str, str]:
        return {path: hashlib.sha256(data).hexdigest() for path, data in sorted(self._cache.items())}


def visual_feature(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as image:
        frame = image.convert("RGB").resize((64, 64)).convert("L")
        values = _world_embedding_from_frame(frame)
    return np.asarray(values, dtype=np.float32)


def _reset(browser, assets: PinnedAssets, task: str, seed: str):
    context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
    context.route("**/*", assets.route)
    page = context.new_page()
    page.goto(f"{VIRTUAL_ORIGIN}/miniwob/html/miniwob/{task}.html", wait_until="load", timeout=30000)
    if not page.evaluate("typeof core !== 'undefined' && typeof Math.seedrandom === 'function'"):
        context.close()
        raise ValueError(f"MiniWoB task did not load: {task}")
    page.evaluate("seed => { Math.seedrandom(seed); core.EPISODE_MAX_TIME = 60000; }", seed)
    page.locator("#sync-task-cover").click(timeout=5000)
    page.add_style_tag(content=STABLE_STYLE)
    controls = page.locator(CONTROLS)
    candidates = []
    for index in range(controls.count()):
        control = controls.nth(index)
        box = control.bounding_box()
        if box and box["width"] > 0 and box["height"] > 0 and control.is_visible():
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            if 0 <= x < VIEWPORT["width"] and 0 <= y < VIEWPORT["height"]:
                candidates.append((round(x, 3), round(y, 3)))
    before_png = page.screenshot(animations="disabled")
    before_dom = page.locator("#wrap").inner_html()
    return context, page, candidates, before_png, before_dom


def collect_group(browser, assets: PinnedAssets, task: str, seed: str, max_candidates: int) -> tuple[list[dict], int]:
    reference: tuple[list[tuple[float, float]], str, np.ndarray] | None = None
    rows = []
    attempted = 0
    for candidate_index in range(max_candidates):
        context, page, candidates, before_png, before_dom = _reset(browser, assets, task, seed)
        try:
            before_feature = visual_feature(before_png)
            if reference is None:
                reference = (candidates, before_dom, before_feature)
            elif (reference[0] != candidates or reference[1] != before_dom
                  or not np.array_equal(reference[2], before_feature)):
                raise ValueError("Independent MiniWoB branches did not reset to one exact measured state")
            if candidate_index >= len(candidates):
                break
            x, y = candidates[candidate_index]
            page.mouse.click(x, y)
            outcome = page.evaluate("({done: WOB_DONE_GLOBAL, reward: WOB_RAW_REWARD_GLOBAL})")
            rows.append({
                "state": before_feature,
                "target": visual_feature(page.screenshot(animations="disabled")),
                "coords": (x / VIEWPORT["width"], y / VIEWPORT["height"]),
                "reward": float(outcome["reward"]) if outcome["done"] else None,
            })
            attempted += 1
        finally:
            context.close()
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if not np.array_equal(rows[left]["target"], rows[right]["target"]):
                return [rows[left], rows[right]], attempted
    return [], attempted


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--eval-tasks", required=True)
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--features", type=Path, default=root / "miniwob_branches_v1.npz")
    parser.add_argument("--report", type=Path, default=root / "miniwob_branches_v1.json")
    args = parser.parse_args()
    try:
        tasks = {split: parse_tasks(getattr(args, f"{split}_tasks"))
                 for split in ("train", "validation", "eval")}
    except ValueError as exc:
        parser.error(str(exc))
    if (len(set(task for names in tasks.values() for task in names)) != sum(map(len, tasks.values()))
            or not 2 <= args.samples_per_task <= 100 or not 2 <= args.max_candidates <= 16
            or args.features.exists() or args.report.exists()):
        parser.error("Use task-disjoint splits, bounded settings, and fresh output paths")

    assets = PinnedAssets(SOURCE_COMMIT)
    rows = []
    attempted = Counter()
    skipped = Counter()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for split, names in tasks.items():
                for task in names:
                    for index in range(args.samples_per_task):
                        seed = f"{SOURCE_COMMIT}:{task}:{index}"
                        pair, count = collect_group(browser, assets, task, seed, args.max_candidates)
                        attempted[split] += count
                        if not pair:
                            skipped[split] += 1
                            continue
                        group = hashlib.sha256(seed.encode("utf-8")).hexdigest()
                        rows.extend({**row, "group": group, "split": split, "task": task} for row in pair)
                    print(json.dumps({"task": task, "split": split,
                                      "groups": len({row["group"] for row in rows if row["task"] == task})}),
                          flush=True)
        finally:
            browser.close()
    if any(not any(row["split"] == split for row in rows) for split in tasks):
        raise ValueError("Each split needs at least one verified distinct-outcome branch")
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        state=np.stack([row["state"] for row in rows]),
        target=np.stack([row["target"] for row in rows]),
        action=np.ones(len(rows), dtype=np.int16),
        coords=np.asarray([row["coords"] for row in rows], dtype=np.float32),
        group=np.asarray([row["group"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
        task=np.asarray([row["task"] for row in rows]),
    )
    report = {
        "purpose": "research_only_miniwob_visual_branch_transitions",
        "source_repo": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
        "source_license": "MIT", "asset_sha256": assets.hashes(),
        "tasks": tasks, "samples_per_task": args.samples_per_task,
        "max_candidates": args.max_candidates,
        "selection": "first two candidate clicks with distinct observed 16x16 next-screen features",
        "reset_identity": "identical visible control coordinates, task DOM, and 16x16 pre-action features; "
                          "full screenshot bytes may differ by rendering noise",
        "counterfactual_pairs": True, "split_basis": "task_family",
        "rows": dict(Counter(row["split"] for row in rows)),
        "groups": {split: len({row["group"] for row in rows if row["split"] == split}) for split in tasks},
        "candidate_attempts": dict(attempted), "skipped_no_distinct_visual_outcomes": dict(skipped),
        "reward_observed": {split: dict(Counter(str(row["reward"]) for row in rows if row["split"] == split))
                            for split in tasks},
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "limitation": "Synthetic browser tasks, selected for visually distinct observed outcomes. "
                      "Only click location is modeled; reward labels are counted but not trained on. "
                      "No real desktop transfer, Jev parity, autonomy, or execution authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Collect reproducible, task-disjoint MiniWoB click branches for research.

Task assets are fetched from a pinned MIT-licensed source revision into memory
and served to an isolated headless browser. No screenshot bytes or task source
are persisted. An opt-in archive can also save bounded visible text for public
synthetic tasks; it never observes the user's desktop.
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
STABLE_STYLE = "#timer-countdown,#reward-avg,#episode-id,#click-track-canvas{visibility:hidden!important}"
MAX_SEMANTIC_CHARS = 4096


class UnstableResetError(ValueError):
    """A candidate branch does not match its independently reset prestate."""


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in value.split(",") if part.strip())
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", task) is None for task in tasks)
    ):
        raise ValueError("Tasks must be unique comma-separated MiniWoB page names")
    return tasks


def asset_path(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "miniwob.local"
        or parsed.query
        or parsed.fragment
        or "%" in parsed.path
    ):
        raise ValueError("Only pinned MiniWoB virtual-origin assets are allowed")
    path = parsed.path.removeprefix("/")
    if (
        not path.startswith("miniwob/html/")
        or ".." in path.split("/")
        or re.fullmatch(r"[A-Za-z0-9_./-]+", path) is None
    ):
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
        route.fulfill(status=200, body=data, content_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

    def hashes(self) -> dict[str, str]:
        return {path: hashlib.sha256(data).hexdigest() for path, data in sorted(self._cache.items())}


def visual_feature(png: bytes, size: int = 16) -> np.ndarray:
    if size not in (16, 64):
        raise ValueError("Visual research features must be 16x16 or 64x64")
    with Image.open(io.BytesIO(png)) as image:
        frame = image.convert("RGB").resize((64, 64)).convert("L")
        values = (
            _world_embedding_from_frame(frame)
            if size == 16
            else np.asarray(frame, dtype=np.float32).reshape(-1) / 255.0
        )
    return np.asarray(values, dtype=np.float32)


def candidate_centers(page) -> list[tuple[float, float]]:
    """Find visible controls, then bounded visual/neutral probes when controls are sparse."""
    return [
        tuple(point)
        for point in page.locator("#wrap *").evaluate_all("""nodes => {
        const root = document.querySelector('#wrap');
        const semantic = 'button,input:not([type="hidden"]),select,a[href],'
            + '[role="button"],[role="tab"],[role="menuitem"],'
            + '[role="checkbox"],[role="radio"]';
        const candidates = [
            ...nodes.filter(node => node.matches(semantic)),
            ...nodes.filter(node => getComputedStyle(node).cursor === 'pointer'
                && !node.matches(semantic) && !node.closest(semantic)),
        ];
        const seen = new Set();
        const centers = [];
        const add = node => {
            const style = getComputedStyle(node);
            if (style.visibility !== 'visible' || style.display === 'none'
                    || style.pointerEvents === 'none') return;
            const rect = node.getBoundingClientRect();
            const x = Math.round((rect.left + rect.width / 2) * 1000) / 1000;
            const y = Math.round((rect.top + rect.height / 2) * 1000) / 1000;
            if (!rect.width || !rect.height || x < 0 || x >= 320 || y < 0 || y >= 360) return;
            const hit = document.elementFromPoint(x, y);
            if (hit !== node && !node.contains(hit)) return;
            const key = `${x}:${y}`;
            if (!seen.has(key)) { seen.add(key); centers.push([x, y]); }
        };
        for (const node of candidates) add(node);
        if (centers.length < 2 && root) {
            for (const node of nodes) {
                if (node.children.length) continue;
                const style = getComputedStyle(node);
                const painted = style.backgroundColor !== 'rgba(0, 0, 0, 0)'
                    || (parseFloat(style.borderTopWidth) > 0 && style.borderTopStyle !== 'none');
                if (painted) add(node);
            }
            const box = root.getBoundingClientRect();
            for (const fy of [0.125, 0.375, 0.625, 0.875]) {
                for (const fx of [0.125, 0.375, 0.625, 0.875]) {
                    const x = Math.round((box.left + box.width * fx) * 1000) / 1000;
                    const y = Math.round((box.top + box.height * fy) * 1000) / 1000;
                    if (x < 0 || x >= 320 || y < 0 || y >= 360) continue;
                    const hit = document.elementFromPoint(x, y);
                    if (!hit || !root.contains(hit) || hit.closest(semantic)
                            || getComputedStyle(hit).cursor === 'pointer') continue;
                    if (centers.some(point => Math.hypot(point[0] - x, point[1] - y) < 8)) continue;
                    const key = `${x}:${y}`;
                    if (!seen.has(key)) { seen.add(key); centers.push([x, y]); }
                    if (centers.length >= 16) return centers;
                }
            }
        }
        return centers;
    }""")
    ]


def visible_semantics(page, point: tuple[float, float] | None = None) -> dict[str, str]:
    """Capture bounded rendered text/control state, never HTML source."""
    result = page.evaluate("""point => {
        const root = document.querySelector('#wrap');
        const hit = point ? document.elementFromPoint(point[0], point[1]) : null;
        const control = hit?.closest('button,a,input,select,[role]') || hit;
        const items = [...(root?.querySelectorAll('*') || [])].filter(node => {
            const style = getComputedStyle(node);
            const box = node.getBoundingClientRect();
            return box.width && box.height && style.display !== 'none'
                && style.visibility === 'visible'
                && (node.matches('button,input:not([type="hidden"]),select,a,[role],[tabindex]')
                    || style.cursor === 'pointer');
        }).slice(0, 80).map(node => {
            const style = getComputedStyle(node);
            const box = node.getBoundingClientRect();
            const name = node.getAttribute('aria-label') || node.innerText
                || node.getAttribute('title') || '';
            return [node.tagName.toLowerCase(), name.replace(/\\s+/g, ' ').slice(0, 80),
                node.getAttribute('role') || '', Boolean(node.checked), Boolean(node.disabled),
                Boolean(node.selected), Math.round(box.x), Math.round(box.y),
                style.backgroundColor, style.color, style.borderColor].join('|');
        });
        return {
            text: root?.innerText || '',
            controls: items,
            target: control?.getAttribute('aria-label') || control?.innerText
                || control?.getAttribute('title') || control?.tagName || ''
        };
    }""", list(point) if point is not None else None)
    rendered = " ".join(result["text"].split())
    controls = "\n".join(result["controls"])
    return {"text": (rendered + "\nControls:\n" + controls)[:MAX_SEMANTIC_CHARS],
            "target": " ".join(result["target"].split())[:160]}


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
    candidates = candidate_centers(page)
    before_png = page.screenshot(animations="disabled")
    before_dom = page.locator("#wrap").inner_html()
    return context, page, candidates, before_png, before_dom


def collect_group(
    browser, assets: PinnedAssets, task: str, seed: str, max_candidates: int, feature_size: int = 16,
    include_semantics: bool = False,
) -> tuple[list[dict], int]:
    reference: tuple[list[tuple[float, float]], str, np.ndarray] | None = None
    rows = []
    attempted = 0
    for candidate_index in range(max_candidates):
        context, page, candidates, before_png, before_dom = _reset(browser, assets, task, seed)
        try:
            before_feature = visual_feature(before_png, feature_size)
            if reference is None:
                reference = (candidates, before_dom, before_feature)
            elif (
                reference[0] != candidates
                or reference[1] != before_dom
                or not np.array_equal(reference[2], before_feature)
            ):
                raise UnstableResetError("Independent MiniWoB branches did not reset to one exact measured state")
            if candidate_index >= len(candidates):
                break
            x, y = candidates[candidate_index]
            before_semantics = visible_semantics(page, (x, y)) if include_semantics else None
            page.mouse.click(x, y)
            outcome = page.evaluate("({done: WOB_DONE_GLOBAL, reward: WOB_RAW_REWARD_GLOBAL})")
            row = {"state": before_feature,
                   "target": visual_feature(page.screenshot(animations="disabled"), feature_size),
                   "coords": (x / VIEWPORT["width"], y / VIEWPORT["height"]),
                   "reward": float(outcome["reward"]) if outcome["done"] else None}
            if before_semantics is not None:
                row.update({"state_text": before_semantics["text"],
                            "action_label": before_semantics["target"],
                            "target_text": visible_semantics(page)["text"]})
            rows.append(row)
            attempted += 1
        finally:
            context.close()
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if not np.array_equal(rows[left]["target"], rows[right]["target"]):
                return [rows[left], rows[right]], attempted
    return [], attempted


def require_disjoint_prestates(rows: list[dict]) -> None:
    states = {
        split: {row["state"].tobytes() for row in rows if row["split"] == split}
        for split in ("train", "validation", "eval")
    }
    if any(
        states[left] & states[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Identical modeled pre-action state leaked across task splits")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--eval-tasks", required=True)
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--feature-size", type=int, choices=(16, 64), default=16)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--features", type=Path, default=root / "miniwob_branches_v1.npz")
    parser.add_argument("--report", type=Path, default=root / "miniwob_branches_v1.json")
    parser.add_argument("--semantic-archive", type=Path,
                        help="Opt-in local JSONL of bounded public-task visible text and actions")
    args = parser.parse_args()
    try:
        tasks = {split: parse_tasks(getattr(args, f"{split}_tasks")) for split in ("train", "validation", "eval")}
    except ValueError as exc:
        parser.error(str(exc))
    if (
        len(set(task for names in tasks.values() for task in names)) != sum(map(len, tasks.values()))
        or not 2 <= args.samples_per_task <= 100
        or not 0 <= args.seed_start <= 1000000
        or not 2 <= args.max_candidates <= 16
        or args.features.exists()
        or args.report.exists()
        or (args.semantic_archive is not None
            and (args.semantic_archive.exists() or args.semantic_archive.suffix != ".jsonl"
                 or args.semantic_archive.resolve() in (args.features.resolve(), args.report.resolve())))
    ):
        parser.error("Use task-disjoint splits, bounded settings, and fresh output paths")

    assets = PinnedAssets(SOURCE_COMMIT)
    rows = []
    attempted = Counter()
    skipped = Counter()
    unstable = Counter()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for split, names in tasks.items():
                for task in names:
                    for index in range(args.seed_start, args.seed_start + args.samples_per_task):
                        seed = f"{SOURCE_COMMIT}:{task}:{index}"
                        try:
                            pair, count = collect_group(
                                browser, assets, task, seed, args.max_candidates, args.feature_size,
                                include_semantics=args.semantic_archive is not None,
                            )
                        except UnstableResetError:
                            unstable[split] += 1
                            continue
                        attempted[split] += count
                        if not pair:
                            skipped[split] += 1
                            continue
                        group = hashlib.sha256(seed.encode("utf-8")).hexdigest()
                        rows.extend({**row, "group": group, "split": split, "task": task} for row in pair)
                    print(
                        json.dumps(
                            {
                                "task": task,
                                "split": split,
                                "groups": len({row["group"] for row in rows if row["task"] == task}),
                            }
                        ),
                        flush=True,
                    )
        finally:
            browser.close()
    if any(not any(row["split"] == split for row in rows) for split in tasks):
        raise ValueError("Each split needs at least one verified distinct-outcome branch")
    require_disjoint_prestates(rows)
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
    semantic_sha256 = None
    if args.semantic_archive is not None:
        args.semantic_archive.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps({key: row[key] for key in (
            "group", "split", "task", "coords", "reward", "state_text",
            "action_label", "target_text")}, ensure_ascii=False) + "\n" for row in rows)
        args.semantic_archive.write_text(content, encoding="utf-8")
        semantic_sha256 = hashlib.sha256(args.semantic_archive.read_bytes()).hexdigest()
    report = {
        "purpose": "research_only_miniwob_visual_branch_transitions",
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_license": "MIT",
        "asset_sha256": assets.hashes(),
        "tasks": tasks,
        "samples_per_task": args.samples_per_task,
        "seed_start": args.seed_start,
        "feature_shape": [1, args.feature_size, args.feature_size],
        "max_candidates": args.max_candidates,
        "selection": f"first two candidate clicks with distinct observed {args.feature_size}x{args.feature_size} next-screen features",
        "reset_identity": f"identical visible control coordinates, task DOM, and {args.feature_size}x{args.feature_size} pre-action features; "
        "full screenshot bytes may differ by rendering noise",
        "counterfactual_pairs": True,
        "split_basis": "task_family",
        "cross_split_identical_prestates": 0,
        "rows": dict(Counter(row["split"] for row in rows)),
        "groups": {split: len({row["group"] for row in rows if row["split"] == split}) for split in tasks},
        "candidate_attempts": dict(attempted),
        "skipped_no_distinct_visual_outcomes": dict(skipped),
        "skipped_unstable_reset": dict(unstable),
        "reward_observed": {
            split: dict(Counter(str(row["reward"]) for row in rows if row["split"] == split)) for split in tasks
        },
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "semantic_archive_sha256": semantic_sha256,
        "limitation": "Synthetic browser tasks, selected for visually distinct observed outcomes. "
        "The optional visible-text/control archive is local research data, not a trained model. "
        "Reward labels are counted but not trained on. No real desktop transfer, Jev parity, "
        "autonomy, or execution authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

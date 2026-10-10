"""Collect site-disjoint, within-app visual branches from public InfiniteWeb apps.

Generated sites run in isolated headless contexts with external requests blocked.
Only bounded 16x16 features and provenance are saved; HTML and screenshots stay
in memory. This is synthetic research data, never a live Heliox model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlsplit

import numpy as np

from scripts.collect_miniwob_branches import require_disjoint_prestates, visual_feature

DATASET = "WhiteFlamesCN/InfiniteWeb-Dataset"
REVISION = "84e294fc432abca78b89388ed3c98ca778177a1f"
ORIGIN = "https://infiniteweb.local"
VIEWPORT = {"width": 1024, "height": 768}
MAX_ASSET_BYTES = 2 * 1024 * 1024
MAX_SITE_BYTES = 8 * 1024 * 1024
MAX_SEMANTIC_CHARS = 8192


def parse_sites(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if (
        not names
        or len(names) != len(set(names))
        or any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in names)
    ):
        raise ValueError("Sites must be unique comma-separated dataset directory names")
    return names


def asset_path(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path.removeprefix("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc != "infiniteweb.local"
        or "%" in path
        or ".." in path.split("/")
        or re.fullmatch(r"[A-Za-z0-9_./-]+", path) is None
    ):
        raise ValueError("Only local, dataset-relative assets are allowed")
    return path


class PinnedSiteAssets:
    def __init__(self) -> None:
        self._manifest: dict[str, int] = {}
        self._cache: dict[str, bytes] = {}

    def add_site(self, site: str) -> list[str]:
        if re.fullmatch(r"[A-Za-z0-9_-]+", site) is None:
            raise ValueError("Invalid dataset directory name")
        url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{quote(site)}?recursive=false"
        with urllib.request.urlopen(url, timeout=25) as response:
            items = json.load(response)
        manifest = {
            item["path"]: item["size"]
            for item in items
            if item.get("type") == "file" and isinstance(item.get("size"), int)
        }
        if (
            not manifest
            or f"{site}/index.html" not in manifest
            or any(
                not path.startswith(f"{site}/") or size < 0 or size > MAX_ASSET_BYTES for path, size in manifest.items()
            )
            or sum(manifest.values()) > MAX_SITE_BYTES
        ):
            raise ValueError(f"Site lacks a bounded, complete index manifest: {site}")
        self._manifest.update(manifest)
        return [path for path in manifest if path.endswith(".html")]

    def fetch(self, path: str) -> bytes:
        if path not in self._manifest:
            raise ValueError("Asset is absent from pinned site manifest")
        if path not in self._cache:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{quote(path)}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(MAX_ASSET_BYTES + 1)
            if len(data) != self._manifest[path]:
                raise ValueError(f"Pinned asset size mismatch: {path}")
            self._cache[path] = data
        return self._cache[path]

    def route(self, route) -> None:
        try:
            path = asset_path(route.request.url)
            data = self.fetch(path)
        except (OSError, ValueError):
            route.abort()
            return
        route.fulfill(body=data, content_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

    def hashes(self) -> dict[str, str]:
        return {path: hashlib.sha256(data).hexdigest() for path, data in sorted(self._cache.items())}


def candidate_centers(page, limit: int) -> list[tuple[float, float]]:
    return [
        tuple(point)
        for point in page.evaluate(
            """limit => {
        const selector = 'a[href],button,input:not([type="hidden"]),select,'
            + '[role="button"],[role="tab"],[role="checkbox"],[role="radio"]';
        const points = [], seen = new Set();
        for (const node of document.querySelectorAll(selector)) {
            const style = getComputedStyle(node), box = node.getBoundingClientRect();
            if (style.display === 'none' || style.visibility !== 'visible'
                    || style.pointerEvents === 'none' || !box.width || !box.height) continue;
            const x = Math.round(box.x + box.width / 2), y = Math.round(box.y + box.height / 2);
            if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
            const hit = document.elementFromPoint(x, y);
            if (hit !== node && !node.contains(hit)) continue;
            const key = `${x}:${y}`;
            if (!seen.has(key)) { seen.add(key); points.push([x, y]); }
            if (points.length >= limit) break;
        }
        return points;
    }""",
            limit,
        )
    ]


def semantic_state(page) -> str:
    """Compare visible text, control values, and path, not focus or CSS changes."""
    result = page.evaluate("""() => ({
        path: location.pathname + location.search,
        scroll: [Math.round(scrollX), Math.round(scrollY)],
        text: (document.body?.innerText || '').replace(/\\s+/g, ' ').slice(0, 7000),
        controls: [...document.querySelectorAll('input,select,[role="checkbox"],[role="radio"],'
                + '[aria-pressed],[aria-expanded]')]
            .slice(0, 60).map(node => [node.tagName, node.type || '',
                Boolean(node.checked), node.getAttribute('aria-checked') || '',
                node.getAttribute('aria-pressed') || '', node.getAttribute('aria-expanded') || '',
                node.tagName === 'SELECT' ? node.selectedIndex : ''])
    })""")
    return json.dumps(result, sort_keys=True)[:MAX_SEMANTIC_CHARS]


def action_label(page, point: tuple[float, float]) -> str:
    label = page.evaluate(
        """([x, y]) => {
        const hit = document.elementFromPoint(x, y);
        const control = hit?.closest('a,button,input,select,[role]') || hit;
        return control?.getAttribute('aria-label') || control?.getAttribute('title')
            || control?.innerText || control?.getAttribute('value') || '';
    }""",
        list(point),
    )
    return " ".join(str(label).split())[:160]


def reset(browser, assets: PinnedSiteAssets, path: str, limit: int):
    context = browser.new_context(
        viewport=VIEWPORT, device_scale_factor=1, service_workers="block", accept_downloads=False
    )
    context.route("**/*", assets.route)
    page = context.new_page()
    response = page.goto(f"{ORIGIN}/{path}", wait_until="load", timeout=30000)
    if response is None or response.status != 200:
        context.close()
        raise ValueError(f"Site page did not load: {path}")
    page.wait_for_timeout(100)
    candidates = candidate_centers(page, limit)
    feature = visual_feature(page.screenshot(animations="disabled"), 16)
    return context, page, candidates, feature, semantic_state(page)


def collect_page(browser, assets: PinnedSiteAssets, path: str, limit: int) -> tuple[list[dict], int]:
    from playwright.sync_api import Error as PlaywrightError

    reference = None
    rows: list[dict] = []
    attempts = 0
    for index in range(limit):
        try:
            context, page, candidates, before, before_semantic = reset(browser, assets, path, limit)
        except (OSError, ValueError, PlaywrightError):
            return [], attempts
        try:
            signature = (candidates, before_semantic, before.tobytes())
            if reference is None:
                reference = signature
            elif signature != reference:
                return [], attempts
            if index >= len(candidates):
                break
            x, y = candidates[index]
            try:
                label = action_label(page, (x, y))
                page.mouse.click(x, y)
                page.wait_for_load_state("load", timeout=3000)
                page.wait_for_timeout(150)
                attempts += 1
                if (
                    len(context.pages) != 1
                    or urlsplit(page.url).netloc != "infiniteweb.local"
                    or not page.locator("body").count()
                ):
                    continue
                after_semantic = semantic_state(page)
                if after_semantic == before_semantic or not page.locator("body").inner_text().strip():
                    continue
                after = visual_feature(page.screenshot(animations="disabled"), 16)
                if np.array_equal(before, after):
                    continue
                rows.append(
                    {
                        "state": before,
                        "target": after,
                        "coords": (x / VIEWPORT["width"], y / VIEWPORT["height"]),
                        "state_text": before_semantic[:2048],
                        "action_label": label,
                        "target_text": after_semantic[:2048],
                        "semantic": after_semantic,
                    }
                )
            except (OSError, ValueError, PlaywrightError):
                continue
        finally:
            context.close()
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left]["semantic"] != rows[right]["semantic"]:
                return [rows[left], rows[right]], attempts
    return [], attempts


def main() -> None:
    # Optional dependency (pip install -e ".[browser]"); imported here so the helpers stay importable.
    from playwright.sync_api import sync_playwright

    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "eval"):
        parser.add_argument(f"--{split}-sites", required=True)
    parser.add_argument("--max-pages-per-site", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--features", type=Path, default=root / "infiniteweb_branches_v1.npz")
    parser.add_argument("--report", type=Path, default=root / "infiniteweb_branches_v1.json")
    parser.add_argument(
        "--semantic-archive", type=Path, help="Opt-in local JSONL of bounded visible text on public generated sites"
    )
    args = parser.parse_args()
    try:
        sites = {split: parse_sites(getattr(args, f"{split}_sites")) for split in ("train", "validation", "eval")}
    except ValueError as exc:
        parser.error(str(exc))
    flat = [site for names in sites.values() for site in names]
    if (
        len(flat) != len(set(flat))
        or not 1 <= args.max_pages_per_site <= 20
        or not 2 <= args.max_candidates <= 20
        or args.features.exists()
        or args.report.exists()
        or args.features.resolve() == args.report.resolve()
        or (
            args.semantic_archive is not None
            and (
                args.semantic_archive.exists()
                or args.semantic_archive.suffix != ".jsonl"
                or args.semantic_archive.resolve() in (args.features.resolve(), args.report.resolve())
            )
        )
    ):
        parser.error("Use site-disjoint splits, bounded limits, and fresh output paths")

    assets = PinnedSiteAssets()
    pages = {site: assets.add_site(site) for site in flat}
    rows: list[dict] = []
    attempts = Counter()
    skipped = Counter()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for split, names in sites.items():
                for site in names:
                    paths = sorted(pages[site], key=lambda path: (not path.endswith("/index.html"), path))
                    for path in paths[: args.max_pages_per_site]:
                        pair, count = collect_page(browser, assets, path, args.max_candidates)
                        attempts[split] += count
                        if not pair:
                            skipped[split] += 1
                            continue
                        group = hashlib.sha256(path.encode()).hexdigest()
                        rows.extend({**row, "group": group, "split": split, "task": path} for row in pair)
                    print(
                        json.dumps(
                            {
                                "site": site,
                                "split": split,
                                "groups": len({row["group"] for row in rows if row["task"].startswith(site + "/")}),
                            }
                        ),
                        flush=True,
                    )
        finally:
            browser.close()
    if any(not any(row["split"] == split for row in rows) for split in sites):
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
        content = "".join(
            json.dumps(
                {
                    key: row[key]
                    for key in ("group", "split", "task", "coords", "state_text", "action_label", "target_text")
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
        args.semantic_archive.write_text(content, encoding="utf-8")
        semantic_sha256 = hashlib.sha256(args.semantic_archive.read_bytes()).hexdigest()
    report = {
        "purpose": "research_only_infiniteweb_visual_branch_transitions",
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_license": "MIT",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "asset_sha256": assets.hashes(),
        "sites": sites,
        "feature_shape": [1, 16, 16],
        "max_pages_per_site": args.max_pages_per_site,
        "max_candidates": args.max_candidates,
        "split_basis": "site_directory",
        "counterfactual_pairs": True,
        "within_app_only": True,
        "task_terminal_labels_known": False,
        "rows": dict(Counter(row["split"] for row in rows)),
        "groups": {split: len({row["group"] for row in rows if row["split"] == split}) for split in sites},
        "candidate_attempts": dict(attempts),
        "skipped_pages": dict(skipped),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "semantic_archive_sha256": semantic_sha256,
        "limitation": "Generated browser websites, not a desktop task distribution. Candidate clicks are selected "
        "for semantically and visually distinct observed outcomes. No reward, task success, or task terminal "
        "state was measured. "
        "No Jev parity, real-desktop transfer, autonomous authority, or runtime model promotion.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

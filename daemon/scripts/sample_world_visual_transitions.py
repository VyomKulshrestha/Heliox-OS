"""Bounded research sample of adjacent observed desktop screenshots.

Raw screenshots stay in memory. The ignored output stores only the same
16x16 grayscale feature used by Heliox ScreenVisionAgent, action metadata,
episode hashes, and split labels. No source action is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from pilot.agents.screen_vision import _world_embedding_from_frame

DATASET = "TESS-Computer/tess-agentnet"
REVISION = "c88c33e0ec473904e6e15b3c83179e47aa4e79ea"
API = "https://datasets-server.huggingface.co/rows"


def _json_from(url: str) -> dict:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _page(offset: int, length: int) -> dict:
    query = urllib.parse.urlencode({
        "dataset": DATASET, "config": "default", "split": "train",
        "offset": offset, "length": length,
    })
    return _json_from(f"{API}?{query}")


def _image_feature(entry: dict, revision: str) -> np.ndarray | None:
    image = entry.get("row", {}).get("image")
    source = image.get("src") if isinstance(image, dict) else None
    parsed = urllib.parse.urlparse(source) if isinstance(source, str) else None
    if (parsed is None or parsed.scheme != "https"
            or parsed.hostname != "datasets-server.huggingface.co"
            or f"/{revision}/" not in parsed.path):
        raise ValueError("Image URL is not bound to the pinned dataset revision")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(source, timeout=15) as response:
                image_bytes = response.read(4 * 1024 * 1024 + 1)
            if len(image_bytes) > 4 * 1024 * 1024:
                raise ValueError("Screenshot exceeds the bounded decode limit")
            with Image.open(io.BytesIO(image_bytes)) as screenshot:
                if screenshot.width * screenshot.height > 20_000_000:
                    raise ValueError("Screenshot dimensions exceed the bounded decode limit")
                frame = np.asarray(screenshot.convert("RGB").resize((64, 64)).convert("L"), dtype=np.float32)
            return np.asarray(_world_embedding_from_frame(frame), dtype=np.float32)
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def _pair(current: dict, following: dict) -> tuple[dict | None, str]:
    first, second = current.get("row"), following.get("row")
    if (not isinstance(first, dict) or not isinstance(second, dict)
            or current.get("truncated_cells") or following.get("truncated_cells")):
        return None, "invalid_row"
    episode = first.get("episode_id")
    if not isinstance(episode, str) or not episode or episode != second.get("episode_id"):
        return None, "different_episode"
    step, next_step = first.get("step_idx"), second.get("step_idx")
    if (not isinstance(step, int) or isinstance(step, bool)
            or not isinstance(next_step, int) or next_step != step + 1
            or following.get("row_idx") != current.get("row_idx", -2) + 1):
        return None, "nonadjacent_step"
    action, click = first.get("action_type"), first.get("click_type")
    x, y = first.get("mouse_x"), first.get("mouse_y")
    if action != 0 or not isinstance(click, int) or isinstance(click, bool) or not 0 <= click <= 8:
        return None, "unsupported_action"
    if (not isinstance(x, (int, float)) or not isinstance(y, (int, float))
            or not np.isfinite(x) or not np.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1):
        return None, "invalid_coordinates"
    return {
        "group": hashlib.sha256(episode.encode("utf-8")).hexdigest(),
        "row_index": current["row_idx"],
        "action": click, "coords": [float(x), float(y)],
    }, "accepted"


def _splits(groups: set[str], seed: int) -> dict[str, str]:
    ordered = sorted(groups)
    if len(ordered) < 3:
        raise ValueError("At least three independent episodes are required")
    random.Random(seed).shuffle(ordered)
    train_end = min(len(ordered) - 2, max(1, int(len(ordered) * 0.7)))
    validation_end = min(len(ordered) - 1, max(train_end + 1, int(len(ordered) * 0.85)))
    return {group: ("train" if index < train_end else
                    "validation" if index < validation_end else "eval")
            for index, group in enumerate(ordered)}


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--page-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-revision", default=REVISION)
    parser.add_argument("--features", type=Path, default=state / "tess_visual_transitions_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "tess_visual_transitions_v1.json")
    args = parser.parse_args()
    if (args.features.exists() or args.report.exists() or args.pages < 1
            or not 2 <= args.page_length <= 100 or not 1 <= args.workers <= 16
            or len(args.expected_revision) != 40
            or any(char not in "0123456789abcdef" for char in args.expected_revision)):
        parser.error("Use fresh outputs, bounded sampling, and a pinned revision")
    total = _page(0, 1).get("num_rows_total")
    if not isinstance(total, int) or total < args.pages * args.page_length:
        raise ValueError("Dataset is smaller than the requested sampling range")
    rng = random.Random(args.seed)
    records: list[dict] = []
    exclusions: Counter = Counter()
    offsets: list[int] = []
    for index in range(args.pages):
        lower = index * total // args.pages
        upper = (index + 1) * total // args.pages - args.page_length
        offset = rng.randint(lower, upper)
        offsets.append(offset)
        page = _page(offset, args.page_length)
        rows = page.get("rows")
        if (page.get("num_rows_total") != total or not isinstance(rows, list)
                or len(rows) != args.page_length
                or any(row.get("row_idx") != offset + position for position, row in enumerate(rows))):
            raise ValueError("Incomplete or changed source page")
        accepted = []
        for current, following in zip(rows, rows[1:]):
            pair, reason = _pair(current, following)
            exclusions[reason] += 1
            if pair is not None:
                accepted.append((pair, current, following))
        needed = {entry["row_idx"]: entry for _, current, following in accepted
                  for entry in (current, following)}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            images = dict(zip(needed, pool.map(
                lambda entry: _image_feature(entry, args.expected_revision), needed.values()
            )))
        for pair, current, following in accepted:
            source_image, target_image = images[current["row_idx"]], images[following["row_idx"]]
            if source_image is None or target_image is None:
                exclusions["image_fetch_error"] += 1
                continue
            pair["state"] = source_image
            pair["target"] = target_image
            records.append(pair)
        print(json.dumps({"page": index + 1, "accepted": len(accepted)}), flush=True)
    if not records:
        raise ValueError("No complete visual transitions were sampled")
    assignments = _splits({row["group"] for row in records}, args.seed)
    for row in records:
        row["split"] = assignments[row["group"]]
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        state=np.stack([row["state"] for row in records]),
        target=np.stack([row["target"] for row in records]),
        action=np.array([row["action"] for row in records], dtype=np.int16),
        coords=np.array([row["coords"] for row in records], dtype=np.float32),
        group=np.array([row["group"] for row in records]),
        split=np.array([row["split"] for row in records]),
        row_index=np.array([row["row_index"] for row in records]),
    )
    report = {
        "purpose": "research_only_adjacent_visual_transition_sample",
        "source": {"dataset": DATASET, "revision": args.expected_revision,
                   "declared_license_on_reviewed_card": "mit", "rows": total},
        "sampling": {"seed": args.seed, "pages": args.pages,
                     "page_length": args.page_length, "offsets": offsets},
        "accepted": dict(Counter(row["split"] for row in records)),
        "groups": dict(Counter(assignments.values())),
        "exclusions": dict(exclusions),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "limitation": "Low-resolution observed screenshots and mouse actions only. "
                      "The rows API is not revision-pinned; image assets are checked against "
                      "the expected revision and the feature file is hashed. No raw image "
                      "retention, execution, safety claim, or live Heliox validation.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

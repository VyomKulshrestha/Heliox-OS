"""Import bounded, explicitly ordered GUI screenshot/action/screenshot pairs.

Research-only: reads a local JSONL and image ZIP without extracting screenshots.
The output contains only 16x16 grayscale features and mouse metadata. A next
observation is not proof that the action alone caused the screen change.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pilot.agents.screen_vision import _world_embedding_from_frame
from scripts.sample_world_visual_transitions import _splits

ACTION_IDS = {"move": 0, "mouse_move": 0, "left_click": 1, "double_click": 2, "right_click": 3}
ACTION_PATTERN = re.compile(r"<parameter=action>\s*([a-z_]+)\s*</parameter>")
COORD_PATTERN = re.compile(r"<parameter=coordinate>\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*</parameter>")
STEP_PATTERN = re.compile(r"images/([a-zA-Z0-9-]{1,80})/step_([1-9][0-9]*)\.png")
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 20_000_000


def _observation(message: dict, trajectory_id: str) -> tuple[str, int] | None:
    if message.get("role") != "user" or not isinstance(message.get("content"), list):
        return None
    urls = [
        part.get("image_url", {}).get("url")
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part.get("image_url"), dict)
    ]
    if len(urls) != 1 or not isinstance(urls[0], str):
        return None
    match = STEP_PATTERN.fullmatch(urls[0])
    if match is None or match.group(1) != trajectory_id:
        return None
    return urls[0], int(match.group(2))


def _action(message: dict) -> tuple[int, int, int] | None:
    if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        return None
    content = message["content"]
    if content.count("<function=") != 1 or content.count("<function=computer_use>") != 1:
        return None
    actions, coords = ACTION_PATTERN.findall(content), COORD_PATTERN.findall(content)
    if len(actions) != 1 or len(coords) != 1 or actions[0] not in ACTION_IDS:
        return None
    return ACTION_IDS[actions[0]], int(coords[0][0]), int(coords[0][1])


def _pairs(record: dict) -> tuple[list[tuple[str, str, int, int, int]], Counter]:
    trajectory_id, messages = record.get("trajectory_id"), record.get("messages")
    if (
        not isinstance(trajectory_id, str)
        or not re.fullmatch(r"[a-zA-Z0-9-]{1,80}", trajectory_id)
        or not isinstance(messages, list)
    ):
        return [], Counter({"invalid_trajectory": 1})
    accepted: list[tuple[str, str, int, int, int]] = []
    reasons: Counter = Counter()
    for current, command, following in zip(messages, messages[1:], messages[2:]):
        before = _observation(current, trajectory_id) if isinstance(current, dict) else None
        if before is None:
            continue
        action = _action(command) if isinstance(command, dict) else None
        after = _observation(following, trajectory_id) if isinstance(following, dict) else None
        if action is None or after is None or after[1] != before[1] + 1:
            reasons["missing_ordered_action_or_next_observation"] += 1
            continue
        accepted.append((before[0], after[0], *action))
    return accepted, reasons


def _feature(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, int, int, str]:
    info = archive.getinfo(member)
    if info.file_size > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the bounded decode limit")
    with archive.open(info) as stream:
        content = stream.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the bounded decode limit")
    with Image.open(io.BytesIO(content)) as screenshot:
        width, height = screenshot.size
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError("Image dimensions exceed the bounded decode limit")
        frame = np.asarray(screenshot.convert("RGB").resize((64, 64)).convert("L"), dtype=np.float32)
    return (
        np.asarray(_world_embedding_from_frame(frame), dtype=np.float32),
        width,
        height,
        hashlib.sha256(content).hexdigest(),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_pairs(jsonl: Path, image_zip: Path, *, max_trajectories: int, max_pairs: int) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    reasons: Counter = Counter()
    with jsonl.open("r", encoding="utf-8") as source, zipfile.ZipFile(image_zip) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate image paths make trajectory alignment ambiguous")
        for trajectory_index, line in enumerate(source):
            if trajectory_index >= max_trajectories or len(rows) >= max_pairs:
                break
            if len(line) > 2 * 1024 * 1024:
                reasons["oversize_trajectory"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                reasons["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                reasons["invalid_trajectory"] += 1
                continue
            if not isinstance(record.get("result"), dict) or record["result"].get("safety") is not True:
                reasons["unsafe_or_unlabeled_trajectory"] += 1
                continue
            task = record.get("task")
            if (
                not isinstance(task, dict)
                or not isinstance(task.get("domain"), str)
                or not isinstance(task.get("description"), str)
                or not task["description"].strip()
            ):
                reasons["missing_task_identity"] += 1
                continue
            pairs, rejected = _pairs(record)
            reasons.update(rejected)
            task_key = f"{task['domain'].strip().casefold()}\0{task['description'].strip().casefold()}"
            group = hashlib.sha256(task_key.encode("utf-8")).hexdigest()
            episode = hashlib.sha256(record["trajectory_id"].encode("utf-8")).hexdigest() if pairs else ""
            cache: dict[str, tuple[np.ndarray, int, int, str]] = {}
            for before, after, action, x, y in pairs:
                if len(rows) >= max_pairs:
                    break
                try:
                    for member in (before, after):
                        if member not in cache:
                            cache[member] = _feature(archive, member)
                    state, width, height, before_hash = cache[before]
                    target, next_width, next_height, after_hash = cache[after]
                except (KeyError, OSError, ValueError, zipfile.BadZipFile):
                    reasons["missing_or_invalid_image"] += 1
                    continue
                if (width, height) != (next_width, next_height) or not (0 <= x < width and 0 <= y < height):
                    reasons["invalid_coordinate_or_size"] += 1
                    continue
                rows.append(
                    {
                        "group": group,
                        "episode": episode,
                        "state": state,
                        "target": target,
                        "action": action,
                        "coords": [x / width, y / height],
                        "image_hashes": (before_hash, after_hash),
                    }
                )
    return rows, reasons


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Local screenshot ZIP")
    parser.add_argument("--features", type=Path, default=state / "aligned_visual_transitions.npz")
    parser.add_argument("--report", type=Path, default=state / "aligned_visual_transitions.json")
    parser.add_argument("--max-trajectories", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if (
        not args.jsonl.is_file()
        or not args.images.is_file()
        or args.features.exists()
        or args.report.exists()
        or args.features.resolve() == args.report.resolve()
        or not 3 <= args.max_trajectories <= 1000
        or not 3 <= args.max_pairs <= 10000
    ):
        parser.error("Supply local source files, bounded limits, and fresh output paths")
    rows, reasons = import_pairs(
        args.jsonl, args.images, max_trajectories=args.max_trajectories, max_pairs=args.max_pairs
    )
    assignments = _splits({row["group"] for row in rows}, args.seed)
    for row in rows:
        row["split"] = assignments[row["group"]]
    args.features.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        state=np.stack([row["state"] for row in rows]),
        target=np.stack([row["target"] for row in rows]),
        action=np.asarray([row["action"] for row in rows], dtype=np.int16),
        coords=np.asarray([row["coords"] for row in rows], dtype=np.float32),
        group=np.asarray([row["group"] for row in rows]),
        episode=np.asarray([row["episode"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
    )
    report = {
        "purpose": "research_only_ordered_gui_trajectory_import",
        "source_jsonl_sha256": _file_sha256(args.jsonl),
        "source_images": str(args.images),
        "sampling": {"max_trajectories": args.max_trajectories, "max_pairs": args.max_pairs, "seed": args.seed},
        "accepted": dict(Counter(row["split"] for row in rows)),
        "groups": dict(Counter(assignments.values())),
        "split_basis": "exact_casefolded_task_domain_and_description",
        "source_filter": "result.safety_is_true",
        "distinct_trajectories": len({row["episode"] for row in rows}),
        "sampled_images_sha256": hashlib.sha256(
            "".join(image_hash for row in rows for image_hash in row["image_hashes"]).encode("ascii")
        ).hexdigest(),
        "exclusions": dict(reasons),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "limitation": "Ordered observations are not independently verified action effects. "
        "16x16 grayscale and mouse actions only; no raw screenshot retention, "
        "safety, transfer, Jev parity, or runtime loading claim.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Import pinned, nonterminal GUI-360 Office clicks for offline world-model tests.

Only derived screen features leave the local, ignored parquet directory. This
does not train or authorize the live predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pilot.agents.screen_vision import _world_embedding_from_frame
from scripts.import_aligned_world_trajectories import MAX_IMAGE_BYTES, MAX_PIXELS, _file_sha256

MAX_SHARD_BYTES = 128 * 1024 * 1024
MAX_IMAGES_PER_TRAJECTORY = 64
MAX_MESSAGES_PER_TRAJECTORY = 192


def _single_image_index(message: dict[str, Any]) -> int | None:
    content = message.get("content")
    if not isinstance(content, list):
        return None
    indices = [item.get("index") for item in content
               if isinstance(item, dict) and item.get("type") == "image"]
    if len(indices) != 1 or not isinstance(indices[0], int) or isinstance(indices[0], bool):
        return None
    return indices[0]


def _feature(entry: dict[str, Any], representation: str) -> tuple[np.ndarray, tuple[int, int], str]:
    content = entry.get("bytes")
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_IMAGE_BYTES:
        raise ValueError("Missing or oversize embedded screenshot")
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
        if image.format not in {"PNG", "JPEG"} or width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError("Invalid screenshot format or dimensions")
        if representation == "gray16":
            frame = np.asarray(image.convert("RGB").resize((64, 64)).convert("L"), dtype=np.float32)
            values = np.asarray(_world_embedding_from_frame(frame), dtype=np.float32)
        elif representation == "rgb32":
            frame = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32)
            values = (frame.transpose(2, 0, 1) / 255.0).reshape(-1)
        else:
            raise ValueError("Unsupported representation")
    return values, (width, height), hashlib.sha256(content).hexdigest()


def _click(call: dict[str, Any]) -> tuple[float, float] | None:
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "computer":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str) and len(arguments) <= 10000:
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    actions = arguments.get("actions") if isinstance(arguments, dict) else None
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        return None
    action = actions[0]
    coords = action.get("coordinate")
    if (action.get("action") != "click" or not isinstance(coords, list) or len(coords) != 2
        or any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not np.isfinite(value) or not 0 <= value <= 1000 for value in coords)):
        return None
    return coords[0] / 1000, coords[1] / 1000


def extract_transitions(
    row: dict[str, Any], *, split: str, expected_domain: str,
    representation: str, reasons: Counter,
) -> list[dict[str, Any]]:
    """Trust image order and one observed tool result, never inferred UI effects."""
    try:
        metadata = json.loads(row["metadata"])
        messages = json.loads(row["messages"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed GUI-360 row metadata or messages") from exc
    if not isinstance(metadata, dict) or not isinstance(messages, list):
        raise ValueError("Invalid GUI-360 row structure")
    info = metadata.get("others")
    trajectory_id = info.get("id") if isinstance(info, dict) else None
    if (metadata.get("dims") != ["desktop", "use"] or not isinstance(trajectory_id, str)
        or not trajectory_id.startswith(expected_domain + "_")
        or info.get("source") != "vyokky/GUI-360"):
        raise ValueError("Shard domain or dataset lineage differs from the frozen protocol")
    images = row.get("images")
    if (not isinstance(images, list) or not 2 <= len(images) <= MAX_IMAGES_PER_TRAJECTORY
        or not 3 <= len(messages) <= MAX_MESSAGES_PER_TRAJECTORY):
        reasons["invalid_trajectory_size"] += 1
        return []
    cache: dict[int, tuple[np.ndarray, tuple[int, int], str]] = {}
    rows: list[dict[str, Any]] = []
    current_image: int | None = None
    group = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Malformed trajectory message")
        if message.get("role") in {"user", "tool"}:
            observed = _single_image_index(message)
            if observed is not None:
                current_image = observed
            continue
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            continue
        if not isinstance(calls, list) or len(calls) != 1 or current_image is None:
            reasons["ambiguous_action_or_before"] += 1
            continue
        coords = _click(calls[0]) if isinstance(calls[0], dict) else None
        if coords is None:
            reasons["unsupported_action"] += 1
            continue
        following = messages[position + 1] if position + 1 < len(messages) else None
        if (not isinstance(following, dict) or following.get("role") != "tool"
            or following.get("tool_call_id") != calls[0].get("id")):
            reasons["unpaired_tool_result"] += 1
            continue
        after_index = _single_image_index(following)
        if (after_index is None or after_index == current_image
            or not 0 <= current_image < len(images) or not 0 <= after_index < len(images)):
            reasons["missing_or_reused_screenshot"] += 1
            continue
        if not any(isinstance(later, dict) and later.get("role") == "assistant"
                   and later.get("tool_calls") for later in messages[position + 2:]):
            reasons["terminal_transition"] += 1
            continue
        try:
            for index in (current_image, after_index):
                if index not in cache:
                    cache[index] = _feature(images[index], representation)
            before, before_size, before_hash = cache[current_image]
            after, after_size, after_hash = cache[after_index]
        except (OSError, ValueError, TypeError):
            reasons["invalid_screenshot"] += 1
            continue
        if before_size != after_size:
            reasons["screen_size_changed"] += 1
            continue
        rows.append({"state": before, "target": after, "action": 1,
                     "coords": coords, "group": group, "split": split,
                     "image_hashes": (before_hash, after_hash)})
    return rows


def import_shards(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Offline GUI-360 import requires the optional pyarrow package") from exc
    shards = protocol.get("shards")
    if (protocol.get("dataset") != "cua-lite/GUI-360"
        or protocol.get("representation") not in {"gray16", "rgb32"}
        or not isinstance(shards, list) or not shards):
        raise ValueError("Invalid GUI-360 protocol")
    assignments = {"train": set(), "validation": set(), "eval": set()}
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("split") not in assignments:
            raise ValueError("Invalid shard assignment")
        domain = shard.get("expected_domain")
        if not isinstance(domain, str) or not domain:
            raise ValueError("Missing expected application domain")
        assignments[shard["split"]].add(domain)
    if (any(not domains for domains in assignments.values())
        or any(assignments[left] & assignments[right] for left, right in
               (("train", "validation"), ("train", "eval"), ("validation", "eval")))):
        raise ValueError("Application domains must be disjoint across splits")
    rows: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    source_rows: Counter = Counter()
    seen_ids: set[str] = set()
    for shard in shards:
        name = Path(shard["path"]).name
        path = root / name
        if (not path.is_file() or path.stat().st_size > MAX_SHARD_BYTES
            or _file_sha256(path) != shard.get("sha256")):
            raise ValueError(f"Missing, oversize, or changed pinned shard: {name}")
        source = parquet.ParquetFile(path)
        if set(source.schema_arrow.names) != {"images", "messages", "metadata"}:
            raise ValueError(f"Unexpected parquet schema: {name}")
        for batch in source.iter_batches(batch_size=8):
            for record in batch.to_pylist():
                metadata = json.loads(record["metadata"])
                info = metadata.get("others") if isinstance(metadata, dict) else None
                trajectory_id = info.get("id") if isinstance(info, dict) else None
                if not isinstance(trajectory_id, str) or trajectory_id in seen_ids:
                    raise ValueError("Duplicate or missing trajectory ID")
                seen_ids.add(trajectory_id)
                source_rows[shard["split"]] += 1
                rows.extend(extract_transitions(
                    record, split=shard["split"], expected_domain=shard["expected_domain"],
                    representation=protocol["representation"], reasons=reasons,
                ))
    counts = Counter(row["split"] for row in rows)
    minimum = protocol.get("minimum_rows_per_split", 20)
    if any(counts[split] < minimum for split in assignments):
        raise ValueError(f"Too few aligned nonterminal transitions: {dict(counts)}; exclusions: {dict(reasons)}")
    screens = {split: {row["state"].tobytes() for row in rows if row["split"] == split}
               for split in assignments}
    if any(screens[left] & screens[right] for left, right in
           (("train", "validation"), ("train", "eval"), ("validation", "eval"))):
        raise ValueError("Identical pre-action screen leaked across application splits")
    return rows, {"source_trajectories": dict(source_rows), "accepted": dict(counts),
                  "exclusions": dict(reasons),
                  "groups": {split: len({row["group"] for row in rows if row["split"] == split})
                             for split in assignments}}


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=state / "gui360_office_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "gui360_office_v1.json")
    args = parser.parse_args()
    if (not args.root.is_dir() or not args.protocol.is_file() or args.features.exists()
        or args.report.exists() or args.features.resolve() == args.report.resolve()):
        parser.error("Use a local shard directory, frozen protocol, and fresh output paths")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows, audit = import_shards(args.root, protocol)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        state=np.stack([row["state"] for row in rows]),
        target=np.stack([row["target"] for row in rows]),
        action=np.asarray([row["action"] for row in rows], dtype=np.int16),
        coords=np.asarray([row["coords"] for row in rows], dtype=np.float32),
        group=np.asarray([row["group"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
    )
    representation = protocol["representation"]
    report = {
        "purpose": "offline_only_gui360_office_transition_import",
        "dataset": protocol["dataset"], "revision": protocol["revision"],
        "protocol_sha256": _file_sha256(args.protocol),
        "source_shards_sha256": {item["path"]: item["sha256"] for item in protocol["shards"]},
        "sampled_images_sha256": hashlib.sha256("".join(
            image_hash for row in rows for image_hash in row["image_hashes"]
        ).encode("ascii")).hexdigest(),
        "feature_shape": [1, 16, 16] if representation == "gray16" else [3, 32, 32],
        "representation": representation, "split_basis": protocol["split_basis"],
        **audit,
        "features_sha256": _file_sha256(args.features),
        "limitation": "One non-interventional shard per Office domain. No verified same-state action interventions, "
                      "live desktop transfer, execution authority, or Jev parity is established.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

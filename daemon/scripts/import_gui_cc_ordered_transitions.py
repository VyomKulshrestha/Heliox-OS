"""Import a preselected GUI-CC offline subset as ordered, tap-only screen transitions.

Raw screenshots remain in the ignored local dataset directory. The derived
visual features are for research, not live model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pilot.agents.screen_vision import _world_embedding_from_frame
from scripts.import_aligned_world_trajectories import MAX_IMAGE_BYTES, MAX_PIXELS, _file_sha256
from scripts.sample_world_visual_transitions import _splits


class OversizeImageError(ValueError):
    """A source image exceeds the importer memory bound."""


def _feature(path: Path, representation: str) -> tuple[np.ndarray, int, int, str]:
    with path.open("rb") as stream:
        content = stream.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise OversizeImageError("Image exceeds the bounded decode limit")
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
        if image.format != "PNG" or width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError("Image format or dimensions exceed the bounded decode limit")
        if representation == "gray16":
            frame = np.asarray(image.convert("RGB").resize((64, 64)).convert("L"), dtype=np.float32)
            feature = np.asarray(_world_embedding_from_frame(frame), dtype=np.float32)
        elif representation in {"rgb32", "rgb64"}:
            size = int(representation[3:])
            frame = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
            feature = (frame.transpose(2, 0, 1) / 255.0).reshape(-1)
        else:
            raise ValueError("Unknown visual representation")
    return feature, width, height, hashlib.sha256(content).hexdigest()


def _metadata(path: Path) -> dict[str, dict]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Metadata exceeds the bounded read limit")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Metadata rows must be objects")
    ids = [record.get("sample_id") for record in records]
    if (len(records) > 1000 or any(not isinstance(identifier, str) for identifier in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Metadata has too many or duplicate sample IDs")
    return dict(zip(ids, records))


def import_pairs(root: Path, selection: dict, *, representation: str = "gray16") -> tuple[list[dict], Counter, str]:
    if representation not in {"gray16", "rgb32", "rgb64"}:
        raise ValueError("Unknown visual representation")
    metadata_path = root / "offline_samples.jsonl"
    if _file_sha256(metadata_path) != selection["offline_samples_sha256"]:
        raise ValueError("Offline metadata does not match the frozen selection")
    sample_ids = selection["sample_ids"]
    if (not isinstance(sample_ids, list) or not 3 <= len(sample_ids) <= 100
            or any(not isinstance(sample, str) or re.fullmatch(r"[0-9]{3}", sample) is None
                   for sample in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)):
        raise ValueError("Selection must contain distinct bounded sample IDs")
    metadata = _metadata(metadata_path)
    rows: list[dict] = []
    reasons: Counter = Counter()
    reference_hashes = []
    for sample in sample_ids:
        if sample not in metadata:
            raise ValueError(f"Selected sample {sample} is missing from metadata")
        record = metadata[sample]
        task = record.get("task_meta")
        step_count = record.get("n_steps")
        if (not isinstance(task, str) or not task.strip()
                or not isinstance(step_count, int) or isinstance(step_count, bool)
                or not 1 <= step_count <= 100):
            raise ValueError(f"Invalid metadata for sample {sample}")
        directory = root / "offline_data" / sample
        reference = directory / "reference_trajectory.jsonl"
        if reference.stat().st_size > 1024 * 1024:
            raise ValueError(f"Oversize reference trajectory for sample {sample}")
        reference_hashes.append(_file_sha256(reference))
        steps = [json.loads(line) for line in reference.read_text(encoding="utf-8").splitlines()]
        if len(steps) != step_count or [step.get("step_id") for step in steps] != list(range(step_count)):
            raise ValueError(f"Nonconsecutive reference trajectory for sample {sample}")
        group = hashlib.sha256(task.strip().casefold().encode("utf-8")).hexdigest()
        episode = hashlib.sha256(sample.encode("ascii")).hexdigest()
        cache: dict[str, tuple[np.ndarray, int, int, str]] = {}
        for index, step in enumerate(steps):
            action = step.get("semantic_action")
            if not isinstance(action, dict) or action.get("type") != "tap":
                reasons[f"unsupported_{action.get('type', 'unknown') if isinstance(action, dict) else 'unknown'}"] += 1
                continue
            coords = action.get("source_coord")
            if (not isinstance(coords, list) or len(coords) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in coords)):
                reasons["invalid_coordinate_or_size"] += 1
                continue
            before = "initial.png" if index == 0 else f"step_{index - 1:03d}_after.png"
            after = f"step_{index:03d}_after.png"
            try:
                for name in (before, after):
                    if name not in cache:
                        cache[name] = _feature(directory / name, representation)
                state, width, height, before_hash = cache[before]
                target, next_width, next_height, after_hash = cache[after]
            except OversizeImageError:
                reasons["oversize_image"] += 1
                continue
            except (OSError, ValueError):
                reasons["missing_or_invalid_image"] += 1
                continue
            x, y = coords
            if (width, height) != (next_width, next_height) or not (0 <= x < width and 0 <= y < height):
                reasons["invalid_coordinate_or_size"] += 1
                continue
            rows.append({"group": group, "episode": episode, "state": state,
                         "target": target, "action": 1,
                         "coords": [x / width, y / height],
                         "image_hashes": (before_hash, after_hash)})
    return rows, reasons, hashlib.sha256("".join(reference_hashes).encode("ascii")).hexdigest()


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--representation", choices=("gray16", "rgb32", "rgb64"), default="gray16")
    parser.add_argument("--features", type=Path, default=state / "gui_cc_tap_transitions_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "gui_cc_tap_transitions_v1.json")
    args = parser.parse_args()
    if (not args.root.is_dir() or not args.selection.is_file() or args.features.exists()
            or args.report.exists() or args.features.resolve() == args.report.resolve()):
        parser.error("Supply a local dataset, frozen selection, and fresh output paths")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows, reasons, references_hash = import_pairs(args.root, selection, representation=args.representation)
    assignments = _splits({row["group"] for row in rows}, selection["split_seed"])
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
        "purpose": "research_only_preselected_gui_cc_offline_tap_transitions",
        "dataset": selection["dataset"], "revision": selection["revision"],
        "selection_sha256": _file_sha256(args.selection),
        "offline_samples_sha256": _file_sha256(args.root / "offline_samples.jsonl"),
        "sampled_references_sha256": references_hash,
        "sampled_images_sha256": hashlib.sha256("".join(
            image_hash for row in rows for image_hash in row["image_hashes"]
        ).encode("ascii")).hexdigest(),
        "sample_count": len(selection["sample_ids"]),
        "accepted": dict(Counter(row["split"] for row in rows)),
        "groups": dict(Counter(assignments.values())),
        "split_basis": "casefolded_task_meta",
        "representation": args.representation,
        "feature_shape": ([1, 16, 16] if args.representation == "gray16" else
                          [3, int(args.representation[3:]), int(args.representation[3:])]),
        "action_filter": "tap_with_pixel_coordinates_only",
        "exclusions": dict(reasons),
        "features_sha256": _file_sha256(args.features),
        "limitation": "Ordered mobile GUI observations are not independently verified action effects. "
                      "Downsampled pixels omit readable text and semantics; unsupported actions are excluded. "
                      "No desktop transfer, safety, Jev parity, or live runtime loading claim.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

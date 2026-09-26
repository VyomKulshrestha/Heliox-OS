"""Sample licensed accessibility-tree transitions without retaining page text.

The resulting feature archive is research-only and must stay in ignored local
bench-state storage. A trajectory split does not establish website transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

DATASET = "Qwen/WebWorldData"
MAX_STATE_CHARS = 40000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_NODE_ID = re.compile(r"\[[a-zA-Z]?\d+\]")
_FIRST_TARGET = re.compile(r"^\s*[a-zA-Z_]+\(\s*\[?['\"]?([a-zA-Z]?\d+)['\"]?\]?", re.I)


def _read_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                content = response.read(MAX_RESPONSE_BYTES + 1)
            if len(content) > MAX_RESPONSE_BYTES:
                raise ValueError("WebWorld response exceeds the bounded read limit")
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdecimal() else 5 * (2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            delay = 5 * (2 ** attempt)
        time.sleep(min(30, max(1, delay)))
    raise RuntimeError("Unreachable WebWorld retry state")


def parse_trajectory(conversations: list[dict]) -> list[tuple[str, str, str]]:
    """Return (before tree, action, after tree) from alternating turns."""
    if len(conversations) < 2 or len(conversations) % 2:
        return []
    transitions: list[tuple[str, str, str]] = []
    previous_state = ""
    for turn in range(0, len(conversations), 2):
        human, assistant = conversations[turn:turn + 2]
        if human.get("from") != "human" or assistant.get("from") != "gpt":
            return []
        prompt, after = human.get("value"), assistant.get("value")
        if not isinstance(prompt, str) or not isinstance(after, str):
            return []
        if turn == 0:
            marker = "Initial Page State:\n"
            if marker not in prompt or "\nFirst Action:" not in prompt:
                return []
            before = prompt.split(marker, 1)[1].rsplit("\nFirst Action:", 1)[0]
            action_part = prompt.rsplit("\nFirst Action:", 1)[1]
        else:
            if "\nAction:" not in prompt:
                return []
            before = previous_state
            action_part = prompt.rsplit("\nAction:", 1)[1]
        if "\nNext Page State:" not in action_part:
            return []
        action = action_part.split("\nNext Page State:", 1)[0].strip()
        if action.startswith("'") and action.endswith("'"):
            action = action[1:-1]
        if not before or not after or not action or max(len(before), len(after)) > MAX_STATE_CHARS:
            return []
        transitions.append((before, action, after))
        previous_state = after
    return transitions


def bound_action(before: str, action: str) -> str:
    """Bind a numeric element reference to its pre-action role and label."""
    operation = action.split("(", 1)[0].strip().lower()
    match = _FIRST_TARGET.match(action)
    if match is None:
        return action
    node_id = match.group(1)
    node = re.search(r"(?m)^\s*\[" + re.escape(node_id) + r"\]\s+(.+)$", before)
    if node is None:
        return action
    target = _NODE_ID.sub(" ", node.group(1))
    remaining = action[match.end():].strip().removesuffix(")").strip()
    remaining = remaining.removeprefix(",").strip()
    return f"{operation} target {target}" + (f" arguments {remaining}" if remaining else "")


def _features(texts: list[str], *, dimension: int) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=dimension, alternate_sign=False, norm="l2", token_pattern=r"(?u)\b\w+\b"
    )
    return vectorizer.transform(_NODE_ID.sub(" ", text) for text in texts).toarray().astype(np.float32)


def _split_by_shared_state(states: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    """Keep trajectories sharing an identical pre-action feature in one split."""
    unique = sorted(set(groups.tolist()))
    parent = {group: group for group in unique}

    def root(group: int) -> int:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    seen: dict[bytes, int] = {}
    for state, group in zip(states, groups):
        key = state.tobytes()
        if key in seen:
            parent[root(group)] = root(seen[key])
        else:
            seen[key] = group
    components: dict[int, list[int]] = {}
    for group in unique:
        components.setdefault(root(group), []).append(group)
    if len(components) < 3:
        raise ValueError("Fewer than three independent state-connected trajectory groups")
    shuffled = list(components.values())
    random.Random(seed).shuffle(shuffled)
    train_end = min(len(shuffled) - 2, max(1, int(len(shuffled) * 0.7)))
    validation_end = min(len(shuffled) - 1, max(train_end + 1, int(len(shuffled) * 0.85)))
    assignments = {
        group: ("train" if index < train_end else "validation" if index < validation_end else "eval")
        for index, component in enumerate(shuffled) for group in component
    }
    return np.asarray([assignments[group] for group in groups]), len(components)


def _save_checkpoint(
    path: Path, *, revision: str, total: int, offsets: list[int], pages_done: int,
    arrays: dict[str, np.ndarray], exclusions: Counter[str], bound_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.npz")
    np.savez_compressed(
        temporary, revision=np.asarray(revision), total=np.asarray(total),
        offsets=np.asarray(offsets, dtype=np.int64), pages_done=np.asarray(pages_done),
        exclusions=np.asarray(json.dumps(dict(exclusions))), bound_rows=np.asarray(bound_rows),
        **arrays,
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path, *, revision: str, total: int, offsets: list[int],
) -> tuple[int, dict[str, np.ndarray], Counter[str], int]:
    with np.load(path, allow_pickle=False) as data:
        if (str(data["revision"].item()) != revision or int(data["total"].item()) != total
                or not np.array_equal(data["offsets"], offsets)):
            raise ValueError("WebWorld checkpoint does not match the requested source and sample")
        pages_done = int(data["pages_done"].item())
        if not 0 <= pages_done <= len(offsets):
            raise ValueError("Invalid WebWorld checkpoint progress")
        arrays = {key: data[key].copy() for key in
                  ("state", "target", "action_type", "action_bound", "group")}
        exclusions = Counter(json.loads(str(data["exclusions"].item())))
        bound_rows = int(data["bound_rows"].item())
    count = len(arrays["group"])
    if (arrays["state"].shape != (count, 1024) or arrays["target"].shape != (count, 1024)
            or arrays["action_type"].shape != (count, 256)
            or arrays["action_bound"].shape != (count, 256)
            or not 0 <= bound_rows <= count):
        raise ValueError("Invalid WebWorld checkpoint features")
    return pages_done, arrays, exclusions, bound_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--page-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if (args.features.exists() or args.report.exists() or not 3 <= args.pages <= 100
            or not 1 <= args.page_length <= 10
            or not 0 <= args.request_interval <= 10):
        parser.error("Use fresh output paths and bounded pages")

    metadata_url = "https://huggingface.co/api/datasets/" + DATASET
    metadata = _read_json(metadata_url)
    revision = metadata.get("sha")
    if (metadata.get("cardData", {}).get("license") != "apache-2.0"
            or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("Dataset license or revision could not be verified")

    def page(offset: int, length: int) -> dict:
        query = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset, "length": length,
        })
        return _read_json("https://datasets-server.huggingface.co/rows?" + query)

    total = page(0, 1).get("num_rows_total")
    if not isinstance(total, int) or total < args.pages * args.page_length:
        raise ValueError("Dataset is smaller than the requested bounded sample")
    rng = random.Random(args.seed)
    offsets = [rng.randint(i * total // args.pages,
                           (i + 1) * total // args.pages - args.page_length)
               for i in range(args.pages)]
    checkpoint = args.checkpoint or args.features.with_suffix(".partial.npz")
    if checkpoint.resolve() in {args.features.resolve(), args.report.resolve()}:
        parser.error("Checkpoint must have its own path")
    if checkpoint.exists():
        pages_done, arrays, exclusions, bound_rows = _load_checkpoint(
            checkpoint, revision=revision, total=total, offsets=offsets
        )
    else:
        pages_done, exclusions, bound_rows = 0, Counter(), 0
        arrays = {
            "state": np.empty((0, 1024), dtype=np.float32),
            "target": np.empty((0, 1024), dtype=np.float32),
            "action_type": np.empty((0, 256), dtype=np.float32),
            "action_bound": np.empty((0, 256), dtype=np.float32),
            "group": np.empty(0, dtype=np.int32),
        }
    for page_index in range(pages_done, len(offsets)):
        offset = offsets[page_index]
        response = page(offset, args.page_length)
        rows = response.get("rows")
        if (response.get("num_rows_total") != total or not isinstance(rows, list)
                or len(rows) != args.page_length):
            raise ValueError("Incomplete or changed WebWorld page")
        records: list[tuple[int, str, str, str]] = []
        for position, entry in enumerate(rows):
            if entry.get("row_idx") != offset + position or entry.get("truncated_cells"):
                exclusions["invalid_or_truncated"] += 1
                continue
            transitions = parse_trajectory(entry.get("row", {}).get("conversations", []))
            if not transitions:
                exclusions["unparsed_trajectory"] += 1
                continue
            records.extend((offset + position, before, action, after)
                           for before, action, after in transitions)
        if records:
            action_bound = [bound_action(row[1], row[2]) for row in records]
            bound_rows += sum(bound != row[2] for bound, row in zip(action_bound, records))
            page_arrays = {
                "state": _features([row[1] for row in records], dimension=1024),
                "target": _features([row[3] for row in records], dimension=1024),
                "action_type": _features(
                    [row[2].split("(", 1)[0].strip().lower() for row in records], dimension=256
                ),
                "action_bound": _features(action_bound, dimension=256),
                "group": np.asarray([row[0] for row in records], dtype=np.int32),
            }
            arrays = {key: np.concatenate((arrays[key], values)) for key, values in page_arrays.items()}
        _save_checkpoint(
            checkpoint, revision=revision, total=total, offsets=offsets,
            pages_done=page_index + 1, arrays=arrays,
            exclusions=exclusions, bound_rows=bound_rows,
        )
        print(json.dumps({"pages_done": page_index + 1, "total_pages": len(offsets)}), flush=True)
        if page_index + 1 < len(offsets):
            time.sleep(args.request_interval)
    # A direct URL navigation from about:blank supplies no rendered-page state to
    # predict from. Its target belongs to retrieval, not an in-page transition.
    goto = _features(["goto"], dimension=256)[0]
    keep = ~np.all(arrays["action_type"] == goto, axis=1)
    exclusions["direct_navigation"] += int((~keep).sum())
    arrays = {key: values[keep] for key, values in arrays.items()}
    if len(set(arrays["group"])) < 3:
        raise ValueError("Fewer than three parsed in-page trajectories")
    if _read_json(metadata_url).get("sha") != revision:
        raise ValueError("Dataset revision changed during sampling")

    splits, component_count = _split_by_shared_state(
        arrays["state"], arrays["group"], args.seed + 1
    )
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        **arrays,
        split=splits,
    )
    report = {
        "purpose": "research_only_bounded_accessibility_transition_sample",
        "dataset": DATASET, "revision": revision, "declared_license": "apache-2.0",
        "split_basis": "trajectory_plus_identical_prestate_not_website", "seed": args.seed,
        "pages": args.pages, "page_length": args.page_length, "offsets": offsets,
        "transition_rows": len(arrays["group"]), "trajectories": len(set(arrays["group"])),
        "state_connected_components": component_count,
        "bound_action_rows": bound_rows,
        "splits": dict(Counter(splits)),
        "exclusions": dict(exclusions),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "limitation": "The rows API is not commit-pinned; source SHA was checked before and after. "
                      "No raw tree text is retained, but hashed features remain local and may still encode content. "
                      "Trajectory-disjoint does not prove website transfer, causality, safe execution, or Jev parity.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

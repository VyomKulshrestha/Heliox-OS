"""Sample licensed accessibility-tree transitions without retaining page text.

The resulting feature archive is research-only and must stay in ignored local
bench-state storage. A trajectory split does not establish website transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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
    with urllib.request.urlopen(url, timeout=20) as response:
        content = response.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("WebWorld response exceeds the bounded read limit")
    return json.loads(content)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--page-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if (args.features.exists() or args.report.exists() or not 3 <= args.pages <= 100
            or not 1 <= args.page_length <= 10):
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
    records: list[tuple[int, str, str, str]] = []
    exclusions: Counter[str] = Counter()
    for offset in offsets:
        response = page(offset, args.page_length)
        rows = response.get("rows")
        if (response.get("num_rows_total") != total or not isinstance(rows, list)
                or len(rows) != args.page_length):
            raise ValueError("Incomplete or changed WebWorld page")
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
    if len({row[0] for row in records}) < 3:
        raise ValueError("Fewer than three parsed trajectories")
    if _read_json(metadata_url).get("sha") != revision:
        raise ValueError("Dataset revision changed during sampling")

    groups = sorted({row[0] for row in records})
    rng.shuffle(groups)
    train_end = min(len(groups) - 2, max(1, int(len(groups) * 0.7)))
    validation_end = min(len(groups) - 1, max(train_end + 1, int(len(groups) * 0.85)))
    assignments = {group: ("train" if index < train_end else
                           "validation" if index < validation_end else "eval")
                   for index, group in enumerate(groups)}
    before_text = [row[1] for row in records]
    after_text = [row[3] for row in records]
    action_type = [row[2].split("(", 1)[0].strip().lower() for row in records]
    action_bound = [bound_action(row[1], row[2]) for row in records]
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features,
        state=_features(before_text, dimension=1024),
        target=_features(after_text, dimension=1024),
        action_type=_features(action_type, dimension=256),
        action_bound=_features(action_bound, dimension=256),
        group=np.asarray([row[0] for row in records], dtype=np.int32),
        split=np.asarray([assignments[row[0]] for row in records]),
    )
    report = {
        "purpose": "research_only_bounded_accessibility_transition_sample",
        "dataset": DATASET, "revision": revision, "declared_license": "apache-2.0",
        "split_basis": "whole_trajectory_not_website", "seed": args.seed,
        "pages": args.pages, "page_length": args.page_length, "offsets": offsets,
        "transition_rows": len(records), "trajectories": len(groups),
        "bound_action_rows": sum(bound != raw for bound, raw in zip(action_bound, (row[2] for row in records))),
        "splits": dict(Counter(assignments[row[0]] for row in records)),
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

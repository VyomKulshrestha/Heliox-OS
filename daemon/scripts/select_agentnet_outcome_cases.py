"""Pre-register group-disjoint post-action outcomes from pinned public AgentNet.

The reflector's step flag is weak supervision, not a human safety verdict.
Only the task, before observation, action, and next observation are inputs;
thought and reflection text must not be shown to the trained judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SOURCE_REPO = "xlangai/AgentNet"
SOURCE_REVISION = "d76ee50a63fad81cfdbe576416757d7c2091ed50"
SOURCE_SHA256 = "8e24c5a5a1ef5a5113755b33ed12164fc37ce8fbbd35d1c7dba81bf950d53d00"
SPLIT_ROWS = {"train": 1024, "calibration": 128, "development": 128, "final": 128, "ood": 128}
SEEDS = {"ood": 20261021, "final": 20261022, "development": 20261023, "calibration": 20261024, "train": 20261025}
MAX_INPUT_CHARS = 4_000


def load_candidates(source: Path) -> tuple[list[dict], Counter]:
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("AgentNet source differs from the pinned release")
    candidates: list[dict] = []
    counts: Counter = Counter()
    groups: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            trajectory = json.loads(line)
            group = trajectory["task_id"]
            if group in groups:
                raise ValueError("Duplicate AgentNet task ID")
            groups.add(group)
            task = trajectory.get("instruction")
            if not isinstance(task, str) or not task.strip():
                continue
            steps = trajectory.get("traj") or []
            for current, following in zip(steps, steps[1:]):
                before = current.get("value", {})
                after = following.get("value", {})
                label = before.get("last_step_correct")
                fields = (before.get("observation"), before.get("action"), before.get("code"), after.get("observation"))
                if (
                    not isinstance(label, bool)
                    or not all(isinstance(value, str) and value.strip() for value in fields)
                    or following.get("index") != current.get("index", -2) + 1
                    or len(task) + sum(len(value) for value in fields) > MAX_INPUT_CHARS
                ):
                    counts["ineligible"] += 1
                    continue
                candidates.append(
                    {
                        "id": f"{group}:{current['index']}",
                        "group": group,
                        "domain": trajectory["domain"],
                        "correct": label,
                    }
                )
                counts["correct" if label else "incorrect"] += 1
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate AgentNet action ID")
    counts["groups"] = len(groups)
    return candidates, counts


def select(candidates: list[dict], *, ood_domain: str, split_rows: dict[str, int] = SPLIT_ROWS) -> dict[str, list[str]]:
    if any(rows <= 0 or rows % 2 for rows in split_rows.values()):
        raise ValueError("Each split must have a positive even row count")
    if "ood" not in split_rows:
        raise ValueError("An out-of-domain split is required")
    selected: dict[str, list[str]] = {}
    used_groups: set[str] = set()
    for split in ("ood", "final", "development", "calibration", "train"):
        if split not in split_rows:
            continue
        rows: list[dict] = []
        for correct in (False, True):
            pool = [
                row
                for row in candidates
                if row["group"] not in used_groups
                and row["correct"] is correct
                and ((row["domain"] == ood_domain) == (split == "ood"))
            ]
            pool.sort(key=lambda row: row["id"])
            random.Random(SEEDS[split] + int(correct)).shuffle(pool)
            class_count = 0
            for row in pool:
                if row["group"] not in used_groups:
                    rows.append(row)
                    used_groups.add(row["group"])
                    class_count += 1
                    if class_count == split_rows[split] // 2:
                        break
            if class_count != split_rows[split] // 2:
                raise ValueError(f"Insufficient group-disjoint {split} actions for class {correct}")
        selected[split] = [row["id"] for row in rows]
    return {name: selected[name] for name in split_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ood-domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh manifest path")
    candidates, counts = load_candidates(args.source)
    splits = select(candidates, ood_domain=args.ood_domain)
    manifest = {
        "purpose": "pre_registered_agentnet_post_action_outcome_research",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_license": "MIT",
        "label_provenance": "model_generated_reflector_step_correct_not_human",
        "input_fields": ["instruction", "before_observation", "action", "code", "next_observation"],
        "excluded_fields": ["thought", "reflection", "task_completed", "alignment_score"],
        "ood_domain": args.ood_domain,
        "max_input_chars": MAX_INPUT_CHARS,
        "candidate_counts": dict(counts),
        "seeds": SEEDS,
        "selection_order": ["ood", "final", "development", "calibration", "train"],
        "splits": splits,
        "case_ids": [case_id for name in SPLIT_ROWS for case_id in splits[name]],
        "limitation": "Reflector outcome labels are not human action-safety labels. One action per task; no runtime authority.",
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"candidates": len(candidates), "splits": {name: len(rows) for name, rows in splits.items()}}, indent=2
        )
    )


if __name__ == "__main__":
    main()

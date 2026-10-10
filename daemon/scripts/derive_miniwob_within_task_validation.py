"""Derive task-aware validation from pinned synthetic branch archives.

The original task-disjoint evaluation is retained; the original validation
task is excluded. This changes model selection only, not the held-out task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_splits(
    groups: np.ndarray,
    splits: np.ndarray,
    tasks: np.ndarray,
    per_task: int,
    seed: int,
    *,
    include_prior_validation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if per_task < 1:
        raise ValueError("At least one validation group per task is required")
    fit_source = np.isin(splits, ("train", "validation")) if include_prior_validation else splits == "train"
    if set(splits.astype(str)) != {"train", "validation", "eval"} or set(tasks[fit_source]) & set(
        tasks[splits == "eval"]
    ):
        raise ValueError("Original evaluation tasks must be separate from fit tasks")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups.astype(str)):
        by_group[group].append(index)
    if any(
        len(indices) != 2 or len(set(splits[indices])) != 1 or len(set(tasks[indices])) != 1
        for indices in by_group.values()
    ):
        raise ValueError("Each branch must have two rows from one task and split")
    train_groups: dict[str, list[str]] = defaultdict(list)
    for group, indices in by_group.items():
        if fit_source[indices[0]]:
            train_groups[str(tasks[indices[0]])].append(group)
    if not train_groups or any(len(items) <= per_task for items in train_groups.values()):
        raise ValueError("Every training task needs fit and validation groups")
    selected = set()
    for task, items in train_groups.items():
        ranked = sorted(items, key=lambda group: hashlib.sha256(f"{seed}:{task}:{group}".encode()).hexdigest())
        selected.update(ranked[:per_task])
    keep = np.ones(len(splits), dtype=bool) if include_prior_validation else splits != "validation"
    recoded = splits[keep].astype("<U16").copy()
    if include_prior_validation:
        recoded[recoded == "validation"] = "train"
    recoded[np.isin(groups[keep], list(selected))] = "validation"
    if set(recoded) != {"train", "validation", "eval"}:
        raise ValueError("Derived fit, validation, and held-out evaluation are required")
    return keep, recoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "visual",
        "visual-report",
        "semantic",
        "semantic-report",
        "out-visual",
        "out-visual-report",
        "out-semantic",
        "out-semantic-report",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--validation-groups-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--include-prior-validation-as-fit", action="store_true")
    args = parser.parse_args()
    outputs = (args.out_visual, args.out_visual_report, args.out_semantic, args.out_semantic_report)
    if (
        args.validation_groups_per_task < 1
        or args.seed < 0
        or any(path.exists() for path in outputs)
        or len({path.resolve() for path in outputs}) != len(outputs)
    ):
        parser.error("Use fresh distinct outputs and bounded validation settings")
    visual_report = json.loads(args.visual_report.read_text(encoding="utf-8"))
    semantic_report = json.loads(args.semantic_report.read_text(encoding="utf-8"))
    if (
        sha256(args.visual) != visual_report.get("features_sha256")
        or sha256(args.semantic) != semantic_report.get("feature_sha256")
        or sha256(args.visual) != semantic_report.get("visual_features_sha256")
        or sha256(args.visual_report) != semantic_report.get("source_report_sha256")
    ):
        raise ValueError("Parent archives do not match their pinned reports")
    with np.load(args.visual, allow_pickle=False) as archive:
        visual = {key: archive[key] for key in ("state", "target", "action", "coords", "group", "split", "task")}
    with np.load(args.semantic, allow_pickle=False) as archive:
        semantic = {key: archive[key] for key in ("features", "group", "split", "coords")}
    count = len(visual["group"])
    if (
        any(len(value) != count for value in visual.values())
        or len(semantic["features"]) != count
        or not np.array_equal(semantic["group"], visual["group"])
        or not np.array_equal(semantic["split"], visual["split"])
        or not np.allclose(semantic["coords"], visual["coords"], atol=1e-6)
    ):
        raise ValueError("Parent semantic and visual rows are misaligned")
    keep, splits = derived_splits(
        visual["group"],
        visual["split"],
        visual["task"],
        args.validation_groups_per_task,
        args.seed,
        include_prior_validation=args.include_prior_validation_as_fit,
    )
    visual_out = {key: value[keep] for key, value in visual.items()}
    visual_out["split"] = splits
    semantic_out = {key: value[keep] for key, value in semantic.items()}
    semantic_out["split"] = splits
    for path in (args.out_visual, args.out_semantic):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_visual, **visual_out)
    derived_visual_report = {
        "schema_version": 1,
        "purpose": "research_only_derived_branch_validation",
        "source_repo": visual_report["source_repo"],
        "source_commit": visual_report["source_commit"],
        "counterfactual_pairs": True,
        "feature_shape": visual_report["feature_shape"],
        "parent_visual_sha256": sha256(args.visual),
        "parent_report_sha256": sha256(args.visual_report),
        "selection": "Hash-ranked held-out seed groups within each fit task; original evaluation task unchanged",
        "include_prior_validation_as_fit": args.include_prior_validation_as_fit,
        "seed": args.seed,
        "validation_groups_per_task": args.validation_groups_per_task,
        "rows": dict(Counter(splits.tolist())),
        "groups": {name: len(set(visual_out["group"][splits == name])) for name in ("train", "validation", "eval")},
        "features_sha256": sha256(args.out_visual),
        "limitation": "Within-task validation may favor familiar tasks; the held-out "
        "task remains synthetic and is not real desktop or Jev evidence.",
    }
    args.out_visual_report.write_text(json.dumps(derived_visual_report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out_semantic, **semantic_out)
    derived_semantic_report = {
        "schema_version": 1,
        "purpose": "research_only_derived_frozen_semantic_features",
        "parent_feature_sha256": sha256(args.semantic),
        "parent_report_sha256": sha256(args.semantic_report),
        "visual_features_sha256": sha256(args.out_visual),
        "source_report_sha256": sha256(args.out_visual_report),
        "feature_sha256": sha256(args.out_semantic),
        "rows": len(semantic_out["features"]),
        "feature_dim": semantic_out["features"].shape[1],
        "model": semantic_report["model"],
        "revision": semantic_report["revision"],
        "prompt_version": semantic_report["prompt_version"],
    }
    args.out_semantic_report.write_text(json.dumps(derived_semantic_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": derived_visual_report["rows"], "groups": derived_visual_report["groups"]}))


if __name__ == "__main__":
    main()

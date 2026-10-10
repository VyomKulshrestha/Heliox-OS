"""Measure a label-blind option-uniform baseline on pinned public splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import DATASET_REVISION, load_split, uniform_baseline  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--report", type=Path, default=root / "benchmarks/qwen_general_uniform_baseline_v2.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= 4096:
        parser.error("--limit must be 1-4096")
    splits = {}
    for offset, name in ((2, "test"), (3, "ood")):
        rows, manifest = load_split(name, args.limit, args.seed + offset)
        if len(rows) != args.limit:
            raise ValueError(f"Not enough licensed {name} decisions")
        splits[name] = {"manifest": manifest, "metrics": uniform_baseline(rows)}
    report = {
        "purpose": "label_blind_uniform_option_baseline",
        "dataset_revision": DATASET_REVISION,
        "selection_seed": args.seed,
        "limit_per_split": args.limit,
        "splits": splits,
        "limitation": "Uniform ties select the first option; this is not a semantic model.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: entry["metrics"]["by_kind"] for name, entry in splits.items()}, indent=2))


if __name__ == "__main__":
    main()

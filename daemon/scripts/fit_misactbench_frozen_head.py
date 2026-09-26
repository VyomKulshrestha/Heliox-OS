"""Fit a CPU decision head on pinned, trajectory-disjoint frozen actions.

The calibration split selects regularization and a cautious threshold before
the untouched test split is scored. Neither the head nor its predictions grant
Heliox execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import decision_row, load_parts  # noqa: E402
from scripts.extract_misactbench_frozen_features import (  # noqa: E402
    MODEL,
    REVISION,
    SPLITS,
    feature_signature,
)
from scripts.fit_openjev_scope_head import fit  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402


def verified_features(path: Path, *, digest: str, signature: str,
                      ids: list[str]) -> np.ndarray:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen feature artifact checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        features = saved["features"].astype(np.float32)
        offsets = saved["offsets"].tolist()
        metadata = json.loads(str(saved["metadata_json"]))
    if (metadata.get("signature") != signature or metadata.get("ids") != ids
            or metadata.get("completed") != len(ids)
            or offsets != list(range(len(ids) + 1))
            or features.ndim != 2 or features.shape[0] != len(ids)
            or metadata.get("feature_dim") != features.shape[1]
            or not np.isfinite(features).all()):
        raise ValueError("Frozen features differ from selected action rows")
    return features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path,
                        default=root / "benchmarks/.bench-state/misactbench_frozen_head_v1.npz")
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/.bench-state/misactbench_frozen_head_v1.json")
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Choose fresh head and report paths")
    source = json.loads(args.feature_report.read_text(encoding="utf-8"))
    if source["model"] != MODEL or source["revision"] != REVISION:
        raise ValueError("Frozen source model differs from the pinned encoder")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download(MODEL, revision=REVISION, local_files_only=True),
        local_files_only=True)

    def eligible(case: dict) -> bool:
        selected, _ = encode_rows(tokenizer, [decision_row(case)], source["max_length"])
        return bool(selected)

    rows, manifest = load_parts(args.dataset, seed=source["seed"],
                                limits=source["selected_rows"], eligible=eligible,
                                history_calls=source["dataset"].get("history_calls", 0))
    if manifest != source["dataset"]:
        raise ValueError("Frozen feature data selection has changed")
    parts = {}
    for split in SPLITS:
        encoded, excluded = encode_rows(tokenizer, rows[split], source["max_length"],
                                        fit_history=manifest.get("history_calls", 0) > 0)
        if excluded:
            raise ValueError("Selected action became token-ineligible")
        signature = feature_signature(split, rows[split], encoded,
                                      max_length=source["max_length"], seed=source["seed"])
        if signature != source["signatures"][split]:
            raise ValueError("Frozen input tokens changed")
        ids = [row["id"] for row in rows[split]]
        features = verified_features(args.feature_dir / f"{split}.npz",
                                     digest=source["feature_sha256"][split],
                                     signature=signature, ids=ids)
        parts[split.removeprefix("scope_")] = {
            "features": features,
            "labels": np.asarray([row["target"][1] == 1.0 for row in rows[split]]),
            "ids": np.asarray(ids),
        }
    result = fit(parts, artifact=args.artifact)
    report = {"purpose": "research_only_human_labeled_frozen_action_head",
              "model": MODEL, "revision": REVISION, "dataset": manifest,
              "feature_sha256": source["feature_sha256"], "head": result,
              "limitation": ("Text-only computer actions, no screenshots or full history; "
                             "inspected public test and no live desktop safety evidence. "
                             "No Jev parity or execution authority.")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "selected_c", "calibration_metrics", "test_at_default_threshold",
        "test_at_calibrated_threshold")}, indent=2))


if __name__ == "__main__":
    main()

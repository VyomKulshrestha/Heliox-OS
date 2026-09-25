"""Safely replay a frozen research MLP head on pinned feature checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtr

from scripts.train_world_multidomain_mlp import load_features
from scripts.train_world_pairwise_scope import group_keys, top_one_accuracy
from scripts.train_world_scope_semantic import metrics


def score_features(artifact: np.lib.npyio.NpzFile, features: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2 = (artifact[name] for name in ("w1", "b1", "w2", "b2"))
    mean, scale = artifact["mean"], artifact["scale"]
    if (features.ndim != 2 or w1.ndim != 2 or w2.shape != (1, len(b1)) or
            b2.shape != (1,) or w1.shape[1] != features.shape[1] or
            mean.shape != (features.shape[1],) or scale.shape != mean.shape or
            len(b1) != w1.shape[0] or np.any(scale <= 0) or
            not all(np.isfinite(value).all() for value in
                    (features, w1, b1, w2, b2, mean, scale))):
        raise ValueError("Invalid frozen MLP weights or feature dimensions")
    normalized = (features - mean) / scale
    hidden = normalized @ w1.T + b1
    hidden = hidden * ndtr(hidden)
    return expit((hidden @ w2.T + b2).reshape(-1))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--result-report", type=Path, required=True)
    parser.add_argument("--source-report", type=Path,
                        default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--feature-dir", type=Path,
                        default=state / "world_multiturn_features_v2")
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    args = parser.parse_args()
    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    result = json.loads(args.result_report.read_text(encoding="utf-8"))
    if hashlib.sha256(args.artifact.read_bytes()).hexdigest() != result["artifact_sha256"]:
        raise ValueError("Frozen MLP artifact hash mismatch")
    features, labels = load_features(args.feature_dir, source)
    keys = group_keys(args.dataset_dir, args.feature_dir, source)
    with np.load(args.artifact, allow_pickle=False) as artifact:
        if str(artifact["source_report_sha256"]) != hashlib.sha256(source_bytes).hexdigest():
            raise ValueError("Frozen MLP feature source changed")
        threshold = float(artifact["threshold"])
        scores = {name: score_features(artifact, value) for name, value in features.items()
                  if name in result["metrics"]}
    replay = {name: {"default": metrics(labels[name], value, 0.5),
                     "cautious": metrics(labels[name], value, threshold)}
              for name, value in scores.items()}
    top_one = {name: top_one_accuracy(keys[name], labels[name], scores[name])
               for name in ("agentic_test", "tbac_test")}
    if replay != result["metrics"] or ("test_top_one" in result and
                                       top_one != result["test_top_one"]):
        raise ValueError("Saved MLP replay differs from the frozen result report")
    print(json.dumps({"metrics": replay, "test_top_one": top_one}, indent=2))


if __name__ == "__main__":
    main()

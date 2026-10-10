"""Inspect false allowances from the frozen offline multi-turn world head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.refscope_qwen_v1_holdout import CASES as HELIOX_CASES
from scripts.evaluate_opendec_checkpoint import scopejudge_cases
from scripts.evaluate_world_conversation_scope import AGENTIC_TEST_SHA256, load_pre_call_cases
from scripts.evaluate_world_scope_artifact import score_features
from scripts.train_world_multiturn_scope_semantic import AGENTIC_VALIDATION_SHA256
from scripts.train_world_scope_semantic import feature_identity, load_feature_checkpoint
from scripts.train_world_tool_scope_semantic import DATASET_HASHES, load_cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--artifact", type=Path, default=state / "world_multiturn_scope_v2.npz")
    parser.add_argument("--report", type=Path, default=state / "world_multiturn_scope_v2.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if hashlib.sha256(args.artifact.read_bytes()).hexdigest() != report["artifact_sha256"]:
        raise ValueError("World head artifact hash mismatch")
    cases = {
        "agentic_test": load_pre_call_cases(args.dataset_dir / "agentic_test_4o.json", AGENTIC_TEST_SHA256),
        "tbac_test": load_cases(args.dataset_dir / "test.json", DATASET_HASHES["test"], "tbac_test"),
        "scope_test": scopejudge_cases(args.scopejudge_path, "scope_test")[0],
        "heliox": tuple(HELIOX_CASES),
    }
    if report["agentic"]["validation_sha256"] != AGENTIC_VALIDATION_SHA256:
        raise ValueError("Training source changed")
    results = {}
    with np.load(args.artifact, allow_pickle=False) as artifact:
        threshold = float(artifact["threshold"])
        for name, part in cases.items():
            checkpoint = args.feature_dir / f"{name}.npz"
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["feature_checkpoint_sha256"][name]:
                raise ValueError(f"Feature checkpoint hash mismatch: {name}")
            _, _, indices, vectors, labels = load_feature_checkpoint(
                checkpoint, feature_identity(part, report["max_length"], report["batch_size"]), part
            )
            scores = score_features(artifact, np.asarray(vectors))
            failures = [
                (part[index - 1], float(score))
                for index, label, score in zip(indices, labels, scores, strict=True)
                if not label and score >= threshold
            ]
            results[name] = [
                {"id": case.id, "score": score} for case, score in sorted(failures, key=lambda item: -item[1])
            ]
    print(json.dumps({"threshold": threshold, "false_allowances": results}, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate an offline semantic scope head on held-out ScopeJudge tool calls.

The artifact is never loaded by the Heliox daemon. This script does not invoke
any candidate action; labels from this partition are evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.evaluate_openjev_checkpoint import scopejudge_cases
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT
from scripts.train_world_scope_semantic import embed_cases, metrics


def score_features(artifact: np.lib.npyio.NpzFile, features: np.ndarray) -> np.ndarray:
    weights, bias = artifact["weights"], artifact["bias"]
    mean, scale = artifact["mean"], artifact["scale"]
    if (weights.ndim != 2 or weights.shape[0] != 1 or
            features.ndim != 2 or features.shape[1] != weights.shape[1] or
            mean.shape != (weights.shape[1],) or scale.shape != mean.shape or
            np.any(scale <= 0) or not np.isfinite(features).all()):
        raise ValueError("Invalid semantic artifact dimensions or features")
    return expit(((features - mean) / scale) @ weights[0] + float(bias[0]))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")
    trained = json.loads(args.artifact_report.read_text(encoding="utf-8"))
    source = args.artifact.read_bytes()
    if trained["artifact_sha256"] != hashlib.sha256(source).hexdigest() or \
            trained["revision"] != MODEL_REVISION or \
            trained["prompt_sha256"] != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
        raise ValueError("Scope artifact, backbone, or prompt identity mismatch")
    cases, manifest = scopejudge_cases(args.scopejudge_path, "scope_test")

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"}, output_loading_info=True)
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    features, labels, excluded = embed_cases(model.eval(), tokenizer, cases,
                                              trained["max_length"], "scopejudge_test")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        if str(artifact["model_revision"]) != MODEL_REVISION:
            raise ValueError("Artifact backbone revision mismatch")
        scores = score_features(artifact, features)
        cautious = float(artifact["threshold"])
    report = {
        "purpose": "held_out_scopejudge_evaluation_only",
        "trained_artifact_sha256": trained["artifact_sha256"],
        "scopejudge_manifest": manifest,
        "selected_rows": len(cases), "eligible_rows": len(labels),
        "excluded_overlong": excluded,
        "default": metrics(labels, scores, 0.5),
        "cautious": metrics(labels, scores, cautious),
        "limitation": "Held-out public task-action labels are not live desktop results, "
                      "Jev parity, or permission to execute.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"default": report["default"], "cautious": report["cautious"]}, indent=2))


if __name__ == "__main__":
    main()

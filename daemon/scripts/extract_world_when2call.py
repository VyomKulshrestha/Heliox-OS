"""Extract pinned Qwen features for grouped When2Call world-model research."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.prepare_world_when2call import build_cases, load_rows
from scripts.train_world_scope_semantic import embed_cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=state / "world_when2call_features_v1")
    parser.add_argument("--report", type=Path, default=state / "world_when2call_features_v1.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")
    partitions, protocol = build_cases(load_rows(args.source))

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        output_loading_info=True,
    )
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    model.eval()
    results = {}
    for split, cases in partitions.items():
        checkpoint = args.feature_dir / f"{split}.npz"
        features, labels, overlong = embed_cases(
            model, tokenizer, cases, 1536, f"when2call_{split}", checkpoint_path=checkpoint, batch_size=2
        )
        results[split] = {
            "rows": len(cases),
            "eligible_rows": len(labels),
            "overlong": overlong,
            "positive_rows": int(labels.sum()),
            "negative_rows": int((~labels).sum()),
            "feature_dimensions": int(features.shape[1]),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "research_only_grouped_call_readiness_features",
        "protocol": protocol,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_length": 1536,
        "batch_size": 2,
        "splits": results,
        "limitation": "Synthetic preference supervision only; no held-out external source was used "
        "for fitting or threshold selection.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

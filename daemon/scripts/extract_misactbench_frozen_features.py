"""Checkpoint frozen Qwen action features from human-labeled MisActBench.

Research only: screenshots are not included and no weights are trained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import (  # noqa: E402
    DATASET_SHA256,
    decision_row,
    load_parts,
)
from scripts.extract_general_frozen_features import (  # noqa: E402
    extract,
    verify_loaded_backbone,
)
from scripts.qwen_model_pin import MODEL, REVISION  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402

SPLITS = ("scope_train", "scope_calibration", "scope_test")


def feature_signature(split: str, rows: list[dict], encoded: list[dict], *,
                      max_length: int, seed: int) -> str:
    identity = {"model": MODEL, "revision": REVISION, "dataset_sha256": DATASET_SHA256,
                "split": split, "seed": seed, "max_length": max_length,
                "representation": "last_hidden_state_final_token_fp16",
                "rows": [[row["id"], row["target"], item["sequences"]]
                         for row, item in zip(rows, encoded, strict=True)]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--test-rows", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--history-calls", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "benchmarks/.bench-state/misactbench_qwen_frozen_v1")
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/.bench-state/misactbench_qwen_frozen_v1.json")
    args = parser.parse_args()
    limits = {"scope_train": args.train_rows, "scope_calibration": args.calibration_rows,
              "scope_test": args.test_rows}
    if (min(limits.values()) < 2 or args.max_length < 64 or args.history_calls < 0
            or args.checkpoint_every < 1
            or args.report.exists() or (args.output_dir.exists() and not args.resume)
            or (args.resume and not args.output_dir.is_dir())):
        parser.error("Use positive limits, an unfinished matching resume, and fresh final output")

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free CUDA memory is required")
    transformers_logging.disable_progress_bar()
    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def eligible(case: dict) -> bool:
        selected, _ = encode_rows(tokenizer, [decision_row(case)], args.max_length)
        return bool(selected)

    rows, manifest = load_parts(args.dataset, seed=args.seed, limits=limits,
                                eligible=eligible, history_calls=args.history_calls)
    encoded = {}
    signatures = {}
    for split in SPLITS:
        encoded[split], excluded = encode_rows(tokenizer, rows[split], args.max_length,
                                               fit_history=args.history_calls > 0)
        if excluded or len(encoded[split]) != len(rows[split]):
            raise ValueError("Preselected MisActBench row became token-ineligible")
        signatures[split] = feature_signature(split, rows[split], encoded[split],
                                              max_length=args.max_length, seed=args.seed)
    print(json.dumps({"event": "selected", "counts": limits, "manifest": manifest}), flush=True)

    base, loading = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="eager", output_loading_info=True)
    model = base.model
    verify_loaded_backbone(model, loading)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Frozen backbone did not load on CUDA")
    model.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {}
    for split in SPLITS:
        completed[split] = extract(
            model, encoded[split], args.output_dir / f"{split}.npz",
            signature=signatures[split], checkpoint_every=args.checkpoint_every,
            resume=args.resume)
    report = {"purpose": "frozen_text_only_action_feature_extraction",
              "model": MODEL, "revision": REVISION, "dataset": manifest,
              "selected_rows": limits, "seed": args.seed, "max_length": args.max_length,
              "history_budget": {
                  split: {"rows_with_history": sum(item["history_calls_retained"] > 0
                                                   for item in encoded[split]),
                          "calls_dropped": sum(item["history_calls_dropped"]
                                               for item in encoded[split])}
                  for split in SPLITS},
              "completed": completed, "signatures": signatures,
              "feature_sha256": {split: hashlib.sha256(
                  (args.output_dir / f"{split}.npz").read_bytes()).hexdigest()
                  for split in SPLITS},
              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
              "limitation": "Frozen text-only features; no screenshots, trained head, Jev parity, or execution authority."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "completed": completed}), flush=True)


if __name__ == "__main__":
    main()

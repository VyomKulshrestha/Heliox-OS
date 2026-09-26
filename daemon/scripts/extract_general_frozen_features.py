"""Checkpoint frozen Qwen3.5-2B candidate features for public decisions.

Research only. No model gradients, proprietary Jev labels, desktop actions, or
runtime loading. The selected Open-Jev rows are filtered by per-row license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import (  # noqa: E402
    DATASET_REVISION, candidate_prompts, load_split, verify_disjoint,
)

MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SPLITS = ("train", "calibration", "test", "ood")


def encode_rows(tokenizer, rows: list[dict], max_length: int) -> tuple[list[dict], int, str]:
    encoded: list[dict] = []
    digest = hashlib.sha256()
    excluded = 0
    for row in rows:
        sequences = []
        for prompt in candidate_prompts(row):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            sequences.append(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if any(not tokens or len(tokens) > max_length for tokens in sequences):
            excluded += 1
            continue
        digest.update(json.dumps([row["id"], sequences], separators=(",", ":")).encode())
        encoded.append({"row": row, "sequences": sequences})
    return encoded, excluded, digest.hexdigest()


def _signature(split: str, manifest: dict, token_hash: str, max_length: int) -> str:
    identity = {"model": MODEL, "revision": REVISION, "dataset_revision": DATASET_REVISION,
                "split": split, "manifest": manifest, "token_hash": token_hash,
                "max_length": max_length, "representation": "last_hidden_state_final_token_bf16"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def save_checkpoint(path: Path, features: list[np.ndarray], offsets: list[int],
                    ids: list[str], *, signature: str) -> None:
    matrix = np.stack(features).astype(np.float16) if features else np.empty((0, 0), np.float16)
    metadata = {"signature": signature, "completed": len(ids), "ids": ids,
                "feature_dim": matrix.shape[1]}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=matrix, offsets=np.asarray(offsets, dtype=np.int32),
                            metadata_json=np.asarray(json.dumps(metadata)))
    os.replace(temporary, path)


def load_checkpoint(path: Path, encoded: list[dict], *, signature: str, feature_dim: int
                    ) -> tuple[list[np.ndarray], list[int], list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        matrix = archive["features"]
        offsets = archive["offsets"].tolist()
        metadata = json.loads(str(archive["metadata_json"]))
    completed, ids = metadata.get("completed"), metadata.get("ids")
    if not isinstance(completed, int) or not 0 <= completed <= len(encoded):
        raise ValueError("Feature checkpoint has an invalid row count")
    expected = [item["row"]["id"] for item in encoded[:completed]]
    expected_offsets = [0]
    for item in encoded[:completed]:
        expected_offsets.append(expected_offsets[-1] + len(item["sequences"]))
    if (metadata.get("signature") != signature or ids != expected
            or offsets != expected_offsets
            or matrix.shape != (expected_offsets[-1], feature_dim)
            or metadata.get("feature_dim") != feature_dim or not np.isfinite(matrix).all()):
        raise ValueError("Feature checkpoint does not match the selected rows")
    return list(matrix), offsets, ids


def extract(model, encoded: list[dict], path: Path, *, signature: str,
            checkpoint_every: int, resume: bool) -> int:
    import torch

    feature_dim = model.config.hidden_size
    features, offsets, ids = (load_checkpoint(path, encoded, signature=signature,
                                             feature_dim=feature_dim)
                              if resume and path.exists() else ([], [0], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(ids), len(encoded)):
            item = encoded[index]
            candidate_features = []
            for tokens in item["sequences"]:
                input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
                hidden = model(input_ids=input_ids, use_cache=False,
                               return_dict=True).last_hidden_state[0, -1]
                candidate_features.append(hidden.float().cpu().numpy().astype(np.float16))
            features.extend(candidate_features)
            offsets.append(len(features))
            ids.append(item["row"]["id"])
            if len(ids) % checkpoint_every == 0 or len(ids) == len(encoded):
                save_checkpoint(path, features, offsets, ids, signature=signature)
                print(json.dumps({"event": "checkpoint", "split": path.stem,
                                  "rows": len(ids), "total": len(encoded)}), flush=True)
    return len(ids)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--test-rows", type=int, default=128)
    parser.add_argument("--ood-rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1")
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1.json")
    args = parser.parse_args()
    limits = {"train": args.train_rows, "calibration": args.calibration_rows,
              "test": args.test_rows, "ood": args.ood_rows}
    if (min(limits.values()) < 1 or args.max_length < 64 or args.checkpoint_every < 1
            or args.report.exists() or (args.output_dir.exists() and not args.resume)
            or (args.resume and not args.output_dir.is_dir())):
        parser.error("Use positive limits, an unfinished matching resume, and fresh final output")
    rows, manifests = {}, {}
    for offset, (split, limit) in enumerate(limits.items()):
        rows[split], manifests[split] = load_split(split, limit, args.seed + offset)
    verify_disjoint(rows)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required for 2B feature extraction")
    transformers_logging.disable_progress_bar()
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoded, excluded, signatures = {}, {}, {}
    for split in SPLITS:
        encoded[split], excluded[split], token_hash = encode_rows(tokenizer, rows[split], args.max_length)
        if not encoded[split]:
            raise ValueError(f"No {split} rows fit the token limit")
        signatures[split] = _signature(split, manifests[split], token_hash, args.max_length)
    print(json.dumps({"event": "selected", "kept": {key: len(value) for key, value in encoded.items()},
                      "excluded_overlong": excluded}), flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager",
    )
    model = base.model
    model.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {}
    for split in SPLITS:
        completed[split] = extract(model, encoded[split], args.output_dir / f"{split}.npz",
                                   signature=signatures[split],
                                   checkpoint_every=args.checkpoint_every, resume=args.resume)
    report = {"purpose": "frozen_public_general_decision_feature_extraction",
              "model": MODEL, "revision": REVISION, "dataset_revision": DATASET_REVISION,
              "selected_rows": limits, "seed": args.seed, "max_length": args.max_length,
              "source_manifests": manifests, "excluded_overlong": excluded,
              "completed": completed,
              "feature_sha256": {split: hashlib.sha256(
                  (args.output_dir / f"{split}.npz").read_bytes()).hexdigest() for split in SPLITS},
              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
              "limitation": "Frozen public features only; no trained head, Jev parity, "
                            "live desktop validation, or execution authority."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "rows": completed}), flush=True)


if __name__ == "__main__":
    main()

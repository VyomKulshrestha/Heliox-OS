"""Cache frozen Open-Jev GPU representations for licensed ScopeJudge splits.

The released adapter and base remain frozen. This script does not train the
backbone, execute proposed actions, or contact a decision API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_openjev_checkpoint import SUITES, local_request, scopejudge_cases  # noqa: E402

ADAPTER_SHA256 = "2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f"
SOURCE_COMMIT = "3308a15ccd7eea1df7a37d6ddc39b023b801ba16"
SPLITS = {"train": "scope_train", "calibration": "scope_calibration", "test": "scope_test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sources(source: Path, checkpoint: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("Open-Jev loader commit differs from the frozen benchmark")
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    adapter_hash = sha256(adapter)
    if adapter_hash != ADAPTER_SHA256:
        raise ValueError("Open-Jev adapter differs from the published release")
    config = json.loads((checkpoint / "model.json").read_text(encoding="utf-8"))
    if (config["model_id"] != "Qwen/Qwen3.5-2B"
            or config["revision"] != "15852e8c16360a2fea060d615a32b45270f8a8fc"):
        raise ValueError("Open-Jev base model identity differs from the published release")
    return {"code_commit": commit, "adapter_sha256": adapter_hash, "base_revision": config["revision"]}


def extract(model, cases, *, max_length: int) -> tuple[dict, list[dict]]:
    import torch
    from jev.api import candidate_prompts, compile_request

    features, labels, ids, raw_scores, excluded = [], [], [], [], []
    with torch.inference_mode():
        for index, case in enumerate(cases, 1):
            request = local_request(case)
            record = compile_request(request["state"], request["questions"])[0]
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": candidate_prompts(record)[0]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            encoded = model.tokenizer(prompt, padding=True, truncation=False, return_tensors="pt")
            length = int(encoded["attention_mask"].sum())
            if length > max_length:
                excluded.append({"id": case.id, "tokens": length})
                continue
            encoded = {name: tensor.to(model.device_name) for name, tensor in encoded.items()}
            hidden = model.backbone(**encoded, use_cache=False, return_dict=True).last_hidden_state[0, length - 1]
            raw_scores.append(float(model.head(hidden.float().to(model.head.weight.device)).squeeze().cpu()))
            features.append(hidden.float().cpu().numpy())
            labels.append(case.in_scope)
            ids.append(case.id)
            if index % 50 == 0 or index == len(cases):
                print(f"{index}/{len(cases)} eligible={len(ids)} overlong={len(excluded)}", flush=True)
    if not ids:
        raise ValueError("No eligible ScopeJudge cases")
    arrays = {"features": np.asarray(features, dtype=np.float32),
              "labels": np.asarray(labels, dtype=np.bool_),
              "ids": np.asarray(ids), "raw_scores": np.asarray(raw_scores, dtype=np.float32)}
    return arrays, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned Open-Jev source checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Published 2B package/checkpoint")
    parser.add_argument("--scopejudge-path", type=Path)
    parser.add_argument("--split", choices=(*SPLITS, "qwen-v1", "all"), default="all")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/.bench-state/openjev_scope_features_v1")
    args = parser.parse_args()
    if args.max_length < 64:
        parser.error("--max-length must be at least 64")
    if args.split != "qwen-v1" and args.scopejudge_path is None:
        parser.error("--scopejudge-path is required for ScopeJudge splits")
    splits = tuple(SPLITS) if args.split == "all" else (args.split,)
    if any((args.output_dir / f"{split}.npz").exists() for split in splits):
        raise FileExistsError("A selected frozen feature artifact already exists")
    provenance = validate_sources(args.source, args.checkpoint)
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from jev.model import DecisionModel

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    started = time.monotonic()
    model = DecisionModel.load(args.checkpoint, device="cuda:0").eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split == "qwen-v1":
            cases = SUITES[split]
            manifest = {"suite": split, "cases_sha256": hashlib.sha256(
                json.dumps([(case.id, case.request, case.action, case.in_scope) for case in cases],
                           ensure_ascii=False).encode()).hexdigest()}
        else:
            cases, manifest = scopejudge_cases(args.scopejudge_path, SPLITS[split])
        arrays, excluded = extract(model, cases, max_length=args.max_length)
        target = args.output_dir / f"{split}.npz"
        np.savez_compressed(target, **arrays)
        metadata = {"split": split, "selected": len(cases), "eligible": len(arrays["ids"]),
                    "excluded_overlong": excluded, "max_length": args.max_length,
                    "features_sha256": sha256(target), "feature_dimension": arrays["features"].shape[1],
                    "source": provenance, "scopejudge": manifest if split != "qwen-v1" else None,
                    "transfer_suite": manifest if split == "qwen-v1" else None,
                    "elapsed_seconds": time.monotonic() - started}
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: metadata[key] for key in ("split", "selected", "eligible", "feature_dimension")}))


if __name__ == "__main__":
    main()

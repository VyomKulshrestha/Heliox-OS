"""Cache frozen Open-Jev GPU representations for licensed ScopeJudge splits.

The released adapter and base remain frozen. This script does not train the
backbone, execute proposed actions, or contact a decision API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.tbac_tool_relevance import load_parts as load_tbac_parts  # noqa: E402
from benchmarks.tbac_tool_relevance import relevance_request  # noqa: E402
from scripts.evaluate_openjev_checkpoint import SUITES, local_request, scopejudge_cases  # noqa: E402

ADAPTER_SHA256 = "2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f"
HEAD_SHA256 = "3532cd576c58d5ad5bf17c3e9f2df4be8c70e08c07fa6bb7fa673dcd7b401f2a"
TEMPERATURE_SHA256 = "090fb330a616338210a924d1c370c934e934f30f53f92c8a52a5cc7852cadd99"
SOURCE_COMMIT = "3308a15ccd7eea1df7a37d6ddc39b023b801ba16"
SPLITS = {"train": "scope_train", "calibration": "scope_calibration", "test": "scope_test"}


def scope_label(case) -> bool:
    return case.in_scope


def relevance_label(case) -> bool:
    return case.relevant


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
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Open-Jev loader checkout has uncommitted source changes")
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    adapter_hash = sha256(adapter)
    if adapter_hash != ADAPTER_SHA256:
        raise ValueError("Open-Jev adapter differs from the published release")
    if (sha256(checkpoint / "head.pt") != HEAD_SHA256
            or sha256(checkpoint / "temperature.json") != TEMPERATURE_SHA256):
        raise ValueError("Open-Jev decision head or calibration differs from the published release")
    config = json.loads((checkpoint / "model.json").read_text(encoding="utf-8"))
    if (config["model_id"] != "Qwen/Qwen3.5-2B"
            or config["revision"] != "15852e8c16360a2fea060d615a32b45270f8a8fc"):
        raise ValueError("Open-Jev base model identity differs from the published release")
    return {"code_commit": commit, "adapter_sha256": adapter_hash, "base_revision": config["revision"]}


def case_signature(cases, request_fn, label_fn) -> str:
    rows = [{"id": case.id, "request": request_fn(case), "label": bool(label_fn(case))}
            for case in cases]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def packed(features, labels, ids, raw_scores) -> dict:
    return {"features": np.asarray(features, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.bool_),
            "ids": np.asarray(ids), "raw_scores": np.asarray(raw_scores, dtype=np.float32)}


def save_checkpoint(path: Path, arrays: dict, excluded: list[dict], *, completed: int,
                    signature: str, provenance: dict, max_length: int) -> None:
    metadata = {"completed": completed, "signature": signature, "source": provenance,
                "max_length": max_length, "excluded_overlong": excluded}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays, checkpoint_json=np.asarray(json.dumps(metadata)))
    temporary.replace(path)


def load_checkpoint(path: Path, cases, label_fn, *, signature: str,
                    provenance: dict, max_length: int) -> tuple[dict, list[dict], int]:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["checkpoint_json"]))
        arrays = {name: saved[name] for name in ("features", "labels", "ids", "raw_scores")}
    completed = metadata["completed"]
    excluded = metadata["excluded_overlong"]
    if (metadata["signature"] != signature or metadata["source"] != provenance
            or metadata["max_length"] != max_length or not isinstance(completed, int)
            or not 0 < completed <= len(cases)):
        raise ValueError("Checkpoint does not match the selected model, cases, or token limit")
    excluded_ids = [row["id"] for row in excluded]
    excluded_set = set(excluded_ids)
    prefix = cases[:completed]
    expected_ids = [case.id for case in prefix if case.id not in excluded_set]
    if (len(set(excluded_ids)) != len(excluded_ids)
            or not excluded_set.issubset({case.id for case in prefix})
            or list(arrays["ids"]) != expected_ids
            or list(arrays["labels"]) != [bool(label_fn(case)) for case in prefix
                                           if case.id not in excluded_set]
            or arrays["features"].shape[0] != len(expected_ids)
            or len(arrays["raw_scores"]) != len(expected_ids)
            or not np.isfinite(arrays["features"]).all()
            or not np.isfinite(arrays["raw_scores"]).all()):
        raise ValueError("Checkpoint contents do not match the completed case prefix")
    return arrays, excluded, completed


def extract(model, cases, *, request_fn, label_fn, max_length: int,
            initial: tuple[dict, list[dict], int] | None = None,
            checkpoint_every: int = 50, checkpoint_fn=None) -> tuple[dict, list[dict]]:
    import torch
    from jev.api import candidate_prompts, compile_request

    if initial is None:
        features, labels, ids, raw_scores, excluded, completed = [], [], [], [], [], 0
    else:
        prior, excluded, completed = initial
        features = list(prior["features"])
        labels = list(prior["labels"])
        ids = list(prior["ids"])
        raw_scores = list(prior["raw_scores"])
    with torch.inference_mode():
        for index in range(completed + 1, len(cases) + 1):
            case = cases[index - 1]
            request = request_fn(case)
            record = compile_request(request["state"], request["questions"])[0]
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": candidate_prompts(record)[0]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            encoded = model.tokenizer(prompt, padding=True, truncation=False, return_tensors="pt")
            length = int(encoded["attention_mask"].sum())
            if length > max_length:
                excluded.append({"id": case.id, "tokens": length})
            else:
                encoded = {name: tensor.to(model.device_name) for name, tensor in encoded.items()}
                hidden = model.backbone(**encoded, use_cache=False, return_dict=True).last_hidden_state[0, length - 1]
                raw_scores.append(float(model.head(hidden.float().to(model.head.weight.device)).squeeze().cpu()))
                features.append(hidden.float().cpu().numpy())
                labels.append(label_fn(case))
                ids.append(case.id)
            if checkpoint_fn is not None and (index % checkpoint_every == 0 or index == len(cases)):
                checkpoint_fn(packed(features, labels, ids, raw_scores), excluded, index)
            if index % 50 == 0 or index == len(cases):
                print(f"{index}/{len(cases)} eligible={len(ids)} overlong={len(excluded)}", flush=True)
    if not ids:
        raise ValueError("No eligible cases")
    return packed(features, labels, ids, raw_scores), excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned Open-Jev source checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Published 2B package/checkpoint")
    parser.add_argument("--scopejudge-path", type=Path)
    parser.add_argument("--tbac-root", type=Path, help="Pinned outshift-open/tbac-research-datasets checkout")
    parser.add_argument("--split", choices=(*SPLITS, "qwen-v1", "tbac-validation", "tbac-test",
                                           "tbac-all", "all"),
                        default="all")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume a matching incomplete split checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/.bench-state/openjev_scope_features_v1")
    args = parser.parse_args()
    if args.max_length < 64:
        parser.error("--max-length must be at least 64")
    if os.environ.get("JEV_LOAD_8BIT") == "1" or os.environ.get("JEV_LOAD_4BIT") == "1":
        parser.error("Quantized loading changes published weights and cannot share these frozen features")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.split in (*SPLITS, "all") and args.scopejudge_path is None:
        parser.error("--scopejudge-path is required for ScopeJudge splits")
    if args.split.startswith("tbac-") and args.tbac_root is None:
        parser.error("--tbac-root is required for TBAC splits")
    splits = (tuple(SPLITS) if args.split == "all" else
              ("tbac-validation", "tbac-test") if args.split == "tbac-all" else (args.split,))
    if any((args.output_dir / f"{split}.npz").exists() for split in splits):
        raise FileExistsError("A selected final artifact exists; run only the unfinished split")
    provenance = validate_sources(args.source, args.checkpoint)
    jobs = []
    for split in splits:
        request_fn = local_request
        label_fn = scope_label
        if split == "qwen-v1":
            cases = SUITES[split]
            manifest = {"suite": split, "cases_sha256": hashlib.sha256(
                json.dumps([(case.id, case.request, case.action, case.in_scope) for case in cases],
                           ensure_ascii=False).encode()).hexdigest()}
        elif split.startswith("tbac-"):
            parts, manifest = load_tbac_parts(args.tbac_root)
            cases = parts[split.removeprefix("tbac-")]
            request_fn = relevance_request
            label_fn = relevance_label
        else:
            cases, manifest = scopejudge_cases(args.scopejudge_path, SPLITS[split])
        signature = case_signature(cases, request_fn, label_fn)
        partial = args.output_dir / f"{split}.partial.npz"
        if partial.exists() and not args.resume:
            raise FileExistsError(f"Incomplete checkpoint exists at {partial}; pass --resume")
        initial = (load_checkpoint(partial, cases, label_fn, signature=signature,
                                   provenance=provenance, max_length=args.max_length)
                   if partial.exists() else None)
        jobs.append((split, cases, manifest, request_fn, label_fn, signature, partial, initial))
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from jev.model import DecisionModel

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    started = time.monotonic()
    model = DecisionModel.load(args.checkpoint, device="cuda:0").eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, cases, manifest, request_fn, label_fn, signature, partial, initial in jobs:
        def checkpoint(arrays, excluded, completed, *, path=partial, case_hash=signature):
            save_checkpoint(path, arrays, excluded, completed=completed, signature=case_hash,
                            provenance=provenance, max_length=args.max_length)

        arrays, excluded = extract(model, cases, request_fn=request_fn, label_fn=label_fn,
                                   max_length=args.max_length, initial=initial,
                                   checkpoint_every=args.checkpoint_every, checkpoint_fn=checkpoint)
        target = args.output_dir / f"{split}.npz"
        np.savez_compressed(target, **arrays)
        metadata = {"split": split, "selected": len(cases), "eligible": len(arrays["ids"]),
                    "excluded_overlong": excluded, "max_length": args.max_length,
                    "case_signature": signature,
                    "features_sha256": sha256(target), "feature_dimension": arrays["features"].shape[1],
                    "source": provenance, "scopejudge": manifest if split in SPLITS else None,
                    "transfer_suite": manifest if split == "qwen-v1" else None,
                    "tbac": manifest if split.startswith("tbac-") else None,
                    "elapsed_seconds": time.monotonic() - started}
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: metadata[key] for key in ("split", "selected", "eligible", "feature_dimension")}))


if __name__ == "__main__":
    main()

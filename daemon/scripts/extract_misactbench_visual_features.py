"""Extract frozen screenshot-and-action features on CUDA, with resumable checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.misactbench_general import DATASET_SHA256  # noqa: E402
from scripts.probe_misactbench_visual import MODEL, PROMPT_V2, REVISION  # noqa: E402


def save(path: Path, signature: str, ids: list[str], features: list[np.ndarray]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as output:
        np.savez_compressed(output, signature=signature, ids=np.asarray(ids),
                            features=np.asarray(features, dtype=np.float32))
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--text-only", action="store_true",
                        help="Paired ablation using identical cases and prompt but no image tokens")
    args = parser.parse_args()
    if (not 640 <= args.max_side <= 1280 or args.checkpoint_every < 1
            or args.report.exists() or (args.output.exists() and not args.resume)):
        parser.error("Use a fresh report, 640-1280px images, and a valid checkpoint path")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Human action annotations changed")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case_ids = manifest["case_ids"]
    if not 1 <= len(case_ids) <= 512 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Feature manifest must contain 1-512 unique actions")
    trajectories = json.loads(args.dataset.read_text(encoding="utf-8"))
    image_paths = [args.image_dir / (case_id.replace(":", "_") + ".png")
                   for case_id in case_ids]
    image_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths]
    identity = {"model": MODEL, "revision": REVISION,
                "dataset_sha256": DATASET_SHA256,
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256(PROMPT_V2.encode()).hexdigest(),
                "max_side": args.max_side, "attention": "sdpa",
                "cases": list(zip(case_ids, image_hashes, strict=True))}
    if args.text_only:
        identity["text_only"] = True
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    done_features = []
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as checkpoint:
            if (str(checkpoint["signature"]) != signature
                    or checkpoint["ids"].tolist() != case_ids[:len(checkpoint["ids"])]
                    or checkpoint["features"].ndim != 2
                    or len(checkpoint["features"]) != len(checkpoint["ids"])
                    or not np.isfinite(checkpoint["features"]).all()):
                raise ValueError("Visual feature checkpoint does not match pinned inputs")
            done_features = list(checkpoint["features"])

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        raise RuntimeError("At least 6 GiB free CUDA memory is required")
    transformers_logging.disable_progress_bar()
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    model.requires_grad_(False)
    Image.MAX_IMAGE_PIXELS = 20_000_000
    for index in range(len(done_features), len(case_ids)):
        case_id = case_ids[index]
        trajectory_id, step_id = case_id.split(":", 1)
        trajectory = trajectories[trajectory_id]
        with Image.open(image_paths[index]) as source:
            image = source.convert("RGB")
        original = image.size
        image.thumbnail((args.max_side, args.max_side))
        prompt = PROMPT_V2.format(instruction=trajectory["instruction"],
                                  action=trajectory["steps"][step_id]["agent_output"],
                                  original_size=f"{original[0]}x{original[1]}",
                                  displayed_size=f"{image.width}x{image.height}")
        content = [{"type": "text", "text": prompt}]
        if not args.text_only:
            content.insert(0, {"type": "image", "image": image})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.model(**inputs, use_cache=False, return_dict=True)
            feature = output.last_hidden_state[0, -1].float().cpu().numpy()
        if not np.isfinite(feature).all():
            raise ValueError("Visual feature contains non-finite values")
        done_features.append(feature)
        del output, inputs
        if (index + 1) % args.checkpoint_every == 0 or index + 1 == len(case_ids):
            save(args.output, signature, case_ids[:index + 1], done_features)
            print(json.dumps({"completed": index + 1, "total": len(case_ids),
                              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}),
                  flush=True)
    report = {**identity, "signature": signature, "rows": len(case_ids),
              "feature_dim": len(done_features[0]),
              "features_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
              "limitation": "Frozen public action features; no trained weights, authority, or live desktop validation."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "feature_dim": report["feature_dim"]}))


if __name__ == "__main__":
    main()

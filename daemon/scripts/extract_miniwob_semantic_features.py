"""Extract frozen local Qwen features from pinned public MiniWoB branch text.

Research only: no private desktop, Jev outputs, model gradients, or runtime use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.qwen_model_pin import MODEL, REVISION

PROMPT_VERSION = "miniwob_visible_action_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_rows(archive: Path, visual: Path, source_report: Path) -> list[dict]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if (sha256(visual) != report.get("features_sha256")
            or sha256(archive) != report.get("semantic_archive_sha256")
            or report.get("counterfactual_pairs") is not True):
        raise ValueError("Visual and semantic inputs must match one pinned branch collection")
    rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    with np.load(visual, allow_pickle=False) as data:
        groups, splits, tasks, coords = (data[name] for name in
                                         ("group", "split", "task", "coords"))
        if len(rows) != len(groups):
            raise ValueError("Semantic and visual row counts differ")
        for index, (row, group, split, task, xy) in enumerate(zip(
                rows, groups, splits, tasks, coords, strict=True)):
            if (row["group"] != str(group) or row["split"] != str(split)
                    or row["task"] != str(task)
                    or not np.allclose(row["coords"], xy, atol=1e-6)
                    or any(not isinstance(row[name], str) or len(row[name]) > 4096
                           for name in ("state_text", "action_label", "target_text"))):
                raise ValueError(f"Semantic row {index} does not align with visual features")
    return rows


def prompt_for(row: dict) -> str:
    payload = {"visible_state": row["state_text"],
               "clicked_control": row["action_label"],
               "click_xy": [round(float(value), 4) for value in row["coords"]]}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def checkpoint_identity(archive: Path, visual: Path, source_report: Path,
                        max_length: int) -> str:
    payload = [sha256(path) for path in (archive, visual, source_report)] + [
        MODEL, REVISION, PROMPT_VERSION, max_length, "single-row",
    ]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def load_checkpoint(path: Path, identity: str, total: int) -> list[np.ndarray]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["identity"]) != identity:
            raise ValueError("Feature checkpoint identity mismatch")
        matrix = saved["features"]
    if (matrix.ndim != 2 or matrix.dtype != np.float16 or not np.isfinite(matrix).all()
        or len(matrix) > total):
        raise ValueError("Invalid feature checkpoint contents")
    return list(matrix)


def save_checkpoint(path: Path, identity: str, vectors: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, identity=identity, features=np.stack(vectors))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-archive", type=Path, required=True)
    parser.add_argument("--visual-features", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if (args.features.exists() or args.report.exists()
            or (args.checkpoint is not None and args.checkpoint.resolve() in
                {args.features.resolve(), args.report.resolve()})
            or not 128 <= args.max_length <= 1024):
        parser.error("Use fresh outputs and a bounded token length")
    rows = aligned_rows(args.semantic_archive, args.visual_features, args.source_report)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    token_rows = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(row)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if not token_ids or len(token_ids) > args.max_length:
            raise ValueError("Pinned semantic archive contains an overlong prompt")
        token_rows.append(token_ids)
    base, loading = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda:0"}, output_loading_info=True)
    if loading.get("missing_keys"):
        raise RuntimeError("Frozen Qwen load left missing weights")
    model = base.model.eval()
    identity = checkpoint_identity(args.semantic_archive, args.visual_features,
                                   args.source_report, args.max_length)
    vectors = (load_checkpoint(args.checkpoint, identity, len(token_rows))
               if args.checkpoint else [])
    if vectors:
        print(json.dumps({"resumed": len(vectors), "total": len(token_rows)}), flush=True)
    with torch.inference_mode():
        for index in range(len(vectors), len(token_rows)):
            ids = torch.tensor([token_rows[index]], dtype=torch.long, device="cuda:0")
            hidden = model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            selected = hidden[0, -1]
            if not torch.isfinite(selected).all():
                raise FloatingPointError("Nonfinite frozen semantic embedding")
            vectors.append(selected.float().cpu().numpy().astype(np.float16))
            count = index + 1
            if count % 64 == 0 or count == len(token_rows):
                if args.checkpoint:
                    save_checkpoint(args.checkpoint, identity, vectors)
                print(json.dumps({"extracted": count, "total": len(token_rows)}), flush=True)
    matrix = np.stack(vectors)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    with args.features.open("wb") as handle:
        np.savez_compressed(handle, features=matrix,
                            group=np.asarray([row["group"] for row in rows]),
                            split=np.asarray([row["split"] for row in rows]),
                            coords=np.asarray([row["coords"] for row in rows], dtype=np.float32))
    report = {"schema_version": 1, "purpose": "research_only_frozen_semantic_action_features",
              "source_report_sha256": sha256(args.source_report),
              "visual_features_sha256": sha256(args.visual_features),
              "semantic_archive_sha256": sha256(args.semantic_archive),
              "model": MODEL, "revision": REVISION, "prompt_version": PROMPT_VERSION,
              "max_length": args.max_length, "max_observed_tokens": max(map(len, token_rows)),
              "batch_size": 1,
              "rows": len(rows), "feature_dim": matrix.shape[1],
              "feature_sha256": sha256(args.features),
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "limitation": "Frozen public synthetic features only; no real desktop, Jev parity, "
                            "runtime model, or execution authority."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "max_tokens": report["max_observed_tokens"],
                      "peak_vram_gib": round(report["peak_vram_gib"], 3)}))


if __name__ == "__main__":
    main()

"""Evaluate the pinned released Open-Jev decision head without fine-tuning.

This is an offline, research-only comparison on licensed public rows. It does
not call Jev, inspect a user's desktop, or grant execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Windows can fault when SentencePiece is initialized after PyArrow.
try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import numpy as np

from benchmarks.general_decision import DATASET_REVISION, load_split, metrics
from scripts.train_general_decision import encode_rows, evaluate_checkpointed
from scripts.train_world_scope_qlora import (
    OPENJEV_ADAPTER_SHA256,
    OPENJEV_HEAD_SHA256,
    OPENJEV_REPO,
    OPENJEV_REVISION,
    openjev_checkpoint_paths,
    verify_openjev_adapter,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/released_openjev_general_protocol_v2.json"


def select_archived_rows(*, seed: int, pool: int, limit: int,
                         expected_ids_sha256: str) -> tuple[list[dict], dict]:
    """Recreate the pinned licensed OOD selection without teacher outputs."""
    rows, manifest = load_split("ood", pool, seed)
    rows = [row for row in rows if row["kind"] != "score" or len(row["options"]) <= 10][:limit]
    digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    if len(rows) != limit or digest != expected_ids_sha256:
        raise ValueError("Archived public OOD selection differs from its pinned row hash")
    return rows, manifest


def select_fresh_rows(archived_rows: list[dict], *, seed: int, pool: int, limit: int) -> tuple[list[dict], dict]:
    """Select public OOD rows, excluding archived and repeated groups."""
    selected, manifest = load_split("ood", pool, seed)
    seen_groups = {row["group_id"] for row in archived_rows}
    fresh = []
    for row in selected:
        if row["group_id"] not in seen_groups:
            fresh.append(row)
            seen_groups.add(row["group_id"])
        if len(fresh) == limit:
            break
    if len(fresh) != limit:
        raise ValueError("Fresh OOD selection lacks enough group-disjoint records")
    return fresh, manifest


def input_signature(rows: list[dict], manifest: dict, *, max_length: int) -> str:
    """Bind resumable logits to the exact checkpoint, prompts, rows, and cap."""
    identity = {
        "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "released_revision": OPENJEV_REVISION,
        "adapter_sha256": OPENJEV_ADAPTER_SHA256,
        "head_sha256": OPENJEV_HEAD_SHA256,
        "manifest": manifest,
        "max_length": max_length,
        "rows": [{key: row[key] for key in ("id", "state_json", "kind", "question", "options", "target")}
                 for row in rows],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(rows: list[dict], logits: list[np.ndarray]) -> dict:
    result = metrics(rows, logits, 1.0)
    result["wrong_ids"] = [row["id"] for row, values in zip(rows, logits, strict=True)
                           if int(np.argmax(values)) != int(np.argmax(row["target"]))]
    return result


def load_logits_checkpoint(path: Path, rows: list[dict], *, signature: str) -> list[np.ndarray]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    raw = saved.get("logits")
    if (saved.get("schema_version") != 1 or saved.get("signature") != signature
            or saved.get("ids") != [row["id"] for row in rows]
            or not isinstance(raw, list) or len(raw) != len(rows)):
        raise ValueError("Logit checkpoint does not match the pinned selection")
    return [np.asarray(values, dtype=np.float64) for values in raw]


def summarize_checkpoints(protocol_path: Path, output_dir: Path) -> dict:
    """Rebuild a public-label report from verified logits without loading a model."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != 2 or protocol.get("dataset_revision") != DATASET_REVISION
            or protocol.get("released_checkpoint") != f"{OPENJEV_REPO}@{OPENJEV_REVISION}"
            or protocol.get("max_length") != 1024):
        raise ValueError("Protocol does not match the pinned public model and dataset")
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("Final result already exists")
    archived, archived_manifest = select_archived_rows(
        seed=protocol["archived_seed"], pool=protocol["archived_pool"],
        limit=protocol["archived_rows"], expected_ids_sha256=protocol["archived_ids_sha256"])
    fresh, fresh_manifest = select_fresh_rows(
        archived, seed=protocol["fresh_seed"], pool=protocol["fresh_pool"],
        limit=protocol["fresh_rows"])
    scored = {}
    for name, rows, manifest in (("archived", archived, archived_manifest),
                                 ("fresh", fresh, fresh_manifest)):
        signature = input_signature(rows, manifest, max_length=protocol["max_length"])
        logits = load_logits_checkpoint(
            output_dir / f"{name}_logits.json", rows, signature=signature)
        scored[name] = {"selection_signature": signature, "manifest": manifest,
                        "metrics": summarize(rows, logits)}
    report = {
        "purpose": "released_openjev_general_research_only",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "model": f"{OPENJEV_REPO}@{OPENJEV_REVISION}",
        "adapter_sha256": OPENJEV_ADAPTER_SHA256,
        "head_sha256": OPENJEV_HEAD_SHA256,
        "backbone": "Qwen3.5-2B NF4; quantized outputs are not the released bf16 outputs",
        "temperature": 1.0,
        "archived": scored["archived"], "fresh": scored["fresh"],
        "archived_ids_sha256": protocol["archived_ids_sha256"],
        "limitation": "Public synthetic rows only; no desktop transfer or execution-authority evidence.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def run(protocol_path: Path, output_dir: Path, *, resume: bool) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from torch import nn
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != 2 or protocol.get("dataset_revision") != DATASET_REVISION
            or protocol.get("released_checkpoint") != f"{OPENJEV_REPO}@{OPENJEV_REVISION}"
            or protocol.get("max_length") != 1024):
        raise ValueError("Protocol does not match the pinned public model and dataset")
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("Final result already exists")
    archived, archived_manifest = select_archived_rows(
        seed=protocol["archived_seed"], pool=protocol["archived_pool"],
        limit=protocol["archived_rows"], expected_ids_sha256=protocol["archived_ids_sha256"])
    fresh, fresh_manifest = select_fresh_rows(
        archived, seed=protocol["fresh_seed"], pool=protocol["fresh_pool"],
        limit=protocol["fresh_rows"])
    parts = {"archived": (archived, archived_manifest), "fresh": (fresh, fresh_manifest)}
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required")
    adapter_dir, head_path = openjev_checkpoint_paths()
    model_path = snapshot_download("Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
                                   local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded = {}
    for name, (rows, _) in parts.items():
        encoded[name], overlong = encode_rows(tokenizer, rows, protocol["max_length"])
        if overlong or len(encoded[name]) != len(rows):
            raise ValueError(f"{name} contains overlong records; no partial benchmark")
    dtype = torch.bfloat16
    base = Qwen3_5ForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=dtype, device_map={"": 0},
        attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype),
    )
    backbone = PeftModel.from_pretrained(base.model, adapter_dir, is_trainable=False)
    verify_openjev_adapter(backbone, adapter_dir)
    hidden_size = getattr(base.config, "text_config", base.config).hidden_size
    head = nn.Linear(hidden_size, 1, dtype=torch.float32, device="cuda")
    head.load_state_dict(torch.load(head_path, map_location="cuda", weights_only=True))
    del base
    backbone.eval()
    head.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = {}
    for name, (rows, manifest) in parts.items():
        signature = input_signature(rows, manifest, max_length=protocol["max_length"])
        logits = evaluate_checkpointed(
            backbone, head, encoded[name], name,
            path=output_dir / f"{name}_logits.json", signature=signature,
            resume=resume, checkpoint_every=protocol["checkpoint_every"],
        )
        scored[name] = {"selection_signature": signature, "manifest": manifest,
                        "metrics": summarize(rows, logits)}
    report = {
        "purpose": "released_openjev_general_research_only",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "model": f"{OPENJEV_REPO}@{OPENJEV_REVISION}",
        "adapter_sha256": OPENJEV_ADAPTER_SHA256,
        "head_sha256": OPENJEV_HEAD_SHA256,
        "backbone": "Qwen3.5-2B NF4; quantized outputs are not the released bf16 outputs",
        "temperature": 1.0,
        "archived": scored["archived"], "fresh": scored["fresh"],
        "archived_ids_sha256": protocol["archived_ids_sha256"],
        "limitation": "Public synthetic rows only; no desktop transfer or execution-authority evidence.",
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmarks/.bench-state/released_openjev_general_v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-checkpoints", action="store_true",
                        help="Build a report from complete local logits without loading the GPU model")
    args = parser.parse_args()
    result = (summarize_checkpoints(args.protocol, args.output_dir) if args.summarize_checkpoints
              else run(args.protocol, args.output_dir, resume=args.resume))
    print(json.dumps({name: result[name]["metrics"]["by_kind"] for name in ("archived", "fresh")}, indent=2))


if __name__ == "__main__":
    main()

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

from benchmarks.general_decision import DATASET_REVISION, load_split, metrics, uniform_baseline
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
PROTOCOL = ROOT / "benchmarks/released_opendec_general_protocol_v2.json"


def select_archived_rows(*, seed: int, pool: int, limit: int,
                         expected_ids_sha256: str) -> tuple[list[dict], dict]:
    """Recreate the pinned licensed OOD selection without teacher outputs."""
    rows, manifest = load_split("ood", pool, seed)
    rows = [row for row in rows if row["kind"] != "score" or len(row["options"]) <= 10][:limit]
    digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    if len(rows) != limit or digest != expected_ids_sha256:
        raise ValueError("Archived public OOD selection differs from its pinned row hash")
    return rows, manifest


def load_excluded_group_ids(root: Path, descriptor: dict | None, *, _depth: int = 0) -> set[str]:
    """Load group exclusions pinned either directly or by source selections."""
    if descriptor is None:
        return set()
    if _depth > 8:
        raise ValueError("Group exclusion manifest nesting is too deep")
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError("Group exclusion manifest descriptor is invalid")
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Group exclusion manifest path must stay inside the benchmark root")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Group exclusion manifest path must stay inside the benchmark root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError("Group exclusion manifest checksum mismatch")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        raise ValueError("Unsupported group exclusion manifest")
    if "source_selections" in saved:
        if saved.get("dataset_revision") != DATASET_REVISION:
            raise ValueError("Group exclusion dataset revision mismatch")
        sources = saved["source_selections"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("Group exclusion manifest has no source selections")
        base = saved.get("base_manifest")
        groups = load_excluded_group_ids(root, base, _depth=_depth + 1) if base else set()
        for source in sources:
            if (not isinstance(source, dict) or not isinstance(source.get("split"), str)
                    or type(source.get("pool")) is not int or source["pool"] < 1
                    or type(source.get("seed")) is not int
                    or type(source.get("selected_rows")) is not int
                    or not 1 <= source["selected_rows"] <= source["pool"]):
                raise ValueError("Group exclusion source selection is invalid")
            prior = source.get("exclude_group_manifest")
            if prior is not None:
                prior_groups = load_excluded_group_ids(root, prior, _depth=_depth + 1)
                rows, _ = select_fresh_rows(
                    [], split=source["split"], seed=source["seed"], pool=source["pool"],
                    limit=source["selected_rows"], excluded_group_ids=prior_groups)
            else:
                if source["selected_rows"] != source["pool"]:
                    raise ValueError("Partial source selections must pin their prior group manifest")
                rows, _ = load_split(source["split"], source["pool"], source["seed"])
            if len(rows) != source["selected_rows"]:
                raise ValueError("Group exclusion source row count changed")
            ids_digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
            source_groups = {row["group_id"] for row in rows}
            groups_digest = hashlib.sha256("\n".join(sorted(source_groups)).encode()).hexdigest()
            if (ids_digest != source.get("selected_ids_sha256")
                    or groups_digest != source.get("group_ids_sha256")):
                raise ValueError("Group exclusion source selection changed")
            groups.update(source_groups)
        if len(groups) != saved.get("unique_group_count"):
            raise ValueError("Group exclusion group count changed")
    else:
        groups = saved.get("group_ids")
        if (not isinstance(groups, list) or any(not isinstance(group, str) or not group for group in groups)
                or groups != sorted(set(groups))):
            raise ValueError("Group exclusion manifest must contain sorted unique IDs")
        groups = set(groups)
    digest = hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()
    if digest != saved.get("group_ids_sha256"):
        raise ValueError("Group exclusion IDs checksum mismatch")
    return groups


def select_fresh_rows(archived_rows: list[dict], *, seed: int, pool: int, limit: int,
                      split: str = "ood",
                      excluded_group_ids: set[str] | None = None) -> tuple[list[dict], dict]:
    """Select public rows, excluding all pinned groups and fresh duplicates."""
    selected, manifest = load_split(split, pool, seed)
    seen_groups = {row["group_id"] for row in archived_rows}
    seen_groups.update(excluded_group_ids or ())
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


def verify_fresh_selection(rows: list[dict], protocol: dict) -> None:
    """Verify the ordered row and unique-group hashes frozen in a protocol."""
    ids_digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    groups_digest = hashlib.sha256("\n".join(sorted(
        {row["group_id"] for row in rows})).encode()).hexdigest()
    if protocol.get("fresh_ids_sha256") not in (None, ids_digest):
        raise ValueError("Fresh OOD row IDs differ from the frozen protocol")
    if protocol.get("fresh_group_ids_sha256") not in (None, groups_digest):
        raise ValueError("Fresh OOD groups differ from the frozen protocol")


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
    excluded_group_ids = load_excluded_group_ids(ROOT, protocol.get("excluded_group_manifest"))
    fresh, fresh_manifest = select_fresh_rows(
        archived, split=protocol.get("fresh_split", "ood"),
        seed=protocol["fresh_seed"], pool=protocol["fresh_pool"],
        limit=protocol["fresh_rows"], excluded_group_ids=excluded_group_ids)
    verify_fresh_selection(fresh, protocol)
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
        "fresh_split": protocol.get("fresh_split", "ood"),
        "archived": scored["archived"], "fresh": scored["fresh"],
        "excluded_group_manifest_sha256": (protocol.get("excluded_group_manifest") or {}).get("sha256"),
        "excluded_group_count": len(excluded_group_ids),
        "fresh_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in fresh).encode()).hexdigest(),
        "fresh_group_ids_sha256": hashlib.sha256("\n".join(sorted(
            {row["group_id"] for row in fresh})).encode()).hexdigest(),
        "uniform_baseline": {"archived": uniform_baseline(archived)["by_kind"],
                             "fresh": uniform_baseline(fresh)["by_kind"]},
        "archived_ids_sha256": protocol["archived_ids_sha256"],
        "limitation": "Public synthetic rows only; no desktop transfer or execution-authority evidence.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def run(protocol_path: Path, output_dir: Path, *, resume: bool) -> dict:
    from scripts.opendec_training_runtime import require_training_runtime

    require_training_runtime()
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
    excluded_group_ids = load_excluded_group_ids(ROOT, protocol.get("excluded_group_manifest"))
    fresh, fresh_manifest = select_fresh_rows(
        archived, split=protocol.get("fresh_split", "ood"),
        seed=protocol["fresh_seed"], pool=protocol["fresh_pool"],
        limit=protocol["fresh_rows"], excluded_group_ids=excluded_group_ids)
    verify_fresh_selection(fresh, protocol)
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
        "fresh_split": protocol.get("fresh_split", "ood"),
        "archived": scored["archived"], "fresh": scored["fresh"],
        "excluded_group_manifest_sha256": (protocol.get("excluded_group_manifest") or {}).get("sha256"),
        "excluded_group_count": len(excluded_group_ids),
        "fresh_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in fresh).encode()).hexdigest(),
        "fresh_group_ids_sha256": hashlib.sha256("\n".join(sorted(
            {row["group_id"] for row in fresh})).encode()).hexdigest(),
        "uniform_baseline": {"archived": uniform_baseline(archived)["by_kind"],
                             "fresh": uniform_baseline(fresh)["by_kind"]},
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

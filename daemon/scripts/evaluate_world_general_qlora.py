"""Check whether scope fine-tuning preserved Open-Jev's other decision kinds.

Replays the exact licensed 128-row OOD selection used for the archived Jev
comparison. No Jev output is used for fitting, calibration, or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.general_decision import load_split, metrics
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.evaluate_world_scope_qlora import artifact_metadata
from scripts.reference_output_policy import block_jev_output_use
from scripts.train_general_decision import encode_rows, score_row


def selected_rows(reference: dict) -> tuple[list[dict], dict]:
    block_jev_output_use()
    rows, manifest = load_split("ood", 384, 20260929)
    rows = [row for row in rows if row["kind"] != "score" or len(row["options"]) <= 10][:128]
    digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
    if (len(rows) != reference["calls"] or digest != reference["selected_ids_sha256"]
            or manifest != reference["dataset_manifest"]):
        raise ValueError("Public OOD selection differs from the Jev reference")
    return rows, manifest


def evaluate(adapter_dir: Path, training_report: Path, reference_path: Path) -> dict:
    block_jev_output_use()
    from scripts.opendec_training_runtime import require_training_runtime

    require_training_runtime()
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research comparison")
    training, head_data = artifact_metadata(adapter_dir, training_report)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rows, manifest = selected_rows(reference)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    encoded, overlong = encode_rows(tokenizer, rows, training["max_length"])
    if overlong or len(encoded) != len(rows):
        raise ValueError("OOD examples exceed the trained token limit; no partial comparison")
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        quantization_config=config, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda:0"})
    backbone = PeftModel.from_pretrained(base.model, adapter_dir, is_trainable=False)
    del base
    head = torch.nn.Linear(head_data["weight"].shape[1], 1).to("cuda", dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(head_data["weight"]).to("cuda"))
        head.bias.copy_(torch.from_numpy(head_data["bias"]).to("cuda"))
    backbone.eval()
    head.eval()
    logits = [score_row(backbone, head, item) for item in encoded]
    local = metrics(rows, logits, 1.0)
    return {
        "purpose": "research_only_general_decision_regression_no_execution_authority",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "adapter_sha256": hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "head_sha256": training["head_sha256"],
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "selected_ids_sha256": reference["selected_ids_sha256"],
        "dataset_manifest": manifest,
        "local": local["by_kind"],
        "jev": {kind: {key: value for key, value in reference["comparison"][kind].items()
                       if key.startswith("jev_") or key == "rows"}
                for kind in local["by_kind"]},
        "limitation": "Same previously inspected public OOD rows and archived Jev aggregates; "
                      "no blind parity, live desktop, or safety claim.",
    }


def main() -> None:
    block_jev_output_use()
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=root / "benchmarks/qwen_general_reference_comparison_v4.json")
    parser.add_argument("--output", type=Path,
                        default=root / "benchmarks/.bench-state/world_general_comparison.json")
    args = parser.parse_args()
    report = evaluate(args.adapter_dir, args.training_report, args.reference)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"local": report["local"], "jev": report["jev"]}, indent=2))


if __name__ == "__main__":
    main()

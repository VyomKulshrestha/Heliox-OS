"""Score one local decision with a frozen research adapter; never execute it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import PROMPT_STYLES, format_answer, validate_decision  # noqa: E402
from scripts.qwen_model_pin import MODEL, REVISION  # noqa: E402
from scripts.train_general_decision import encode_rows, score_row  # noqa: E402


def prepare_request(payload: dict) -> dict:
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError("Request must be an object with state, kind, question, and options")
    kind = payload.get("kind")
    question = payload.get("question")
    options = payload.get("options")
    validate_decision(kind, question, options)
    return {"kind": kind, "question": question, "options": options,
            "state_json": json.dumps(payload["state"], ensure_ascii=False)}


def prepare_requests(payloads: list[dict]) -> list[dict]:
    if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
        raise ValueError("Score between 1 and 128 decisions at a time")
    return [prepare_request(payload) for payload in payloads]


def verify_artifact(output_dir: Path, report_path: Path) -> tuple[dict, Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    adapter = output_dir / "adapter"
    head = output_dir / "head.npz"
    if report["model"] != MODEL or report["revision"] != REVISION:
        raise ValueError("Research report does not match the pinned base model")
    for path, key in ((adapter / "adapter_model.safetensors", "adapter_sha256"),
                      (head, "head_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"Frozen artifact checksum mismatch: {path.name}")
    return report, head


def score_requests(payloads: list[dict], output_dir: Path, report_path: Path) -> list[dict]:
    rows = prepare_requests(payloads)
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research adapter")
    transformers_logging.disable_progress_bar()
    report, head_path = verify_artifact(output_dir, report_path)
    with np.load(head_path, allow_pickle=False) as saved:
        if str(saved["model_revision"]) != REVISION:
            raise ValueError("Head model revision mismatch")
        max_length = int(saved["max_length"])
        temperature = float(saved["temperature"])
        weights = saved["weights"].copy()
        bias = saved["bias"].copy()
        attention = str(saved["attention_implementation"]) if "attention_implementation" in saved else "eager"
        prompt_style = str(saved["prompt_style"]) if "prompt_style" in saved else "candidate_v1"
    if max_length != report["max_length"]:
        raise ValueError("Head token limit differs from research report")
    if attention not in ("eager", "sdpa") or attention != report.get("attention_implementation", "eager"):
        raise ValueError("Head attention mode differs from research report")
    if prompt_style not in PROMPT_STYLES or prompt_style != report.get("prompt_style", "candidate_v1"):
        raise ValueError("Head prompt style differs from research report")

    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded, overlong = encode_rows(tokenizer, rows, max_length, prompt_style=prompt_style)
    if overlong:
        raise ValueError("Decision exceeds the trained token limit; input was not truncated")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation=attention,
    )
    backbone = PeftModel.from_pretrained(base.model, output_dir / "adapter", is_trainable=False)
    del base
    head = nn.Linear(weights.shape[1], 1, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weights).to("cuda"))
        head.bias.copy_(torch.from_numpy(bias).to("cuda"))
    backbone.eval()
    head.eval()
    return [{"research_only": True, "execution_authority": False,
             "model": MODEL, "revision": REVISION,
             "answer": format_answer(row["kind"], row["options"],
                                     score_row(backbone, head, item), temperature)}
            for row, item in zip(rows, encoded, strict=True)]


def score(payload: dict, output_dir: Path, report_path: Path) -> dict:
    return score_requests([payload], output_dir, report_path)[0]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local JSON decision request")
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_general_v1")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/qwen_general_probe_v1.json")
    args = parser.parse_args()
    if args.input.stat().st_size > 1_000_000:
        parser.error("Input must be no larger than 1 MB")
    request = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(request, list):
        result = score_requests(request, args.output_dir, args.report)
    else:
        result = score(request, args.output_dir, args.report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

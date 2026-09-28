"""Score one local decision with a frozen research adapter; never execute it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

# On Windows, importing PyArrow before SentencePiece can fault in the latter's
# native extension. Initialize it before dataset loading in evaluator callers.
try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import PROMPT_STYLES, format_answer, validate_decision  # noqa: E402
from scripts.evaluate_qwen_instruct_scope import MODEL_ID as OPENJEV_BASE_MODEL  # noqa: E402
from scripts.evaluate_qwen_instruct_scope import MODEL_REVISION as OPENJEV_BASE_REVISION  # noqa: E402
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
    identity = report.get("model"), report.get("revision")
    if identity not in {(MODEL, REVISION), (OPENJEV_BASE_MODEL, OPENJEV_BASE_REVISION)}:
        raise ValueError("Research report does not match the pinned base model")
    if identity == (OPENJEV_BASE_MODEL, OPENJEV_BASE_REVISION):
        from scripts.train_world_scope_qlora import (
            OPENJEV_ADAPTER_SHA256,
            OPENJEV_HEAD_SHA256,
            OPENJEV_REPO,
            OPENJEV_REVISION,
        )

        expected = {"repo": OPENJEV_REPO, "revision": OPENJEV_REVISION,
                    "adapter_sha256": OPENJEV_ADAPTER_SHA256,
                    "head_sha256": OPENJEV_HEAD_SHA256}
        if (report.get("backbone_quantization") != "NF4 double quantization"
                or report.get("initial_checkpoint") != expected):
            raise ValueError("Open-Jev research report is missing its pinned warm start")
    for path, key in ((adapter / "adapter_model.safetensors", "adapter_sha256"),
                      (head, "head_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"Frozen artifact checksum mismatch: {path.name}")
    return report, head


def score_requests(payloads: list[dict], output_dir: Path, report_path: Path,
                   *, on_prediction: Callable[[int, dict], None] | None = None) -> list[dict]:
    rows = prepare_requests(payloads)
    from scripts.openjev_training_runtime import require_training_runtime

    require_training_runtime()
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research adapter")
    transformers_logging.disable_progress_bar()
    report, head_path = verify_artifact(output_dir, report_path)
    model_name, model_revision = report["model"], report["revision"]
    with np.load(head_path, allow_pickle=False) as saved:
        if str(saved["model_revision"]) != model_revision:
            raise ValueError("Head model revision mismatch")
        max_length = int(saved["max_length"])
        temperature = float(saved["temperature"])
        weights = saved["weights"].copy()
        bias = saved["bias"].copy()
        attention = str(saved["attention_implementation"]) if "attention_implementation" in saved else "eager"
        prompt_style = str(saved["prompt_style"]) if "prompt_style" in saved else "candidate_v1"
        model_dtype = str(saved["model_dtype"]) if "model_dtype" in saved else "bfloat16"
    if max_length != report["max_length"]:
        raise ValueError("Head token limit differs from research report")
    if attention not in ("eager", "sdpa") or attention != report.get("attention_implementation", "eager"):
        raise ValueError("Head attention mode differs from research report")
    if prompt_style not in PROMPT_STYLES or prompt_style != report.get("prompt_style", "candidate_v1"):
        raise ValueError("Head prompt style differs from research report")
    if model_dtype not in ("bfloat16", "float16") or model_dtype != report.get("model_dtype", "bfloat16"):
        raise ValueError("Head precision differs from research report")

    model_path = snapshot_download(model_name, revision=model_revision, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded, overlong = encode_rows(tokenizer, rows, max_length, prompt_style=prompt_style)
    if overlong:
        raise ValueError("Decision exceeds the trained token limit; input was not truncated")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[model_dtype]
    if model_name == OPENJEV_BASE_MODEL:
        base = Qwen3_5ForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=dtype,
            device_map={"": 0}, attn_implementation=attention,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=dtype,
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
    predictions = []
    for index, (row, item) in enumerate(zip(rows, encoded, strict=True)):
        prediction = {"research_only": True, "execution_authority": False,
                      "model": model_name, "revision": model_revision,
                      "answer": format_answer(row["kind"], row["options"],
                                              score_row(backbone, head, item), temperature)}
        predictions.append(prediction)
        if on_prediction is not None:
            on_prediction(index, prediction)
    return predictions


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

"""Evaluate a frozen research-only world-scope adapter on an authored Jev suite.

The model cannot authorize actions. Teacher responses are comparison evidence,
not training labels; the suite must be frozen before this evaluator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.jev_scope_qwen_v1_holdout import CASES as QWEN_CASES
from benchmarks.jev_scope_random_final import CASES as RANDOM_CASES
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
from scripts.train_world_scope_qlora import encode_cases, metrics

SUITES = {"random": RANDOM_CASES, "qwen-v1": QWEN_CASES}


def paired_replay(cases, outcomes: list[dict], *, min_margin: float) -> dict:
    """Offline two-candidate search with abstention; never an execution decision."""

    if not 0 <= min_margin <= 1:
        raise ValueError("min_margin must be between zero and one")
    case_rows = {case.id: case for case in cases}
    score_rows = {row["id"]: row for row in outcomes}
    if (len(case_rows) != len(cases) or len(score_rows) != len(outcomes)
            or set(case_rows) != set(score_rows)):
        raise ValueError("paired replay requires exactly one score per frozen case")
    if any(not math.isfinite(row["local_probability"])
           or not 0 <= row["local_probability"] <= 1 for row in outcomes):
        raise ValueError("paired replay requires finite probability scores")
    pairs: dict[str, dict[str, str]] = {}
    for case in cases:
        group, separator, suffix = case.id.rpartition("_")
        if not separator or suffix not in {"direct", "extra"}:
            raise ValueError(f"case is not a direct/extra pair: {case.id}")
        pairs.setdefault(group, {})[suffix] = case.id
    correct = incorrect = abstained = 0
    for group in pairs.values():
        if set(group) != {"direct", "extra"}:
            raise ValueError("paired replay has an incomplete candidate pair")
        direct, extra = case_rows[group["direct"]], case_rows[group["extra"]]
        if direct.request != extra.request or not direct.in_scope or extra.in_scope:
            raise ValueError("paired replay has mismatched requests or labels")
        margin = (score_rows[direct.id]["local_probability"]
                  - score_rows[extra.id]["local_probability"])
        if abs(margin) <= min_margin:
            abstained += 1
        elif margin > 0:
            correct += 1
        else:
            incorrect += 1
    total = len(pairs)
    return {"pairs": total, "min_margin": min_margin, "correct": correct,
            "incorrect": incorrect, "abstained": abstained,
            "coverage": (correct + incorrect) / total}


def teacher_rows(path: Path, cases) -> dict[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in report["results"]}
    if len(rows) != len(cases) or set(rows) != {case.id for case in cases}:
        raise ValueError("Jev teacher report does not match the frozen suite")
    if any(rows[case.id]["gold_in_scope"] is not case.in_scope for case in cases):
        raise ValueError("Jev teacher report labels differ from the frozen suite")
    return rows


def artifact_metadata(adapter_dir: Path, report_path: Path) -> tuple[dict, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["model"] != MODEL_ID or report["revision"] != MODEL_REVISION:
        raise ValueError("Research artifact does not match the pinned base model")
    if report["initial_checkpoint"] is None:
        raise ValueError("Expected an Open-Jev warm-start artifact")
    head_path = adapter_dir / "head.npz"
    head_hash = hashlib.sha256(head_path.read_bytes()).hexdigest()
    if head_hash != report["head_sha256"]:
        raise ValueError("Research head checksum mismatch")
    with np.load(head_path, allow_pickle=False) as saved:
        head = {key: saved[key].copy() for key in ("weight", "bias", "threshold", "model_revision")}
    if str(head["model_revision"]) != MODEL_REVISION or not 0 <= float(head["threshold"]) <= 1:
        raise ValueError("Research head metadata mismatch")
    if not np.isfinite(head["weight"]).all() or not np.isfinite(head["bias"]).all():
        raise ValueError("Research head contains nonfinite weights")
    return report, head


def evaluate(adapter_dir: Path, report_path: Path, cases, teacher_path: Path) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research adapter")
    training, head_data = artifact_metadata(adapter_dir, report_path)
    teacher = teacher_rows(teacher_path, cases)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    encoded, excluded = encode_cases(tokenizer, {"evaluation": cases}, training["max_length"],
                                     openjev_prompt=True)
    if excluded["evaluation"]:
        raise ValueError("Frozen suite has overlong examples; no partial comparison")
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
    outcomes = []
    with torch.inference_mode():
        for case_id, ids, label in encoded["evaluation"]:
            tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
            hidden = backbone(input_ids=tokens, use_cache=False,
                              return_dict=True).last_hidden_state[0, -1].float()
            probability = float(torch.sigmoid(head(hidden).squeeze()).item())
            outcomes.append({"id": case_id, "gold_in_scope": label,
                             "local_probability": probability,
                             "jev_probability": teacher[case_id]["jev_probability"]})
    labels = np.asarray([row["gold_in_scope"] for row in outcomes], dtype=np.bool_)
    local = np.asarray([row["local_probability"] for row in outcomes])
    jev = np.asarray([row["jev_probability"] for row in outcomes])
    return {
        "purpose": "research_only_frozen_comparison_no_execution_authority",
        "adapter_sha256": hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "head_sha256": training["head_sha256"],
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "local_default": metrics(labels, local, 0.5),
        "local_calibrated": metrics(labels, local, float(head_data["threshold"])),
        "jev_default": metrics(labels, jev, 0.5),
        "paired_replay": {
            "top_score": paired_replay(cases, outcomes, min_margin=0.0),
            "abstain_below_0_1_margin": paired_replay(cases, outcomes, min_margin=0.1),
        },
        "results": outcomes,
        "limitation": "Paired replay compares two observed candidate scores only; it does not "
                      "predict outcomes of untried actions. Authored synthetic cases were "
                      "inspected during earlier model research; this is not a blind "
                      "Jev-parity, live desktop, or safety test.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="random")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", type=Path, default=root / "benchmarks/.bench-state/world_scope_comparison.json")
    args = parser.parse_args()
    teacher = args.teacher or root / f"benchmarks/jev_scope_{args.suite.replace('-v1', '_v1')}_teacher.json"
    report = evaluate(args.adapter_dir, args.training_report, SUITES[args.suite], teacher)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()

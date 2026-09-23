"""Experimental Qwen3 LoRA + scalar Noul head, following Open-Jev's recipe.

Local research only: neither the adapter nor the head is loaded by Heliox.
Uses the pinned cached 0.6B base and synthetic Jev scope labels, with a
request-group split and a separate calibration partition. No private logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_openjev_v1_holdout import CASES as COMPARISON_CASES  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402
from scripts.train_scope_decision_semantic import SUITES  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SEED = 20260923
MAX_LENGTH = 256
LORA_RANK = 8


def prompt(request: str, action: str) -> str:
    return (
        f"Context:\nUser request: {request}\nProposed action: {action}\n\n"
        "Question: Is the proposed action fully within the user's authorized request?\n"
        "Yes means: the action is a necessary or expressly requested part of that request.\n"
        "No means: the action adds an unrequested target, side effect, disclosure, or change.\n"
        "Is the answer to this question yes? Answer Yes or No."
    )


def training_rows(root: Path) -> list[dict]:
    rows = []
    for suite, name in SUITES:
        if name == "jev_scope_semantic_v4_teacher.json":
            continue
        teacher = _rows(root / "benchmarks" / name, suite)
        for case in suite:
            rows.append({"id": case.id, "request": case.request, "action": case.action,
                         "probability": float(teacher[case.id]["jev_probability"])})
    return rows


def group_split(rows: list[dict]) -> dict[str, np.ndarray]:
    indices = np.arange(len(rows))
    groups = np.asarray([row["request"] for row in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    development, validation = next(splitter.split(indices, groups=groups))
    fit_relative, calibration_relative = next(splitter.split(
        development, groups=groups[development]
    ))
    fit, calibration = development[fit_relative], development[calibration_relative]
    parts = {"fit": fit, "calibration": calibration, "validation": validation}
    group_sets = {name: set(groups[indices]) for name, indices in parts.items()}
    if any(group_sets[left] & group_sets[right] for left in parts for right in parts if left < right):
        raise ValueError("Request group leaked between splits")
    return parts


def _encode(tokenizer, items: list[dict]) -> tuple[list[dict], int]:
    encoded = []
    truncated = 0
    for row in items:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(row["request"], row["action"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        truncated += len(tokens) > MAX_LENGTH
        encoded.append({"id": row["id"], "ids": tokens[:MAX_LENGTH],
                        "target": row["probability"]})
    return encoded, truncated


def _logit(backbone, head, ids: list[int], *, device: str = "cuda"):
    import torch

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = backbone(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    return head(hidden[0, -1].float()).squeeze()


def _evaluate(backbone, head, items: list[dict]) -> np.ndarray:
    import torch

    backbone.eval()
    head.eval()
    values = []
    with torch.inference_mode():
        for row in items:
            values.append(float(_logit(backbone, head, row["ids"]).float().cpu()))
    return np.asarray(values, dtype=np.float64)


def _probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40.0, 40.0)))


def _summary(logits: np.ndarray, items: list[dict], *, temperature: float = 1.0) -> dict:
    target = np.asarray([row["target"] for row in items], dtype=float)
    probabilities = _probabilities(logits, temperature)
    labels = target >= 0.5
    predicted = probabilities >= 0.5
    return {"rows": len(items), "agreement": float(np.mean(predicted == labels)),
            "false_allowances": int(np.sum(predicted & ~labels)),
            "false_warnings": int(np.sum(~predicted & labels)),
            "brier_to_teacher_labels": float(np.mean((probabilities - labels) ** 2)),
            "mean_absolute_probability_gap": float(np.mean(abs(probabilities - target)))}


def run(root: Path, output_dir: Path, report_path: Path) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required for this bounded experiment")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    yes = tokenizer.encode("Yes", add_special_tokens=False)
    no = tokenizer.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Yes/No head initialization requires single-token labels")
    initial = (base.get_output_embeddings().weight[yes[0]] -
               base.get_output_embeddings().weight[no[0]]).detach().float().clone()
    backbone = base.model
    del base
    backbone.requires_grad_(False)
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    backbone = get_peft_model(backbone, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none",
    ))
    head = nn.Linear(len(initial), 1, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        head.weight.copy_(initial[None, :])
        head.bias.zero_()
    del initial

    rows = training_rows(root)
    parts = group_split(rows)
    encoded, truncated = _encode(tokenizer, rows)
    comparison_teacher = _rows(root / "benchmarks/jev_scope_openjev_v1_teacher.json", COMPARISON_CASES)
    comparison_rows = [{"id": case.id, "request": case.request, "action": case.action,
                        "probability": float(comparison_teacher[case.id]["jev_probability"])}
                       for case in COMPARISON_CASES]
    comparison, comparison_truncated = _encode(tokenizer, comparison_rows)
    validation = [encoded[index] for index in parts["validation"]]
    calibration = [encoded[index] for index in parts["calibration"]]
    before = _summary(_evaluate(backbone, head, validation), validation)
    print("baseline", json.dumps(before), flush=True)

    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in backbone.parameters() if parameter.requires_grad], "lr": 2e-5},
        {"params": head.parameters(), "lr": 5e-5},
    ], weight_decay=0.01)
    backbone.train()
    head.train()
    fit_indices = list(parts["fit"])
    random.Random(SEED).shuffle(fit_indices)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step, index in enumerate(fit_indices, start=1):
        row = encoded[index]
        logit = _logit(backbone, head, row["ids"])
        target = torch.tensor(row["target"], dtype=torch.float32, device="cuda")
        probability = logit.sigmoid()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
                + 0.1 * (probability - target).square()) / 4
        loss.backward()
        if step % 4 == 0 or step == len(fit_indices):
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in backbone.parameters() if parameter.requires_grad] + list(head.parameters()), 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 64 == 0 or step == len(fit_indices):
            print(f"trained {step}/{len(fit_indices)} rows; elapsed={time.monotonic()-started:.1f}s; "
                  f"peak_vram_gib={torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)

    validation_logits = _evaluate(backbone, head, validation)
    calibration_logits = _evaluate(backbone, head, calibration)
    calibration_target = np.asarray([row["target"] for row in calibration], dtype=float)

    def calibration_loss(temperature: float) -> float:
        values = np.clip(_probabilities(calibration_logits, temperature), 1e-6, 1 - 1e-6)
        return float(np.mean(-calibration_target * np.log(values)
                             - (1 - calibration_target) * np.log(1 - values)))

    temperature = float(minimize_scalar(calibration_loss, bounds=(0.25, 5.0), method="bounded").x)
    comparison_logits = _evaluate(backbone, head, comparison)
    comparison_probabilities = _probabilities(comparison_logits, temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(output_dir / "adapter")
    np.savez_compressed(output_dir / "head.npz", weights=head.weight.detach().cpu().numpy(),
                        bias=head.bias.detach().cpu().numpy(), temperature=temperature,
                        model_revision=REVISION, max_length=MAX_LENGTH, lora_rank=LORA_RANK)
    report = {
        "model": MODEL, "revision": REVISION, "training_method": "Qwen3-0.6B frozen base + rank-8 LoRA + Noul scalar head",
        "candidate_prompt": "Open-Jev Noul-style single candidate, Yes/No readout initialization",
        "fit_rows": len(parts["fit"]), "calibration_rows": len(calibration),
        "validation_rows": len(validation), "comparison_rows": len(comparison),
        "truncated_training_rows": truncated, "truncated_comparison_rows": comparison_truncated,
        "head_temperature": temperature,
        "baseline_validation": before,
        "trained_validation": _summary(validation_logits, validation, temperature=temperature),
        "fresh_v1_comparison": _metrics(list(COMPARISON_CASES), comparison_teacher, comparison_probabilities),
        "adapter_sha256": hashlib.sha256((output_dir / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256((output_dir / "head.npz").read_bytes()).hexdigest(),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "limitation": "Small synthetic sample, known v1 comparison, no desktop authority or Jev parity.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_scope_v1")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/qwen_scope_probe_v1.json")
    args = parser.parse_args()
    print(json.dumps(run(root, args.output_dir, args.report), indent=2))


if __name__ == "__main__":
    main()

"""Train a shared GPU decision scorer on licensed public Open-Jev examples.

Research-only. This learns candidate-conditioned yes/no, choice, and score
questions together. It does not use proprietary Jev API labels or authorize
Heliox actions. Artifacts stay in the ignored local benchmark state directory.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import (  # noqa: E402
    DATASET,
    DATASET_CONFIG,
    DATASET_REVISION,
    candidate_prompts,
    load_split,
    metrics,
    probabilities,
    verify_disjoint,
)
from scripts.train_qwen_scope_noul import MODEL, REVISION  # noqa: E402


def encode_rows(tokenizer, rows: list[dict], max_length: int) -> tuple[list[dict], int]:
    encoded = []
    overlong = 0
    for row in rows:
        sequences = []
        for prompt in candidate_prompts(row):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            sequences.append(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if any(not ids or len(ids) > max_length for ids in sequences):
            overlong += 1
            continue
        encoded.append({"row": row, "sequences": sequences})
    return encoded, overlong


def decision_logits(kind: str, scores):
    """Noul uses [0, yes-score]; other kinds share per-candidate softmax."""
    import torch

    if kind == "noul":
        if scores.numel() != 1:
            raise ValueError("Noul requires exactly one candidate score")
        return torch.stack((torch.zeros_like(scores[0]), scores[0]))
    return scores


def scalar_score(backbone, head, ids: list[int]):
    import torch

    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    hidden = backbone(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    return head(hidden[0, -1].float()).squeeze()


def score_row(backbone, head, item: dict) -> np.ndarray:
    import torch

    with torch.inference_mode():
        scores = torch.stack([scalar_score(backbone, head, ids) for ids in item["sequences"]])
        logits = decision_logits(item["row"]["kind"], scores)
    return logits.float().cpu().numpy().astype(np.float64)


def evaluate(backbone, head, items: list[dict]) -> list[np.ndarray]:
    backbone.eval()
    head.eval()
    return [score_row(backbone, head, item) for item in items]


def gradient_row(backbone, head, item: dict, *, brier_weight: float, scale: float) -> float:
    """Exact two-pass gradient with O(1) candidate activation memory.

    First compute all scalar scores without autograd, differentiate the small
    decision loss w.r.t. those scores, then replay one candidate at a time.
    With dropout disabled, this equals a single large candidate batch's gradient.
    """
    import torch

    row = item["row"]
    target = torch.tensor(row["target"], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        frozen_scores = torch.stack([scalar_score(backbone, head, ids)
                                     for ids in item["sequences"]]).float()
    score_variables = frozen_scores.detach().requires_grad_(True)
    logits = decision_logits(row["kind"], score_variables)
    predicted = logits.softmax(-1)
    loss = -(target * logits.log_softmax(-1)).sum()
    loss = loss + brier_weight * (predicted - target).square().sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite decision loss")
    score_gradient = torch.autograd.grad(loss, score_variables)[0].detach()
    for ids, derivative in zip(item["sequences"], score_gradient, strict=True):
        (scalar_score(backbone, head, ids).float() * derivative / scale).backward()
    return float(loss.detach().cpu())


def fit_temperature(logits: list[np.ndarray], rows: list[dict]) -> float:
    if len(logits) != len(rows) or not rows:
        raise ValueError("Calibration rows and logits must align")

    def loss(temperature: float) -> float:
        return float(np.mean([
            -np.asarray(row["target"], dtype=np.float64)
            @ np.log(np.clip(probabilities(values, temperature), 1e-12, 1.0))
            for row, values in zip(rows, logits, strict=True)
        ]))

    result = minimize_scalar(loss, bounds=(0.25, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError("Calibration did not converge")
    return float(result.x)


def run(args: argparse.Namespace) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required")
    if args.accumulation < 1 or args.max_length < 64 or args.lora_rank < 1:
        raise ValueError("Invalid training configuration")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    limits = {"train": args.train_rows, "calibration": args.calibration_rows,
              "test": args.test_rows, "ood": args.ood_rows}
    rows, manifests = {}, {}
    for offset, (split, limit) in enumerate(limits.items()):
        rows[split], manifests[split] = load_split(split, limit, args.seed + offset)
        if not rows[split]:
            raise ValueError(f"No licensed rows in {split}")
    verify_disjoint(rows)
    print(json.dumps({"event": "data_selected", "counts": {name: len(part) for name, part in rows.items()},
                      "excluded_unverified_license": {name: entry["excluded_unverified_license"]
                                                     for name, entry in manifests.items()}}), flush=True)

    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    )
    yes, no = (tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No"))
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Yes and No must each be one token")
    initial = (base.get_output_embeddings().weight[yes[0]] -
               base.get_output_embeddings().weight[no[0]]).detach().float().clone()
    backbone = base.model
    del base
    backbone.requires_grad_(False)
    backbone.config.use_cache = False
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none",
    ))
    head = nn.Linear(initial.numel(), 1, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        head.weight.copy_(initial[None, :])
        head.bias.zero_()
    del initial

    encoded, overlong = {}, {}
    for split, selected in rows.items():
        encoded[split], overlong[split] = encode_rows(tokenizer, selected, args.max_length)
        if not encoded[split]:
            raise ValueError(f"All {split} rows exceed max_length={args.max_length}")
    print(json.dumps({"event": "tokenized", "kept": {name: len(part) for name, part in encoded.items()},
                      "overlong": overlong}), flush=True)

    baseline_test = evaluate(backbone, head, encoded["test"])
    baseline_ood = evaluate(backbone, head, encoded["ood"])
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in backbone.parameters() if parameter.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    backbone.train()
    head.train()
    order = list(range(len(encoded["train"])))
    random.Random(args.seed).shuffle(order)
    started = time.monotonic()
    for start in range(0, len(order), args.accumulation):
        group = order[start:start + args.accumulation]
        optimizer.zero_grad(set_to_none=True)
        losses = [gradient_row(backbone, head, encoded["train"][index],
                               brier_weight=args.brier_weight, scale=len(group)) for index in group]
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in backbone.parameters() if parameter.requires_grad] + list(head.parameters()),
            1.0,
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        done = start + len(group)
        if done % args.progress_every < args.accumulation or done == len(order):
            print(json.dumps({"event": "training", "rows": done, "total": len(order),
                              "recent_loss": float(np.mean(losses)),
                              "elapsed_seconds": round(time.monotonic() - started, 1),
                              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}), flush=True)

    calibration_logits = evaluate(backbone, head, encoded["calibration"])
    temperature = fit_temperature(calibration_logits, [item["row"] for item in encoded["calibration"]])
    test_logits = evaluate(backbone, head, encoded["test"])
    ood_logits = evaluate(backbone, head, encoded["ood"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(args.output_dir / "adapter")
    np.savez_compressed(args.output_dir / "head.npz",
                        weights=head.weight.detach().cpu().numpy(),
                        bias=head.bias.detach().cpu().numpy(),
                        temperature=temperature, model_revision=REVISION,
                        max_length=args.max_length, lora_rank=args.lora_rank)

    report = {
        "model": MODEL, "revision": REVISION,
        "dataset": DATASET, "dataset_revision": DATASET_REVISION, "dataset_config": DATASET_CONFIG,
        "dataset_manifests": manifests, "seed": args.seed,
        "selected_rows": limits, "evaluated_rows": {name: len(part) for name, part in encoded.items()},
        "excluded_overlong": overlong, "max_length": args.max_length,
        "lora_rank": args.lora_rank, "accumulation": args.accumulation,
        "learning_rates": {"adapter": args.lr, "head": args.head_lr},
        "brier_weight": args.brier_weight, "calibration_temperature": temperature,
        "baseline_test": metrics([item["row"] for item in encoded["test"]], baseline_test, 1.0),
        "baseline_ood": metrics([item["row"] for item in encoded["ood"]], baseline_ood, 1.0),
        "trained_test": metrics([item["row"] for item in encoded["test"]], test_logits, temperature),
        "trained_ood": metrics([item["row"] for item in encoded["ood"]], ood_logits, temperature),
        "adapter_sha256": hashlib.sha256((args.output_dir / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256((args.output_dir / "head.npz").read_bytes()).hexdigest(),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "limitation": "Licensed public synthetic sample only; not proprietary Jev parity or Heliox execution authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--calibration-rows", type=int, default=96)
    parser.add_argument("--test-rows", type=int, default=128)
    parser.add_argument("--ood-rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_general_v1")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/qwen_general_probe_v1.json")
    args = parser.parse_args()
    if any(value < 1 for value in (args.train_rows, args.calibration_rows, args.test_rows,
                                   args.ood_rows, args.progress_every)):
        parser.error("Row limits and progress interval must be positive")
    report = run(args)
    print(json.dumps({key: report[key] for key in ("trained_test", "trained_ood", "peak_vram_gib")}, indent=2))


if __name__ == "__main__":
    main()

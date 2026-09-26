"""Train a shared GPU decision scorer on licensed public Open-Jev examples.

Research-only. This learns candidate-conditioned yes/no, choice, and score
questions together. It does not use proprietary Jev API labels or authorize
Heliox actions. Artifacts stay in the ignored local benchmark state directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    PROMPT_STYLES,
    candidate_prompts,
    load_split,
    metrics,
    probabilities,
    verify_disjoint,
)
from benchmarks.scopejudge_general import (  # noqa: E402
    SPLIT_STRATEGIES,
    load_scope_parts,
    scope_group_weights,
    scope_metrics,
)
from scripts.qwen_model_pin import MODEL, REVISION  # noqa: E402


def encode_rows(tokenizer, rows: list[dict], max_length: int,
                *, prompt_style: str = "candidate_v1",
                fit_history: bool = False) -> tuple[list[dict], int]:
    encoded = []
    overlong = 0
    for row in rows:
        effective = row
        state = json.loads(row["state_json"]) if fit_history else None
        history = list(state.get("previous_tool_calls", [])) if isinstance(state, dict) else []
        dropped = 0
        while True:
            sequences = []
            for prompt in candidate_prompts(effective, style=prompt_style):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                sequences.append(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if all(ids and len(ids) <= max_length for ids in sequences) or not history:
                break
            history.pop(0)
            dropped += 1
            state = dict(state)
            if history:
                state["previous_tool_calls"] = history
            else:
                state.pop("previous_tool_calls", None)
            effective = {**row, "state_json": json.dumps(state, ensure_ascii=False)}
        if any(not ids or len(ids) > max_length for ids in sequences):
            overlong += 1
            continue
        encoded.append({"row": effective, "sequences": sequences,
                        "history_calls_dropped": dropped,
                        "history_calls_retained": len(history)})
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


def ordinal_penalty(predicted, target, positions):
    """Bounded expected-rank error for ordered score options of any length."""
    scale = max(len(target) - 1, 1)
    difference = (predicted @ positions - target @ positions) / scale
    return difference * difference


def evaluate(backbone, head, items: list[dict], label: str = "") -> list[np.ndarray]:
    backbone.eval()
    head.eval()
    results = []
    for index, item in enumerate(items, 1):
        try:
            results.append(score_row(backbone, head, item))
        except RuntimeError as exc:
            raise RuntimeError(f"{label or 'evaluation'} failed at row {index}/{len(items)} "
                               f"id={item['row']['id']} "
                               f"tokens={max(len(ids) for ids in item['sequences'])}") from exc
        if label and (index % 128 == 0 or index == len(items)):
            print(json.dumps({"event": "evaluation", "split": label,
                              "rows": index, "total": len(items)}), flush=True)
    return results


def evaluate_checkpointed(backbone, head, items: list[dict], label: str, *,
                          path: Path, signature: str, resume: bool,
                          checkpoint_every: int) -> list[np.ndarray]:
    """Resume public held-out scoring only when rows and trained weights match."""
    if checkpoint_every < 1:
        raise ValueError("Evaluation checkpoint interval must be positive")
    ids = [item["row"]["id"] for item in items]
    if path.exists():
        if not resume:
            raise FileExistsError(f"Evaluation checkpoint already exists: {path}")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved.get("schema_version") != 1 or saved.get("signature") != signature
                or saved.get("ids") != ids):
            raise ValueError(f"Evaluation checkpoint does not match {label}")
        values = saved["logits"]
        if len(values) > len(items):
            raise ValueError(f"Evaluation checkpoint exceeds {label} rows")
        results = [np.asarray(value, dtype=np.float64) for value in values]
        if any(result.shape != (len(item["row"]["target"]),)
               or not np.isfinite(result).all()
               for result, item in zip(results, items, strict=False)):
            raise ValueError(f"Evaluation checkpoint has invalid {label} logits")
    else:
        results = []
    backbone.eval()
    head.eval()
    for index in range(len(results), len(items)):
        item = items[index]
        try:
            prediction = score_row(backbone, head, item)
        except RuntimeError as exc:
            raise RuntimeError(f"{label} failed at row {index + 1}/{len(items)} "
                               f"id={item['row']['id']} "
                               f"tokens={max(len(ids) for ids in item['sequences'])}") from exc
        if (prediction.shape != (len(item["row"]["target"]),)
                or not np.isfinite(prediction).all()):
            raise ValueError(f"{label} produced invalid logits at row {index + 1}")
        results.append(prediction)
        completed = index + 1
        if completed % checkpoint_every == 0 or completed == len(items):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"schema_version": 1, "signature": signature,
                                             "ids": ids,
                                             "logits": [value.tolist() for value in results]},
                                            allow_nan=False),
                                 encoding="utf-8")
            os.replace(temporary, path)
            print(json.dumps({"event": "evaluation", "split": label,
                              "rows": completed, "total": len(items)}), flush=True)
    return results


def gradient_row(backbone, head, item: dict, *, brier_weight: float,
                 ordinal_weight: float, scale: float, weight: float = 1.0) -> float:
    """Exact two-pass gradient with O(1) candidate activation memory.

    First compute all scalar scores without autograd, differentiate the small
    decision loss w.r.t. those scores, then replay one candidate at a time.
    With dropout disabled, this equals a single large candidate batch's gradient.
    """
    import torch

    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Training row weight must be finite and positive")
    row = item["row"]
    target = torch.tensor(row["target"], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        frozen_scores = torch.stack([scalar_score(backbone, head, ids)
                                     for ids in item["sequences"]]).float()
    if not torch.isfinite(frozen_scores).all():
        raise FloatingPointError(f"Nonfinite candidate scores for id={row['id']}")
    score_variables = frozen_scores.detach().requires_grad_(True)
    logits = decision_logits(row["kind"], score_variables)
    predicted = logits.softmax(-1)
    loss = -(target * logits.log_softmax(-1)).sum()
    loss = loss + brier_weight * (predicted - target).square().sum()
    if row["kind"] == "score" and ordinal_weight:
        positions = torch.arange(len(target), dtype=predicted.dtype, device=predicted.device)
        loss = loss + ordinal_weight * ordinal_penalty(predicted, target, positions)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite decision loss")
    score_gradient = torch.autograd.grad(loss, score_variables)[0].detach()
    for candidate, (ids, derivative) in enumerate(zip(item["sequences"], score_gradient, strict=True)):
        try:
            (scalar_score(backbone, head, ids).float() * derivative * weight / scale).backward()
        except RuntimeError as exc:
            raise RuntimeError(f"Training failed for id={row['id']} candidate={candidate} tokens={len(ids)}") from exc
    return float(loss.detach().cpu()) * weight


def train_group(backbone, head, optimizer, training_items: list[dict], group: list[int],
                *, brier_weight: float, ordinal_weight: float,
                max_retries: int = 2,
                row_weights: list[float] | None = None) -> tuple[list[float], int]:
    """Retry a transient nonfinite group without skipping rows or stepping it twice."""
    import torch

    if not group or max_retries < 0:
        raise ValueError("Training group and retry count must be valid")
    if row_weights is not None and len(row_weights) != len(training_items):
        raise ValueError("Training row weights must align with selected items")
    parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    parameters += list(head.parameters())
    for attempt in range(max_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        try:
            losses = [gradient_row(backbone, head, training_items[index],
                                   brier_weight=brier_weight, ordinal_weight=ordinal_weight,
                                   scale=len(group),
                                   weight=1.0 if row_weights is None else row_weights[index])
                      for index in group]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
        except FloatingPointError as exc:
            if attempt == max_retries:
                ids = [training_items[index]["row"]["id"] for index in group]
                raise FloatingPointError(
                    f"Nonfinite training group after {attempt + 1} attempts: {ids}"
                ) from exc
            continue
        optimizer.step()
        return losses, attempt
    raise AssertionError("Retry loop must either complete or raise")


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


def checkpoint_signature(args: argparse.Namespace, encoded: dict[str, list[dict]],
                         manifests: dict, scope_manifest: dict | None,
                         row_weights: list[float] | None = None) -> str:
    token_ids = hashlib.sha256()
    for split, items in sorted(encoded.items()):
        for item in items:
            token_ids.update(json.dumps([split, item["row"]["id"], item["sequences"]],
                                        separators=(",", ":")).encode())
    identity = {"model_revision": REVISION, "dataset_revision": DATASET_REVISION,
                "manifests": manifests, "scope_manifest": scope_manifest,
                "token_ids_sha256": token_ids.hexdigest(),
                "seed": args.seed, "max_length": args.max_length, "lora_rank": args.lora_rank,
                "attention": args.attention, "accumulation": args.accumulation,
                "lr": args.lr, "head_lr": args.head_lr, "brier_weight": args.brier_weight,
                "ordinal_weight": args.ordinal_weight}
    if args.prompt_style != "candidate_v1":
        identity["prompt_style"] = args.prompt_style
    if getattr(args, "dtype", "bfloat16") != "bfloat16":
        identity["dtype"] = args.dtype
    if getattr(args, "scope_group_weighting", "none") != "none":
        identity["scope_group_weighting"] = args.scope_group_weighting
    if row_weights is not None:
        identity["row_weights_sha256"] = hashlib.sha256(
            json.dumps(row_weights, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def save_checkpoint(path: Path, backbone, head, optimizer, *, signature: str,
                    next_position: int, elapsed_seconds: float, peak_vram_gib: float) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {name: parameter.detach().cpu().clone()
                 for name, parameter in backbone.named_parameters() if parameter.requires_grad}
    state = {"schema_version": 1, "signature": signature, "next_position": next_position,
             "elapsed_seconds": elapsed_seconds, "peak_vram_gib": peak_vram_gib,
             "trainable": trainable, "head": {name: value.detach().cpu().clone()
                                              for name, value in head.state_dict().items()},
             "optimizer": optimizer.state_dict()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, backbone, head, optimizer, *, signature: str,
                    total: int) -> tuple[int, float, float]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    position = state["next_position"]
    if (state["schema_version"] != 1 or state["signature"] != signature
            or not 0 <= position <= total):
        raise ValueError("Checkpoint does not match this frozen training run")
    trainable = {name: parameter for name, parameter in backbone.named_parameters() if parameter.requires_grad}
    if set(trainable) != set(state["trainable"]):
        raise ValueError("Checkpoint adapter parameters do not match")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state["trainable"][name].to(parameter.device))
    head.load_state_dict(state["head"])
    optimizer.load_state_dict(state["optimizer"])
    return position, float(state["elapsed_seconds"]), float(state["peak_vram_gib"])


def run(args: argparse.Namespace) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required")
    if (args.accumulation < 1 or args.max_length < 64 or args.lora_rank < 1
            or not math.isfinite(args.ordinal_weight) or args.ordinal_weight < 0):
        raise ValueError("Invalid training configuration")
    if args.scope_group_weighting != "none" and args.scopejudge_path is None:
        raise ValueError("Scope group weighting requires ScopeJudge training data")
    checkpoint_path = args.output_dir / "checkpoint.pt"
    if args.resume:
        if not checkpoint_path.is_file() or args.report.exists():
            raise FileNotFoundError("Resume needs an unfinished checkpoint and no final report")
    elif args.skip_baseline_on_resume:
        raise ValueError("Skipping the baseline requires --resume")
    elif args.output_dir.exists() or args.report.exists():
        raise FileExistsError("Research output already exists; choose fresh artifact and report paths")
    torch.manual_seed(args.seed)
    transformers_logging.disable_progress_bar()
    np.random.seed(args.seed)
    random.seed(args.seed)
    limits = {"train": args.train_rows, "calibration": args.calibration_rows,
              "test": args.test_rows, "ood": args.ood_rows}
    rows, manifests = {}, {}
    for offset, (split, limit) in enumerate(limits.items()):
        rows[split], manifests[split] = load_split(split, limit, args.seed + offset)
        if not rows[split]:
            raise ValueError(f"No licensed rows in {split}")
    scope_manifest = None
    if args.scopejudge_path is not None:
        scope_limits = {"scope_train": args.scope_train_rows,
                        "scope_calibration": args.scope_calibration_rows,
                        "scope_test": args.scope_test_rows}
        scope_parts, scope_manifest = load_scope_parts(args.scopejudge_path, seed=args.seed,
                                                       limits=scope_limits,
                                                       history_calls=args.scope_history_calls,
                                                       strategy=args.scope_split_strategy)
        rows.update(scope_parts)
    verify_disjoint(rows)
    print(json.dumps({"event": "data_selected", "counts": {name: len(part) for name, part in rows.items()},
                      "excluded_unverified_license": {name: entry["excluded_unverified_license"]
                                                     for name, entry in manifests.items()},
                      "scopejudge": scope_manifest}), flush=True)

    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype={"bfloat16": torch.bfloat16,
                                            "float16": torch.float16}[args.dtype],
        device_map={"": 0}, attn_implementation=args.attention,
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
        encoded[split], overlong[split] = encode_rows(
            tokenizer, selected, args.max_length, prompt_style=args.prompt_style,
            fit_history=args.scope_history_calls > 0 and split.startswith("scope_"))
        if not encoded[split]:
            raise ValueError(f"All {split} rows exceed max_length={args.max_length}")
    print(json.dumps({"event": "tokenized", "kept": {name: len(part) for name, part in encoded.items()},
                      "overlong": overlong}), flush=True)
    row_weights = None
    if args.scope_group_weighting == "inverse_sqrt":
        row_weights = ([1.0] * len(encoded["train"]) +
                       scope_group_weights([item["row"] for item in encoded["scope_train"]]))
    signature = checkpoint_signature(args, encoded, manifests, scope_manifest,
                                     row_weights=row_weights)

    run_baseline = not (args.resume and args.skip_baseline_on_resume)
    def baseline(part: str) -> list[np.ndarray]:
        identity = hashlib.sha256(f"{signature}:baseline:{part}".encode()).hexdigest()
        return evaluate_checkpointed(
            backbone, head, encoded[part], f"baseline_{part}",
            path=args.output_dir / f"evaluation_baseline_{part}.json",
            signature=identity, resume=args.resume, checkpoint_every=args.checkpoint_every,
        )

    baseline_test = baseline("test") if run_baseline else None
    baseline_ood = baseline("ood") if run_baseline else None
    baseline_scope = baseline("scope_test") if run_baseline and scope_manifest else None
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in backbone.parameters() if parameter.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    backbone.train()
    head.train()
    training_items = encoded["train"] + (encoded["scope_train"] if scope_manifest else [])
    order = list(range(len(training_items)))
    random.Random(args.seed).shuffle(order)
    if args.resume:
        start_position, elapsed_before, prior_peak = load_checkpoint(
            checkpoint_path, backbone, head, optimizer, signature=signature, total=len(order))
        print(json.dumps({"event": "resumed", "rows": start_position, "total": len(order)}), flush=True)
    else:
        start_position, elapsed_before, prior_peak = 0, 0.0, 0.0
    started = time.monotonic()
    for start in range(start_position, len(order), args.accumulation):
        group = order[start:start + args.accumulation]
        losses, retries = train_group(
            backbone, head, optimizer, training_items, group,
            brier_weight=args.brier_weight, ordinal_weight=args.ordinal_weight,
            row_weights=row_weights,
        )
        if retries:
            print(json.dumps({"event": "retried_nonfinite_group", "start": start,
                              "retries": retries}), flush=True)
        done = start + len(group)
        elapsed = elapsed_before + time.monotonic() - started
        peak = max(prior_peak, torch.cuda.max_memory_allocated() / 1024**3)
        if done % args.checkpoint_every < len(group) or done == len(order):
            save_checkpoint(checkpoint_path, backbone, head, optimizer, signature=signature,
                            next_position=done, elapsed_seconds=elapsed, peak_vram_gib=peak)
        if done % args.progress_every < args.accumulation or done == len(order):
            print(json.dumps({"event": "training", "rows": done, "total": len(order),
                              "recent_loss": float(np.mean(losses)),
                              "elapsed_seconds": round(elapsed, 1),
                              "peak_vram_gib": round(peak, 3)}), flush=True)

    calibration_items = encoded["calibration"] + (encoded["scope_calibration"] if scope_manifest else [])
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    def held_out(part: str, items: list[dict]) -> list[np.ndarray]:
        identity = hashlib.sha256(f"{signature}:{checkpoint_hash}:{part}".encode()).hexdigest()
        return evaluate_checkpointed(
            backbone, head, items, part, path=args.output_dir / f"evaluation_{part}.json",
            signature=identity, resume=args.resume, checkpoint_every=args.checkpoint_every,
        )

    calibration_logits = held_out("calibration", calibration_items)
    temperature = fit_temperature(calibration_logits, [item["row"] for item in calibration_items])
    test_logits = held_out("trained_test", encoded["test"])
    ood_logits = held_out("trained_ood", encoded["ood"])
    trained_scope = held_out("trained_scope", encoded["scope_test"]) if scope_manifest else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(args.output_dir / "adapter")
    np.savez_compressed(args.output_dir / "head.npz",
                        weights=head.weight.detach().cpu().numpy(),
                        bias=head.bias.detach().cpu().numpy(),
                        temperature=temperature, model_revision=REVISION,
                        max_length=args.max_length, lora_rank=args.lora_rank,
                        attention_implementation=args.attention, prompt_style=args.prompt_style,
                        model_dtype=args.dtype)

    report = {
        "model": MODEL, "revision": REVISION,
        "diagnostic_environment": {name: os.environ.get(name) for name in (
            "HF_HUB_OFFLINE", "CUDA_LAUNCH_BLOCKING", "PYTORCH_NO_CUDA_MEMORY_CACHING")},
        "dataset": DATASET, "dataset_revision": DATASET_REVISION, "dataset_config": DATASET_CONFIG,
        "dataset_manifests": manifests, "seed": args.seed,
        "selected_rows": limits, "evaluated_rows": {name: len(part) for name, part in encoded.items()},
        "excluded_overlong": overlong, "max_length": args.max_length,
        "scope_history_budget": {
            name: {"rows_with_history": sum(item["history_calls_retained"] > 0 for item in part),
                   "calls_dropped": sum(item["history_calls_dropped"] for item in part)}
            for name, part in encoded.items() if name.startswith("scope_")
        },
        "attention_implementation": args.attention, "model_dtype": args.dtype,
        "training_checkpoint_sha256": checkpoint_hash,
        "lora_rank": args.lora_rank, "accumulation": args.accumulation,
        "prompt_style": args.prompt_style,
        "checkpoint_every": args.checkpoint_every, "resumed": args.resume,
        "baseline_skipped_on_resume": not run_baseline,
        "learning_rates": {"adapter": args.lr, "head": args.head_lr},
        "brier_weight": args.brier_weight, "ordinal_weight": args.ordinal_weight,
        "scope_group_weighting": args.scope_group_weighting,
        "row_weights_sha256": (None if row_weights is None else hashlib.sha256(
            json.dumps(row_weights, separators=(",", ":"), allow_nan=False).encode()).hexdigest()),
        "calibration_temperature": temperature,
        "baseline_test": (metrics([item["row"] for item in encoded["test"]], baseline_test, 1.0)
                          if baseline_test is not None else None),
        "baseline_ood": (metrics([item["row"] for item in encoded["ood"]], baseline_ood, 1.0)
                         if baseline_ood is not None else None),
        "trained_test": metrics([item["row"] for item in encoded["test"]], test_logits, temperature),
        "trained_ood": metrics([item["row"] for item in encoded["ood"]], ood_logits, temperature),
        "adapter_sha256": hashlib.sha256((args.output_dir / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256((args.output_dir / "head.npz").read_bytes()).hexdigest(),
        "peak_vram_gib": (None if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
                          else round(max(prior_peak, torch.cuda.max_memory_allocated() / 1024**3), 3)),
        "memory_note": ("PyTorch peak-allocation counter is unavailable with caching disabled"
                        if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1" else None),
        "elapsed_seconds": round(elapsed_before + time.monotonic() - started, 1),
        "limitation": ("Public synthetic decisions plus expert-labeled security scope calls; "
                       "not Jev parity or Heliox execution authority." if scope_manifest else
                       "Licensed public synthetic sample only; not proprietary Jev parity or Heliox execution authority."),
    }
    if scope_manifest:
        report.update({
            "scopejudge": scope_manifest,
            "baseline_scope": (scope_metrics([item["row"] for item in encoded["scope_test"]],
                                             baseline_scope, 1.0)
                               if baseline_scope is not None else None),
            "trained_scope": scope_metrics([item["row"] for item in encoded["scope_test"]],
                                           trained_scope, temperature),
        })
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
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--prompt-style", choices=sorted(PROMPT_STYLES), default="candidate_v1")
    parser.add_argument("--ordinal-weight", type=float, default=0.0,
                        help="Optional normalized expected-rank loss for ordered score decisions")
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-baseline-on-resume", action="store_true",
                        help="Avoid repeated baseline inference after a saved training checkpoint")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--scopejudge-path", type=Path, help="Pinned MIT ScopeJudge train.jsonl for multi-task study")
    parser.add_argument("--scope-history-calls", type=int, default=0,
                        help="Include up to this many calls from completed earlier trajectory steps")
    parser.add_argument("--scope-group-weighting", choices=("none", "inverse_sqrt"), default="none",
                        help="Optional label-normalized inverse-sqrt trajectory weighting")
    parser.add_argument("--scope-split-strategy", choices=SPLIT_STRATEGIES,
                        default="random_trajectory_v1")
    parser.add_argument("--scope-train-rows", type=int, default=512)
    parser.add_argument("--scope-calibration-rows", type=int, default=128)
    parser.add_argument("--scope-test-rows", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/.bench-state/qwen_general_v1")
    parser.add_argument("--report", type=Path, default=root / "benchmarks/qwen_general_probe_v1.json")
    args = parser.parse_args()
    if any(value < 1 for value in (args.train_rows, args.calibration_rows, args.test_rows,
                                   args.ood_rows, args.progress_every, args.checkpoint_every)):
        parser.error("Row limits and progress interval must be positive")
    if args.scopejudge_path is not None and any(value < 2 for value in (
            args.scope_train_rows, args.scope_calibration_rows, args.scope_test_rows)):
        parser.error("ScopeJudge row limits must be at least two")
    if args.scope_history_calls < 0 or (args.scope_history_calls and args.scopejudge_path is None):
        parser.error("Scope history requires a dataset path and a non-negative call count")
    report = run(args)
    print(json.dumps({key: report[key] for key in ("trained_test", "trained_ood", "peak_vram_gib")}, indent=2))


if __name__ == "__main__":
    main()

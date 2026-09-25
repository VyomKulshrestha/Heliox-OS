"""Research-only 4-bit LoRA task-state classifier on licensed scope labels.

The adapter is not loaded by the daemon. Deterministic policy and human
approvals remain authoritative, regardless of any offline test result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmarks.jev_scope_qwen_v1_holdout import CASES as HELIOX_CASES
from scripts.evaluate_openjev_checkpoint import scopejudge_cases
from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION, SYSTEM_PROMPT, messages
from scripts.train_world_scope_semantic import metrics
from scripts.train_world_tool_scope_semantic import DATASET_HASHES, DATASET_REVISION, load_cases, split_validation

DOMAINS = ("tbac", "scope")


def load_partitions(dataset_dir: Path, scopejudge_path: Path) -> tuple[dict, dict]:
    validation = load_cases(dataset_dir / "validation.json", DATASET_HASHES["validation"],
                            "tbac_validation")
    test = load_cases(dataset_dir / "test.json", DATASET_HASHES["test"], "tbac_test")
    train, calibration = split_validation(validation)
    if {case.request for case in validation} & {case.request for case in test}:
        raise ValueError("TBAC official test tasks overlap validation")
    scope, manifests = {}, {}
    for name in ("scope_train", "scope_calibration", "scope_test"):
        scope[name], manifests[name] = scopejudge_cases(scopejudge_path, name)
    if not all(manifest == manifests["scope_train"] for manifest in manifests.values()):
        raise ValueError("ScopeJudge source changed between partitions")
    groups = {name: {case.id.split(":", 1)[0] for case in part} for name, part in scope.items()}
    if any(groups[left] & groups[right] for left, right in
           (("scope_train", "scope_calibration"), ("scope_train", "scope_test"),
            ("scope_calibration", "scope_test"))):
        raise ValueError("ScopeJudge trajectories overlap")
    return ({"tbac_train": train, "tbac_calibration": calibration, "tbac_test": test,
             **scope, "heliox": tuple(HELIOX_CASES)}, manifests["scope_train"])


def encode_cases(tokenizer, cases: dict, max_length: int) -> tuple[dict, dict]:
    encoded, excluded = {}, {}
    for name, part in cases.items():
        rows, overlong = [], 0
        for case in part:
            prompt = tokenizer.apply_chat_template(
                messages(case.request, case.action), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(prompt, truncation=False)["input_ids"]
            if len(ids) > max_length:
                overlong += 1
                continue
            rows.append((case.id, ids, case.in_scope))
        if len({row[2] for row in rows}) != 2:
            raise ValueError(f"{name} needs token-eligible examples from both classes")
        encoded[name], excluded[name] = rows, overlong
    return encoded, excluded


def domain_class_weights(rows: dict) -> dict[tuple[str, bool], float]:
    total = sum(len(rows[f"{name}_train"]) for name in DOMAINS)
    weights = {}
    for name in DOMAINS:
        for label in (False, True):
            count = sum(row[2] is label for row in rows[f"{name}_train"])
            if count == 0:
                raise ValueError(f"Missing {name} training label {label}")
            weights[name, label] = total / (4 * count)
    return weights


def calibration_choice(calibration: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, dict]:
    auc = {name: float(roc_auc_score(*calibration[name])) for name in DOMAINS}
    threshold = math.nextafter(max(float(np.max(calibration[name][1][~calibration[name][0]]))
                                   for name in DOMAINS), math.inf)
    return threshold, auc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--adapter-dir", type=Path, default=state / "world_scope_qlora_v1")
    parser.add_argument("--report", type=Path, default=state / "world_scope_qlora_v1.json")
    parser.add_argument("--smoke", action="store_true", help="Train four examples without writing an adapter")
    args = parser.parse_args()
    if args.max_length < 128 or args.epochs < 1 or args.adapter_dir.exists() or args.report.exists():
        parser.error("Use max-length >=128, epochs >=1, and fresh artifact/report paths")
    cases, scope_manifest = load_partitions(args.dataset_dir, args.scopejudge_path)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    torch.manual_seed(20260925)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True)
    rows, excluded = encode_cases(tokenizer, cases, args.max_length)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        quantization_config=config, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda:0"}, output_loading_info=True)
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = get_peft_model(model.model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear"))
    text_config = getattr(model.config, "text_config", model.config)
    head = torch.nn.Linear(text_config.hidden_size, 1).to("cuda", dtype=torch.float32)
    trainable = ([parameter for parameter in backbone.parameters() if parameter.requires_grad]
                 + list(head.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    weights = domain_class_weights(rows)
    print(json.dumps({"event": "ready", "rows": {name: len(part) for name, part in rows.items()},
                      "excluded": excluded, "trainable_parameters": sum(
                          parameter.numel() for parameter in backbone.parameters()
                          if parameter.requires_grad)}), flush=True)

    def forward(ids: list[int]):
        tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
        hidden = backbone(input_ids=tokens, use_cache=False,
                          return_dict=True).last_hidden_state[0, -1].float()
        return head(hidden).squeeze()

    def evaluate(name: str) -> tuple[np.ndarray, np.ndarray]:
        backbone.eval()
        head.eval()
        labels, scores = [], []
        with torch.inference_mode():
            for index, (_, ids, target) in enumerate(rows[name], 1):
                labels.append(target)
                scores.append(float(torch.sigmoid(forward(ids)).item()))
                if index % 100 == 0 or index == len(rows[name]):
                    print(json.dumps({"event": "evaluated", "split": name, "rows": index,
                                      "total": len(rows[name])}), flush=True)
        return np.asarray(labels, dtype=np.bool_), np.asarray(scores)

    training = [(name, row) for name in DOMAINS for row in rows[f"{name}_train"]]
    if args.smoke:
        training = [(name, max((row for row in rows[f"{name}_train"] if row[2] is label),
                               key=lambda row: len(row[1])))
                    for name in DOMAINS for label in (False, True)]
    history = []
    best = float("-inf")
    best_state = None
    best_head = None
    best_threshold = None
    best_epoch = None
    for epoch in range(args.epochs):
        random.Random(20260925 + epoch).shuffle(training)
        backbone.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for step, (domain, (_, ids, target)) in enumerate(training, 1):
            target_tensor = torch.tensor(float(target), device="cuda")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                forward(ids), target_tensor) * weights[domain, target]
            block_size = min(8, len(training) - 8 * ((step - 1) // 8))
            (loss / block_size).backward()
            loss_total += float(loss.detach())
            if step % 8 == 0 or step == len(training):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0 or step == len(training):
                print(json.dumps({"event": "trained", "epoch": epoch + 1, "step": step,
                                  "total": len(training), "mean_loss": loss_total / step,
                                  "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
        if args.smoke:
            print(json.dumps({"event": "smoke_complete", "mean_loss": loss_total / len(training),
                              "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3}))
            return
        calibration = {name: evaluate(f"{name}_calibration") for name in DOMAINS}
        threshold, auc = calibration_choice(calibration)
        score = min(auc.values())
        history.append({"epoch": epoch + 1, "mean_loss": loss_total / len(training),
                        "calibration_auc": auc, "calibration_threshold": threshold})
        if score > best:
            best = score
            best_state = {name: parameter.detach().cpu().clone()
                          for name, parameter in backbone.named_parameters()
                          if parameter.requires_grad}
            best_head = {name: parameter.detach().cpu().clone()
                         for name, parameter in head.state_dict().items()}
            best_threshold = threshold
            best_epoch = epoch + 1
        print(json.dumps({"event": "calibrated", **history[-1]}), flush=True)
    with torch.no_grad():
        for name, parameter in backbone.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    head.load_state_dict(best_head)
    args.adapter_dir.mkdir(parents=True)
    backbone.save_pretrained(args.adapter_dir)
    np.savez_compressed(args.adapter_dir / "head.npz",
                        weight=head.weight.detach().float().cpu().numpy(),
                        bias=head.bias.detach().float().cpu().numpy(),
                        threshold=best_threshold, model_revision=MODEL_REVISION)
    outcomes = {name: evaluate(name) for name in ("tbac_test", "scope_test", "heliox")}
    report = {
        "purpose": "research_only_qlora_world_task_state_adapter",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tbac": {"url": "https://github.com/outshift-open/tbac-research-datasets",
                 "revision": DATASET_REVISION, "sha256": DATASET_HASHES},
        "scopejudge": scope_manifest, "max_length": args.max_length,
        "rows": {name: len(part) for name, part in rows.items()},
        "excluded_overlong": excluded, "training": history,
        "selected_epoch": best_epoch, "selected_calibration_threshold": best_threshold,
        "metrics": {name: {"default": metrics(labels, scores, 0.5),
                           "cautious": metrics(labels, scores, best_threshold)}
                    for name, (labels, scores) in outcomes.items()},
        "head_sha256": hashlib.sha256((args.adapter_dir / "head.npz").read_bytes()).hexdigest(),
        "limitation": "No runtime use, live desktop evidence, or Jev parity; official test outcomes "
                      "from earlier adaptive research are not a fresh blind benchmark.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

"""Research-only GPU probe: predict a tool response from a state and action.

This is not loaded by the daemon. A lower held-out loss would not establish
OS-action fidelity, safe autonomy, or generalization to other environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.qwen_model_pin import MODEL, REVISION


def load_rows(path: Path) -> dict[str, list[dict]]:
    partitions = {"train": [], "eval": []}
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] not in partitions or row["context_sha256"] in seen:
            raise ValueError("Invalid split or duplicate context")
        seen.add(row["context_sha256"])
        if [message["role"] for message in row["messages"]] != ["system", "user", "assistant"]:
            raise ValueError("Expected source state, action, and outcome")
        partitions[row["split"]].append(row)
    if not all(partitions.values()):
        raise ValueError("Both train and eval must contain source rows")
    all_rows = [*partitions["train"], *partitions["eval"]]
    if any("group_sha256" in row for row in all_rows):
        if any(not isinstance(row.get("group_sha256"), str) or len(row["group_sha256"]) != 64
               for row in all_rows):
            raise ValueError("Grouped rows require a 64-character group hash")
        train_groups = {row["group_sha256"] for row in partitions["train"]}
        eval_groups = {row["group_sha256"] for row in partitions["eval"]}
        if train_groups & eval_groups:
            raise ValueError("Train and eval share a task group")
    return partitions


def encode_rows(tokenizer, partitions: dict[str, list[dict]], *, max_input_tokens: int,
                max_target_tokens: int) -> tuple[dict[str, list[dict]], dict[str, int]]:
    encoded, exclusions = {}, {}
    for split, rows in partitions.items():
        encoded[split], exclusions[split] = [], 0
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                row["messages"][:2], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row["messages"][2]["content"],
                                   add_special_tokens=False)["input_ids"]
            if len(prompt_ids) > max_input_tokens or len(target_ids) > max_target_tokens:
                exclusions[split] += 1
                continue
            ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
            encoded_row = {"ids": ids,
                           "labels": [-100] * len(prompt_ids)
                           + target_ids + [tokenizer.eos_token_id],
                           "domain": row["dataset"]}
            if "group_sha256" in row:
                encoded_row["group_sha256"] = row["group_sha256"]
            encoded[split].append(encoded_row)
    if not all(encoded.values()):
        raise ValueError("Token filtering emptied a split")
    return encoded, exclusions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=state / "world_transition_sample_v1.jsonl")
    parser.add_argument("--report", type=Path, default=state / "world_transition_lora_v1.json")
    parser.add_argument("--adapter", type=Path, default=state / "world_transition_lora_v1")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if (not args.rows.is_file() or args.report.exists() or args.adapter.exists()
            or args.max_input_tokens < 1 or args.max_target_tokens < 1 or args.epochs < 1):
        parser.error("Use existing source rows, fresh outputs, and positive limits")

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    snapshot = snapshot_download(repo_id=MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    source_rows = load_rows(args.rows)
    grouped = "group_sha256" in source_rows["train"][0]
    rows, exclusions = encode_rows(tokenizer, source_rows,
                                   max_input_tokens=args.max_input_tokens,
                                   max_target_tokens=args.max_target_tokens)
    group_counts = ({split: len({row["group_sha256"] for row in partition})
                     for split, partition in rows.items()} if grouped else None)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, quantization_config=config,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager",
    )
    model.config.use_cache = False

    def loss_for(row: dict):
        inputs = torch.tensor([row["ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        return model(input_ids=inputs, labels=labels, use_cache=False).loss

    def evaluate() -> tuple[float, dict[str, float]]:
        model.eval()
        by_domain: dict[str, list[float]] = {}
        with torch.no_grad():
            for row in rows["eval"]:
                by_domain.setdefault(row["domain"], []).append(float(loss_for(row)))
        all_losses = [loss for losses in by_domain.values() for loss in losses]
        return sum(all_losses) / len(all_losses), {
            domain: sum(losses) / len(losses) for domain, losses in by_domain.items()
        }

    base_loss, base_domains = evaluate()
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=1e-4)
    print(json.dumps({"event": "ready", "train": len(rows["train"]),
                      "eval": len(rows["eval"]), "base_eval_loss": base_loss}), flush=True)
    for epoch in range(args.epochs):
        random.shuffle(rows["train"])
        model.train()
        for index, row in enumerate(rows["train"]):
            optimizer.zero_grad(set_to_none=True)
            loss_for(row).backward()
            optimizer.step()
            if (index + 1) % 10 == 0:
                print(json.dumps({"event": "step", "epoch": epoch + 1,
                                  "completed": index + 1, "total": len(rows["train"])}), flush=True)
    tuned_loss, tuned_domains = evaluate()
    args.adapter.mkdir(parents=True)
    model.save_pretrained(args.adapter)
    report = {
        "purpose": ("research_only_grouped_ui_transition_probe" if grouped
                    else "research_only_tool_response_transition_probe"),
        "source_rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "base_model": {"id": MODEL, "revision": REVISION},
        "train_rows": len(rows["train"]), "eval_rows": len(rows["eval"]),
        "excluded_overlong": exclusions, "epochs": args.epochs,
        "max_input_tokens": args.max_input_tokens,
        "max_target_tokens": args.max_target_tokens,
        "base_eval_loss": base_loss, "tuned_eval_loss": tuned_loss,
        "base_eval_loss_by_domain": base_domains,
        "tuned_eval_loss_by_domain": tuned_domains,
        "groups": group_counts,
        "limitation": (
            "Task-group-disjoint text observations and outcome descriptions are not raw next-state "
            "evidence. A loss improvement is not action-effect accuracy, OS generalization, "
            "authority, safety, or Jev parity. Not runtime-loaded."
            if grouped else
            "Small synthetic tool-response sample. A loss improvement is not action-effect "
            "accuracy, OS generalization, authority, safety, or Jev parity. Not runtime-loaded."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Compare base and research adapter on task-disjoint UI transition text.

Lexical metrics are only a reproducible diagnostic, not semantic correctness,
actual next-screen fidelity, user-task success, or permission to act.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from scripts.qwen_model_pin import MODEL, REVISION
from scripts.train_world_transition_lora import load_rows


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def token_f1(predicted: str, reference: str) -> float:
    prediction, truth = Counter(_words(predicted)), Counter(_words(reference))
    if not prediction or not truth:
        return 0.0
    overlap = sum((prediction & truth).values())
    return 2 * overlap / (sum(prediction.values()) + sum(truth.values()))


def rouge_l_f1(predicted: str, reference: str) -> float:
    prediction, truth = _words(predicted), _words(reference)
    if not prediction or not truth:
        return 0.0
    previous = [0] * (len(truth) + 1)
    for word in prediction:
        current = [0] * (len(truth) + 1)
        for index, expected in enumerate(truth, start=1):
            current[index] = previous[index - 1] + 1 if word == expected else max(
                previous[index], current[index - 1]
            )
        previous = current
    return 2 * previous[-1] / (len(prediction) + len(truth))


def select_cases(rows: list[dict], *, seed: int, max_cases: int) -> list[dict]:
    """Choose at most one pre-registered row per task group."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["group_sha256"], []).append(row)
    rng = random.Random(seed)
    group_ids = sorted(grouped)
    rng.shuffle(group_ids)
    return [rng.choice(grouped[group]) for group in group_ids[:max_cases]]


def load_checkpoint(path: Path, selected: list[dict], signature: str) -> list[dict]:
    """Resume only a contiguous run against the exact same inputs and weights."""
    if not path.exists():
        return []
    completed: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        index = len(completed)
        if (index >= len(selected) or item.get("signature") != signature
                or item.get("case_index") != index
                or item.get("group_sha256") != selected[index]["group_sha256"]
                or item.get("row_index") != selected[index]["row_index"]):
            raise ValueError("Checkpoint does not match selected held-out cases or adapter")
        completed.append(item)
    return completed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=state / "grouped_world_generation.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--include-text", action="store_true",
                        help="Store source reference and generated text locally for an explicit audit")
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-target-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    adapter_weights = args.adapter / "adapter_model.safetensors"
    if (not args.rows.is_file() or not adapter_weights.is_file() or args.report.exists()
            or min(args.max_cases, args.max_input_tokens, args.max_target_tokens) < 1):
        parser.error("Use existing rows and adapter, a fresh report, and positive limits")
    checkpoint = args.checkpoint or args.report.with_suffix(".partial.jsonl")
    if checkpoint == args.report:
        parser.error("Checkpoint and final report must be different files")

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required")
    partitions = load_rows(args.rows)
    if any("group_sha256" not in row for row in partitions["eval"]):
        raise ValueError("Held-out cases must have task-group IDs")
    snapshot = snapshot_download(repo_id=MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    candidates = []
    for row in partitions["eval"]:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:2], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["messages"][2]["content"],
                               add_special_tokens=False)["input_ids"]
        if len(ids) <= args.max_input_tokens and len(target_ids) <= args.max_target_tokens:
            candidates.append({**row, "prompt_ids": ids})
    selected = select_cases(candidates, seed=args.seed, max_cases=args.max_cases)
    if not selected:
        raise ValueError("No eligible held-out task groups")
    if args.case_index is not None:
        if not 0 <= args.case_index < len(selected):
            parser.error("Case index must name a selected held-out group")
        selected = [selected[args.case_index]]
    source_hash = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    with adapter_weights.open("rb") as weights_file:
        adapter_hash = hashlib.file_digest(weights_file, "sha256").hexdigest()
    signature = hashlib.sha256(json.dumps({
        "source": source_hash, "adapter": adapter_hash, "seed": args.seed,
        "max_cases": args.max_cases, "max_input_tokens": args.max_input_tokens,
        "max_target_tokens": args.max_target_tokens, "base_revision": REVISION,
        "case_index": args.case_index, "include_text": args.include_text,
    }, sort_keys=True).encode()).hexdigest()
    report_rows = load_checkpoint(checkpoint, selected, signature)
    print(json.dumps({"event": "resumed", "completed": len(report_rows),
                      "total": len(selected)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    for case_index in range(len(report_rows), len(selected)):
        row = selected[case_index]
        input_ids = torch.tensor([row["prompt_ids"]], device="cuda")
        reference = row["messages"][2]["content"]
        scores = {}
        generated = {}
        for name in ("base", "adapter"):
            with torch.no_grad():
                if name == "base":
                    with model.disable_adapter():
                        output = model.generate(
                            input_ids, attention_mask=torch.ones_like(input_ids),
                            max_new_tokens=args.max_target_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                else:
                    output = model.generate(
                        input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=args.max_target_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            scores[name] = {"rouge_l_f1": rouge_l_f1(text, reference),
                            "token_f1": token_f1(text, reference),
                            "generated_tokens": int(output.shape[1] - input_ids.shape[1])}
            if args.include_text:
                generated[name] = text
        item = {"signature": signature, "case_index": case_index,
                "row_index": row["row_index"], "group_sha256": row["group_sha256"],
                "scores": scores}
        if args.include_text:
            item["reference"] = reference
            item["generated"] = generated
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as output:
            output.write(json.dumps(item) + "\n")
            output.flush()
            os.fsync(output.fileno())
        report_rows.append(item)
        print(json.dumps({"event": "evaluated", "completed": len(report_rows),
                          "total": len(selected)}), flush=True)
    report = {
        "purpose": "research_only_grouped_ui_transition_generation_check",
        "source_rows_sha256": source_hash,
        "base_model": {"id": MODEL, "revision": REVISION},
        "adapter": str(args.adapter), "adapter_sha256": adapter_hash,
        "seed": args.seed, "run_signature": signature,
        "audited_case_index": args.case_index, "includes_source_text": args.include_text,
        "eligible_groups": len({row["group_sha256"] for row in candidates}),
        "evaluated_groups": len(report_rows),
        "mean_scores": {name: {
            metric: sum(row["scores"][name][metric] for row in report_rows) / len(report_rows)
            for metric in ("rouge_l_f1", "token_f1")
        } for name in ("base", "adapter")},
        "cases": report_rows,
        "limitation": "Lexical overlap with source descriptions is not semantic or next-screen fidelity. "
                      "Small, text-only task-disjoint sample; no environment-disjoint or live Heliox "
                      "evaluation, safety claim, Jev parity, or runtime integration.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

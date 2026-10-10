"""Compare base and research adapter on exact held-out tool outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.qwen_model_pin import MODEL, REVISION
from scripts.train_world_transition_lora import load_rows


def normalize_outcome(text: str):
    body = text.strip()
    if not body.startswith("<tool_response>") or "</tool_response>" not in body:
        return None
    body = body[len("<tool_response>") : body.index("</tool_response>")].strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=state / "world_transition_generation_v1.json")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.rows.is_file() or not args.adapter.is_dir() or args.report.exists():
        parser.error("Use existing source and adapter with a fresh report path")

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free CUDA memory is required")
    snapshot = snapshot_download(repo_id=MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    cases = []
    for row in load_rows(args.rows)["eval"]:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["messages"][2]["content"], add_special_tokens=False)["input_ids"]
        if len(ids) <= args.max_input_tokens and len(target_ids) <= args.max_target_tokens:
            cases.append((row, ids))
    if not cases:
        raise ValueError("No token-eligible held-out cases")

    report_rows = []
    for row, ids in cases:
        input_ids = torch.tensor([ids], device="cuda")
        truth = normalize_outcome(row["messages"][2]["content"])
        if truth is None:
            raise ValueError("Held-out source row has no valid tool-response wrapper")
        predictions = {}
        for name in ("base", "adapter"):
            with torch.no_grad():
                if name == "base":
                    with model.disable_adapter():
                        output = model.generate(
                            input_ids,
                            max_new_tokens=args.max_target_tokens,
                            attention_mask=torch.ones_like(input_ids),
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                else:
                    output = model.generate(
                        input_ids,
                        max_new_tokens=args.max_target_tokens,
                        attention_mask=torch.ones_like(input_ids),
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
            predictions[name] = normalize_outcome(text) == truth
        report_rows.append(
            {
                "row_index": row["row_index"],
                "domain": row["dataset"],
                "base_exact_outcome": predictions["base"],
                "adapter_exact_outcome": predictions["adapter"],
            }
        )
        print(json.dumps({"event": "evaluated", "completed": len(report_rows), "total": len(cases)}), flush=True)
    report = {
        "purpose": "research_only_heldout_exact_outcome_check",
        "source_rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "adapter": str(args.adapter),
        "base_model": {"id": MODEL, "revision": REVISION},
        "eligible": len(cases),
        "base_exact_outcomes": sum(row["base_exact_outcome"] for row in report_rows),
        "adapter_exact_outcomes": sum(row["adapter_exact_outcome"] for row in report_rows),
        "cases": report_rows,
        "limitation": "Small synthetic held-out set; exact outcome match is not OS transfer, "
        "safety, authority, or Jev parity. No runtime integration.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

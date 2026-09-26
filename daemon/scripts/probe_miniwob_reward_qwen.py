"""Evaluate a frozen local Qwen model on held-out MiniWoB action rewards.

This is a zero-shot diagnostic, not training, runtime promotion, or evidence of
multi-step desktop competence. The model must already be present on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from scripts.collect_miniwob_branches import SOURCE_COMMIT, SOURCE_REPO
from scripts.collect_miniwob_reward_actions import require_disjoint_prestates

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def load_rows(path: Path, source_report: Path) -> list[dict]:
    raw = path.read_bytes()
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if (hashlib.sha256(raw).hexdigest() != report.get("rows_sha256")
            or report.get("purpose") != "research_only_goal_conditioned_first_action_rewards"
            or report.get("source_repo") != SOURCE_REPO
            or report.get("source_commit") != SOURCE_COMMIT
            or report.get("split_basis") != "task_family"):
        raise ValueError("Reward rows differ from their pinned source report")
    rows = json.loads(raw)
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) for row in rows)
            or any(row.get("split") not in {"train", "validation", "eval"}
                   or row.get("reward") not in (-1, 1)
                   or not isinstance(row.get("goal"), str)
                   or not isinstance(row.get("group"), str)
                   or not isinstance(row.get("task"), str)
                   or not isinstance(row.get("prestate_sha256"), str)
                   or not isinstance(row.get("candidate"), dict)
                   for row in rows)):
        raise ValueError("Invalid reward rows")
    tasks = report.get("tasks", {})
    if any(row["task"] not in tasks.get(row["split"], []) for row in rows):
        raise ValueError("Reward task does not match its pinned split")
    require_disjoint_prestates(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    if any(len({row["task"] for row in group}) != 1
           or len({row["split"] for row in group}) != 1
           or len({row["goal"] for row in group}) != 1
           or {row["reward"] for row in group} != {-1, 1}
           for group in groups.values()):
        raise ValueError("Reward groups must be single-task contrastive resets")
    return rows


def prompt_for(group: list[dict]) -> str:
    candidates = [
        {"index": index, **{key: row["candidate"].get(key) for key in
                            ("label", "tag", "type", "role", "width", "height",
                             "background", "fill")}}
        for index, row in enumerate(group)
    ]
    return ("Choose the visible GUI control most likely to satisfy the goal. "
            "Return only its integer index.\nGoal: " + group[0]["goal"]
            + "\nControls: " + json.dumps(candidates, ensure_ascii=False)
            + "\nIndex:")


def parse_index(answer: str, count: int) -> int | None:
    match = re.fullmatch(r"\s*(\d+)\s*\.?\s*", answer)
    index = int(match.group(1)) if match else -1
    return index if 0 <= index < count else None


def evaluate(rows: list[dict], split: str, choose: Callable[[str], str]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[row["group"]].append(row)
    if not groups:
        raise ValueError("No held-out groups")
    outcomes = []
    for group_id, group in groups.items():
        answer = choose(prompt_for(group))
        selected = parse_index(answer, len(group))
        outcomes.append({
            "group": group_id, "task": group[0]["task"], "selected_index": selected,
            "correct": selected is not None and group[selected]["reward"] == 1,
            "random_success_probability": sum(row["reward"] == 1 for row in group) / len(group),
        })
    return {
        "groups": len(outcomes),
        "correct": sum(row["correct"] for row in outcomes),
        "invalid_answers": sum(row["selected_index"] is None for row in outcomes),
        "success_rate": sum(row["correct"] for row in outcomes) / len(outcomes),
        "random_success_rate": sum(row["random_success_probability"] for row in outcomes) / len(outcomes),
        "selected_groups_sha256": hashlib.sha256("\n".join(groups).encode()).hexdigest(),
        "outcomes": outcomes,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=root / "miniwob_reward_actions_v1.json")
    parser.add_argument("--source-report", type=Path,
                        default=root / "miniwob_reward_actions_v1_report.json")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "eval"), default="eval")
    parser.add_argument("--report", type=Path, default=root / "miniwob_reward_qwen_v1.json")
    args = parser.parse_args()
    if (not args.rows.is_file() or not args.source_report.is_file()
            or args.model_snapshot.name != REVISION or not args.model_snapshot.is_dir()
            or args.report.exists()):
        parser.error("Use existing pinned rows/model snapshot and a fresh report path")
    rows = load_rows(args.rows, args.source_report)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_snapshot, local_files_only=True, torch_dtype="auto",
        device_map="cuda:0", attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True)
    model.eval()

    def choose(prompt: str) -> str:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        ).to("cuda:0")
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=12, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    result = evaluate(rows, args.split, choose)
    report = {
        "purpose": "research_only_frozen_local_goal_action_probe",
        "model": MODEL, "revision": REVISION, "device": "cuda:0",
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "split": args.split, **result,
        "limitation": "Exploratory zero-shot first-action choices on synthetic task-disjoint MiniWoB. "
                      "No training, multi-step execution, real desktop transfer, Jev parity, "
                      "or Heliox runtime promotion.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Fit a research-only action-reward head on frozen local Qwen features.

MiniWoB first-action rewards, not Jev answers, are the only training labels.
The task-family evaluation split is never used for fitting or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from scripts.probe_miniwob_reward_qwen import MODEL, REVISION, load_rows

PENALTIES = (0.01, 0.1, 1.0, 10.0)


def candidate_prompts(rows: list[dict]) -> list[str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group"]].append(index)
    prompts = [""] * len(rows)
    for indices in groups.values():
        group = [rows[index] for index in indices]
        options = [
            {"index": index, **{key: row["candidate"].get(key) for key in
                                ("label", "tag", "type", "role", "width", "height",
                                 "background", "fill")}}
            for index, row in enumerate(group)
        ]
        shared = ("Goal: " + group[0]["goal"] + "\nVisible controls: "
                  + json.dumps(options, ensure_ascii=False))
        for candidate_index, row_index in enumerate(indices):
            prompts[row_index] = (shared + f"\nWould clicking control {candidate_index} "
                                  "satisfy the goal? Answer yes or no.")
    return prompts


def binary_loss_gradient(parameters: np.ndarray, matrix: np.ndarray,
                         labels: np.ndarray, penalty: float) -> tuple[float, np.ndarray]:
    logits = matrix @ parameters[:-1] + parameters[-1]
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    loss = np.logaddexp(0, logits).mean() - np.mean(labels * logits)
    loss += 0.5 * penalty * float(parameters[:-1] @ parameters[:-1])
    error = probabilities - labels
    gradient = np.r_[matrix.T @ error / len(labels) + penalty * parameters[:-1], error.mean()]
    return float(loss), gradient


def fit_head(matrix: np.ndarray, labels: np.ndarray, penalty: float) -> np.ndarray:
    initial = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    result = minimize(binary_loss_gradient, initial, args=(matrix, labels, penalty),
                      jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Action-reward head fit failed: {result.message}")
    return result.x


def group_metrics(rows: list[dict], scores: np.ndarray, split: str) -> dict[str, float | int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == split:
            groups[row["group"]].append(index)
    if not groups:
        raise ValueError("No groups in split")
    correct = 0
    random_rate = 0.0
    for indices in groups.values():
        chosen = indices[int(np.argmax(scores[indices]))]
        correct += rows[chosen]["reward"] == 1
        random_rate += sum(rows[index]["reward"] == 1 for index in indices) / len(indices)
    return {"groups": len(groups), "correct": correct, "success_rate": correct / len(groups),
            "random_success_rate": random_rate / len(groups)}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=root / "miniwob_reward_actions_v1.json")
    parser.add_argument("--source-report", type=Path,
                        default=root / "miniwob_reward_actions_v1_report.json")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=root / "miniwob_reward_head_v1.npz")
    parser.add_argument("--report", type=Path, default=root / "miniwob_reward_head_v1.json")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if (not args.rows.is_file() or not args.source_report.is_file()
            or args.model_snapshot.name != REVISION or not args.model_snapshot.is_dir()
            or args.weights.suffix != ".npz" or args.weights.exists() or args.report.exists()
            or not 128 <= args.max_length <= 2048):
        parser.error("Use pinned input/model, bounded context, and fresh outputs")
    rows = load_rows(args.rows, args.source_report)
    prompts = candidate_prompts(rows)
    if len(prompts) != len(rows):
        raise ValueError("Candidate prompts do not align with reward rows")
    prompt_sha256 = hashlib.sha256("\n".join(prompts).encode()).hexdigest()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen feature extraction")
    tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True)
    sequences = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if not tokens or len(tokens) > args.max_length:
            raise ValueError("A candidate prompt exceeds the fixed context limit")
        sequences.append(tokens)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_snapshot, local_files_only=True, torch_dtype="auto",
        device_map="cuda:0", attn_implementation="eager",
    )
    model = base.model.eval()
    features = []
    with torch.inference_mode():
        for index, tokens in enumerate(sequences, 1):
            tensor = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
            state = model(input_ids=tensor, use_cache=False, return_dict=True).last_hidden_state[0, -1]
            features.append(state.float().cpu().numpy())
            if index % 25 == 0 or index == len(sequences):
                print(json.dumps({"extracted": index, "total": len(sequences)}), flush=True)
    matrix = np.asarray(features, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Frozen model produced nonfinite features")
    labels = np.asarray([row["reward"] == 1 for row in rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in rows])
    validation = np.asarray([row["split"] == "validation" for row in rows])
    mean = matrix[train].mean(axis=0)
    scale = matrix[train].std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    trials = []
    for penalty in PENALTIES:
        weights = fit_head(normalized[train], labels[train], penalty)
        loss, _ = binary_loss_gradient(weights, normalized[validation], labels[validation], 0)
        trials.append((loss, penalty, weights))
    validation_loss, penalty, weights = min(trials, key=lambda trial: trial[0])
    scores = normalized @ weights[:-1] + weights[-1]
    result = {split: group_metrics(rows, scores, split) for split in ("validation", "eval")}
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.weights, weights=weights, mean=mean, scale=scale,
                        model_revision=np.asarray(REVISION), prompt_sha256=np.asarray(prompt_sha256))
    report = {
        "purpose": "research_only_frozen_qwen_first_action_reward_head",
        "model": MODEL, "revision": REVISION, "device": "cuda:0",
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "prompt_sha256": prompt_sha256, "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "feature_dimension": int(matrix.shape[1]),
        "rows": {split: sum(row["split"] == split for row in rows)
                 for split in ("train", "validation", "eval")},
        "regularization_selected_on_validation": penalty,
        "validation_binary_loss": validation_loss,
        "metrics": result,
        "limitation": "Small, contrastive, synthetic first-action corpus; task families were explored "
                      "before this run. Evaluation was not used to fit the head, but is not a "
                      "preregistered untouched benchmark. No Jev labels, multi-step autonomy, "
                      "real desktop transfer, or runtime promotion.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

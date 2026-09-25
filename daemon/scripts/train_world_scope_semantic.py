"""Offline semantic task-state head adjacent to Heliox's world model.

The pinned language backbone only extracts features. A numeric head is fit on
MIT ScopeJudge training labels; calibration selects its regularization and
caution threshold. Nothing here is loaded by the runtime or grants approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_qwen_v1_holdout import CASES as HELIOX_CASES  # noqa: E402
from scripts.evaluate_openjev_checkpoint import scopejudge_cases  # noqa: E402
from scripts.evaluate_qwen_instruct_scope import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
    messages,
)

REGULARIZATION = (0.001, 0.01, 0.1, 1.0)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive, negative = labels, ~labels
    return {"rows": len(labels), "in_scope_rows": int(np.sum(positive)),
            "out_of_scope_rows": int(np.sum(negative)),
            "auc": float(roc_auc_score(labels, scores)),
            "accuracy": float(np.mean(predicted == labels)),
            "balanced_accuracy": float((np.mean(predicted[positive]) +
                                        np.mean(~predicted[negative])) / 2),
            "false_allowances": int(np.sum(predicted & negative)),
            "false_warnings": int(np.sum(~predicted & positive))}


def normalize(features: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean = features["train"].mean(axis=0)
    scale = features["train"].std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {name: (values - mean) / scale for name, values in features.items()}, mean, scale


def select_head(train: np.ndarray, train_labels: np.ndarray,
                calibration: np.ndarray, calibration_labels: np.ndarray):
    trials = []
    for penalty in REGULARIZATION:
        head = LogisticRegression(C=penalty, max_iter=1000)
        head.fit(train, train_labels)
        scores = head.predict_proba(calibration)[:, 1]
        trials.append((float(roc_auc_score(calibration_labels, scores)), penalty, head, scores))
    selected = max(trials, key=lambda item: (item[0], -item[1]))
    threshold = math.nextafter(float(np.max(selected[3][~calibration_labels])), math.inf)
    return selected[2], threshold, [{"C": item[1], "calibration_auc": item[0]} for item in trials]


def embed_cases(model, tokenizer, cases, max_length: int, label: str) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    vectors, targets, overlong = [], [], 0
    with torch.inference_mode():
        for index, case in enumerate(cases, 1):
            prompt = tokenizer.apply_chat_template(
                messages(case.request, case.action), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > max_length:
                overlong += 1
                continue
            hidden = model.model(input_ids=encoded["input_ids"].to("cuda"),
                                 use_cache=False, return_dict=True).last_hidden_state[0, -1]
            if not torch.isfinite(hidden).all():
                raise FloatingPointError(f"Nonfinite semantic representation in {label} row {index}")
            vectors.append(hidden.float().cpu().numpy())
            targets.append(case.in_scope)
            if index % 100 == 0 or index == len(cases):
                print(json.dumps({"event": "embedded", "split": label, "rows": index,
                                  "total": len(cases), "overlong": overlong}), flush=True)
    if not vectors or len(set(targets)) != 2:
        raise ValueError(f"{label} needs token-eligible examples from both classes")
    return np.asarray(vectors), np.asarray(targets, dtype=np.bool_), overlong


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--artifact", type=Path,
                        default=root / "benchmarks/.bench-state/world_scope_semantic_v1.npz")
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/.bench-state/world_scope_semantic_v1.json")
    args = parser.parse_args()
    if args.max_length < 64 or args.artifact.exists() or args.report.exists():
        parser.error("Use max-length >=64 and fresh artifact/report paths")
    cases, manifests = {}, {}
    for name, partition in (("train", "scope_train"),
                            ("calibration", "scope_calibration"), ("test", "scope_test")):
        cases[name], manifests[name] = scopejudge_cases(args.scopejudge_path, partition)
    if not all(manifest == manifests["train"] for manifest in manifests.values()):
        raise ValueError("ScopeJudge source changed between partitions")
    groups = {name: {case.id.split(":", 1)[0] for case in part} for name, part in cases.items()}
    if any(groups[left] & groups[right] for left, right in
           (("train", "calibration"), ("train", "test"), ("calibration", "test"))):
        raise ValueError("ScopeJudge trajectories overlap")
    cases["heliox"] = tuple(HELIOX_CASES)

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free CUDA memory is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"}, output_loading_info=True)
    if loading.get("missing_keys"):
        raise RuntimeError("Pinned text backbone has missing weights")
    model.eval()
    features, labels, excluded = {}, {}, {}
    for name, part in cases.items():
        features[name], labels[name], excluded[name] = embed_cases(
            model, tokenizer, part, args.max_length, name)
    del model
    normalized, mean, scale = normalize(features)
    head, threshold, trials = select_head(normalized["train"], labels["train"],
                                           normalized["calibration"], labels["calibration"])
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, weights=head.coef_.astype(np.float32),
                        bias=head.intercept_.astype(np.float32), mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32), threshold=threshold,
                        model_revision=MODEL_REVISION)
    scores = {name: head.predict_proba(values)[:, 1] for name, values in normalized.items()}
    report = {
        "purpose": "research_only_semantic_world_task_state_head",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "scopejudge": manifests["train"], "max_length": args.max_length,
        "rows": {name: len(value) for name, value in labels.items()},
        "excluded_overlong": excluded, "regularization_trials": trials,
        "selected_C": float(head.C), "calibration_threshold": threshold,
        "calibration": metrics(labels["calibration"], scores["calibration"], threshold),
        "test_default": metrics(labels["test"], scores["test"], 0.5),
        "test_cautious": metrics(labels["test"], scores["test"], threshold),
        "heliox_transfer_default": metrics(labels["heliox"], scores["heliox"], 0.5),
        "heliox_transfer_cautious": metrics(labels["heliox"], scores["heliox"], threshold),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "limitation": "Public expert labels and inspected synthetic transfer only; no Jev parity, "
                      "live desktop evidence, world-model promotion, or execution authority.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in
                      ("test_default", "test_cautious", "heliox_transfer_cautious")}, indent=2))


if __name__ == "__main__":
    main()

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


def feature_identity(cases, max_length: int, batch_size: int = 1) -> str:
    identity = {"model": MODEL_ID, "revision": MODEL_REVISION,
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "max_length": max_length,
                "cases": [(case.id, case.request, case.action, case.in_scope) for case in cases]}
    if batch_size != 1:
        identity["batch_size"] = batch_size
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def save_feature_checkpoint(path: Path, identity: str, processed: int, overlong: int,
                            indices: list[int], vectors: list[np.ndarray],
                            targets: list[bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, identity=identity, processed=processed,
                            overlong=overlong, indices=np.asarray(indices, dtype=np.int32),
                            features=np.asarray(vectors, dtype=np.float32),
                            labels=np.asarray(targets, dtype=np.bool_))
    temporary.replace(path)


def load_feature_checkpoint(path: Path, identity: str, cases) -> tuple[int, int, list,
                                                                         list[np.ndarray], list[bool]]:
    if not path.exists():
        return 0, 0, [], [], []
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["identity"]) != identity:
            raise ValueError(f"Feature checkpoint identity mismatch: {path}")
        processed = int(saved["processed"])
        overlong = int(saved["overlong"])
        indices = saved["indices"].astype(int).tolist()
        vectors = list(saved["features"])
        targets = saved["labels"].astype(bool).tolist()
    if (not 0 <= processed <= len(cases) or overlong != processed - len(indices) or
            len(indices) != len(vectors) or len(indices) != len(targets) or
            indices != sorted(set(indices)) or any(index < 1 or index > processed for index in indices) or
            any(not np.isfinite(vector).all() for vector in vectors) or
            any(cases[index - 1].in_scope is not target
                for index, target in zip(indices, targets, strict=True))):
        raise ValueError(f"Feature checkpoint contents invalid: {path}")
    return processed, overlong, indices, vectors, targets


def embed_batch(model, token_ids: list, pad_token_id: int) -> list[np.ndarray]:
    import torch

    lengths = [len(ids) for ids in token_ids]
    device = next(model.parameters()).device
    input_ids = torch.full((len(token_ids), max(lengths)), pad_token_id,
                           dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(token_ids):
        input_ids[row, :len(ids)] = ids.to(device)
        attention_mask[row, :len(ids)] = 1
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False, return_dict=True).last_hidden_state
    selected = hidden[torch.arange(len(lengths), device=device),
                      torch.tensor(lengths, device=device) - 1]
    if not torch.isfinite(selected).all():
        raise FloatingPointError("Nonfinite semantic representation")
    return [vector.float().cpu().numpy() for vector in selected]


def embed_cases(model, tokenizer, cases, max_length: int, label: str,
                checkpoint_path: Path | None = None,
                batch_size: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has no pad token")
    cases = tuple(cases)
    identity = feature_identity(cases, max_length, batch_size) if checkpoint_path else None
    processed, overlong, indices, vectors, targets = (
        load_feature_checkpoint(checkpoint_path, identity, cases) if checkpoint_path
        else (0, 0, [], [], []))
    if processed:
        print(json.dumps({"event": "resumed", "split": label, "rows": processed,
                          "total": len(cases), "overlong": overlong}), flush=True)
    pending_ids, pending_indices = [], []

    def flush() -> None:
        if pending_ids:
            for row, vector in zip(pending_indices,
                                   embed_batch(model, pending_ids, pad_token_id), strict=True):
                vectors.append(vector)
                targets.append(cases[row - 1].in_scope)
                indices.append(row)
            pending_ids.clear()
            pending_indices.clear()

    with torch.inference_mode():
        for index in range(processed + 1, len(cases) + 1):
            case = cases[index - 1]
            prompt = tokenizer.apply_chat_template(
                messages(case.request, case.action), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > max_length:
                overlong += 1
            else:
                pending_ids.append(encoded["input_ids"][0])
                pending_indices.append(index)
                if len(pending_ids) == batch_size:
                    flush()
            if index % 100 == 0 or index == len(cases):
                flush()
            if checkpoint_path and (index % 100 == 0 or index == len(cases)):
                save_feature_checkpoint(checkpoint_path, identity, index, overlong,
                                        indices, vectors, targets)
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

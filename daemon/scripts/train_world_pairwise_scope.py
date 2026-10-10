"""Test Jev-style within-task candidate ranking in the offline world head.

The head sees public training pairs and expert scope labels only. It cannot
authorize tools, and official tests are evaluated after calibration selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.evaluate_world_conversation_scope import (
    AGENTIC_TEST_SHA256,
    load_pre_call_cases,
)
from scripts.train_world_multidomain_mlp import (
    DOMAINS,
    EVALUATION,
    balanced_weights,
    load_features,
)
from scripts.train_world_multiturn_scope_semantic import (
    AGENTIC_VALIDATION_SHA256,
    split_agentic,
)
from scripts.train_world_scope_semantic import metrics, normalize
from scripts.train_world_tool_scope_semantic import DATASET_HASHES, load_cases, split_validation

SEEDS = (19, 73)
PAIRWISE_WEIGHTS = (0.0, 0.5, 2.0)
EPOCHS = 120
HIDDEN = 64


def paired_indices(groups: list[str], labels: np.ndarray) -> np.ndarray:
    if len(groups) != len(labels):
        raise ValueError("Groups and labels differ")
    grouped = defaultdict(lambda: {True: [], False: []})
    for index, (group, label) in enumerate(zip(groups, labels, strict=True)):
        grouped[group][bool(label)].append(index)
    pairs = [
        (positive, negative) for items in grouped.values() for positive in items[True] for negative in items[False]
    ]
    if not pairs:
        raise ValueError("No within-task positive-negative pairs")
    return np.asarray(pairs, dtype=np.int64)


def top_one_accuracy(groups: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(groups) != len(labels) or len(groups) != len(scores):
        raise ValueError("Groups, labels, and scores differ")
    grouped = defaultdict(list)
    for group, label, score in zip(groups, labels, scores, strict=True):
        grouped[group].append((bool(label), float(score)))
    mixed = [
        items for items in grouped.values() if any(label for label, _ in items) and any(not label for label, _ in items)
    ]
    return {
        "mixed_tasks": len(mixed),
        "correct": sum(max(items, key=lambda item: item[1])[0] for items in mixed),
        "accuracy": float(np.mean([max(items, key=lambda item: item[1])[0] for items in mixed])) if mixed else None,
    }


def group_keys(dataset_dir: Path, feature_dir: Path, source: dict) -> dict[str, list[str]]:
    validation_source = dataset_dir / "agentic_validation_4o.json"
    agentic_validation = load_pre_call_cases(validation_source, AGENTIC_VALIDATION_SHA256)
    agentic_train, agentic_cal, _ = split_agentic(agentic_validation, validation_source)
    test_source = dataset_dir / "agentic_test_4o.json"
    agentic_test = load_pre_call_cases(test_source, AGENTIC_TEST_SHA256)
    tbac_validation = load_cases(dataset_dir / "validation.json", DATASET_HASHES["validation"], "tbac_validation")
    tbac_train, tbac_cal = split_validation(tbac_validation)
    tbac_test = load_cases(dataset_dir / "test.json", DATASET_HASHES["test"], "tbac_test")
    cases = {
        "agentic_train": agentic_train,
        "agentic_calibration": agentic_cal,
        "agentic_test": agentic_test,
        "tbac_train": tbac_train,
        "tbac_calibration": tbac_cal,
        "tbac_test": tbac_test,
    }
    task_by_id = {}
    for source_path in (validation_source, test_source):
        for row in json.loads(source_path.read_bytes()):
            task_by_id[row["id"]] = row["metadata"]["request"]["conversation"]["seed_task_request"][
                "task_tool_request"
            ]["task"]
    keys = {}
    for name, part in cases.items():
        with np.load(feature_dir / f"{name}.npz", allow_pickle=False) as saved:
            indices = saved["indices"].astype(int)
            labels = saved["labels"].astype(bool)
        if not np.array_equal(labels, [part[index - 1].in_scope for index in indices]):
            raise ValueError(f"Feature labels do not match source cases: {name}")
        keys[name] = [
            (task_by_id[part[index - 1].id] if name.startswith("agentic") else part[index - 1].request)
            for index in indices
        ]
    if (
        not set(keys["agentic_train"]) <= set(keys["tbac_train"])
        or not set(keys["agentic_calibration"]) <= set(keys["tbac_calibration"])
        or not set(keys["agentic_test"]) <= set(keys["tbac_test"])
        or set(keys["agentic_train"]) & set(keys["agentic_calibration"])
        or set(keys["tbac_test"]) & (set(keys["tbac_train"]) | set(keys["tbac_calibration"]))
    ):
        raise ValueError("Task partitions overlap or fail to align")
    if source["agentic"]["validation_sha256"] != AGENTIC_VALIDATION_SHA256:
        raise ValueError("Agentic source changed")
    return keys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=state / "tbac_6385bd3")
    parser.add_argument("--source-report", type=Path, default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--artifact", type=Path, default=state / "world_pairwise_scope_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "world_pairwise_scope_v1.json")
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")
    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    if source["domains"] != list(DOMAINS):
        raise ValueError("Expected frozen three-domain feature report")
    features, labels = load_features(args.feature_dir, source)
    keys = group_keys(args.dataset_dir, args.feature_dir, source)
    features["train"] = np.vstack([features[f"{name}_train"] for name in DOMAINS])
    normalized, mean, scale = normalize(features)
    offsets = {
        domain: sum(len(labels[f"{prior}_train"]) for prior in DOMAINS[:index]) for index, domain in enumerate(DOMAINS)
    }
    pairs = {
        domain: paired_indices(keys[f"{domain}_train"], labels[f"{domain}_train"]) + offsets[domain]
        for domain in ("agentic", "tbac")
    }

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research training run")
    device = torch.device("cuda")
    train_x = torch.as_tensor(normalized["train"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(
        np.concatenate([labels[f"{name}_train"] for name in DOMAINS]), dtype=torch.float32, device=device
    )
    weights = torch.as_tensor(balanced_weights(labels), device=device)
    pair_tensors = {name: torch.as_tensor(value, device=device) for name, value in pairs.items()}
    calibration_x = {
        name: torch.as_tensor(normalized[f"{name}_calibration"], dtype=torch.float32, device=device) for name in DOMAINS
    }
    candidates = []
    selected = None
    for pairwise_weight in PAIRWISE_WEIGHTS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(train_x.shape[1], HIDDEN),
                torch.nn.GELU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(HIDDEN, 1),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            for _ in range(EPOCHS):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x).squeeze(1)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y, reduction="none")
                loss = (bce * weights).mean()
                if pairwise_weight:
                    ranking = [
                        torch.nn.functional.softplus(-(logits[indices[:, 0]] - logits[indices[:, 1]])).mean()
                        for indices in pair_tensors.values()
                    ]
                    loss = loss + pairwise_weight * sum(ranking) / len(ranking)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                calibration_scores = {
                    name: torch.sigmoid(model(value)).squeeze(1).cpu().numpy() for name, value in calibration_x.items()
                }
            auc = {
                name: float(roc_auc_score(labels[f"{name}_calibration"], values))
                for name, values in calibration_scores.items()
            }
            top_one = {
                name: top_one_accuracy(
                    keys[f"{name}_calibration"], labels[f"{name}_calibration"], calibration_scores[name]
                )
                for name in ("agentic", "tbac")
            }
            threshold = math.nextafter(
                max(float(np.max(calibration_scores[name][~labels[f"{name}_calibration"]])) for name in DOMAINS),
                math.inf,
            )
            rank = (
                min(auc.values()),
                sum(top_one[name]["accuracy"] for name in top_one) / len(top_one),
                sum(auc.values()) / len(DOMAINS),
            )
            candidates.append(
                {
                    "pairwise_weight": pairwise_weight,
                    "seed": seed,
                    "calibration_auc": auc,
                    "calibration_top_one": top_one,
                    "threshold": threshold,
                }
            )
            print(json.dumps({"event": "candidate", **candidates[-1]}), flush=True)
            if selected is None or rank > selected[0]:
                selected = (
                    rank,
                    pairwise_weight,
                    seed,
                    threshold,
                    {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                )
    assert selected is not None
    _, pairwise_weight, seed, threshold, state_dict = selected
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], HIDDEN), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(HIDDEN, 1)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        scores = {
            name: torch.sigmoid(model(torch.as_tensor(value, dtype=torch.float32, device=device)))
            .squeeze(1)
            .cpu()
            .numpy()
            for name, value in normalized.items()
            if name != "train"
        }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        w1=state_dict["0.weight"].numpy(),
        b1=state_dict["0.bias"].numpy(),
        w2=state_dict["3.weight"].numpy(),
        b2=state_dict["3.bias"].numpy(),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        threshold=threshold,
        source_report_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    report = {
        "purpose": "research_only_pairwise_world_task_state_head",
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "pair_counts": {name: len(value) for name, value in pairs.items()},
        "epochs": EPOCHS,
        "hidden": HIDDEN,
        "candidates": candidates,
        "selected": {"pairwise_weight": pairwise_weight, "seed": seed, "threshold": threshold},
        "metrics": {
            name: {
                "default": metrics(labels[name], scores[name], 0.5),
                "cautious": metrics(labels[name], scores[name], threshold),
            }
            for name in EVALUATION
        },
        "test_top_one": {
            name: top_one_accuracy(keys[name], labels[name], scores[name]) for name in ("agentic_test", "tbac_test")
        },
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "limitation": "Public within-task relevance is not authorization, live desktop "
        "safety evidence, or Jev parity; tests were inspected in prior research.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"selected": report["selected"], "test_top_one": report["test_top_one"], "metrics": report["metrics"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

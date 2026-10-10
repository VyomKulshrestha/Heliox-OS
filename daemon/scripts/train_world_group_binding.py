"""Research-only GPU fit of a grouped target-binding head.

The public EntityBindingFailures cases are training/calibration material here,
not a fresh test. Previously inspected tau-bench retail cases are transfer
probes only. Nothing is loaded by the Heliox daemon or grants tool permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.evaluate_world_entity_binding import build_cases as build_entity_cases
from scripts.evaluate_world_entity_binding import load_rows as load_entity_rows
from scripts.evaluate_world_tau_binding import build_cases as build_tau_cases
from scripts.evaluate_world_tau_binding import load_source as load_tau_source
from scripts.train_world_scope_semantic import feature_identity, load_feature_checkpoint

SEED = 20260925


def group_indices(cases: tuple, *, entity: bool) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        key = case.id.split(":", 1)[0] if entity else case.id.rsplit(":", 1)[0]
        grouped[key].append(index)
    return dict(sorted(grouped.items()))


def split_entity_groups(rows: list[dict], cases: tuple, groups: dict[str, list[int]]) -> dict[str, list[str]]:
    families: dict[tuple[str, str], list[str]] = defaultdict(list)
    if len(rows) != len(groups) or {row["task_id"] for row in rows} != set(groups):
        raise ValueError("Source tasks and candidate groups differ")
    for row in rows:
        families[row["domain"], row["ambiguity"]].append(row["task_id"])
    positive, ambiguous = [], []
    for family, keys in sorted(families.items()):
        label_counts = {sum(bool(cases[index].in_scope) for index in groups[key]) for key in keys}
        if label_counts not in ({0}, {1}):
            raise ValueError(f"Family mixes gold and ambiguous cases: {family}")
        (positive if label_counts == {1} else ambiguous).append(family)
    if len(positive) != 7 or len(ambiguous) != 2:
        raise ValueError("Pinned entity-binding family mix changed")
    for key, indices in groups.items():
        count = sum(bool(cases[index].in_scope) for index in indices)
        if count > 1:
            raise ValueError(f"Expected at most one gold candidate in {key}")
    rng = random.Random(SEED)
    rng.shuffle(positive)
    rng.shuffle(ambiguous)

    def expand(selected: list[tuple[str, str]]) -> list[str]:
        return [key for family in selected for key in sorted(families[family])]

    return {
        "train": expand(positive[:5] + ambiguous[:1]),
        "calibration": expand(positive[5:6]),
        "holdout": expand(positive[6:] + ambiguous[1:]),
    }


def load_complete_features(path: Path, cases: tuple, max_tokens: int, batch_size: int) -> np.ndarray:
    processed, overlong, indices, vectors, labels = load_feature_checkpoint(
        path, feature_identity(cases, max_tokens, batch_size), cases
    )
    if (
        processed != len(cases)
        or overlong
        or len(vectors) != len(cases)
        or not np.array_equal(np.asarray(indices), np.arange(1, len(cases) + 1))
        or not np.array_equal(np.asarray(labels, dtype=bool), np.asarray([case.in_scope for case in cases], dtype=bool))
    ):
        raise ValueError(f"Incomplete or mismatched feature checkpoint: {path}")
    features = np.asarray(vectors, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature vectors: {path}")
    return features


def normalize_features(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 0.1)
    return (mean, scale, *(np.clip((values - mean) / scale, -5.0, 5.0).astype(np.float32) for values in others))


def group_arrays(
    features: np.ndarray, cases: tuple, groups: dict[str, list[int]], keys: list[str], width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((len(keys), width, features.shape[1]), dtype=np.float32)
    mask = np.zeros((len(keys), width), dtype=bool)
    gold = np.full(len(keys), width, dtype=np.int64)
    for row, key in enumerate(keys):
        indices = groups[key]
        if len(indices) > width:
            raise ValueError("Group exceeds candidate width")
        values[row, : len(indices)] = features[indices]
        mask[row, : len(indices)] = True
        positives = [column for column, index in enumerate(indices) if cases[index].in_scope]
        if len(positives) > 1:
            raise ValueError("Grouped training requires at most one gold target")
        if positives:
            gold[row] = positives[0]
    return values, mask, gold


def group_metrics(
    cases: tuple, groups: dict[str, list[int]], keys: list[str], scores: np.ndarray, abstain: float
) -> dict[str, int]:
    correct = wrong_target = false_warning = false_allowance = 0
    correct_top_rank = resolvable = 0
    for key in keys:
        indices = groups[key]
        winner = max(indices, key=lambda index: scores[index])
        selected = winner if scores[winner] > abstain else None
        gold = {index for index in indices if cases[index].in_scope}
        resolvable += bool(gold)
        correct_top_rank += winner in gold
        if selected is None:
            correct += not gold
            false_warning += bool(gold)
        elif selected in gold:
            correct += 1
        else:
            wrong_target += bool(gold)
            false_allowance += not gold
    return {
        "groups": len(keys),
        "resolvable": resolvable,
        "correct_top_rank": correct_top_rank,
        "exact_or_valid_selection": correct,
        "wrong_target": wrong_target,
        "false_warning": false_warning,
        "false_allowance_on_ambiguous": false_allowance,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-dataset", type=Path)
    parser.add_argument("--tau-source-root", type=Path, default=state / "tau-bench-source")
    parser.add_argument("--entity-features", type=Path, default=state / "world_entity_binding_features_v1.npz")
    parser.add_argument("--tau-features", type=Path, default=state / "world_tau_binding_features_v1.npz")
    parser.add_argument("--artifact", type=Path, default=state / "world_group_binding_family_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "world_group_binding_family_v1.json")
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")

    entity_rows = load_entity_rows(args.entity_dataset)
    entity_cases, _ = build_entity_cases(entity_rows)
    tau_cases, _ = build_tau_cases(*load_tau_source(args.tau_source_root))
    entity = load_complete_features(args.entity_features, entity_cases, 1024, 4)
    tau = load_complete_features(args.tau_features, tau_cases, 1536, 2)
    if entity.shape[1] != tau.shape[1]:
        raise ValueError("Frozen feature widths differ")
    groups = group_indices(entity_cases, entity=True)
    splits = split_entity_groups(entity_rows, entity_cases, groups)
    train_rows = [index for key in splits["train"] for index in groups[key]]
    mean, scale, entity_normalized, tau_normalized = normalize_features(entity[train_rows], entity, tau)
    width = max(len(indices) for indices in groups.values())

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research training run")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = torch.nn.Linear(entity.shape[1], 1, bias=False).to(device)
    abstain = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.AdamW([*model.parameters(), abstain], lr=3e-4, weight_decay=0.1)
    tensors = {}
    for name, keys in splits.items():
        values, mask, gold = group_arrays(entity_normalized, entity_cases, groups, keys, width)
        tensors[name] = (
            torch.as_tensor(values, device=device),
            torch.as_tensor(mask, device=device),
            torch.as_tensor(gold, device=device),
        )

    def logits(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        values, mask, gold = tensors[name]
        candidate = model(values).squeeze(-1).masked_fill(~mask, -1e9)
        return torch.cat((candidate, abstain.expand(len(values), 1)), dim=1), gold

    best_loss = float("inf")
    best_state = None
    selected_epoch = 0
    stale = 0
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits, train_gold = logits("train")
        loss = torch.nn.functional.cross_entropy(train_logits, train_gold)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            calibration_logits, calibration_gold = logits("calibration")
            calibration_loss = float(torch.nn.functional.cross_entropy(calibration_logits, calibration_gold).item())
        if calibration_loss < best_loss - 1e-5:
            best_loss = calibration_loss
            best_state = (model.weight.detach().cpu().numpy().copy(), float(abstain.detach().cpu()))
            selected_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    assert best_state is not None
    weight, abstain_score = best_state
    entity_scores = (entity_normalized @ weight.T).ravel()
    tau_scores = (tau_normalized @ weight.T).ravel()
    tau_groups = group_indices(tau_cases, entity=False)
    report = {
        "purpose": "research_only_grouped_world_task_binding_head",
        "model": "Qwen/Qwen3.5-2B frozen features; CUDA linear listwise head with abstention",
        "seed": SEED,
        "selected_epoch": selected_epoch,
        "selection": "minimum source-family-disjoint calibration cross-entropy; no holdout or tau selection",
        "splits": {name: len(keys) for name, keys in splits.items()},
        "split_families": {
            name: sorted({(row["domain"], row["ambiguity"]) for row in entity_rows if row["task_id"] in set(keys)})
            for name, keys in splits.items()
        },
        "metrics": {
            name: group_metrics(entity_cases, groups, keys, entity_scores, abstain_score)
            for name, keys in splits.items()
        },
        "tau_transfer": group_metrics(tau_cases, tau_groups, list(tau_groups), tau_scores, abstain_score),
        "sources": {
            "entity_binding": "CC BY 4.0; pinned 60-task public source; previously inspected",
            "tau_retail": "MIT; pinned public source; previously inspected; transfer only",
            "entity_features_sha256": hashlib.sha256(args.entity_features.read_bytes()).hexdigest(),
            "tau_features_sha256": hashlib.sha256(args.tau_features.read_bytes()).hexdigest(),
        },
        "limitations": "Tiny adaptive public-data probe. Published retail actions may depend on earlier "
        "workflow steps. Scores are not calibrated execution probabilities; no live desktop "
        "validation, Jev parity, or runtime action authority.",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact, weight=weight.astype(np.float32), abstain=np.float32(abstain_score), mean=mean, scale=scale
    )
    report["artifact_sha256"] = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"selected_epoch": selected_epoch, **report["metrics"], "tau_transfer": report["tau_transfer"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()

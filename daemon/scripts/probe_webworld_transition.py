"""Compare type-only and bound-target browser transition heads on CUDA.

This is an offline research probe on hashed accessibility features, not a
browser simulator or a live Heliox model. Raw page content is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.sample_webworld_transitions import _state_connected_labels


def load_features(path: Path, source_report: Path) -> dict[str, dict[str, np.ndarray]]:
    metadata = json.loads(source_report.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("features_sha256"):
        raise ValueError("WebWorld feature archive does not match its source report")
    with np.load(path, allow_pickle=False) as data:
        states = data["state"].astype(np.float32)
        targets = data["target"].astype(np.float32)
        types = data["action_type"].astype(np.float32)
        bound = data["action_bound"].astype(np.float32)
        path_actions = data["action_path"].astype(np.float32) if "action_path" in data.files else None
        state_digest = data["state_digest"].copy() if "state_digest" in data.files else None
        groups = data["group"].astype(np.int64)
        splits = data["split"].astype(str)
    count = len(states)
    if (
        states.shape != (count, 1024)
        or targets.shape != (count, 1024)
        or types.shape != (count, 256)
        or bound.shape != (count, 256)
        or groups.shape != (count,)
        or splits.shape != (count,)
        or (path_actions is not None and path_actions.shape != (count, 256))
        or (state_digest is not None and (state_digest.shape != (count, 32) or state_digest.dtype != np.uint8))
        or set(splits) != {"train", "validation", "eval"}
        or not all(
            np.all(np.isfinite(array)) and np.all(array >= 0)
            for array in (states, targets, types, bound) + ((path_actions,) if path_actions is not None else ())
        )
    ):
        raise ValueError("Invalid bounded WebWorld features")
    clusters = _state_connected_labels(states, groups)
    result = {}
    group_sets = {}
    for split in ("train", "validation", "eval"):
        indices = np.flatnonzero(splits == split)
        group_sets[split] = set(groups[indices])
        result[split] = {
            "state": states[indices],
            "target": targets[indices],
            "action_type": types[indices],
            "action_bound": bound[indices],
            "groups": groups[indices],
            "clusters": clusters[indices],
        }
        if path_actions is not None:
            result[split]["action_path"] = path_actions[indices]
        if state_digest is not None:
            result[split]["state_digest"] = state_digest[indices]
    if any(
        group_sets[left] & group_sets[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Trajectory leaked across WebWorld partitions")
    state_sets = {split: {state.tobytes() for state in partition["state"]} for split, partition in result.items()}
    if any(
        state_sets[left] & state_sets[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Identical pre-action feature leaked across WebWorld partitions")
    return result


def action_permutation(types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle actions only among rows with the same type feature."""
    permutation = np.arange(len(types))
    _, inverse = np.unique(types, axis=0, return_inverse=True)
    for group in np.unique(inverse):
        indices = np.flatnonzero(inverse == group)
        permutation[indices] = rng.permutation(indices)
    return permutation


def weighted_mse(predicted: np.ndarray, state: np.ndarray, target: np.ndarray) -> float:
    weights = 1 + 4 * (np.abs(target - state) >= 0.01)
    return float((((predicted - target) ** 2) * weights).mean())


def feature_equivalent_action_pairs(partition: dict[str, np.ndarray]) -> np.ndarray:
    """Find distinct observed outcomes for different actions from one feature state.

    New archives compare digests of the exact raw accessibility tree. Older
    archives fall back to feature equality; neither establishes shared history.
    """
    buckets: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
    identities = partition.get("state_digest", partition["state"])
    for index, (state, action_type) in enumerate(zip(identities, partition["action_type"])):
        buckets[(state.tobytes(), action_type.tobytes())].append(index)
    pairs = [
        (left, right)
        for indices in buckets.values()
        for left, right in combinations(indices, 2)
        if not np.array_equal(partition["action_bound"][left], partition["action_bound"][right])
        and not np.array_equal(partition["target"][left], partition["target"][right])
    ]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def paired_outcome_diagnostic(
    predicted: np.ndarray,
    target: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, float | int]:
    """Score whether each action prediction matches its outcome over the alternative."""
    if predicted.shape != target.shape or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Predictions, targets, and pairs must align")
    if len(pairs) == 0:
        return {"pairs": 0, "correct_action_wins": 0, "ties": 0, "mean_swapped_minus_correct_mse": 0.0}
    left, right = pairs.T
    if np.any(pairs < 0) or np.any(pairs >= len(target)):
        raise ValueError("Pair index is outside the prediction partition")
    correct = ((predicted[left] - target[left]) ** 2).mean(axis=1) + ((predicted[right] - target[right]) ** 2).mean(
        axis=1
    )
    swapped = ((predicted[left] - target[right]) ** 2).mean(axis=1) + ((predicted[right] - target[left]) ** 2).mean(
        axis=1
    )
    advantage = swapped - correct
    return {
        "pairs": len(pairs),
        "correct_action_wins": int(np.sum(advantage > 1e-9)),
        "ties": int(np.sum(np.abs(advantage) <= 1e-9)),
        "mean_swapped_minus_correct_mse": float(advantage.mean()),
    }


def paired_matching_loss(predicted, target):
    """Cross-entropy for matching each of two actions to its observed outcome."""
    import torch
    from torch.nn import functional as F

    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1] != 2:
        raise ValueError("Paired predictions and targets must have shape (pairs, 2, features)")
    scores = -1000.0 * (predicted[:, :, None, :] - target[:, None, :, :]).square().mean(dim=-1)
    labels = torch.arange(2, device=predicted.device).expand(len(predicted), -1)
    return F.cross_entropy(scores.reshape(-1, 2), labels.reshape(-1))


def clustered_advantage(
    type_errors: np.ndarray,
    bound_errors: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap state-connected trajectory components for a paired MSE gap."""
    if type_errors.shape != bound_errors.shape or groups.shape != type_errors.shape or not len(groups):
        raise ValueError("Paired errors and trajectory groups must align")
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = []
    difference = type_errors - bound_errors
    for _ in range(1000):
        selected = rng.integers(0, len(unique), size=len(unique))
        sampled_rows = np.concatenate([indices[index] for index in selected])
        samples.append(float(difference[sampled_rows].mean()))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "groups": len(unique),
        "type_minus_bound_weighted_mse": float(difference.mean()),
        "cluster_bootstrap_95_low": float(lower),
        "cluster_bootstrap_95_high": float(upper),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument(
        "--branch-loss-weight",
        type=float,
        default=0.0,
        help="Optional paired-outcome training loss; model selection remains validation MSE",
    )
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if (
        args.report.exists()
        or args.epochs < 1
        or args.permutations < 1
        or not np.isfinite(args.branch_loss_weight)
        or args.branch_loss_weight < 0
    ):
        parser.error("Use a fresh report and positive epoch/permutation counts")
    rows = load_features(args.features, args.source_report)
    training_pairs = feature_equivalent_action_pairs(rows["train"])
    if args.branch_loss_weight and not len(training_pairs):
        parser.error("Paired-outcome training requires feature-equivalent action pairs")

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    results = {}
    per_row_errors: dict[str, dict[str, np.ndarray]] = {}
    paired_outcomes: dict[str, dict[str, dict[str, float | int]]] = {}
    action_keys = ("action_type", "action_bound") + (("action_path",) if "action_path" in rows["train"] else ())
    for action_key in action_keys:
        torch.manual_seed(args.seed)
        model = nn.Sequential(nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 1024)).to(device)
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
        arrays = {
            split: {
                key: torch.from_numpy(value).to(device)
                for key, value in partition.items()
                if key not in {"groups", "clusters", "state_digest"}
            }
            for split, partition in rows.items()
        }

        def predict(partition, action, model=model):
            return torch.clamp(partition["state"] + model(torch.cat((partition["state"], action), dim=1)), 0, 1)

        def loss(predicted, partition):
            weights = 1 + 4 * ((partition["target"] - partition["state"]).abs() >= 0.01)
            return (((predicted - partition["target"]) ** 2) * weights).mean()

        best_loss, best_epoch, best_weights = float("inf"), 0, None
        train = arrays["train"]
        pair_indices = torch.as_tensor(training_pairs.reshape(-1), dtype=torch.long, device=device)
        for epoch in range(args.epochs):
            model.train()
            for indices in torch.randperm(len(train["state"]), device=device).split(32):
                batch = {key: value[indices] for key, value in train.items()}
                output = predict(batch, batch[action_key])
                training_loss = loss(output, batch)
                optimizer.zero_grad(set_to_none=True)
                training_loss.backward()
                optimizer.step()
            if args.branch_loss_weight and action_key != "action_type":
                pair_batch = {key: value[pair_indices] for key, value in train.items()}
                pair_prediction = predict(pair_batch, pair_batch[action_key]).reshape(-1, 2, 1024)
                pair_target = pair_batch["target"].reshape(-1, 2, 1024)
                branch_loss = paired_matching_loss(pair_prediction, pair_target)
                optimizer.zero_grad(set_to_none=True)
                (args.branch_loss_weight * branch_loss).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation = arrays["validation"]
                score = float(loss(predict(validation, validation[action_key]), validation))
            if score < best_loss:
                best_loss, best_epoch = score, epoch + 1
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert best_weights is not None
        model.load_state_dict(best_weights)
        model.eval()
        split_scores = {}
        per_row_errors[action_key] = {}
        paired_outcomes[action_key] = {}
        with torch.no_grad():
            for split in ("validation", "eval"):
                partition, tensors = rows[split], arrays[split]
                state, target = partition["state"], partition["target"]
                correct = predict(tensors, tensors[action_key]).cpu().numpy()
                paired_outcomes[action_key][split] = paired_outcome_diagnostic(
                    correct, target, feature_equivalent_action_pairs(partition)
                )
                weights = 1 + 4 * (np.abs(target - state) >= 0.01)
                per_row_errors[action_key][split] = (((correct - target) ** 2) * weights).mean(axis=1)
                rng = np.random.default_rng(args.seed + (1 if split == "validation" else 2))
                shuffle_scores = []
                for _ in range(args.permutations):
                    indices = action_permutation(partition["action_type"], rng)
                    shuffled = tensors[action_key][indices]
                    shuffle_scores.append(weighted_mse(predict(tensors, shuffled).cpu().numpy(), state, target))
                split_scores[split] = {
                    "rows": len(state),
                    "unchanged": weighted_mse(state, state, target),
                    "correct_action": weighted_mse(correct, state, target),
                    "within_type_shuffled_action_mean": float(np.mean(shuffle_scores)),
                    "within_type_shuffle_rank": float(
                        (1 + sum(score <= weighted_mse(correct, state, target) for score in shuffle_scores))
                        / (len(shuffle_scores) + 1)
                    ),
                }
        results[action_key] = {"selected_epoch": best_epoch, "scores": split_scores}
    report = {
        "purpose": "research_only_browser_action_binding_probe",
        "source_features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "branch_loss_weight": args.branch_loss_weight,
        "training_feature_equivalent_pairs": len(training_pairs),
        "action_permutations": args.permutations,
        "split_basis": "trajectory_plus_identical_prestate_not_website",
        "paired_state_basis": (
            "exact_raw_tree_sha256" if "state_digest" in rows["train"] else "hashed_accessibility_features"
        ),
        "results": results,
        "feature_equivalent_paired_outcomes": paired_outcomes,
        "paired_type_vs_bound": {
            split: clustered_advantage(
                per_row_errors["action_type"][split],
                per_row_errors["action_bound"][split],
                rows[split]["clusters"],
                seed=args.seed + (11 if split == "validation" else 12),
            )
            for split in ("validation", "eval")
        },
        "paired_type_vs_path": (
            {
                split: clustered_advantage(
                    per_row_errors["action_type"][split],
                    per_row_errors["action_path"][split],
                    rows[split]["clusters"],
                    seed=args.seed + (11 if split == "validation" else 12),
                )
                for split in ("validation", "eval")
            }
            if "action_path" in action_keys
            else None
        ),
        "paired_bound_vs_path": (
            {
                split: clustered_advantage(
                    per_row_errors["action_bound"][split],
                    per_row_errors["action_path"][split],
                    rows[split]["clusters"],
                    seed=args.seed + (11 if split == "validation" else 12),
                )
                for split in ("validation", "eval")
            }
            if "action_path" in action_keys
            else None
        ),
        "paired_bootstrap_unit": "state_connected_trajectory_component",
        "limitation": "Hashed accessibility features, small sampled trajectories, no site-disjoint transfer. "
        "Feature-equivalent pairs are not verified same-history counterfactuals, and older "
        "archives may alias different raw screens. Shuffling compares actions from other states. "
        "No runtime model, safety, autonomy, or Jev parity claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Research-only GPU probe on observed screenshot/action/next-screenshot features.

Episode-disjoint evaluation compares action-conditioned predictions with both
an unchanged-screen baseline and an action-ablated version of the same model.
An optional linear export matches the live UiJepaPredictor contract, but is
always shadow-only and never grants action authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_spatial_joint_model(nn: Any, channels: int) -> Any:
    """Share visual features between next-screen and observed-action prediction."""
    import torch

    class SpatialJoint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.GELU(),
            )
            self.forward_head = nn.Sequential(
                nn.Conv2d(44, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, channels, 3, padding=1),
            )
            self.inverse_head = nn.Sequential(
                nn.Conv2d(66, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 2),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.forward_head[-1].weight)
            nn.init.zeros_(self.forward_head[-1].bias)

        def forward(self, state: Any, action_planes: Any) -> Any:
            return self.forward_head(torch.cat((self.encoder(state), action_planes), dim=1))

        def inverse(self, state: Any, target: Any) -> Any:
            height, width = state.shape[-2:]
            y, x = torch.meshgrid(
                torch.linspace(0, 1, height, device=state.device),
                torch.linspace(0, 1, width, device=state.device),
                indexing="ij",
            )
            coordinates = torch.stack((x, y)).expand(len(state), -1, -1, -1)
            return self.inverse_head(
                torch.cat(
                    (
                        self.encoder(state),
                        self.encoder(target),
                        coordinates,
                    ),
                    dim=1,
                )
            )

    return SpatialJoint()


def feature_shape(source_report: Path) -> tuple[int, int, int]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    shape = report.get("feature_shape", [1, 16, 16])
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) for value in shape)
        or not 1 <= shape[0] <= 4
        or not 8 <= shape[1] <= 64
        or not 8 <= shape[2] <= 64
        or np.prod(shape) > 16384
    ):
        raise ValueError("Invalid bounded visual feature shape")
    return tuple(shape)


def load_features(path: Path, source_report: Path) -> dict[str, dict[str, np.ndarray]]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    shape = feature_shape(source_report)
    dimension = int(np.prod(shape))
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("features_sha256"):
        raise ValueError("Visual features do not match the pinned source report")
    with np.load(path, allow_pickle=False) as data:
        states = data["state"].astype(np.float32)
        targets = data["target"].astype(np.float32)
        actions = data["action"].astype(np.int64)
        coords = data["coords"].astype(np.float32)
        groups = data["group"].astype(str)
        splits = data["split"].astype(str)
    count = len(states)
    if (
        states.shape != (count, dimension)
        or targets.shape != (count, dimension)
        or actions.shape != (count,)
        or coords.shape != (count, 2)
        or groups.shape != (count,)
        or splits.shape != (count,)
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(coords))
        or np.any(states < 0)
        or np.any(states > 1)
        or np.any(targets < 0)
        or np.any(targets > 1)
        or np.any(coords < 0)
        or np.any(coords > 1)
        or np.any(actions < 0)
        or np.any(actions > 8)
    ):
        raise ValueError("Invalid visual transition features")
    result = {}
    group_sets = {}
    for split in ("train", "validation", "eval"):
        indices = np.where(splits == split)[0]
        if not len(indices):
            raise ValueError(f"Missing {split} rows")
        group_sets[split] = set(groups[indices])
        onehot = np.eye(9, dtype=np.float32)[actions[indices]]
        result[split] = {
            "state": states[indices],
            "target": targets[indices],
            "action": np.concatenate((onehot, coords[indices]), axis=1),
            "groups": groups[indices],
        }
    if any(
        group_sets[left] & group_sets[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Episode leaked across partitions")
    if set(splits) != set(result):
        raise ValueError("Unexpected split labels")
    return result


def combine_training_features(
    primary: dict[str, dict[str, np.ndarray]],
    additional: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Add train/validation observations while retaining the primary eval set."""
    if any(
        primary[split]["state"].shape[1:] != additional[split]["state"].shape[1:]
        for split in ("train", "validation", "eval")
    ):
        raise ValueError("Additional visual features must match the primary feature shape")
    merged = {}
    for split in ("train", "validation", "eval"):
        left = primary[split]
        right = additional[split]
        merged[split] = {
            key: (np.concatenate((left[key], right[key]), axis=0) if split != "eval" else left[key].copy())
            for key in ("state", "target", "action")
        }
        groups = [f"primary:{group}" for group in left["groups"]]
        if split != "eval":
            groups.extend(f"additional:{group}" for group in right["groups"])
        merged[split]["groups"] = np.asarray(groups)
    screens = {split: {state.tobytes() for state in merged[split]["state"]} for split in merged}
    if any(
        screens[left] & screens[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Identical pre-action screen leaked across visual partitions")
    return merged


def metrics(predicted: np.ndarray, state: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = (predicted - target) ** 2
    changed = np.abs(target - state) >= 0.02
    weights = 1 + 4 * changed
    delta = target - state
    predicted_delta = predicted - state
    delta_norm = np.linalg.norm(delta, axis=1)
    predicted_norm = np.linalg.norm(predicted_delta, axis=1)
    eligible = (delta_norm > 0.02) & (predicted_norm > 1e-6)
    cosine = np.zeros(len(state), dtype=np.float32)
    cosine[eligible] = np.sum(delta[eligible] * predicted_delta[eligible], axis=1) / (
        delta_norm[eligible] * predicted_norm[eligible]
    )
    return {
        "mse": float(error.mean()),
        "changed_pixel_mse": float(error[changed].mean()) if np.any(changed) else 0.0,
        "weighted_mse": float((error * weights).mean()),
        "mean_delta_cosine": float(cosine.mean()),
        "changed_pixel_fraction": float(changed.mean()),
    }


def weighted_pixel_mse_by_row(predicted: np.ndarray, state: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = (predicted - target) ** 2
    weights = 1 + 4 * (np.abs(target - state) >= 0.02)
    return (error * weights).mean(axis=1).astype(np.float64)


def action_shuffle_diagnostic(correct: float, shuffled: list[float]) -> dict[str, float]:
    """Positive advantage beats shuffles; rank is descriptive, not a p-value."""
    if not shuffled or not np.isfinite(correct) or not np.all(np.isfinite(shuffled)):
        raise ValueError("Action-shuffle scores must be finite and nonempty")
    return {
        "weighted_mse_advantage": float(np.mean(shuffled) - correct),
        "one_sided_shuffle_rank": float((1 + sum(score <= correct for score in shuffled)) / (len(shuffled) + 1)),
    }


def group_bootstrap_loss_delta(
    reference_losses: np.ndarray, candidate_losses: np.ndarray, groups: np.ndarray, *, seed: int, replicates: int = 5000
) -> dict[str, float | int]:
    """Paired row-weighted candidate-minus-reference delta resampling whole groups."""
    reference = np.asarray(reference_losses, dtype=np.float64)
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    group_ids = np.asarray(groups, dtype=str)
    if (
        reference.ndim != 1
        or candidate.shape != reference.shape
        or group_ids.shape != reference.shape
        or not len(reference)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(candidate))
        or np.any(group_ids == "")
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 100
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("Group bootstrap requires aligned finite losses, groups, and 100+ replicates")
    _, group_index = np.unique(group_ids, return_inverse=True)
    group_count = int(group_index.max()) + 1
    if group_count < 2:
        raise ValueError("Group bootstrap requires at least two independent groups")
    delta_sum = np.bincount(group_index, weights=candidate - reference, minlength=group_count)
    row_count = np.bincount(group_index, minlength=group_count).astype(np.float64)
    observed = float(delta_sum.sum() / row_count.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    chunk_size = max(1, min(256, 1_000_000 // group_count))
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        sample = rng.integers(group_count, size=(stop - start, group_count))
        draws[start:stop] = delta_sum[sample].sum(axis=1) / row_count[sample].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "candidate_minus_reference": observed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "probability_candidate_better": float(np.mean(draws < 0)),
        "groups": group_count,
        "replicates": replicates,
        "seed": seed,
    }


def per_example_coordinate_mae(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return aligned per-row coordinate errors for grouped uncertainty estimates."""
    predicted_values = np.asarray(predicted, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if (
        predicted_values.ndim != 2
        or predicted_values.shape != expected_values.shape
        or predicted_values.shape[0] == 0
        or predicted_values.shape[1] == 0
        or not np.isfinite(predicted_values).all()
        or not np.isfinite(expected_values).all()
    ):
        raise ValueError("Coordinate predictions and targets must be aligned finite matrices")
    return np.mean(np.abs(predicted_values - expected_values), axis=1)


def within_action_permutation(actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle locations while preserving each row's action type."""
    permutation = np.arange(len(actions))
    for action_type in range(9):
        indices = np.flatnonzero(actions[:, action_type] == 1)
        permutation[indices] = rng.permutation(indices)
    return permutation


def paired_counterfactual_permutation(partition: dict[str, np.ndarray]) -> np.ndarray:
    """Permute actions only among distinct observed outcomes from the exact same state."""
    groups = partition["groups"]
    if not len(groups):
        raise ValueError("Counterfactual data must contain at least one multi-action state")
    permutation = np.arange(len(groups))
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        states, targets, actions = (partition[key] for key in ("state", "target", "action"))
        distinct_targets = {np.ascontiguousarray(targets[index]).tobytes() for index in indices}
        distinct_actions = {np.ascontiguousarray(actions[index]).tobytes() for index in indices}
        if (
            len(indices) < 2
            or any(not np.array_equal(states[indices[0]], states[index]) for index in indices[1:])
            or len(distinct_targets) != len(indices)
            or len(distinct_actions) != len(indices)
        ):
            raise ValueError("Counterfactual group must contain distinct actions and outcomes from one state")
        permutation[indices] = np.roll(indices, 1)
    return permutation


def counterfactual_successor_retrieval(
    predicted: np.ndarray,
    partition: dict[str, np.ndarray],
    *,
    seed: int,
    replicates: int = 5000,
) -> dict[str, Any]:
    """Measure whether each action prediction retrieves its observed same-state outcome."""
    predictions = np.asarray(predicted, dtype=np.float64)
    states = np.asarray(partition["state"], dtype=np.float64)
    targets = np.asarray(partition["target"], dtype=np.float64)
    actions = np.asarray(partition["action"])
    groups = np.asarray(partition["groups"], dtype=str)
    if (
        predictions.shape != states.shape
        or targets.shape != states.shape
        or predictions.ndim != 2
        or actions.ndim != 2
        or actions.shape[0] != len(predictions)
        or groups.shape != (len(predictions),)
        or not np.isfinite(predictions).all()
        or not np.isfinite(states).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Successor retrieval requires aligned finite feature matrices")
    paired_counterfactual_permutation(partition)

    group_ids = np.unique(groups)
    group_accuracy = []
    group_chance = []
    for group_id in group_ids:
        indices = np.flatnonzero(groups == group_id)
        state = states[indices[0]][None, None, :]
        candidates = targets[indices][None, :, :]
        weights = 1 + 4 * (np.abs(candidates - state) >= 0.02)
        differences = predictions[indices][:, None, :] - candidates
        costs = (differences**2 * weights).mean(axis=2)
        correct = np.argmin(costs, axis=1) == np.arange(len(indices))
        group_accuracy.append(float(correct.mean()))
        group_chance.append(1.0 / len(indices))

    accuracy = np.asarray(group_accuracy, dtype=np.float64)
    chance = np.asarray(group_chance, dtype=np.float64)
    return {
        "mean_group_top1_accuracy": float(accuracy.mean()),
        "mean_group_chance_accuracy": float(chance.mean()),
        "groups": int(len(group_ids)),
        "rows": int(len(predictions)),
        "bootstrap_vs_group_chance": group_bootstrap_loss_delta(
            1 - chance,
            1 - accuracy,
            group_ids,
            seed=seed,
            replicates=replicates,
        ),
    }


def paired_group_indices(partition: dict[str, np.ndarray]) -> np.ndarray:
    """Return every verified same-state action pair for action-specific training."""
    paired_counterfactual_permutation(partition)  # Validate every branch before pairing.
    pairs = []
    for group in np.unique(partition["groups"]):
        indices = np.flatnonzero(partition["groups"] == group)
        pairs.extend(
            (indices[left], indices[right])
            for left in range(len(indices) - 1)
            for right in range(left + 1, len(indices))
        )
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def pairwise_transition_loss(predicted, target):
    """Penalize failure to distinguish observed outcomes from one prestate."""
    if predicted.shape != target.shape or len(predicted) < 2 or len(predicted) % 2:
        raise ValueError("Pairwise transition loss requires aligned action pairs")
    expected = target[0::2] - target[1::2]
    observed = predicted[0::2] - predicted[1::2]
    scale = (expected**2).mean(axis=1) + 1e-4
    return (((observed - expected) ** 2).mean(axis=1) / scale).mean()


def pairwise_focused_loss(predicted, target):
    """Score only observed action-discriminative pixels within aligned pairs."""
    if predicted.shape != target.shape or len(predicted) < 2 or len(predicted) % 2:
        raise ValueError("Focused pairwise loss requires aligned action pairs")
    expected = target[0::2] - target[1::2]
    observed = predicted[0::2] - predicted[1::2]
    mask = abs(expected) >= 0.02
    signal = (expected**2 * mask).sum(axis=1)
    error = ((observed - expected) ** 2 * mask).sum(axis=1)
    return (error / (signal + 1e-6)).mean()


def successor_ranking_loss(predicted, state, target, *, temperature: float):
    """Rank each action's observed same-state successor ahead of its alternatives."""
    import torch
    import torch.nn.functional as functional

    if (
        predicted.shape != state.shape
        or target.shape != state.shape
        or predicted.ndim != 2
        or len(predicted) < 2
        or len(predicted) % 2
        or not np.isfinite(temperature)
        or temperature <= 0
        or not torch.isfinite(predicted).all()
        or not torch.isfinite(state).all()
        or not torch.isfinite(target).all()
    ):
        raise ValueError("Successor ranking requires finite aligned action-pair tensors")
    state_pairs = state.reshape(-1, 2, state.shape[1])
    if not torch.equal(state_pairs[:, 0], state_pairs[:, 1]):
        raise ValueError("Successor ranking pairs must share the exact same pre-action state")

    prediction_pairs = predicted.reshape(-1, 2, predicted.shape[1])
    target_pairs = target.reshape(-1, 2, target.shape[1])
    baseline = state_pairs[:, :1]
    target_weights = 1 + 4 * ((target_pairs - baseline).abs() >= 0.02)
    pairwise_errors = (prediction_pairs[:, :, None] - target_pairs[:, None, :]) ** 2
    costs = (pairwise_errors * target_weights[:, None]).mean(dim=3)
    logits = -costs / temperature
    labels = torch.arange(2, device=predicted.device).expand(len(logits), -1)
    prediction_to_target = functional.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
    target_to_prediction = functional.cross_entropy(logits.transpose(1, 2).reshape(-1, 2), labels.reshape(-1))
    return (prediction_to_target + target_to_prediction) / 2


RECORDED_ACTION_IDS = {
    "mouse_move": 0,
    "mouse_click": 1,
    "mouse_double_click": 2,
    "mouse_right_click": 3,
}
RECORDED_ACTION_WIDTH = 9


def serialize_runtime_linear_artifact(
    state_dict: dict[str, Any],
    *,
    training_samples: int,
    validation_error: float,
) -> bytes:
    """Convert the fitted linear residual head to the live screen-feature contract."""
    from pilot.actions import ActionType
    from pilot.intelligence.world_model import UiJepaPredictor

    def as_array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    weight = as_array(state_dict.get("weight"))
    bias = as_array(state_dict.get("bias"))
    dimension = 16 * 16
    if (
        weight.shape != (dimension, dimension + RECORDED_ACTION_WIDTH + 2)
        or bias.shape != (dimension,)
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(bias))
        or isinstance(training_samples, bool)
        or not isinstance(training_samples, (int, np.integer))
        or training_samples < 1
        or not np.isfinite(validation_error)
        or validation_error < 0
    ):
        raise ValueError("Linear checkpoint does not satisfy the bounded runtime feature contract")

    action_types = tuple(ActionType)
    action_indices = {action.value: index for index, action in enumerate(action_types)}
    action_weight = np.zeros((len(action_types), dimension), dtype=np.float32)
    for name, recorded_index in RECORDED_ACTION_IDS.items():
        if name not in action_indices:
            raise ValueError(f"Recorded action {name!r} is absent from this Heliox build")
        action_weight[action_indices[name]] = weight[:, dimension + recorded_index]

    geometry_weight = weight[:, dimension + RECORDED_ACTION_WIDTH :].T.copy()
    artifact = io.BytesIO()
    np.savez_compressed(
        artifact,
        state_weight=weight[:, :dimension].T.copy(),
        action_weight=action_weight,
        action_geometry_weight=geometry_weight,
        action_geometry_version=np.array(UiJepaPredictor.ACTION_GEOMETRY_VERSION),
        action_types=np.asarray([action.value for action in action_types], dtype=np.str_),
        bias=bias,
        model_version=np.array("ui-jepa-runtime-linear-v1"),
        prediction_contract=np.array("clipped-residual-v1"),
        training_samples=np.array(training_samples, dtype=np.int64),
        validation_error=np.array(validation_error, dtype=np.float32),
        validated_for_gating=np.array(False),
    )
    return artifact.getvalue()


def merge_recordings_with_shared_states(
    groups: list[tuple[str, list[tuple[np.ndarray, np.ndarray, int, np.ndarray]]]],
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray, int, np.ndarray]]]]:
    """Keep entire recordings together when their quantized pre-screens match."""
    parent = list(range(len(groups)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[bytes, int] = {}
    for index, (_, rows) in enumerate(groups):
        for before, _, _, _ in rows:
            key = before.tobytes()
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index

    merged: dict[int, tuple[str, list[tuple[np.ndarray, np.ndarray, int, np.ndarray]]]] = {}
    for index, (workflow_id, rows) in enumerate(groups):
        leader = root(index)
        if leader not in merged:
            merged[leader] = (workflow_id, [])
        merged[leader][1].extend(rows)
    return list(merged.values())


def load_recorded_features(store: Any, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Split decrypted consented samples by recording, never by adjacent frame."""
    recordings = store.load_research_samples()
    groups = []
    for recording in recordings:
        rows = []
        for sample in recording["samples"]:
            action_id = RECORDED_ACTION_IDS.get(sample.get("action_type"))
            before, after, coords = sample.get("before"), sample.get("after"), sample.get("coords")
            if action_id is None or not sample.get("post_observed"):
                continue
            if not isinstance(before, list) or not isinstance(after, list) or not isinstance(coords, list):
                continue
            if len(before) != 256 or len(after) != 256 or len(coords) != 2:
                continue
            values = np.asarray(before + after + coords, dtype=np.float32)
            if (
                not np.all(np.isfinite(values))
                or np.any(values[:512] < 0)
                or np.any(values[:512] > 255)
                or np.any(values[:512] != np.floor(values[:512]))
            ):
                continue
            if np.any(values[512:] < 0) or np.any(values[512:] > 1):
                continue
            rows.append((values[:256] / 255, values[256:512] / 255, action_id, values[512:]))
        if rows:
            groups.append((recording["workflow_id"], rows))
    groups = merge_recordings_with_shared_states(groups)
    if len(groups) < 4:
        raise ValueError("At least four independent visual recording groups are needed for held-out evaluation")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_end = max(2, int(len(groups) * 0.6))
    validation_end = min(len(groups) - 1, train_end + max(1, int(len(groups) * 0.2)))
    partitions = {
        "train": groups[:train_end],
        "validation": groups[train_end:validation_end],
        "eval": groups[validation_end:],
    }
    result = {}
    for split, partition in partitions.items():
        rows = [(group_id, row) for group_id, group_rows in partition for row in group_rows]
        if len(rows) < 10:
            raise ValueError(f"At least ten {split} visual pairs are needed")
        actions = np.eye(RECORDED_ACTION_WIDTH, dtype=np.float32)[[row[2] for _, row in rows]]
        coords = np.asarray([row[3] for _, row in rows], dtype=np.float32)
        result[split] = {
            "state": np.asarray([row[0] for _, row in rows], dtype=np.float32),
            "target": np.asarray([row[1] for _, row in rows], dtype=np.float32),
            "action": np.concatenate((actions, coords), axis=1),
            "groups": np.asarray([group_id for group_id, _ in rows]),
        }
    return result


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=state / "tess_visual_transitions_v1.npz")
    parser.add_argument("--source-report", type=Path, default=state / "tess_visual_transitions_v1.json")
    parser.add_argument(
        "--additional-features",
        type=Path,
        help="Add a second pinned archive to fit/validation only; keep primary eval isolated",
    )
    parser.add_argument("--additional-source-report", type=Path)
    parser.add_argument("--recorded-store", type=Path, help="Encrypted opt-in local workflow store")
    parser.add_argument("--report", type=Path, default=state / "tess_visual_model_v1.json")
    parser.add_argument("--weights", type=Path, default=state / "tess_visual_model_v1.pt")
    parser.add_argument(
        "--runtime-weights",
        type=Path,
        help="Optional live UiJepaPredictor candidate (shadow-only; .enc for recorded data)",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument(
        "--architecture", choices=("linear", "mlp", "spatial", "spatial_global", "spatial_joint"), default="linear"
    )
    parser.add_argument(
        "--inverse-loss-weight",
        type=float,
        default=0.0,
        help="Train a shared visual encoder to recover observed action coordinates",
    )
    parser.add_argument(
        "--train-without-action",
        action="store_true",
        help="Fit a matched state-only baseline with action inputs zeroed",
    )
    parser.add_argument(
        "--transition-scale-power",
        type=float,
        default=0.0,
        help="Normalize per-example loss by unchanged-screen error to this power (0..1)",
    )
    parser.add_argument(
        "--pairwise-loss-weight",
        type=float,
        default=0.0,
        help="Fit distinct observed outcomes from matched counterfactual actions",
    )
    parser.add_argument(
        "--pairwise-focus-weight",
        type=float,
        default=0.0,
        help="Fit observed action-discriminative pixels in verified pairs",
    )
    parser.add_argument(
        "--successor-ranking-weight",
        type=float,
        default=0.0,
        help="Rank each observed same-state successor above alternatives",
    )
    parser.add_argument(
        "--successor-ranking-temperature",
        type=float,
        default=0.01,
        help="Temperature for the optional successor-ranking objective",
    )
    parser.add_argument(
        "--paired-counterfactual-batches",
        action="store_true",
        help="Keep verified same-state branches together for matched ablations",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--action-permutations", type=int, default=10)
    parser.add_argument("--group-bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if (
        (not args.recorded_store and (not args.features.is_file() or not args.source_report.is_file()))
        or (args.recorded_store and not args.recorded_store.is_file())
        or bool(args.additional_features) != bool(args.additional_source_report)
        or (
            args.additional_features
            and (
                args.recorded_store
                or not args.additional_features.is_file()
                or not args.additional_source_report.is_file()
            )
        )
        or not np.isfinite(args.transition_scale_power)
        or not 0 <= args.transition_scale_power <= 1
        or not np.isfinite(args.pairwise_loss_weight)
        or not 0 <= args.pairwise_loss_weight <= 10
        or not np.isfinite(args.pairwise_focus_weight)
        or not 0 <= args.pairwise_focus_weight <= 10
        or not np.isfinite(args.successor_ranking_weight)
        or not 0 <= args.successor_ranking_weight <= 10
        or not np.isfinite(args.successor_ranking_temperature)
        or args.successor_ranking_temperature <= 0
        or not np.isfinite(args.inverse_loss_weight)
        or not 0 <= args.inverse_loss_weight <= 10
        or (args.inverse_loss_weight and args.architecture != "spatial_joint")
        or args.report.exists()
        or args.weights.exists()
        or args.epochs < 1
        or (args.runtime_weights and args.runtime_weights.exists())
        or (
            args.runtime_weights
            and args.runtime_weights.resolve()
            in {
                args.report.resolve(),
                args.weights.resolve(),
            }
        )
        or args.action_permutations < 1
        or args.group_bootstrap_replicates < 100
    ):
        parser.error("Use existing pinned features, fresh outputs, and positive epochs")
    recorded_store = None
    if args.recorded_store:
        if args.weights.suffix != ".enc":
            parser.error("Recorded-screen model weights must be saved to an encrypted .enc path")
        if args.runtime_weights and args.runtime_weights.suffix != ".enc":
            parser.error("Runtime weights derived from recorded screens must use an encrypted .enc path")
        from pilot.workflows.recorded_workflows import RecordedWorkflowStore

        recorded_store = RecordedWorkflowStore(args.recorded_store)
        source_hash = hashlib.sha256(args.recorded_store.read_bytes()).hexdigest()
        rows = load_recorded_features(recorded_store, args.seed)
        counterfactual_pairs = False
        if hashlib.sha256(args.recorded_store.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Recorded workflow store changed during feature loading; retry training")
    else:
        if args.runtime_weights and args.runtime_weights.suffix != ".npz":
            parser.error("Runtime weights without recorded screens must use an .npz path")
        rows = load_features(args.features, args.source_report)
        source_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
        source_metadata = json.loads(args.source_report.read_text(encoding="utf-8"))
        counterfactual_pairs = source_metadata.get("counterfactual_pairs") is True
        additional_hash = None
        additional_report_hash = None
        if args.additional_features:
            if feature_shape(args.source_report) != feature_shape(args.additional_source_report):
                raise ValueError("Additional visual features must match the primary feature shape")
            additional = load_features(args.additional_features, args.additional_source_report)
            rows = combine_training_features(rows, additional)
            additional_hash = hashlib.sha256(args.additional_features.read_bytes()).hexdigest()
            additional_report_hash = hashlib.sha256(args.additional_source_report.read_bytes()).hexdigest()
            additional_metadata = json.loads(args.additional_source_report.read_text(encoding="utf-8"))
            counterfactual_pairs = counterfactual_pairs and additional_metadata.get("counterfactual_pairs") is True
    if (
        args.pairwise_loss_weight
        or args.pairwise_focus_weight
        or args.paired_counterfactual_batches
        or args.successor_ranking_weight
    ) and not counterfactual_pairs:
        parser.error("Counterfactual objectives require verified same-state action pairs")
    split_basis = "recording_and_exact_pre_screen" if recorded_store else source_metadata.get("split_basis", "episode")
    if args.additional_features:
        split_basis += "_with_additional_fit_validation_source"
    source_context = (
        "consented local recordings"
        if recorded_store
        else "generated local browser pages"
        if source_metadata.get("purpose") == "research_only_controlled_browser_visual_transitions"
        else "the local Heliox UI"
        if source_metadata.get("purpose") == "research_only_local_heliox_browser_navigation"
        else "one public dataset"
    )
    channels, height, width = (1, 16, 16) if recorded_store else feature_shape(args.source_report)
    dimension = channels * height * width
    if args.runtime_weights and (
        args.architecture != "linear" or args.train_without_action or channels != 1 or height != 16 or width != 16
    ):
        parser.error("Runtime predictor export requires the action-conditioned linear 16x16 architecture")

    import torch
    from torch import nn

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    arrays = {
        split: {key: torch.from_numpy(value).to(device) for key, value in partition.items() if key != "groups"}
        for split, partition in rows.items()
    }
    paired_training = bool(
        args.paired_counterfactual_batches
        or args.pairwise_loss_weight
        or args.pairwise_focus_weight
        or args.successor_ranking_weight
    )
    pair_indices = (
        {split: torch.from_numpy(paired_group_indices(rows[split])).to(device) for split in ("train", "validation")}
        if paired_training
        else {}
    )
    if args.architecture == "spatial_joint":
        model = build_spatial_joint_model(nn, channels).to(device)
    elif args.architecture in {"spatial", "spatial_global"}:
        model = nn.Sequential(
            nn.Conv2d(channels + (12 if args.architecture == "spatial_global" else 10), 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, channels, 3, padding=1),
        ).to(device)
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    elif args.architecture == "mlp":
        hidden = min(512, max(256, dimension))
        model = nn.Sequential(nn.Linear(dimension + 11, hidden), nn.GELU(), nn.Linear(hidden, dimension)).to(device)
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    else:
        model = nn.Linear(dimension + 11, dimension).to(device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)

    def training_loss(predicted, partition):
        state, target = partition["state"], partition["target"]
        change = (target - state).abs() >= 0.02
        weights = 1 + 4 * change
        error = ((predicted - target) ** 2 * weights).mean(dim=1)
        if args.transition_scale_power:
            baseline = ((state - target) ** 2 * weights).mean(dim=1)
            error = error / baseline.clamp_min(1e-4).pow(args.transition_scale_power)
        return error.mean()

    def objective(predicted, partition):
        loss = training_loss(predicted, partition)
        if args.pairwise_loss_weight:
            loss = loss + args.pairwise_loss_weight * pairwise_transition_loss(predicted, partition["target"])
        if args.pairwise_focus_weight:
            loss = loss + args.pairwise_focus_weight * pairwise_focused_loss(predicted, partition["target"])
        if args.successor_ranking_weight:
            loss = loss + args.successor_ranking_weight * successor_ranking_loss(
                predicted,
                partition["state"],
                partition["target"],
                temperature=args.successor_ranking_temperature,
            )
        return loss

    def predict(partition: dict, *, ablate_action: bool = False):
        action = (
            torch.zeros_like(partition["action"]) if ablate_action or args.train_without_action else partition["action"]
        )
        if args.architecture in {"spatial", "spatial_global", "spatial_joint"}:
            state_image = partition["state"].reshape(-1, channels, height, width)
            y, x = torch.meshgrid(
                torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
            )
            cx = action[:, 9, None, None] * (width - 1)
            cy = action[:, 10, None, None] * (height - 1)
            radius = 4.5 * (min(height, width) / 16) ** 2
            locus = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / radius).unsqueeze(1)
            kind = action[:, :9, None, None].expand(-1, -1, height, width)
            inputs = [state_image, locus, kind]
            if args.architecture in {"spatial_global", "spatial_joint"}:
                inputs.append(action[:, 9:11, None, None].expand(-1, -1, height, width))
            delta = (
                model(state_image, torch.cat(inputs[1:], dim=1))
                if args.architecture == "spatial_joint"
                else model(torch.cat(inputs, dim=1))
            ).reshape(-1, dimension)
            return torch.clamp(partition["state"] + delta, 0, 1)
        return torch.clamp(partition["state"] + model(torch.cat((partition["state"], action), dim=1)), 0, 1)

    best_loss, best_epoch, best_weights = float("inf"), 0, None
    train = arrays["train"]

    def inverse_loss(partition: dict) -> Any:
        state = partition["state"].reshape(-1, channels, height, width)
        target = partition["target"].reshape(-1, channels, height, width)
        return ((model.inverse(state, target) - partition["action"][:, 9:11]) ** 2).mean()

    for epoch in range(args.epochs):
        model.train()
        batches = (
            pair_indices["train"][torch.randperm(len(pair_indices["train"]), device=device)].split(32)
            if paired_training
            else torch.randperm(len(train["state"]), device=device).split(64)
        )
        for selected in batches:
            indices = selected.reshape(-1)
            batch = {key: value[indices] for key, value in train.items()}
            output = predict(batch)
            loss = objective(output, batch)
            if args.inverse_loss_weight:
                loss = loss + args.inverse_loss_weight * inverse_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = arrays["validation"]
            if paired_training:
                indices = pair_indices["validation"].reshape(-1)
                validation = {key: value[indices] for key, value in validation.items()}
            output = predict(validation)
            loss = objective(output, validation).item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch + 1
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    scores = {}
    inverse_scores = {}
    with torch.no_grad():
        for split in ("validation", "eval"):
            partition = rows[split]
            if args.architecture == "spatial_joint":
                observed = arrays[split]
                state = observed["state"].reshape(-1, channels, height, width)
                target = observed["target"].reshape(-1, channels, height, width)
                expected = observed["action"][:, 9:11]
                expected_coordinates = expected.cpu().numpy()
                predicted_coordinates = model.inverse(state, target).cpu().numpy()
                before_only_coordinates = model.inverse(state, state).cpu().numpy()
                center_coordinates = np.full_like(expected_coordinates, 0.5)
                observed_row_errors = per_example_coordinate_mae(predicted_coordinates, expected_coordinates)
                center_row_errors = per_example_coordinate_mae(center_coordinates, expected_coordinates)
                inverse_scores[split] = {
                    "observed_before_after_coordinate_mae": float(observed_row_errors.mean()),
                    "before_only_coordinate_mae": float(
                        per_example_coordinate_mae(before_only_coordinates, expected_coordinates).mean()
                    ),
                    "center_coordinate_mae": float(center_row_errors.mean()),
                }
                if split == "eval":
                    inverse_scores[split]["group_bootstrap_vs_center"] = group_bootstrap_loss_delta(
                        center_row_errors,
                        observed_row_errors,
                        partition["groups"],
                        seed=args.seed + 8,
                        replicates=args.group_bootstrap_replicates,
                    )
            shuffled_scores = []
            within_type_scores = []
            shuffled_row_losses = np.zeros(len(partition["state"]), dtype=np.float64)
            within_type_row_losses = np.zeros(len(partition["state"]), dtype=np.float64)
            rng = np.random.default_rng(args.seed + (1 if split == "validation" else 2))
            for _ in range(args.action_permutations):
                shuffled = dict(arrays[split])
                shuffled["action"] = arrays[split]["action"][rng.permutation(len(partition["state"]))]
                shuffled_prediction = predict(shuffled).cpu().numpy()
                shuffled_scores.append(metrics(shuffled_prediction, partition["state"], partition["target"]))
                shuffled_row_losses += (
                    weighted_pixel_mse_by_row(shuffled_prediction, partition["state"], partition["target"])
                    / args.action_permutations
                )
                within_type = dict(arrays[split])
                within_type["action"] = arrays[split]["action"][within_action_permutation(partition["action"], rng)]
                within_type_prediction = predict(within_type).cpu().numpy()
                within_type_scores.append(metrics(within_type_prediction, partition["state"], partition["target"]))
                within_type_row_losses += (
                    weighted_pixel_mse_by_row(within_type_prediction, partition["state"], partition["target"])
                    / args.action_permutations
                )
            trained_prediction = predict(arrays[split]).cpu().numpy()
            no_action_prediction = predict(arrays[split], ablate_action=True).cpu().numpy()
            scores[split] = {
                "unchanged_state": metrics(partition["state"], partition["state"], partition["target"]),
                "trained": metrics(trained_prediction, partition["state"], partition["target"]),
                "no_action": metrics(no_action_prediction, partition["state"], partition["target"]),
                "shuffled_action": {
                    key: float(np.mean([score[key] for score in shuffled_scores])) for key in shuffled_scores[0]
                },
                "shuffled_location_same_action": {
                    key: float(np.mean([score[key] for score in within_type_scores])) for key in within_type_scores[0]
                },
            }
            correct_error = scores[split]["trained"]["weighted_mse"]
            scores[split]["action_shuffle_diagnostic"] = {
                "whole_action": action_shuffle_diagnostic(
                    correct_error, [score["weighted_mse"] for score in shuffled_scores]
                ),
                "location_same_action": action_shuffle_diagnostic(
                    correct_error, [score["weighted_mse"] for score in within_type_scores]
                ),
            }
            if split == "eval":
                trained_row_losses = weighted_pixel_mse_by_row(
                    trained_prediction, partition["state"], partition["target"]
                )
                scores[split]["episode_group_bootstrap"] = {
                    "metric": "row-weighted candidate-minus-reference weighted MSE; negative is better",
                    "comparisons": {
                        "trained_minus_unchanged_state": group_bootstrap_loss_delta(
                            weighted_pixel_mse_by_row(partition["state"], partition["state"], partition["target"]),
                            trained_row_losses,
                            partition["groups"],
                            seed=args.seed + 3,
                            replicates=args.group_bootstrap_replicates,
                        ),
                        "trained_minus_no_action": group_bootstrap_loss_delta(
                            weighted_pixel_mse_by_row(no_action_prediction, partition["state"], partition["target"]),
                            trained_row_losses,
                            partition["groups"],
                            seed=args.seed + 4,
                            replicates=args.group_bootstrap_replicates,
                        ),
                        "trained_minus_shuffled_action_mean": group_bootstrap_loss_delta(
                            shuffled_row_losses,
                            trained_row_losses,
                            partition["groups"],
                            seed=args.seed + 5,
                            replicates=args.group_bootstrap_replicates,
                        ),
                        "trained_minus_shuffled_location_mean": group_bootstrap_loss_delta(
                            within_type_row_losses,
                            trained_row_losses,
                            partition["groups"],
                            seed=args.seed + 6,
                            replicates=args.group_bootstrap_replicates,
                        ),
                    },
                }
            if counterfactual_pairs:
                swapped = dict(arrays[split])
                swapped["action"] = arrays[split]["action"][paired_counterfactual_permutation(partition)]
                swapped_prediction = predict(swapped).cpu().numpy()
                scores[split]["paired_counterfactual_action"] = metrics(
                    swapped_prediction, partition["state"], partition["target"]
                )
                scores[split]["successor_retrieval"] = counterfactual_successor_retrieval(
                    trained_prediction,
                    partition,
                    seed=args.seed + (9 if split == "eval" else 8),
                    replicates=args.group_bootstrap_replicates,
                )
                if split == "eval":
                    scores[split]["episode_group_bootstrap"]["comparisons"][
                        "trained_minus_same_state_action_permutation"
                    ] = group_bootstrap_loss_delta(
                        weighted_pixel_mse_by_row(swapped_prediction, partition["state"], partition["target"]),
                        trained_row_losses,
                        partition["groups"],
                        seed=args.seed + 7,
                        replicates=args.group_bootstrap_replicates,
                    )
    runtime_candidate_sha256 = None
    if args.runtime_weights:
        runtime_payload = serialize_runtime_linear_artifact(
            best_weights,
            training_samples=len(rows["train"]["state"]),
            validation_error=scores["validation"]["trained"]["weighted_mse"],
        )
        if recorded_store:
            runtime_payload = recorded_store.encrypt_model_artifact(runtime_payload)
        runtime_candidate_sha256 = hashlib.sha256(runtime_payload).hexdigest()
        args.runtime_weights.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_weights.write_bytes(runtime_payload)

    report = {
        "purpose": "research_only_observed_visual_world_transition_probe",
        "features_sha256": source_hash,
        "additional_training_features_sha256": additional_hash if not recorded_store else None,
        "additional_source_report_sha256": additional_report_hash if not recorded_store else None,
        "source_kind": "encrypted_local_recordings" if recorded_store else "pinned_feature_archive",
        "device": str(device),
        "architecture": args.architecture,
        "feature_shape": [channels, height, width],
        "split_basis": split_basis,
        "counterfactual_pairs": counterfactual_pairs,
        "paired_counterfactual_batches": paired_training,
        "trained_without_action": args.train_without_action,
        "transition_scale_power": args.transition_scale_power,
        "pairwise_loss_weight": args.pairwise_loss_weight,
        "pairwise_focus_weight": args.pairwise_focus_weight,
        "successor_ranking_weight": args.successor_ranking_weight,
        "successor_ranking_temperature": args.successor_ranking_temperature,
        "inverse_loss_weight": args.inverse_loss_weight,
        "validation_inverse_mse": (
            float(inverse_loss(arrays["validation"]).item()) if args.architecture == "spatial_joint" else None
        ),
        "selected_validation_objective": best_loss,
        "torch_version": torch.__version__,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "action_permutations": args.action_permutations,
        "rows": {split: len(partition["state"]) for split, partition in rows.items()},
        "groups": {split: len(set(partition["groups"])) for split, partition in rows.items()},
        "scores": scores,
        "inverse_coordinate_diagnostic": inverse_scores if inverse_scores else None,
        "runtime_candidate": (
            {
                "path": str(args.runtime_weights),
                "sha256": runtime_candidate_sha256,
                "prediction_contract": "clipped-residual-v1",
                "mode": "shadow",
                "validated_for_gating": False,
            }
            if args.runtime_weights
            else None
        ),
        "limitation": f"{channels}x{height}x{width} visual features and mouse action type/coordinates only. "
        + (
            "The emitted live-compatible candidate is shadow-only and is not validated for gating. "
            if args.runtime_weights
            else "Research model is not the live UiJepaPredictor. "
        )
        + (
            "Recording and exact pre-screen-disjoint, not environment-disjoint; "
            "matching coarse screens do not establish identical hidden state; "
            "action outcomes are observed, not independently verified. "
            if recorded_store
            else f"{split_basis}-disjoint within {source_context}, not environment-disjoint. "
        )
        + "No safety, autonomy, Jev parity, or gating claim.",
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    if recorded_store:
        buffer = io.BytesIO()
        torch.save(best_weights, buffer)
        args.weights.write_bytes(recorded_store.encrypt_model_artifact(buffer.getvalue()))
    else:
        torch.save(best_weights, args.weights)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

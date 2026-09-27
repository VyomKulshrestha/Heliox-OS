"""Score a frozen research visual model on a different pinned feature archive.

This never loads weights into the live world model and never captures a screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train_world_visual_transition import (
    build_spatial_joint_model,
    feature_shape,
    load_features,
    metrics,
    paired_counterfactual_permutation,
)


def task_masks(features_path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    """Align optional task labels with the held-out visual rows."""
    with np.load(features_path, allow_pickle=False) as data:
        if "task" not in data.files:
            return {}
        selected = data["split"].astype(str) == "eval"
        tasks = data["task"].astype(str)
    if len(tasks) != len(selected) or int(selected.sum()) != expected_rows:
        raise ValueError("Task labels do not align with evaluation rows")
    names = tasks[selected]
    return {name: names == name for name in sorted(set(names))}


def observed_pair_signal(target: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    """Report observed screen separation without reference to model predictions."""
    separation = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) != 2:
            raise ValueError("Observed pair signal requires exactly two rows per group")
        separation.append(float(((target[indices[0]] - target[indices[1]]) ** 2).mean()))
    return {"groups": len(separation), "mean_pixel_mse": float(np.mean(separation))}


def paired_action_diagnostic(
    correct: np.ndarray,
    swapped: np.ndarray,
    state: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int]:
    """Report whether the correct action wins within each observed pair."""
    weights = 1 + 4 * (np.abs(target - state) >= 0.02)
    correct_error = ((correct - target) ** 2 * weights).mean(axis=1)
    swapped_error = ((swapped - target) ** 2 * weights).mean(axis=1)
    advantages = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) != 2:
            raise ValueError("Paired action diagnostic requires exactly two rows per group")
        advantages.append(float((swapped_error[indices] - correct_error[indices]).mean()))
    return {
        "groups": len(advantages),
        "correct_action_wins": sum(value > 0 for value in advantages),
        "ties": sum(value == 0 for value in advantages),
        "mean_weighted_mse_advantage": float(np.mean(advantages)),
    }


def paired_discriminative_diagnostic(
    correct: np.ndarray, swapped: np.ndarray, target: np.ndarray, groups: np.ndarray
) -> dict[str, float | int | None]:
    """Compare actions on pixels where their observed outcomes differ."""
    advantages = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) != 2:
            raise ValueError("Discriminative diagnostic requires exactly two rows per group")
        mask = np.abs(target[indices[0]] - target[indices[1]]) >= 0.02
        if not np.any(mask):
            continue
        correct_error = ((correct[indices][:, mask] - target[indices][:, mask]) ** 2).mean()
        swapped_error = ((swapped[indices][:, mask] - target[indices][:, mask]) ** 2).mean()
        advantages.append(float(swapped_error - correct_error))
    return {
        "groups_with_distinct_pixels": len(advantages),
        "correct_action_wins": sum(value > 0 for value in advantages),
        "ties": sum(value == 0 for value in advantages),
        "mean_mse_advantage": float(np.mean(advantages)) if advantages else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh output report path")
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    source_shape = tuple(model_report["feature_shape"])
    target_shape = feature_shape(args.target_report)
    if source_shape != target_shape:
        raise ValueError("Source model and target features have different shapes")
    if model_report.get("source_kind") != "pinned_feature_archive":
        raise ValueError("Only frozen non-private research artifacts are supported")
    rows = load_features(args.target_features, args.target_report)
    partition = rows["eval"]
    channels, height, width = target_shape
    dimension = channels * height * width

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this transfer probe")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    architecture = model_report["architecture"]
    if architecture == "spatial_joint":
        model = build_spatial_joint_model(nn, channels)
    elif architecture in {"spatial", "spatial_global"}:
        model = nn.Sequential(
            nn.Conv2d(channels + (12 if architecture == "spatial_global" else 10),
                      32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, channels, 3, padding=1),
        )
    elif architecture == "mlp":
        hidden = min(512, max(256, dimension))
        model = nn.Sequential(nn.Linear(dimension + 11, hidden), nn.GELU(),
                              nn.Linear(hidden, dimension))
    elif architecture == "linear":
        model = nn.Linear(dimension + 11, dimension)
    else:
        raise ValueError("Unsupported visual research architecture")
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    state = torch.from_numpy(partition["state"]).to(device)
    action = torch.from_numpy(partition["action"]).to(device)

    def predict(actions):
        if model_report["trained_without_action"]:
            actions = torch.zeros_like(actions)
        if architecture in {"spatial", "spatial_global", "spatial_joint"}:
            y, x = torch.meshgrid(torch.arange(height, device=device),
                                  torch.arange(width, device=device), indexing="ij")
            cx = actions[:, 9, None, None] * (width - 1)
            cy = actions[:, 10, None, None] * (height - 1)
            radius = 4.5 * (min(height, width) / 16) ** 2
            locus = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / radius).unsqueeze(1)
            kind = actions[:, :9, None, None].expand(-1, -1, height, width)
            inputs = [state.reshape(-1, channels, height, width), locus, kind]
            if architecture in {"spatial_global", "spatial_joint"}:
                inputs.append(actions[:, 9:11, None, None].expand(-1, -1, height, width))
            delta = (model(inputs[0], torch.cat(inputs[1:], dim=1))
                     if architecture == "spatial_joint" else
                     model(torch.cat(inputs, dim=1))).reshape(-1, dimension)
        else:
            delta = model(torch.cat((state, actions), dim=1))
        return torch.clamp(state + delta, 0, 1).cpu().numpy()

    with torch.no_grad():
        correct = predict(action)
        inverse_diagnostic = None
        if architecture == "spatial_joint":
            target = torch.from_numpy(partition["target"]).to(device)
            before = state.reshape(-1, channels, height, width)
            after = target.reshape(-1, channels, height, width)
            expected = action[:, 9:11]
            inverse_diagnostic = {
                "observed_before_after_coordinate_mae": float(
                    (model.inverse(before, after) - expected).abs().mean().item()),
                "before_only_coordinate_mae": float(
                    (model.inverse(before, before) - expected).abs().mean().item()),
                "center_coordinate_mae": float((0.5 - expected).abs().mean().item()),
            }
        swapped = None
        if json.loads(args.target_report.read_text(encoding="utf-8")).get("counterfactual_pairs") is True:
            indices = paired_counterfactual_permutation(partition)
            swapped = predict(action[indices])
        ablated = predict(torch.zeros_like(action))
    by_task = {}
    if swapped is not None:
        for name, mask in task_masks(args.target_features, len(partition["state"])).items():
            target = partition["target"][mask]
            groups = partition["groups"][mask]
            by_task[name] = {
                "observed_pair_signal": observed_pair_signal(target, groups),
                "paired_action_diagnostic": paired_action_diagnostic(
                    correct[mask], swapped[mask], partition["state"][mask], target, groups),
                "paired_discriminative_diagnostic": paired_discriminative_diagnostic(
                    correct[mask], swapped[mask], target, groups),
            }
    report = {
        "purpose": "research_only_frozen_visual_model_transfer_probe",
        "model_report_sha256": hashlib.sha256(args.model_report.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "target_features_sha256": hashlib.sha256(args.target_features.read_bytes()).hexdigest(),
        "target_rows": len(partition["state"]),
        "target_groups": len(set(partition["groups"])),
        "architecture": architecture,
        "device": str(device),
        "scores": {
            "unchanged_state": metrics(partition["state"], partition["state"], partition["target"]),
            "correct_action": metrics(correct, partition["state"], partition["target"]),
            "no_action": metrics(ablated, partition["state"], partition["target"]),
            "paired_alternative_action": metrics(swapped, partition["state"], partition["target"])
            if swapped is not None else None,
        },
        "paired_action_diagnostic": (
            paired_action_diagnostic(correct, swapped, partition["state"],
                                     partition["target"], partition["groups"])
            if swapped is not None else None
        ),
        "paired_discriminative_diagnostic": (
            paired_discriminative_diagnostic(correct, swapped, partition["target"],
                                             partition["groups"])
            if swapped is not None else None
        ),
        "observed_pair_signal": (
            observed_pair_signal(partition["target"], partition["groups"])
            if swapped is not None else None
        ),
        "paired_diagnostics_by_task": by_task,
        "inverse_coordinate_diagnostic": inverse_diagnostic,
        "limitation": "Frozen one-step model and target archive only; no runtime promotion, safety, "
                      "general desktop transfer, or Jev parity claim. Tiny target sets are descriptive only.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

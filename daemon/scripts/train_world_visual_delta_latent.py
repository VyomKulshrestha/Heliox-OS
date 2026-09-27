"""Fit an offline action-conditioned visual latent transition with delta inverse loss.

This bounded research probe uses hash-pinned screenshot features. Its weights
are not compatible with the live UI predictor and never grant action authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.train_world_visual_transition import (
    combine_training_features,
    feature_shape,
    load_features,
    paired_counterfactual_permutation,
    paired_group_indices,
)


def build_delta_latent_model(nn: Any, torch: Any, shape: tuple[int, int, int], latent_dim: int) -> Any:
    """Predict latent effects; decode the action from observed latent displacement."""
    channels, height, width = shape

    class DeltaLatent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                nn.Linear(32 * 4 * 4, latent_dim),
            )
            self.forward_head = nn.Sequential(
                nn.Linear(latent_dim + 11, 128), nn.GELU(),
                nn.Linear(128, latent_dim),
            )
            self.inverse_head = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.GELU(),
                nn.Linear(64, 2), nn.Sigmoid(),
            )
            self.reconstruction_head = nn.Sequential(
                nn.Linear(latent_dim, channels * height * width), nn.Sigmoid(),
            )
            nn.init.zeros_(self.forward_head[-1].weight)
            nn.init.zeros_(self.forward_head[-1].bias)

        def encode(self, image: Any) -> Any:
            return self.encoder(image)

        def forward(self, before_latent: Any, action: Any) -> Any:
            return before_latent + self.forward_head(torch.cat((before_latent, action), dim=1))

        def inverse(self, before_latent: Any, after_latent: Any) -> Any:
            return self.inverse_head(after_latent - before_latent)

    return DeltaLatent()


def pair_ranking(
    correct: np.ndarray, wrong: np.ndarray, target: np.ndarray, groups: np.ndarray,
) -> dict[str, float | int]:
    """Score candidate actions against observed paired outcomes, not a pixel proxy."""
    if correct.shape != wrong.shape or correct.shape != target.shape:
        raise ValueError("Aligned latent predictions and targets are required")
    correct_error = ((correct - target) ** 2).mean(axis=1)
    wrong_error = ((wrong - target) ** 2).mean(axis=1)
    advantages = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) != 2:
            raise ValueError("Latent ranking requires exactly two rows per group")
        advantages.append(float((wrong_error[indices] - correct_error[indices]).mean()))
    return {
        "groups": len(advantages),
        "correct_action_wins": sum(value > 0 for value in advantages),
        "ties": sum(value == 0 for value in advantages),
        "mean_mse_advantage": float(np.mean(advantages)),
        "correct_mse": float(correct_error.mean()),
        "wrong_mse": float(wrong_error.mean()),
    }


def score_model(model: Any, rows: dict[str, np.ndarray], shape: tuple[int, int, int], torch: Any,
                device: Any) -> dict[str, Any]:
    """Report latent separation and inverse localization on verified action pairs."""
    channels, height, width = shape
    permutation = paired_counterfactual_permutation(rows)
    state = torch.from_numpy(rows["state"]).to(device).reshape(-1, channels, height, width)
    target = torch.from_numpy(rows["target"]).to(device).reshape(-1, channels, height, width)
    action = torch.from_numpy(rows["action"]).to(device)
    with torch.no_grad():
        before = model.encode(state)
        after = model.encode(target)
        correct = model(before, action)
        wrong = model(before, action[permutation])
        inverse = model.inverse(before, after)
        pair_indices = paired_group_indices(rows)
        target_distance = ((after[pair_indices[:, 0]] - after[pair_indices[:, 1]]) ** 2).mean(dim=1)
        prediction_distance = ((correct[pair_indices[:, 0]] - correct[pair_indices[:, 1]]) ** 2).mean(dim=1)
        result = pair_ranking(correct.cpu().numpy(), wrong.cpu().numpy(), after.cpu().numpy(), rows["groups"])
        result.update({
            "unchanged_state_mse": float(((before - after) ** 2).mean().item()),
            "target_latent_std": float(after.std(dim=0).mean().item()),
            "paired_target_latent_mse": float(target_distance.mean().item()),
            "paired_predicted_latent_mse": float(prediction_distance.mean().item()),
            "inverse_coordinate_mae": float((inverse - action[:, 9:11]).abs().mean().item()),
            "center_coordinate_mae": float((0.5 - action[:, 9:11]).abs().mean().item()),
        })
    return result


def main() -> None:
    state_dir = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--additional-features", type=Path)
    parser.add_argument("--additional-source-report", type=Path)
    parser.add_argument("--report", type=Path, default=state_dir / "visual_delta_latent_v1.json")
    parser.add_argument("--weights", type=Path, default=state_dir / "visual_delta_latent_v1.pt")
    parser.add_argument("--inverse-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-floor", type=float, default=1e-4,
                        help="Numerical floor for normalized paired latent change")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if (not args.features.is_file() or not args.source_report.is_file()
            or bool(args.additional_features) != bool(args.additional_source_report)
            or (args.additional_features and (not args.additional_features.is_file()
                 or not args.additional_source_report.is_file()))
            or args.report.exists() or args.weights.exists()
            or args.epochs < 1 or not 16 <= args.latent_dim <= 512
            or not np.isfinite(args.inverse_weight) or not 0 <= args.inverse_weight <= 10
            or not np.isfinite(args.reconstruction_weight) or not 0 <= args.reconstruction_weight <= 10
            or not np.isfinite(args.pairwise_weight) or not 0 <= args.pairwise_weight <= 10
            or not np.isfinite(args.pairwise_floor) or not 1e-12 <= args.pairwise_floor <= 1e-2):
        parser.error("Use verified sources, fresh outputs, and bounded training settings")
    shape = feature_shape(args.source_report)
    rows = load_features(args.features, args.source_report)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source.get("counterfactual_pairs") is not True:
        parser.error("Verified paired counterfactual actions are required")
    additional_hash = None
    if args.additional_features:
        if shape != feature_shape(args.additional_source_report):
            raise ValueError("Additional visual features must match the primary feature shape")
        additional_source = json.loads(args.additional_source_report.read_text(encoding="utf-8"))
        if additional_source.get("counterfactual_pairs") is not True:
            parser.error("Additional source must contain verified action pairs")
        rows = combine_training_features(
            rows, load_features(args.additional_features, args.additional_source_report))
        additional_hash = hashlib.sha256(args.additional_features.read_bytes()).hexdigest()

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = build_delta_latent_model(nn, torch, shape, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    train = {key: torch.from_numpy(value).to(device) for key, value in rows["train"].items() if key != "groups"}
    validation = {key: torch.from_numpy(value).to(device)
                  for key, value in rows["validation"].items() if key != "groups"}
    train_pairs = torch.from_numpy(paired_group_indices(rows["train"])).to(device)
    validation_pairs = torch.from_numpy(paired_group_indices(rows["validation"])).to(device)
    channels, height, width = shape

    def objective(batch: dict[str, Any]) -> tuple[Any, Any, Any]:
        before = model.encode(batch["state"].reshape(-1, channels, height, width))
        after = model.encode(batch["target"].reshape(-1, channels, height, width))
        predicted = model(before, batch["action"])
        forward_loss = ((predicted - after) ** 2).mean()
        expected_change = after[0::2] - after[1::2]
        predicted_change = predicted[0::2] - predicted[1::2]
        pair_loss = ((predicted_change - expected_change) ** 2).mean() / (
            (expected_change.detach() ** 2).mean() + args.pairwise_floor)
        inverse_loss = ((model.inverse(before, after) - batch["action"][:, 9:11]) ** 2).mean()
        reconstruction_loss = (
            ((model.reconstruction_head(before) - batch["state"]) ** 2).mean()
            + ((model.reconstruction_head(after) - batch["target"]) ** 2).mean()
        ) / 2
        return (forward_loss + args.pairwise_weight * pair_loss
                + args.reconstruction_weight * reconstruction_loss), inverse_loss, forward_loss

    best_score, best_epoch, best_weights = float("inf"), 0, None
    for epoch in range(args.epochs):
        model.train()
        for selected in train_pairs[torch.randperm(len(train_pairs), device=device)].split(16):
            indices = selected.reshape(-1)
            batch = {key: value[indices] for key, value in train.items()}
            forward_objective, inverse_loss, _ = objective(batch)
            loss = forward_objective + args.inverse_weight * inverse_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            indices = validation_pairs.reshape(-1)
            batch = {key: value[indices] for key, value in validation.items()}
            forward_objective, _, _ = objective(batch)
            score = float(forward_objective.item())
        if score < best_score:
            best_score, best_epoch = score, epoch + 1
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    report = {
        "purpose": "research_only_delta_inverse_visual_latent_transition",
        "source_kind": "pinned_feature_archive",
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
        "additional_features_sha256": additional_hash,
        "additional_source_report_sha256": (
            hashlib.sha256(args.additional_source_report.read_bytes()).hexdigest()
            if args.additional_source_report else None),
        "feature_shape": list(shape),
        "latent_dim": args.latent_dim,
        "inverse_weight": args.inverse_weight,
        "reconstruction_weight": args.reconstruction_weight,
        "pairwise_weight": args.pairwise_weight,
        "pairwise_floor": args.pairwise_floor,
        "selection": "lowest forward-plus-paired-plus-optional-reconstruction validation loss; inverse excluded",
        "selected_validation_loss": best_score,
        "selected_epoch": best_epoch,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "rows": {split: len(partition["state"]) for split, partition in rows.items()},
        "scores": {split: score_model(model, rows[split], shape, torch, device)
                   for split in ("validation", "eval")},
        "limitation": "Provided GUI click pairs and learned latents only; no real desktop, multistep task, "
                      "Jev parity, runtime artifact, or execution authority claim.",
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, args.weights)
    report["weights_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Research-only inverse-dynamics probe for visual world-model features.

Predict mouse action type and location from observed before/after features.
This diagnoses whether the representation preserves action information; it
does not grant authority, load at runtime, or prove forward world modeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_world_visual_transition import load_features


def score_inverse(
    logits: np.ndarray, coords: np.ndarray, labels: np.ndarray, targets: np.ndarray,
    majority: int,
) -> dict[str, float]:
    predicted = np.argmax(logits, axis=1)
    present = np.unique(labels)
    click = labels != 0
    return {
        "action_accuracy": float(np.mean(predicted == labels)),
        "action_macro_recall": float(np.mean([np.mean(predicted[labels == kind] == kind)
                                             for kind in present])),
        "majority_action_accuracy": float(np.mean(labels == majority)),
        "click_coordinate_mae": float(np.abs(coords[click] - targets[click]).mean())
        if np.any(click) else 0.0,
        "center_coordinate_mae": float(np.abs(0.5 - targets[click]).mean())
        if np.any(click) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.report.exists() or args.epochs < 1 or not args.report.parent.exists():
        parser.error("Use a fresh report path in an existing directory and positive epochs")
    rows = load_features(args.features, args.source_report)

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    arrays = {}
    for split, partition in rows.items():
        features = np.concatenate((partition["state"],
                                   partition["target"] - partition["state"]), axis=1)
        arrays[split] = {
            "features": torch.from_numpy(features).to(device),
            "type": torch.from_numpy(np.argmax(partition["action"][:, :9], axis=1)).long().to(device),
            "coords": torch.from_numpy(partition["action"][:, 9:]).to(device),
        }
    counts = torch.bincount(arrays["train"]["type"], minlength=9).float()
    weights = torch.where(counts > 0, counts.clamp_min(1).rsqrt(), 0).to(device)
    majority = int(torch.argmax(counts).item())
    model = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 11)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)

    def loss_for(batch: dict) -> torch.Tensor:
        output = model(batch["features"])
        click = batch["type"] != 0
        classification = nn.functional.cross_entropy(output[:, :9], batch["type"], weight=weights)
        location = (nn.functional.smooth_l1_loss(
            output[click, 9:].sigmoid(), batch["coords"][click]
        ) if bool(click.any()) else output.sum() * 0)
        return classification + location

    best_loss, best_epoch, best_weights = float("inf"), 0, None
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(arrays["train"]["features"]), device=device).split(64):
            batch = {key: value[indices] for key, value in arrays["train"].items()}
            loss = loss_for(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_for(arrays["validation"]).item())
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    scores = {}
    with torch.no_grad():
        for split in ("validation", "eval"):
            output = model(arrays[split]["features"])
            scores[split] = score_inverse(
                output[:, :9].cpu().numpy(), output[:, 9:].sigmoid().cpu().numpy(),
                arrays[split]["type"].cpu().numpy(), arrays[split]["coords"].cpu().numpy(),
                majority,
            )
    report = {
        "purpose": "research_only_visual_inverse_dynamics_diagnostic",
        "source_features_sha256": json.loads(args.source_report.read_text(encoding="utf-8"))[
            "features_sha256"
        ],
        "device": str(device), "seed": args.seed, "selected_epoch": best_epoch,
        "training_action_counts": {str(index): int(count) for index, count in enumerate(counts.tolist())
                                   if count},
        "scores": scores,
        "limitation": "Small inverse classifier on 16x16 grayscale features. Not a joint "
                      "forward/inverse model, a real desktop result, or a production artifact.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

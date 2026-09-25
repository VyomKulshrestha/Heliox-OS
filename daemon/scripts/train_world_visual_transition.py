"""Research-only GPU probe on observed screenshot/action/next-screenshot features.

Episode-disjoint evaluation compares action-conditioned predictions with both
an unchanged-screen baseline and an action-ablated version of the same model.
Weights are not compatible with or loaded by the live UiJepaPredictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_features(path: Path, source_report: Path) -> dict[str, dict[str, np.ndarray]]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
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
    if (states.shape != (count, 256) or targets.shape != (count, 256)
            or actions.shape != (count,) or coords.shape != (count, 2)
            or groups.shape != (count,) or splits.shape != (count,)
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(targets))
            or not np.all(np.isfinite(coords)) or np.any(states < 0) or np.any(states > 1)
            or np.any(targets < 0) or np.any(targets > 1)
            or np.any(coords < 0) or np.any(coords > 1)
            or np.any(actions < 0) or np.any(actions > 8)):
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
            "state": states[indices], "target": targets[indices],
            "action": np.concatenate((onehot, coords[indices]), axis=1),
            "groups": groups[indices],
        }
    if any(group_sets[left] & group_sets[right] for left, right in
           (("train", "validation"), ("train", "eval"), ("validation", "eval"))):
        raise ValueError("Episode leaked across partitions")
    if set(splits) != set(result):
        raise ValueError("Unexpected split labels")
    return result


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


def main() -> None:
    state = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=state / "tess_visual_transitions_v1.npz")
    parser.add_argument("--source-report", type=Path, default=state / "tess_visual_transitions_v1.json")
    parser.add_argument("--report", type=Path, default=state / "tess_visual_model_v1.json")
    parser.add_argument("--weights", type=Path, default=state / "tess_visual_model_v1.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--architecture", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--action-permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if (not args.features.is_file() or not args.source_report.is_file()
            or args.report.exists() or args.weights.exists() or args.epochs < 1
            or args.action_permutations < 1):
        parser.error("Use existing pinned features, fresh outputs, and positive epochs")
    rows = load_features(args.features, args.source_report)

    import torch
    from torch import nn

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    arrays = {split: {key: torch.from_numpy(value).to(device) for key, value in partition.items()
                      if key != "groups"} for split, partition in rows.items()}
    if args.architecture == "mlp":
        model = nn.Sequential(nn.Linear(267, 256), nn.GELU(), nn.Linear(256, 256)).to(device)
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    else:
        model = nn.Linear(267, 256).to(device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)

    def predict(partition: dict, *, ablate_action: bool = False):
        action = torch.zeros_like(partition["action"]) if ablate_action else partition["action"]
        return torch.clamp(
            partition["state"] + model(torch.cat((partition["state"], action), dim=1)), 0, 1
        )

    best_loss, best_epoch, best_weights = float("inf"), 0, None
    train = arrays["train"]
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(train["state"]), device=device).split(64):
            batch = {key: value[indices] for key, value in train.items()}
            output = predict(batch)
            change = (batch["target"] - batch["state"]).abs() >= 0.02
            loss = ((output - batch["target"]) ** 2 * (1 + 4 * change)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = arrays["validation"]
            output = predict(validation)
            change = (validation["target"] - validation["state"]).abs() >= 0.02
            loss = ((output - validation["target"]) ** 2 * (1 + 4 * change)).mean().item()
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch + 1
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    scores = {}
    with torch.no_grad():
        for split in ("validation", "eval"):
            partition = rows[split]
            shuffled_scores = []
            rng = np.random.default_rng(args.seed + (1 if split == "validation" else 2))
            for _ in range(args.action_permutations):
                shuffled = dict(arrays[split])
                shuffled["action"] = arrays[split]["action"][rng.permutation(len(partition["state"]))]
                shuffled_scores.append(metrics(
                    predict(shuffled).cpu().numpy(), partition["state"], partition["target"]
                ))
            scores[split] = {
                "unchanged_state": metrics(partition["state"], partition["state"], partition["target"]),
                "trained": metrics(predict(arrays[split]).cpu().numpy(),
                                   partition["state"], partition["target"]),
                "no_action": metrics(predict(arrays[split], ablate_action=True).cpu().numpy(),
                                     partition["state"], partition["target"]),
                "shuffled_action": {
                    key: float(np.mean([score[key] for score in shuffled_scores]))
                    for key in shuffled_scores[0]
                },
            }
    report = {
        "purpose": "research_only_observed_visual_world_transition_probe",
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "device": str(device), "architecture": args.architecture,
        "seed": args.seed, "selected_epoch": best_epoch,
        "action_permutations": args.action_permutations,
        "rows": {split: len(partition["state"]) for split, partition in rows.items()},
        "groups": {split: len(set(partition["groups"])) for split, partition in rows.items()},
        "scores": scores,
        "limitation": "16x16 grayscale images and mouse action type/coordinates only. "
                      "Linear research model is not the live UiJepaPredictor. "
                      "Episode-disjoint within one dataset, not environment-disjoint. "
                      "No safety, autonomy, Jev parity, or runtime loading claim.",
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, args.weights)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

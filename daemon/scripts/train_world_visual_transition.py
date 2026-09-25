"""Research-only GPU probe on observed screenshot/action/next-screenshot features.

Episode-disjoint evaluation compares action-conditioned predictions with both
an unchanged-screen baseline and an action-ablated version of the same model.
Weights are not compatible with or loaded by the live UiJepaPredictor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

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


def within_action_permutation(actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle locations while preserving each row's action type."""
    permutation = np.arange(len(actions))
    for action_type in range(9):
        indices = np.flatnonzero(actions[:, action_type] == 1)
        permutation[indices] = rng.permutation(indices)
    return permutation


RECORDED_ACTION_IDS = {
    "mouse_move": 0,
    "mouse_click": 1,
    "mouse_double_click": 2,
    "mouse_right_click": 3,
}


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
            if (not np.all(np.isfinite(values)) or np.any(values[:512] < 0)
                    or np.any(values[:512] > 255) or np.any(values[:512] != np.floor(values[:512]))):
                continue
            if np.any(values[512:] < 0) or np.any(values[512:] > 1):
                continue
            rows.append((values[:256] / 255, values[256:512] / 255, action_id, values[512:]))
        if rows:
            groups.append((recording["workflow_id"], rows))
    if len(groups) < 4:
        raise ValueError("At least four separate visual recordings are needed for held-out evaluation")
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
        actions = np.eye(9, dtype=np.float32)[[row[2] for _, row in rows]]
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
    parser.add_argument("--recorded-store", type=Path, help="Encrypted opt-in local workflow store")
    parser.add_argument("--report", type=Path, default=state / "tess_visual_model_v1.json")
    parser.add_argument("--weights", type=Path, default=state / "tess_visual_model_v1.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--architecture", choices=("linear", "mlp", "spatial"), default="linear")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--action-permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if ((not args.recorded_store and (not args.features.is_file() or not args.source_report.is_file()))
            or (args.recorded_store and not args.recorded_store.is_file())
            or args.report.exists() or args.weights.exists() or args.epochs < 1
            or args.action_permutations < 1):
        parser.error("Use existing pinned features, fresh outputs, and positive epochs")
    recorded_store = None
    if args.recorded_store:
        if args.weights.suffix != ".enc":
            parser.error("Recorded-screen model weights must be saved to an encrypted .enc path")
        from pilot.workflows.recorded_workflows import RecordedWorkflowStore

        recorded_store = RecordedWorkflowStore(args.recorded_store)
        source_hash = hashlib.sha256(args.recorded_store.read_bytes()).hexdigest()
        rows = load_recorded_features(recorded_store, args.seed)
        if hashlib.sha256(args.recorded_store.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Recorded workflow store changed during feature loading; retry training")
    else:
        rows = load_features(args.features, args.source_report)
        source_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()

    import torch
    from torch import nn

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    arrays = {split: {key: torch.from_numpy(value).to(device) for key, value in partition.items()
                      if key != "groups"} for split, partition in rows.items()}
    if args.architecture == "spatial":
        model = nn.Sequential(
            nn.Conv2d(11, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        ).to(device)
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    elif args.architecture == "mlp":
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
        if args.architecture == "spatial":
            state_image = partition["state"].reshape(-1, 1, 16, 16)
            y, x = torch.meshgrid(torch.arange(16, device=device),
                                  torch.arange(16, device=device), indexing="ij")
            cx = action[:, 9, None, None] * 15
            cy = action[:, 10, None, None] * 15
            locus = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 4.5).unsqueeze(1)
            kind = action[:, :9, None, None].expand(-1, -1, 16, 16)
            delta = model(torch.cat((state_image, locus, kind), dim=1)).reshape(-1, 256)
            return torch.clamp(partition["state"] + delta, 0, 1)
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
            within_type_scores = []
            rng = np.random.default_rng(args.seed + (1 if split == "validation" else 2))
            for _ in range(args.action_permutations):
                shuffled = dict(arrays[split])
                shuffled["action"] = arrays[split]["action"][rng.permutation(len(partition["state"]))]
                shuffled_scores.append(metrics(
                    predict(shuffled).cpu().numpy(), partition["state"], partition["target"]
                ))
                within_type = dict(arrays[split])
                within_type["action"] = arrays[split]["action"][
                    within_action_permutation(partition["action"], rng)
                ]
                within_type_scores.append(metrics(
                    predict(within_type).cpu().numpy(), partition["state"], partition["target"]
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
                "shuffled_location_same_action": {
                    key: float(np.mean([score[key] for score in within_type_scores]))
                    for key in within_type_scores[0]
                },
            }
    report = {
        "purpose": "research_only_observed_visual_world_transition_probe",
        "features_sha256": source_hash,
        "source_kind": "encrypted_local_recordings" if recorded_store else "pinned_feature_archive",
        "device": str(device), "architecture": args.architecture,
        "torch_version": torch.__version__, "cudnn_enabled": torch.backends.cudnn.enabled,
        "seed": args.seed, "selected_epoch": best_epoch,
        "action_permutations": args.action_permutations,
        "rows": {split: len(partition["state"]) for split, partition in rows.items()},
        "groups": {split: len(set(partition["groups"])) for split, partition in rows.items()},
        "scores": scores,
        "limitation": "16x16 grayscale images and mouse action type/coordinates only. "
                      "Research model is not the live UiJepaPredictor. "
                      + ("Recording-disjoint, not environment-disjoint; action outcomes are observed, "
                         "not independently verified. " if recorded_store else
                         "Episode-disjoint within one public dataset, not environment-disjoint. ")
                      + "No safety, autonomy, Jev parity, or runtime loading claim.",
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

"""GPU ridge probe for frozen text/action features and observed next-screen deltas.

Fits state-only, coordinate-only, and semantic-action baselines with the same
train-only preprocessing. Validation selects regularization; eval is read once.
Research weights never enter the live Heliox predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train_world_visual_delta_latent import attach_semantic_features
from scripts.train_world_visual_transition import load_features, metrics, paired_counterfactual_permutation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sites(features: Path, source_report: Path, rows: dict) -> dict[str, np.ndarray]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    sites = source.get("sites")
    if source.get("counterfactual_pairs") is not True or not isinstance(sites, dict):
        raise ValueError("Site-disjoint observed branch source is required")
    with np.load(features, allow_pickle=False) as archive:
        tasks = archive["task"].astype(str)
        splits = archive["split"].astype(str)
    result = {}
    for split in ("train", "validation", "eval"):
        names = sites.get(split)
        selected = tasks[splits == split]
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(selected) != len(rows[split]["state"])
            or any(path.split("/", 1)[0] not in names for path in selected)
        ):
            raise ValueError(f"Invalid {split} site partition")
        result[split] = selected
    if len({name for names in sites.values() for name in names}) != sum(map(len, sites.values())):
        raise ValueError("Site identity leaked across splits")
    return result


def semantic_projection(rows: dict, dimensions: int, torch, device) -> dict:
    train = torch.as_tensor(rows["train"]["semantic"], dtype=torch.float32, device=device)
    if dimensions < 1 or dimensions > min(train.shape):
        raise ValueError("PCA dimensions exceed the train-only semantic matrix")
    mean = train.mean(dim=0)
    _, _, directions = torch.linalg.svd(train - mean, full_matrices=False)
    basis = directions[:dimensions].T.contiguous()
    return {
        split: (torch.as_tensor(rows[split]["semantic"], dtype=torch.float32, device=device) - mean) @ basis
        for split in ("train", "validation", "eval")
    }


def feature_matrix(partition: dict, semantic, variant: str, torch, device):
    state = torch.as_tensor(partition["state"], dtype=torch.float32, device=device)
    if variant == "state":
        return state
    coords = torch.as_tensor(partition["action"][:, 9:11], dtype=torch.float32, device=device)
    if variant == "coords":
        return torch.cat((state, coords), dim=1)
    if variant == "semantic":
        return torch.cat((state, coords, semantic), dim=1)
    raise ValueError(f"Unknown model variant: {variant}")


def fit_ridge(train_input, train_delta, penalty: float, torch):
    """Solve the small sample-space ridge system without a free test-set parameter."""
    mean = train_input.mean(dim=0)
    scale = train_input.std(dim=0, unbiased=False).clamp_min(0.05)
    design = (train_input - mean) / scale
    delta_mean = train_delta.mean(dim=0)
    gram = design @ design.T
    alpha = torch.linalg.solve(gram + penalty * torch.eye(len(design), device=design.device), train_delta - delta_mean)
    return {"mean": mean, "scale": scale, "weights": design.T @ alpha, "delta_mean": delta_mean}


def predict(model: dict, features, state, torch):
    delta = (features - model["mean"]) / model["scale"] @ model["weights"] + model["delta_mean"]
    return torch.clamp(state + delta, 0, 1)


def weighted_row_error(prediction: np.ndarray, state: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != state.shape or state.shape != target.shape:
        raise ValueError("Aligned screen predictions are required")
    weights = 1 + 4 * (np.abs(target - state) >= 0.02)
    return (((prediction - target) ** 2) * weights).mean(axis=1)


def paired_score(prediction: np.ndarray, swapped: np.ndarray, partition: dict) -> dict:
    groups = partition["groups"]
    correct_error = weighted_row_error(prediction, partition["state"], partition["target"])
    swapped_error = weighted_row_error(swapped, partition["state"], partition["target"])
    margins = []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        if len(selected) != 2:
            raise ValueError("Each branch group must contain exactly two actions")
        margins.append(float((swapped_error[selected] - correct_error[selected]).mean()))
    return {
        "groups": len(margins),
        "correct_action_wins": sum(value > 0 for value in margins),
        "ties": sum(value == 0 for value in margins),
        "mean_weighted_mse_advantage": float(np.mean(margins)),
        "correct_weighted_mse": float(correct_error.mean()),
        "swapped_weighted_mse": float(swapped_error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "visual_features",
        "source_report",
        "semantic_features",
        "semantic_report",
        "protocol",
        "report",
        "weights",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    files = (args.visual_features, args.source_report, args.semantic_features, args.semantic_report, args.protocol)
    if (
        any(not path.is_file() for path in files)
        or args.report.exists()
        or args.weights.exists()
        or args.weights.suffix != ".npz"
        or args.report.resolve() == args.weights.resolve()
    ):
        parser.error("Use existing pinned inputs and fresh output paths")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = protocol["model"]
    dimensions = config["pca_dimensions"]
    penalties = config["ridge_lambdas"]
    if (
        not isinstance(penalties, list)
        or not penalties
        or any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in penalties)
        or config.get("device") != "cuda"
    ):
        raise ValueError("Protocol requires finite positive penalties and CUDA")
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if (
        source.get("source_dataset") != protocol.get("source_dataset")
        or source.get("source_revision") != protocol.get("source_revision")
        or any(
            source.get("sites", {}).get(split) != protocol[f"{split}_sites"]
            for split in ("train", "validation", "eval")
        )
    ):
        raise ValueError("Collection differs from the frozen site protocol")
    rows = load_features(args.visual_features, args.source_report)
    tasks = validate_sites(args.visual_features, args.source_report, rows)
    attach_semantic_features(
        rows, args.visual_features, args.source_report, args.semantic_features, args.semantic_report
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen protocol")
    device = torch.device("cuda:0")
    semantic = semantic_projection(rows, dimensions, torch, device)
    states = {split: torch.as_tensor(rows[split]["state"], dtype=torch.float32, device=device) for split in rows}
    train_delta = torch.as_tensor(rows["train"]["target"] - rows["train"]["state"], dtype=torch.float32, device=device)
    selected_models = {}
    report_models = {}
    for variant in ("state", "coords", "semantic"):
        inputs = {split: feature_matrix(rows[split], semantic[split], variant, torch, device) for split in rows}
        trials = []
        for penalty in penalties:
            model = fit_ridge(inputs["train"], train_delta, penalty, torch)
            validation = predict(model, inputs["validation"], states["validation"], torch)
            error = weighted_row_error(
                validation.cpu().numpy(), rows["validation"]["state"], rows["validation"]["target"]
            ).mean()
            trials.append((float(error), penalty, model))
        validation_error, chosen_penalty, model = min(trials, key=lambda item: (item[0], item[1]))
        selected_models[variant] = model
        partition = rows["eval"]
        observed = predict(model, inputs["eval"], states["eval"], torch).cpu().numpy()
        permutation = paired_counterfactual_permutation(partition)
        swapped_inputs = inputs["eval"].clone()
        if variant != "state":
            swapped_inputs[:, 256:] = inputs["eval"][permutation, 256:]
        swapped = predict(model, swapped_inputs, states["eval"], torch).cpu().numpy()
        site_scores = {}
        for site in source["sites"]["eval"]:
            mask = np.asarray([task.startswith(site + "/") for task in tasks["eval"]])
            sub = {key: value[mask] for key, value in partition.items()}
            site_scores[site] = paired_score(observed[mask], swapped[mask], sub)
        report_models[variant] = {
            "selected_lambda": chosen_penalty,
            "validation_weighted_mse": validation_error,
            "eval": paired_score(observed, swapped, partition),
            "eval_sites": site_scores,
            "eval_metrics": metrics(observed, partition["state"], partition["target"]),
        }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.weights,
        **{
            f"{variant}_{name}": tensor.detach().cpu().numpy()
            for variant, model in selected_models.items()
            for name, tensor in model.items()
        },
    )
    report = {
        "purpose": "research_only_frozen_semantic_action_gpu_ridge",
        "protocol_sha256": digest(args.protocol),
        "visual_features_sha256": digest(args.visual_features),
        "semantic_features_sha256": digest(args.semantic_features),
        "weights_sha256": digest(args.weights),
        "device": str(device),
        "rows": {split: len(rows[split]["state"]) for split in rows},
        "groups": {split: len(set(rows[split]["groups"])) for split in rows},
        "models": report_models,
        "limitation": "Generated one-step browser actions, not a real desktop or Jev parity. No runtime loading.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "groups": report["groups"],
                "eval_sites": {name: scores["eval_sites"] for name, scores in report_models.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

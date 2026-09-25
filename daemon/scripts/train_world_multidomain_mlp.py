"""Fit a research-only nonlinear task-state head on frozen public features.

Only training and calibration checkpoints select the model. Public test and
previously inspected synthetic cases are scored after selection, never used as
training examples or action authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.train_world_scope_semantic import metrics, normalize

DOMAINS = ("agentic", "tbac", "scope")
EVALUATION = ("agentic_test", "tbac_test", "scope_test", "heliox")
HIDDEN_SIZES = (64, 128)
SEEDS = (19, 41, 73)
EPOCHS = 120


def load_features(feature_dir: Path, source: dict) -> tuple[dict[str, np.ndarray],
                                                            dict[str, np.ndarray]]:
    features, labels = {}, {}
    for name, rows in source["rows"].items():
        path = feature_dir / f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source[
                "feature_checkpoint_sha256"][name]:
            raise ValueError(f"Feature checkpoint hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as saved:
            processed = int(saved["processed"])
            indices = saved["indices"].astype(int)
            values = saved["features"].astype(np.float32)
            targets = saved["labels"].astype(bool)
        if (processed != rows or len(values) != len(targets) or
                len(values) != rows - source["excluded_overlong"][name] or
                len(indices) != len(values) or
                not np.array_equal(indices, np.unique(indices)) or
                np.any(indices < 1) or np.any(indices > processed) or
                values.ndim != 2 or not np.isfinite(values).all() or
                len(np.unique(targets)) != 2):
            raise ValueError(f"Invalid feature checkpoint contents: {name}")
        features[name], labels[name] = values, targets
    return features, labels


def balanced_weights(labels: dict[str, np.ndarray]) -> np.ndarray:
    size = sum(len(labels[f"{domain}_train"]) for domain in DOMAINS)
    weights = []
    for domain in DOMAINS:
        targets = labels[f"{domain}_train"]
        positive, negative = int(targets.sum()), int((~targets).sum())
        if not positive or not negative:
            raise ValueError(f"Training domain lacks a class: {domain}")
        weights.append(np.where(targets, size / (2 * len(DOMAINS) * positive),
                                size / (2 * len(DOMAINS) * negative)))
    return np.concatenate(weights).astype(np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path,
                        default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--feature-dir", type=Path,
                        default=state / "world_multiturn_features_v2")
    parser.add_argument("--artifact", type=Path,
                        default=state / "world_multidomain_mlp_v1.npz")
    parser.add_argument("--report", type=Path,
                        default=state / "world_multidomain_mlp_v1.json")
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Use fresh artifact and report paths")
    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    if source["domains"] != list(DOMAINS) or not source["tbac"]["validation_in_training"]:
        raise ValueError("Expected frozen three-domain feature report")
    features, labels = load_features(args.feature_dir, source)
    features["train"] = np.vstack([features[f"{name}_train"] for name in DOMAINS])
    normalized, mean, scale = normalize(features)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research training run")
    device = torch.device("cuda")
    train_x = torch.as_tensor(normalized["train"], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(np.concatenate([labels[f"{name}_train"] for name in DOMAINS]),
                              dtype=torch.float32, device=device)
    weights = torch.as_tensor(balanced_weights(labels), device=device)
    calibration_x = {name: torch.as_tensor(normalized[f"{name}_calibration"],
                                           dtype=torch.float32, device=device)
                     for name in DOMAINS}
    candidates = []
    selected = None
    for hidden in HIDDEN_SIZES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(train_x.shape[1], hidden),
                torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            for _ in range(EPOCHS):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x).squeeze(1)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, train_y, reduction="none")
                loss = (losses * weights).mean()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                calibration_scores = {name: torch.sigmoid(model(values)).squeeze(1)
                                      .cpu().numpy() for name, values in calibration_x.items()}
            auc = {name: float(roc_auc_score(labels[f"{name}_calibration"], values))
                   for name, values in calibration_scores.items()}
            threshold = math.nextafter(max(float(np.max(calibration_scores[name][
                ~labels[f"{name}_calibration"]])) for name in DOMAINS), math.inf)
            recall = {name: float(np.mean(calibration_scores[name][
                labels[f"{name}_calibration"]] >= threshold)) for name in DOMAINS}
            rank = (min(auc.values()), min(recall.values()), sum(auc.values()) / len(DOMAINS))
            candidates.append({"hidden": hidden, "seed": seed, "calibration_auc": auc,
                               "calibration_cautious_recall": recall,
                               "threshold": threshold})
            print(json.dumps({"event": "candidate", **candidates[-1]}), flush=True)
            if selected is None or rank > selected[0]:
                selected = (rank, hidden, seed, threshold,
                            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    assert selected is not None
    _, hidden, seed, threshold, state_dict = selected
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], hidden),
        torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        scores = {name: torch.sigmoid(model(torch.as_tensor(values, dtype=torch.float32,
                                                           device=device))).squeeze(1).cpu().numpy()
                  for name, values in normalized.items() if name != "train"}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, w1=state_dict["0.weight"].numpy(),
                        b1=state_dict["0.bias"].numpy(), w2=state_dict["3.weight"].numpy(),
                        b2=state_dict["3.bias"].numpy(), mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32), threshold=threshold,
                        source_report_sha256=hashlib.sha256(source_bytes).hexdigest())
    report = {
        "purpose": "research_only_nonlinear_world_task_state_head",
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "training_domains": list(DOMAINS), "epochs": EPOCHS,
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.1},
        "selection": "maximum minimum calibration AUC, then minimum cautious recall; no test selection",
        "candidates": candidates,
        "selected": {"hidden": hidden, "seed": seed, "threshold": threshold},
        "metrics": {name: {"default": metrics(labels[name], scores[name], 0.5),
                           "cautious": metrics(labels[name], scores[name], threshold)}
                    for name in EVALUATION},
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "limitation": "Public relevance/scope and inspected synthetic transfer only; not action "
                      "authorization, blind desktop evidence, or Jev parity.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report["metrics"][name] for name in EVALUATION}, indent=2))


if __name__ == "__main__":
    main()

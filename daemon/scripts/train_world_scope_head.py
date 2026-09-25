"""Train an offline task-state scope head for Heliox's world-model research.

Only pinned MIT ScopeJudge train labels fit weights. Trajectory-disjoint
calibration selects regularization and a caution threshold. Test and Heliox
transfer labels are read only after that selection. No runtime integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_qwen_v1_holdout import CASES as HELIOX_CASES  # noqa: E402
from benchmarks.world_scope_head import world_scope_features  # noqa: E402
from scripts.evaluate_openjev_checkpoint import scopejudge_cases  # noqa: E402

PENALTIES = (0.0001, 0.001, 0.01, 0.1)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive, negative = labels, ~labels
    return {
        "rows": len(labels),
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float((np.mean(predicted[positive]) + np.mean(~predicted[negative])) / 2),
        "auc": float(roc_auc_score(labels, scores)),
        "false_allowances": int(np.sum(predicted & negative)),
        "false_warnings": int(np.sum(~predicted & positive)),
    }


def train_one(x: np.ndarray, labels: np.ndarray, penalty: float, *, epochs: int,
              device: str) -> tuple[np.ndarray, float]:
    import torch

    torch.manual_seed(20260925)
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    targets = torch.as_tensor(labels.astype(np.float32), device=device)
    head = torch.nn.Linear(x.shape[1], 1, device=device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + penalty * head.weight.square().sum()
        loss.backward()
        optimizer.step()
    return head.weight.detach().cpu().numpy().ravel(), float(head.bias.detach().cpu().item())


def scores(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(x @ weights + bias, -40, 40)
    return np.asarray(1 / (1 + np.exp(-logits)), dtype=np.float64)


def run(path: Path, *, artifact: Path, epochs: int, history_calls: int,
        device: str) -> dict:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if artifact.exists():
        raise FileExistsError("World-scope research artifact already exists")
    parts = {}
    manifests = {}
    for name, partition in (("train", "scope_train"),
                            ("calibration", "scope_calibration"), ("test", "scope_test")):
        parts[name], manifests[name] = scopejudge_cases(
            path, partition, history_calls=history_calls
        )
    if not all(manifest == manifests["train"] for manifest in manifests.values()):
        raise ValueError("ScopeJudge source changed between partitions")
    groups = [{case.id.split(":", 1)[0] for case in parts[name]}
              for name in ("train", "calibration", "test")]
    if any(groups[left] & groups[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Trajectory leakage between partitions")
    x = {name: world_scope_features(cases) for name, cases in parts.items()}
    y = {name: np.asarray([case.in_scope for case in cases], dtype=np.bool_)
         for name, cases in parts.items()}
    if any(not np.any(values) or not np.any(~values) for values in y.values()):
        raise ValueError("Each partition needs both scope labels")
    trials = []
    for penalty in PENALTIES:
        weights, bias = train_one(x["train"], y["train"], penalty,
                                  epochs=epochs, device=device)
        calibration_scores = scores(x["calibration"], weights, bias)
        auc = float(roc_auc_score(y["calibration"], calibration_scores))
        trials.append((auc, penalty, weights, bias, calibration_scores))
    auc, penalty, weights, bias, calibration_scores = max(
        trials, key=lambda trial: (trial[0], -trial[1])
    )
    threshold = math.nextafter(float(np.max(calibration_scores[~y["calibration"]])), math.inf)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, weights=weights, bias=bias,
                        threshold=threshold, penalty=penalty,
                        history_calls=history_calls)
    test_scores = scores(x["test"], weights, bias)
    heliox = tuple(HELIOX_CASES)
    heliox_scores = scores(world_scope_features(heliox), weights, bias)
    heliox_labels = np.asarray([case.in_scope for case in heliox], dtype=np.bool_)
    return {
        "purpose": "research_only_world_model_task_state_head",
        "dataset": manifests["train"],
        "training": {"device": device, "epochs": epochs, "rows": len(y["train"]),
                     "features": x["train"].shape[1], "history_calls": history_calls,
                     "regularization_trials": [{"penalty": item[1], "calibration_auc": item[0]}
                                               for item in trials],
                     "selected_penalty": penalty, "calibration_auc": auc},
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "calibration_threshold": threshold,
        "calibration": metrics(y["calibration"], calibration_scores, threshold),
        "test_default": metrics(y["test"], test_scores, 0.5),
        "test_cautious": metrics(y["test"], test_scores, threshold),
        "heliox_transfer_default": metrics(heliox_labels, heliox_scores, 0.5),
        "heliox_transfer_cautious": metrics(heliox_labels, heliox_scores, threshold),
        "limitation": "Public security tool-call labels and inspected synthetic transfer; "
                      "not a production world model, Jev parity, or action approval.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopejudge-path", type=Path, required=True)
    parser.add_argument("--history-calls", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/.bench-state/world_scope_head_v1.npz")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1]
                        / "benchmarks/.bench-state/world_scope_head_v1.json")
    args = parser.parse_args()
    if args.history_calls < 0 or args.epochs < 1:
        parser.error("History count must be nonnegative and epochs positive")
    if args.report.exists():
        raise FileExistsError("World-scope research report already exists")
    report = run(args.scopejudge_path, artifact=args.artifact, epochs=args.epochs,
                 history_calls=args.history_calls, device=args.device)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("training", "calibration", "test_default", "test_cautious",
                       "heliox_transfer_default", "heliox_transfer_cautious")}, indent=2))


if __name__ == "__main__":
    main()

"""Fit an unpromoted scope head from pinned, public ScopeJudge features only.

The cache must match a report that records an MIT ScopeJudge release and
trajectory-disjoint partitions. Training uses only its train split; the
calibration split selects regularization and a cautious threshold, while test
remains untouched until final scoring. No action text is emitted or saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SPLITS = ("scope_train", "scope_calibration", "scope_test")
REGULARIZATION = (0.001, 0.01, 0.1, 1.0)


def load_public_splits(feature_dir: Path, source_report: Path) -> tuple[dict[str, dict], dict]:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    source = report.get("scopejudge", {})
    if (
        source.get("dataset") != "dreadnode/scopejudge"
        or source.get("license") != "MIT"
        or "trajectory-disjoint" not in source.get("selection", "")
    ):
        raise ValueError("Source report must attest to MIT ScopeJudge trajectory-disjoint splits")

    hashes = report.get("feature_checkpoint_sha256", {})
    rows = report.get("rows", {})
    excluded = report.get("excluded_overlong", {})
    splits: dict[str, dict] = {}
    dimension: int | None = None
    for name in SPLITS:
        path = feature_dir / f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name):
            raise ValueError(f"Feature cache does not match source report: {name}")
        with np.load(path, allow_pickle=False) as saved:
            features = np.asarray(saved["features"], dtype=np.float32)
            labels = np.asarray(saved["labels"], dtype=bool)
            identity = str(saved["identity"])
            processed = int(saved["processed"])
            overlong = int(saved["overlong"])
        if (
            features.ndim != 2
            or features.shape[0] != len(labels)
            or not len(labels)
            or not np.all(np.isfinite(features))
            or len(np.unique(labels)) != 2
        ):
            raise ValueError(f"Invalid feature matrix or labels: {name}")
        if dimension is not None and features.shape[1] != dimension:
            raise ValueError("Feature dimensions differ across splits")
        dimension = features.shape[1]
        if not identity or processed != rows.get(name) or overlong != excluded.get(name):
            raise ValueError(f"Feature metadata does not match source report: {name}")
        splits[name] = {"features": features, "labels": labels, "identity": identity}
    if len({split["identity"] for split in splits.values()}) != len(SPLITS):
        raise ValueError("Each feature split must have a distinct identity")
    return splits, {
        **source,
        "representation": {"model": report.get("model"), "revision": report.get("revision")},
    }


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    positive, negative = labels, ~labels
    return {
        "rows": len(labels),
        "in_scope_rows": int(np.sum(positive)),
        "out_of_scope_rows": int(np.sum(negative)),
        "auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float((np.mean(predicted[positive]) + np.mean(~predicted[negative])) / 2),
        "false_allowances": int(np.sum(predicted & negative)),
        "false_warnings": int(np.sum(~predicted & positive)),
        "brier": float(np.mean((scores - labels.astype(np.float64)) ** 2)),
    }


def train_candidate(splits: dict[str, dict], *, device: str = "auto") -> tuple[dict, dict]:
    import torch
    from torch import nn

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(20260927)
    train_x = splits["scope_train"]["features"]
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    tensors = {
        name: (
            torch.as_tensor((split["features"] - mean) / scale, dtype=torch.float32, device=device),
            torch.as_tensor(split["labels"], dtype=torch.float32, device=device),
        )
        for name, split in splits.items()
    }
    train_x_t, train_y_t = tensors["scope_train"]
    calibration_x, calibration_y = tensors["scope_calibration"]
    trials = []
    for strength in REGULARIZATION:
        model = nn.Linear(train_x.shape[1], 1, device=device)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

        def closure(model=model, optimizer=optimizer, strength=strength):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(train_x_t).squeeze(-1), train_y_t)
            loss = loss + 0.5 * strength * model.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            calibration_scores = torch.sigmoid(model(calibration_x).squeeze(-1)).cpu().numpy()
        labels = splits["scope_calibration"]["labels"]
        trials.append((float(roc_auc_score(labels, calibration_scores)), strength, model, calibration_scores))

    _, selected_strength, selected, _ = max(trials, key=lambda item: (item[0], -item[1]))
    calibration_labels = splits["scope_calibration"]["labels"]
    scores: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, (features, _) in tensors.items():
            scores[name] = torch.sigmoid(selected(features).squeeze(-1)).cpu().numpy()
    worst_calibration_negative = np.float32(np.max(scores["scope_calibration"][~calibration_labels]))
    threshold = float(np.nextafter(worst_calibration_negative, np.float32(math.inf)))

    model_state = {
        "weights": selected.weight.detach().cpu().numpy().astype(np.float32).reshape(-1),
        "bias": selected.bias.detach().cpu().numpy().astype(np.float32).reshape(-1),
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "threshold": np.asarray(threshold, dtype=np.float64),
        "model_version": np.asarray("public-scope-head-gpu-v1"),
    }
    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name() if str(device).startswith("cuda") else "CPU",
        "method": "L2-regularized logistic head over frozen Qwen features; train-only normalization",
        "selected_l2": selected_strength,
        "calibration_trials": [{"l2": strength, "auc": auc} for auc, strength, _, _ in trials],
        "calibration_threshold": threshold,
        "calibration": _metrics(calibration_labels, scores["scope_calibration"], threshold),
        "test_default": _metrics(splits["scope_test"]["labels"], scores["scope_test"], 0.5),
        "test_cautious": _metrics(splits["scope_test"]["labels"], scores["scope_test"], threshold),
    }
    return model_state, report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--source-report", type=Path, default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--artifact", type=Path, default=state / "public_scope_head_gpu_v1.npz")
    parser.add_argument("--report", type=Path, default=state / "public_scope_head_gpu_v1.json")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.artifact.exists() or args.report.exists():
        parser.error("Choose fresh artifact and report paths; existing evidence is never overwritten")
    splits, source = load_public_splits(args.feature_dir, args.source_report)
    model_state, result = train_candidate(splits, device=args.device)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, **model_state)
    result.update(
        {
            "purpose": "unpromoted_public_scope_label_research_only",
            "representation": source["representation"],
            "source": {key: source[key] for key in ("dataset", "revision", "sha256", "license", "selection")},
            "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
            "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "feature_cache_sha256": {
                name: hashlib.sha256((args.feature_dir / f"{name}.npz").read_bytes()).hexdigest() for name in SPLITS
            },
            "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
            "rows": {name: len(split["labels"]) for name, split in splits.items()},
            "limitation": "Adaptive public-label research on offensive-security trajectories; not broad desktop-agent safety, live approval, or production authorization evidence.",
        }
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("device", "selected_l2", "calibration", "test_default", "test_cautious")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

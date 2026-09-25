#!/usr/bin/env python3
"""Trains the Learned Risk Gate's transition model — a small MLP predicting
(disk_usage_delta, process_count_delta_normalized) from the encoded
(OS state, action) embedding (pilot/security/risk_model.py's encode()).

Pure numpy, no PyTorch/candle — same pragmatic scoping Ferrum-OS's own
scripts/train_world_model.py uses, and keeps runtime inference
(pilot/security/risk_model.py's RiskTransitionModel) dependency-free
beyond numpy, which this daemon already requires.

Reads collect_risk_training_data.py's output (real telemetry — see that
script's module docstring for exactly what's real vs. why nothing here is
synthetic/fabricated), trains a 2-layer MLP via plain gradient descent,
and writes a flat .npz weights file RiskTransitionModel loads at runtime.

Usage:
    python scripts/train_risk_gate.py [--dataset PATH] [--out PATH] [--hidden N] [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pilot.actions import ActionType  # noqa: E402
from pilot.security.risk_model import (  # noqa: E402
    EMBEDDING_SIZE,
    IDX_ACTION_TYPE_BASE,
    LEARNABLE_ACTION_TYPE_ORDER,
    MODEL_VERSION,
)

OUTPUT_SIZE = 2  # [disk_delta, proc_delta_normalized]


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[list[float]] = []
    targets: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            embedding = list(row["embedding"])
            if len(embedding) == IDX_ACTION_TYPE_BASE:
                # Datasets collected before v2 contain the original state +
                # family embedding. Preserve that real telemetry and append
                # the exact action identity recorded alongside every row.
                action_type = ActionType(row["action_type"])
                action_slots = [0.0] * len(LEARNABLE_ACTION_TYPE_ORDER)
                action_slots[LEARNABLE_ACTION_TYPE_ORDER.index(action_type)] = 1.0
                embedding.extend(action_slots)
            embeddings.append(embedding)
            targets.append([row["disk_delta"], row["proc_delta"]])

    X = np.array(embeddings, dtype=np.float32)
    Y = np.array(targets, dtype=np.float32)
    if X.shape[1] != EMBEDDING_SIZE:
        raise ValueError(f"Dataset embedding width {X.shape[1]} != current EMBEDDING_SIZE {EMBEDDING_SIZE}")
    return X, Y


def load_action_types(path: str) -> np.ndarray:
    action_types: list[str] = []
    with open(path) as dataset:
        for line in dataset:
            line = line.strip()
            if line:
                action_types.append(str(json.loads(line)["action_type"]))
    return np.asarray(action_types)


def stratified_temporal_split(
    action_types: np.ndarray,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every action family into train, calibration, and validation.

    Rows are collected sequentially per action type. Keeping the final two
    contiguous blocks out of training is stricter than a random row split,
    which leaked nearly identical adjacent OS snapshots across both sides.
    """
    train_indices: list[int] = []
    calibration_indices: list[int] = []
    validation_indices: list[int] = []
    for action_type in LEARNABLE_ACTION_TYPE_ORDER:
        indices = np.flatnonzero(action_types == action_type.value)
        holdout_size = max(1, int(len(indices) * holdout_fraction))
        if len(indices) <= holdout_size * 2:
            raise ValueError(f"Not enough {action_type.value} samples for temporal calibration and validation")
        train_indices.extend(indices[: -2 * holdout_size])
        calibration_indices.extend(indices[-2 * holdout_size : -holdout_size])
        validation_indices.extend(indices[-holdout_size:])
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(calibration_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
    )


def init_weights(input_size: int, hidden_size: int, output_size: int, rng: np.random.Generator):
    # Small random init scaled by fan-in, standard practice for a tanh
    # hidden layer to avoid saturating at the start of training.
    w1 = rng.normal(0, 1.0 / np.sqrt(input_size), size=(input_size, hidden_size)).astype(np.float32)
    b1 = np.zeros(hidden_size, dtype=np.float32)
    w2 = rng.normal(0, 1.0 / np.sqrt(hidden_size), size=(hidden_size, output_size)).astype(np.float32)
    b2 = np.zeros(output_size, dtype=np.float32)
    return w1, b1, w2, b2


def train(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_size: int,
    epochs: int,
    lr: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trains on Y normalized per-output-column (disk_delta and
    proc_delta differ by ~4 orders of magnitude in this dataset — without
    normalizing, a shared MSE loss across both outputs is dominated
    entirely by whichever has the larger raw scale, and the smaller-scale
    output's gradient is too small to actually learn anything). The
    output layer is linear (no activation), so the normalization is
    folded directly into w2/b2 before returning — callers and
    RiskTransitionModel's inference code never need to know
    normalization happened at all.
    """
    rng = np.random.default_rng(seed)
    n, input_size = X.shape
    output_size = Y.shape[1]
    w1, b1, w2, b2 = init_weights(input_size, hidden_size, output_size, rng)

    y_scale = Y.std(axis=0)
    y_scale = np.where(y_scale < 1e-8, 1.0, y_scale)  # constant column -> leave unscaled
    Y_norm = Y / y_scale

    for epoch in range(epochs):
        # Full-batch gradient descent — the dataset here is small enough
        # (a few thousand rows, ~11 input dims) that mini-batching or an
        # optimizer beyond plain SGD would be over-engineering.
        hidden_pre = X @ w1 + b1
        hidden = np.tanh(hidden_pre)
        pred = hidden @ w2 + b2

        error = pred - Y_norm
        loss = float(np.mean(error**2))

        d_pred = 2.0 * error / n
        grad_w2 = hidden.T @ d_pred
        grad_b2 = d_pred.sum(axis=0)

        d_hidden = (d_pred @ w2.T) * (1.0 - hidden**2)  # tanh derivative
        grad_w1 = X.T @ d_hidden
        grad_b1 = d_hidden.sum(axis=0)

        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            # Reported in the ORIGINAL (unnormalized) scale so this number
            # is comparable to the baseline MSE printed at the end, not the
            # normalized-space loss actually being optimized above.
            print(f"epoch {epoch:5d}  mse={loss * np.mean(y_scale**2):.8f}")

    # Fold the normalization into the linear output layer: since
    # pred_norm = hidden @ w2 + b2 and pred = pred_norm * y_scale, we have
    # pred = hidden @ (w2 * y_scale) + (b2 * y_scale) — same forward pass,
    # no inference-side changes needed.
    w2 = w2 * y_scale
    b2 = b2 * y_scale

    return w1, b1, w2, b2


def _predict(X: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
    return np.tanh(X @ w1 + b1) @ w2 + b2


def _action_medians(Y: np.ndarray, action_types: np.ndarray) -> np.ndarray:
    medians = np.zeros((len(LEARNABLE_ACTION_TYPE_ORDER), OUTPUT_SIZE), dtype=np.float32)
    for index, action_type in enumerate(LEARNABLE_ACTION_TYPE_ORDER):
        rows = Y[action_types == action_type.value]
        if len(rows):
            medians[index] = np.median(rows, axis=0)
    return medians


def fit_calibration_alpha(
    predictions: np.ndarray,
    targets: np.ndarray,
    action_types: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    """Fit a bounded per-action/output blend of MLP and robust median."""
    alpha = np.zeros_like(medians)
    for index, action_type in enumerate(LEARNABLE_ACTION_TYPE_ORDER):
        mask = action_types == action_type.value
        for output_index in range(OUTPUT_SIZE):
            delta = predictions[mask, output_index] - medians[index, output_index]
            denominator = float(delta @ delta)
            if denominator <= 1e-20:
                alpha[index, output_index] = 0.0
                continue
            numerator = float(delta @ (targets[mask, output_index] - medians[index, output_index]))
            alpha[index, output_index] = float(np.clip(numerator / denominator, 0.0, 1.0))
    return alpha


def apply_calibration(
    predictions: np.ndarray,
    action_types: np.ndarray,
    medians: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    calibrated = np.empty_like(predictions)
    for index, action_type in enumerate(LEARNABLE_ACTION_TYPE_ORDER):
        mask = action_types == action_type.value
        calibrated[mask] = alpha[index] * predictions[mask] + (1.0 - alpha[index]) * medians[index]
    return calibrated


def write_weights(
    path: str,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    samples: int,
    *,
    validation_samples: int,
    validation_mae: np.ndarray,
    baseline_mae: np.ndarray,
    action_medians: np.ndarray,
    calibration_alpha: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> None:
    np.savez(
        path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        model_version=np.array(MODEL_VERSION),
        training_samples=np.array(samples, dtype=np.int64),
        validation_samples=np.array(validation_samples, dtype=np.int64),
        validation_mae=validation_mae.astype(np.float32),
        baseline_mae=baseline_mae.astype(np.float32),
        action_medians=action_medians.astype(np.float32),
        calibration_alpha=calibration_alpha.astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
    )


def _mse(X: np.ndarray, Y: np.ndarray, w1, b1, w2, b2) -> float:
    pred = _predict(X, w1, b1, w2, b2)
    return float(np.mean((pred - Y) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=str(Path(__file__).parent / "risk_dataset.jsonl"))
    parser.add_argument(
        "--out", type=str, default=str(Path(__file__).parent.parent / "pilot" / "security" / "risk_gate_weights.npz")
    )
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Per-action fraction reserved for calibration and again for final validation; "
        "neither split is used to fit the saved MLP.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, Y = load_dataset(args.dataset)
    action_types = load_action_types(args.dataset)
    print(f"Loaded {X.shape[0]} samples, embedding width {X.shape[1]}")

    train_indices, calibration_indices, validation_indices = stratified_temporal_split(
        action_types,
        args.val_frac,
    )
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_calibration, Y_calibration = X[calibration_indices], Y[calibration_indices]
    X_validation, Y_validation = X[validation_indices], Y[validation_indices]
    train_actions = action_types[train_indices]
    calibration_actions = action_types[calibration_indices]
    validation_actions = action_types[validation_indices]

    print(
        "Temporal stratified split: "
        f"{len(train_indices)} train, {len(calibration_indices)} calibration, "
        f"{len(validation_indices)} validation"
    )

    w1, b1, w2, b2 = train(X_train, Y_train, hidden_size=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed)

    medians = _action_medians(Y_train, train_actions)
    calibration_predictions = _predict(X_calibration, w1, b1, w2, b2)
    calibration_alpha = fit_calibration_alpha(
        calibration_predictions,
        Y_calibration,
        calibration_actions,
        medians,
    )
    raw_validation_predictions = _predict(X_validation, w1, b1, w2, b2)
    calibrated_validation_predictions = apply_calibration(
        raw_validation_predictions,
        validation_actions,
        medians,
        calibration_alpha,
    )

    baseline_val_mse = float(np.mean(Y_validation**2))
    train_mse = _mse(X_train, Y_train, w1, b1, w2, b2)
    raw_val_mse = float(np.mean((raw_validation_predictions - Y_validation) ** 2))
    val_mse = float(np.mean((calibrated_validation_predictions - Y_validation) ** 2))
    validation_mae = np.mean(np.abs(calibrated_validation_predictions - Y_validation), axis=0)
    baseline_mae = np.mean(np.abs(Y_validation), axis=0)
    print(f"Baseline (predict zero) val MSE: {baseline_val_mse:.8e}")
    print(f"Train MSE:                       {train_mse:.8e}")
    print(f"Raw held-out val MSE:            {raw_val_mse:.8e}")
    print(f"Calibrated held-out val MSE:     {val_mse:.8e}  (the number that actually matters)")
    print(f"Calibrated held-out MAE:         disk={validation_mae[0]:.10f} proc={validation_mae[1]:.10f}")
    if val_mse > baseline_val_mse:
        raise RuntimeError("Learned model is worse than predicting zero on held-out data; weights not written")

    # Save the exact model measured above. Refitting on calibration and
    # validation rows would invalidate both the calibration blend and the
    # reported held-out metrics for the artifact used at runtime.
    feature_mean = X_train[:, :3].mean(axis=0)
    feature_scale = X_train[:, :3].std(axis=0)
    write_weights(
        args.out,
        w1,
        b1,
        w2,
        b2,
        len(train_indices),
        validation_samples=len(validation_indices),
        validation_mae=validation_mae,
        baseline_mae=baseline_mae,
        action_medians=medians,
        calibration_alpha=calibration_alpha,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    print(
        f"Wrote held-out weights ({len(train_indices)} training, "
        f"{len(calibration_indices)} calibration, {len(validation_indices)} validation) to {args.out}"
    )


if __name__ == "__main__":
    main()

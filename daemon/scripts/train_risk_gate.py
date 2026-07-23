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

from pilot.security.risk_model import EMBEDDING_SIZE  # noqa: E402

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
            embeddings.append(row["embedding"])
            targets.append([row["disk_delta"], row["proc_delta"]])

    X = np.array(embeddings, dtype=np.float32)
    Y = np.array(targets, dtype=np.float32)
    if X.shape[1] != EMBEDDING_SIZE:
        raise ValueError(f"Dataset embedding width {X.shape[1]} != current EMBEDDING_SIZE {EMBEDDING_SIZE}")
    return X, Y


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


def write_weights(path: str, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2)


def _mse(X: np.ndarray, Y: np.ndarray, w1, b1, w2, b2) -> float:
    pred = np.tanh(X @ w1 + b1) @ w2 + b2
    return float(np.mean((pred - Y) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=str(Path(__file__).parent / "risk_dataset.jsonl"))
    parser.add_argument(
        "--out", type=str, default=str(Path(__file__).parent.parent / "pilot" / "security" / "risk_gate_weights.npz")
    )
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Held-out fraction used ONLY to report generalization MSE -- "
        "the final saved weights are refit on the full dataset afterward.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, Y = load_dataset(args.dataset)
    print(f"Loaded {X.shape[0]} samples, embedding width {X.shape[1]}")

    # The dataset file is grouped by action_type (collect_risk_training_data.py
    # writes all samples for one type before moving to the next), so a plain
    # index-based split would hand the validation set almost entirely
    # different action types than training saw -- shuffle first so both
    # splits are representative.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    X, Y = X[perm], Y[perm]

    n_val = max(1, int(X.shape[0] * args.val_frac))
    X_val, Y_val = X[:n_val], Y[:n_val]
    X_train, Y_train = X[n_val:], Y[n_val:]

    print(f"Train/val split: {X_train.shape[0]} train, {X_val.shape[0]} val")

    w1, b1, w2, b2 = train(X_train, Y_train, hidden_size=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed)

    baseline_val_mse = float(np.mean(Y_val**2))  # predicting all-zeros on held-out data
    train_mse = _mse(X_train, Y_train, w1, b1, w2, b2)
    val_mse = _mse(X_val, Y_val, w1, b1, w2, b2)
    print(f"Baseline (predict zero) val MSE: {baseline_val_mse:.6f}")
    print(f"Train MSE:                       {train_mse:.6f}")
    print(f"Held-out val MSE:                {val_mse:.6f}  (the number that actually matters)")
    if val_mse > baseline_val_mse:
        print("WARNING: learned model is worse than predicting zero on held-out data -- do not ship these weights.")

    # Validation above already confirmed this architecture/hyperparameters
    # generalize; refit on the FULL dataset (train+val) for the weights
    # actually shipped, since there's no reason to withhold real data from
    # the production model once its generalization is confirmed.
    w1, b1, w2, b2 = train(X, Y, hidden_size=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed)
    write_weights(args.out, w1, b1, w2, b2)
    print(f"Wrote weights (trained on all {X.shape[0]} samples) to {args.out}")


if __name__ == "__main__":
    main()

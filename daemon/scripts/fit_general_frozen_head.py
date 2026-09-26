"""Fit an unpromoted decision head on frozen, licensed public Qwen features.

The test and OOD splits are never used for fitting or hyperparameter selection.
No proprietary Jev responses, desktop screenshots, or runtime actions enter this
experiment. All model weights stay in ignored local benchmark state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import (  # noqa: E402
    DATASET_REVISION,
    load_split,
    metrics,
    probabilities,
    uniform_baseline,
    verify_disjoint,
)
from scripts.extract_general_frozen_features import MODEL, REVISION, SPLITS  # noqa: E402

PENALTIES = (0.001, 0.01, 0.1, 1.0)


def load_experiment(extraction_report: Path, feature_dir: Path
                    ) -> tuple[dict, dict[str, list[dict]], dict[str, np.ndarray]]:
    record = json.loads(extraction_report.read_text(encoding="utf-8"))
    if (record.get("purpose") != "frozen_public_general_decision_feature_extraction"
            or record.get("model") != MODEL or record.get("revision") != REVISION
            or record.get("dataset_revision") != DATASET_REVISION):
        raise ValueError("Extraction report does not match the pinned experiment")
    rows, features = {}, {}
    for offset, split in enumerate(SPLITS):
        selected, manifest = load_split(split, record["selected_rows"][split],
                                        record["seed"] + offset)
        if manifest != record["source_manifests"][split]:
            raise ValueError(f"Public dataset manifest changed for {split}")
        path = feature_dir / f"{split}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["feature_sha256"][split]:
            raise ValueError(f"Feature file hash changed for {split}")
        with np.load(path, allow_pickle=False) as archive:
            matrix = archive["features"].astype(np.float64)
            offsets = archive["offsets"].tolist()
            metadata = json.loads(str(archive["metadata_json"]))
        identifiers = metadata["ids"]
        by_id = {row["id"]: row for row in selected}
        positions = {row["id"]: position for position, row in enumerate(selected)}
        if (len(positions) != len(selected) or len(identifiers) != record["completed"][split]
                or len(selected) - len(identifiers) != record["excluded_overlong"][split]
                or any(identifier not in by_id for identifier in identifiers)
                or [positions[key] for key in identifiers] != sorted(positions[key] for key in identifiers)
                or len(offsets) != len(identifiers) + 1 or offsets[0] != 0
                or offsets[-1] != len(matrix) or not np.isfinite(matrix).all()
                or matrix.ndim != 2
                or matrix.shape[1] != metadata.get("feature_dim", matrix.shape[1])):
            raise ValueError(f"Feature rows do not match the selected {split} split")
        matched = [by_id[identifier] for identifier in identifiers]
        for index, row in enumerate(matched):
            count = 1 if row["kind"] == "noul" else len(row["options"])
            if offsets[index + 1] - offsets[index] != count:
                raise ValueError(f"Feature candidate count does not match {split} row")
        rows[split] = matched
        features[split] = (matrix, np.asarray(offsets, dtype=np.int64))
    verify_disjoint(rows)
    dimensions = {matrix.shape[1] for matrix, _ in features.values()}
    if len(dimensions) != 1:
        raise ValueError("Feature dimensions differ between splits")
    return record, rows, features


def prepare(features: dict[str, tuple[np.ndarray, np.ndarray]]
            ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    training = features["train"][0]
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = {split: ((matrix - mean) / scale, offsets)
                  for split, (matrix, offsets) in features.items()}
    return normalized, mean, scale


def logits_for_rows(rows: list[dict], data: tuple[np.ndarray, np.ndarray],
                    parameters: np.ndarray) -> list[np.ndarray]:
    matrix, offsets = data
    scores = matrix @ parameters[:-1] + parameters[-1]
    result = []
    for index, row in enumerate(rows):
        values = scores[offsets[index]:offsets[index + 1]]
        result.append(np.r_[0.0, values] if row["kind"] == "noul" else values)
    return result


def loss_gradient(parameters: np.ndarray, rows: list[dict],
                  data: tuple[np.ndarray, np.ndarray], penalty: float
                  ) -> tuple[float, np.ndarray]:
    matrix, offsets = data
    if len(rows) != len(offsets) - 1 or not rows:
        raise ValueError("Features and training rows must align")
    predicted = logits_for_rows(rows, data, parameters)
    score_derivatives = np.empty(len(matrix), dtype=np.float64)
    loss = 0.0
    for index, (row, logits) in enumerate(zip(rows, predicted, strict=True)):
        target = np.asarray(row["target"], dtype=np.float64)
        probs = probabilities(logits)
        loss -= float(target @ np.log(np.clip(probs, 1e-12, 1.0)))
        gradient = probs - target
        score_derivatives[offsets[index]:offsets[index + 1]] = (
            gradient[1:] if row["kind"] == "noul" else gradient)
    count = len(rows)
    loss = loss / count + 0.5 * penalty * float(parameters[:-1] @ parameters[:-1])
    derivative = np.r_[matrix.T @ score_derivatives / count + penalty * parameters[:-1],
                       score_derivatives.sum() / count]
    return loss, derivative


def fit_head(rows: list[dict], data: tuple[np.ndarray, np.ndarray],
             penalty: float) -> tuple[np.ndarray, float]:
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Penalty must be positive and finite")
    initial = np.zeros(data[0].shape[1] + 1, dtype=np.float64)
    result = minimize(loss_gradient, initial, args=(rows, data, penalty), jac=True,
                      method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Decision-head optimization failed: {result.message}")
    return result.x, float(result.fun)


def mean_nll(rows: list[dict], logits: list[np.ndarray], temperature: float) -> float:
    return metrics(rows, logits, temperature)["mean_nll"]


def calibrate(rows: list[dict], logits: list[np.ndarray]) -> float:
    result = minimize_scalar(lambda value: mean_nll(rows, logits, value),
                             bounds=(0.25, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature calibration failed")
    return float(result.x)


def run(extraction_report: Path, feature_dir: Path, weight_path: Path,
        report_path: Path) -> dict:
    if weight_path.exists() or report_path.exists():
        raise FileExistsError("Choose fresh weight and report paths")
    extraction, rows, raw_features = load_experiment(extraction_report, feature_dir)
    features, mean, scale = prepare(raw_features)
    trials = []
    for penalty in PENALTIES:
        weights, train_loss = fit_head(rows["train"], features["train"], penalty)
        calibration_logits = logits_for_rows(rows["calibration"],
                                            features["calibration"], weights)
        trials.append((mean_nll(rows["calibration"], calibration_logits, 1.0),
                       penalty, weights, train_loss))
    calibration_nll, penalty, weights, train_loss = min(trials, key=lambda trial: trial[0])
    logits = {split: logits_for_rows(rows[split], features[split], weights) for split in SPLITS}
    temperature = calibrate(rows["calibration"], logits["calibration"])
    report = {
        "purpose": "unpromoted_frozen_public_general_decision_head",
        "model": MODEL, "revision": REVISION,
        "extraction_report_sha256": hashlib.sha256(extraction_report.read_bytes()).hexdigest(),
        "dataset_revision": DATASET_REVISION,
        "selected_rows": extraction["selected_rows"], "evaluated_rows": {
            split: len(rows[split]) for split in SPLITS},
        "penalties": list(PENALTIES), "selected_penalty": penalty,
        "training_objective": train_loss, "calibration_nll_before_temperature": calibration_nll,
        "temperature": temperature,
        "trained": {split: metrics(rows[split], logits[split], temperature)
                    for split in ("test", "ood")},
        "uniform": {split: uniform_baseline(rows[split]) for split in ("test", "ood")},
        "limitations": "Licensed public frozen-feature research only. No Jev-label training, "
                       "Jev parity, live desktop validation, or execution authority.",
    }
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    with weight_path.open("wb") as stream:
        np.savez_compressed(stream, weights=weights, mean=mean, scale=scale,
                            model_revision=np.asarray(REVISION))
    report["weight_sha256"] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-report", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1.json")
    parser.add_argument("--feature-dir", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1")
    parser.add_argument("--weights", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1/head.npz")
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/qwen2b_frozen_general_v1.json")
    args = parser.parse_args()
    report = run(args.extraction_report, args.feature_dir, args.weights, args.report)
    print(json.dumps({split: report["trained"][split] for split in ("test", "ood")}, indent=2))


if __name__ == "__main__":
    main()

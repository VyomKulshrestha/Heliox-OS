"""Evaluate calibration of a frozen, research-only world task-state head.

The selected MLP and feature checkpoints are immutable inputs. Calibrators fit
only on the designated calibration partitions; test partitions are reporting
only. No artifact from this script is loaded by the Heliox runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, ndtr
from sklearn.metrics import roc_auc_score

from scripts.train_world_multidomain_mlp import DOMAINS, EVALUATION, load_features


def infer_logits(features: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Reproduce the saved PyTorch Linear/GELU/Linear head without dropout."""
    mean, scale = weights["mean"], weights["scale"]
    w1, b1, w2, b2 = (weights[name] for name in ("w1", "b1", "w2", "b2"))
    if (features.ndim != 2 or features.shape[1] != len(mean)
            or scale.shape != mean.shape or np.any(scale <= 0)
            or w1.shape[1] != len(mean) or b1.shape != (w1.shape[0],)
            or w2.shape != (1, w1.shape[0]) or b2.shape != (1,)):
        raise ValueError("Saved head and features have incompatible shapes")
    standardized = (features.astype(np.float32) - mean) / scale
    hidden_input = standardized @ w1.T + b1
    hidden = hidden_input * ndtr(hidden_input)
    logits = (hidden @ w2.T + b2).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError("Saved head produced non-finite logits")
    return logits


def balanced_calibration_weights(labels: dict[str, np.ndarray], domains: tuple[str, ...]) -> np.ndarray:
    """Give every domain/class cell equal calibration mass."""
    parts = []
    for domain in domains:
        target = labels[f"{domain}_calibration"].astype(bool)
        positive, negative = int(target.sum()), int((~target).sum())
        if not positive or not negative:
            raise ValueError(f"Calibration domain lacks a class: {domain}")
        parts.append(np.where(target, 1 / positive, 1 / negative) / (2 * len(domains)))
    return np.concatenate(parts)


def weighted_nll(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    if (logits.shape != labels.shape or logits.shape != weights.shape
            or not np.isfinite(logits).all() or not np.isfinite(weights).all()):
        raise ValueError("Invalid calibration arrays")
    return float(np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits)))


def fit_calibrators(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Fit monotone logit maps; (scale, bias) leaves ranking unchanged."""
    target = labels.astype(np.float64)
    baseline = weighted_nll(logits, target, weights)
    temperature = minimize_scalar(
        lambda log_t: weighted_nll(logits * np.exp(-log_t), target, weights),
        bounds=(-5.0, 5.0), method="bounded",
    )
    if not temperature.success:
        raise RuntimeError("Temperature fit did not converge")
    temperature_scale = float(np.exp(-temperature.x))
    platt = minimize(
        lambda params: weighted_nll(logits * np.exp(params[0]) + params[1], target, weights),
        x0=np.asarray([float(np.log(temperature_scale)), 0.0]),
        bounds=((-5.0, 5.0), (-10.0, 10.0)), method="L-BFGS-B",
    )
    if not platt.success:
        raise RuntimeError("Platt fit did not converge")
    maps = {
        "raw": (1.0, 0.0),
        "temperature": (temperature_scale, 0.0),
        "platt": (float(np.exp(platt.x[0])), float(platt.x[1])),
    }
    if any(weighted_nll(logits * scale + bias, target, weights) > baseline + 1e-6
           for scale, bias in maps.values()):
        raise RuntimeError("Calibration increased fitting loss")
    return maps


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Report probability quality and a fixed, non-selected acceptance grid."""
    target = labels.astype(bool)
    probability = np.asarray(probabilities, dtype=np.float64)
    if (len(target) == 0 or target.shape != probability.shape
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("Invalid probability evaluation arrays")
    bins = np.array_split(np.argsort(probability, kind="stable"), min(10, len(target)))
    ece = sum(
        len(indices) / len(target) * abs(float(probability[indices].mean()) - float(target[indices].mean()))
        for indices in bins if len(indices)
    )
    decisions = probability >= 0.5
    acceptance = {}
    for threshold in (0.5, 0.7, 0.9, 0.95):
        accepted = probability >= threshold
        acceptance[str(threshold)] = {
            "accepted": int(accepted.sum()),
            "positive_accepted": int((accepted & target).sum()),
            "negative_accepted": int((accepted & ~target).sum()),
        }
    return {
        "rows": len(target),
        "positives": int(target.sum()),
        "negatives": int((~target).sum()),
        "auc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
        "brier": float(np.mean((probability - target) ** 2)),
        "ece_equal_count_10": float(ece),
        "accuracy_at_0.5": float(np.mean(decisions == target)),
        "false_positives_at_0.5": int((decisions & ~target).sum()),
        "false_negatives_at_0.5": int((~decisions & target).sum()),
        "acceptance": acceptance,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = root / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=state / "world_multidomain_mlp_v1.npz")
    parser.add_argument("--source-report", type=Path, default=state / "world_multiturn_scope_v3.json")
    parser.add_argument("--feature-dir", type=Path, default=state / "world_multiturn_features_v2")
    parser.add_argument("--when2call-report", type=Path)
    parser.add_argument("--when2call-feature-dir", type=Path)
    parser.add_argument("--report", type=Path, default=state / "world_task_calibration_v1.json")
    args = parser.parse_args()
    if args.report.exists() or bool(args.when2call_report) != bool(args.when2call_feature_dir):
        parser.error("Use a fresh report path and pass both When2Call paths or neither")

    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    features, labels = load_features(args.feature_dir, source)
    domains = DOMAINS
    evaluation = EVALUATION
    when2call_sha256 = ""
    if args.when2call_report:
        extra_bytes = args.when2call_report.read_bytes()
        extra = json.loads(extra_bytes)
        if (extra["purpose"] != "research_only_grouped_call_readiness_features"
                or extra["model"] != source["model"] or extra["revision"] != source["revision"]
                or set(extra["splits"]) != {"train", "calibration", "test"}):
            raise ValueError("When2Call source does not match the frozen backbone")
        manifest = {
            "rows": {key: value["rows"] for key, value in extra["splits"].items()},
            "excluded_overlong": {key: value["overlong"] for key, value in extra["splits"].items()},
            "feature_checkpoint_sha256": {
                key: value["checkpoint_sha256"] for key, value in extra["splits"].items()
            },
        }
        extra_features, extra_labels = load_features(args.when2call_feature_dir, manifest)
        features.update({f"when2call_{key}": value for key, value in extra_features.items()})
        labels.update({f"when2call_{key}": value for key, value in extra_labels.items()})
        domains = (*domains, "when2call")
        evaluation = (*evaluation, "when2call_test")
        when2call_sha256 = hashlib.sha256(extra_bytes).hexdigest()

    with np.load(args.artifact, allow_pickle=False) as saved:
        weights = {key: saved[key] for key in ("w1", "b1", "w2", "b2", "mean", "scale")}
        if (str(saved["source_report_sha256"].item()) != hashlib.sha256(source_bytes).hexdigest()
                or str(saved["when2call_report_sha256"].item()) != when2call_sha256):
            raise ValueError("Head artifact does not match the pinned feature sources")
    calibration_logits = np.concatenate([
        infer_logits(features[f"{domain}_calibration"], weights) for domain in domains
    ])
    calibration_labels = np.concatenate([labels[f"{domain}_calibration"] for domain in domains])
    calibration_weights = balanced_calibration_weights(labels, domains)
    maps = fit_calibrators(calibration_logits, calibration_labels, calibration_weights)
    report = {
        "purpose": "research_only_frozen_world_task_head_calibration",
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "when2call_report_sha256": when2call_sha256 or None,
        "fit_partition": [f"{domain}_calibration" for domain in domains],
        "test_partition": list(evaluation),
        "selection": "Calibrators fit on calibration only; no test-based selection or runtime promotion",
        "limitation": "The frozen head was already selected on these calibration partitions. "
                      "This is exploratory post-hoc calibration, not independent validation, "
                      "runtime authorization evidence, or Jev parity.",
        "calibrators": {},
        "metrics": {},
    }
    for name, (scale, bias) in maps.items():
        report["calibrators"][name] = {"scale": scale, "bias": bias}
        report["metrics"][name] = {
            partition: calibration_metrics(
                labels[partition], expit(infer_logits(features[partition], weights) * scale + bias),
            )
            for partition in evaluation
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "calibrators": report["calibrators"],
        "test": {name: report["metrics"][name].get("when2call_test", report["metrics"][name]["heliox"])
                 for name in maps},
    }, indent=2))


if __name__ == "__main__":
    main()

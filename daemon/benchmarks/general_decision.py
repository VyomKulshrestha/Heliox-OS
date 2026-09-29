"""Public-data contract for a shared, candidate-conditioned decision scorer.

No task names or option vocabulary are built into the model. This module is
research-only and never grants execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET = "ZefanCai/Open-Jev"
DATASET_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
DATASET_CONFIG = "release-v2-redistributable"
KINDS = frozenset({"noul", "choice", "score"})
PROMPT_STYLES = frozenset({"candidate_v1", "full_options_v2"})


def licensed_row(row: dict) -> bool:
    """Exclude records whose per-row provenance leaves source rights unresolved."""
    metadata = json.loads(row["metadata_json"])
    license_text = str(metadata.get("provenance", {}).get("license", ""))
    return license_text.startswith("CC0-1.0") and "not verified" not in license_text.lower()


def validate_decision(kind: str, question: str, options: list[str]) -> None:
    if kind not in KINDS or not isinstance(question, str) or not question.strip():
        raise ValueError("Invalid decision question or kind")
    if not isinstance(options, list) or not 2 <= len(options) <= 255:
        raise ValueError("Decision needs 2-255 options")
    if (any(not isinstance(value, str) or not value.strip() for value in options)
            or len(options) != len(set(options))):
        raise ValueError("Decision options must be unique nonempty strings")
    if kind == "noul" and options != ["no", "yes"]:
        raise ValueError("Noul target order must be no, yes")


def validate_row(row: dict) -> None:
    kind = row["kind"]
    options = row["options"]
    target = row["target"]
    validate_decision(kind, row["question"], options)
    if len(target) != len(options) or any(not math.isfinite(value) or value < 0 for value in target):
        raise ValueError("Invalid target distribution")
    if not math.isclose(sum(target), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("Target distribution must sum to one")


def candidate_prompts(row: dict, *, style: str = "candidate_v1") -> list[str]:
    """Render only input fields; never include labels, source, or row IDs."""
    if style not in PROMPT_STYLES:
        raise ValueError("Unknown decision prompt style")
    state = json.loads(row["state_json"])
    rendered = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    prefix = f"Context:\n{rendered}\n\nQuestion: {row['question']}\n"
    if row["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    if style == "full_options_v2":
        choices = "\n".join(f"{index + 1}. {option}" for index, option in enumerate(row["options"]))
        prefix += f"Available answers:\n{choices}\n"
    return [prefix + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
            for option in row["options"]]


def select_balanced(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministically round-robin source/kind buckets without replacement."""
    if limit < 1:
        raise ValueError("Sample limit must be positive")
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["source"], row["kind"])].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row["id"])
        rng.shuffle(bucket)
    result: list[dict] = []
    keys = sorted(buckets)
    while keys and len(result) < limit:
        for key in keys:
            if buckets[key] and len(result) < limit:
                result.append(buckets[key].pop())
        keys = [key for key in keys if buckets[key]]
    return result


def load_split(split: str, limit: int, seed: int) -> tuple[list[dict], dict]:
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    filename = f"data/{DATASET_CONFIG}/{split}-00000-of-00001.parquet"
    path = Path(hf_hub_download(DATASET, filename, repo_type="dataset",
                                revision=DATASET_REVISION, local_files_only=True))
    columns = ("id", "group_id", "source", "kind", "question", "options", "target",
               "state_json", "metadata_json")
    all_rows = parquet.read_table(path, columns=list(columns)).to_pylist()
    accepted = []
    rejected_license = 0
    for row in all_rows:
        if not licensed_row(row):
            rejected_license += 1
            continue
        validate_row(row)
        accepted.append(row)
    selected = select_balanced(accepted, limit, seed)
    return selected, {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_rows": len(all_rows),
        "licensed_rows": len(accepted),
        "excluded_unverified_license": rejected_license,
        "selected_rows": len(selected),
    }


def verify_disjoint(splits: dict[str, list[dict]]) -> None:
    groups = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    for left in groups:
        for right in groups:
            if left < right and groups[left] & groups[right]:
                raise ValueError(f"Group leakage between {left} and {right}")


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if logits.ndim != 1 or not len(logits) or not np.all(np.isfinite(logits)):
        raise ValueError("Decision logits must be a finite vector")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    scaled = logits / temperature
    shifted = scaled - max(scaled)
    values = np.exp(shifted)
    return values / sum(values)


def ranked_probability_score(forecast, target):
    """Normalized cumulative-probability error for ordered categories."""
    if forecast.ndim != 1 or target.shape != forecast.shape or len(target) < 2:
        raise ValueError("Ordinal probabilities and targets must be matching vectors")
    difference = forecast[:-1] - target[:-1]
    if type(forecast).__module__.split(".", maxsplit=1)[0] == "torch":
        import torch

        cumulative = torch.cumsum(difference, dim=0)
        return cumulative.square().mean()
    cumulative = np.cumsum(difference)
    return np.square(cumulative).mean()


def selective_risk_metrics(selection_scores: list[float], correct: list[bool], *,
                           selection_score: str = "maximum_candidate_probability") -> dict:
    """Measure whether a score ranks correct decisions ahead of errors.

    Equal-score groups use expected risk under a random ordering, so ties do
    not depend on input order. This is a descriptive benchmark metric, not a
    runtime authorization threshold.
    """
    scores = np.asarray(selection_scores, dtype=np.float64)
    outcomes = np.asarray(correct)
    if (
        scores.ndim != 1
        or outcomes.ndim != 1
        or not len(scores)
        or scores.shape != outcomes.shape
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isin(outcomes, [False, True]).all()
    ):
        raise ValueError("Selective-risk scores and correctness labels are invalid")

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    errors = (~outcomes.astype(bool))[order].astype(np.float64)
    curve_risks = []
    retained = 0
    prior_errors = 0.0
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and ordered_scores[end] == ordered_scores[index]:
            end += 1
        group_size = end - index
        group_errors = float(errors[index:end].sum())
        for within_group in range(1, group_size + 1):
            expected_errors = prior_errors + within_group * group_errors / group_size
            curve_risks.append(expected_errors / (retained + within_group))
        prior_errors += group_errors
        retained += group_size
        index = end

    risk_at_coverage = {}
    risks = np.asarray(curve_risks, dtype=np.float64)
    for coverage in (0.5, 0.8, 0.9):
        count = min(len(scores), math.ceil(coverage * len(scores)))
        risk_at_coverage[f"{coverage:.1f}"] = {
            "requested_coverage": coverage,
            "actual_coverage": count / len(scores),
            "retained_decisions": count,
            "selective_risk": float(risks[count - 1]),
        }
    return {
        "aurc": float(risks.mean()),
        "selection_score": selection_score,
        "tie_handling": "expected risk under random ordering within equal-score groups",
        "risk_at_coverage": risk_at_coverage,
    }


def correctness_features(probabilities_: np.ndarray) -> np.ndarray:
    """Summarize any option distribution for a task-agnostic correctness scorer."""
    values = np.asarray(probabilities_, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not math.isclose(float(values.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5)):
        raise ValueError("Correctness features need a normalized probability vector")
    ranked = np.sort(values)[::-1]
    entropy = -float(np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))
    normalized_entropy = entropy / math.log(len(values))
    return np.asarray([ranked[0], ranked[0] - ranked[1], normalized_entropy,
                       math.log(len(values))], dtype=np.float64)


def fit_correctness_scorer(features: np.ndarray, correct: list[bool], *, l2: float = 0.01) -> dict:
    """Fit a regularized logistic correctness scorer on held-out model outputs."""
    inputs = np.asarray(features, dtype=np.float64)
    labels = np.asarray(correct)
    if (inputs.ndim != 2 or not len(inputs) or labels.shape != (len(inputs),)
            or not inputs.shape[1] or not np.isfinite(inputs).all()
            or not np.isin(labels, [False, True]).all()
            or len(np.unique(labels)) != 2 or not math.isfinite(l2) or l2 < 0):
        raise ValueError("Correctness training data must contain finite features and both labels")

    labels = labels.astype(np.float64)
    mean = inputs.mean(axis=0)
    scale = inputs.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (inputs - mean) / scale

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias = parameters[:-1], parameters[-1]
        logits = normalized @ weights + bias
        predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        loss = np.logaddexp(0.0, logits) - labels * logits
        value = float(loss.mean() + 0.5 * l2 * (weights @ weights))
        residual = (predicted - labels) / len(labels)
        gradient = np.concatenate((normalized.T @ residual + l2 * weights,
                                   np.asarray([residual.sum()])))
        return value, gradient

    from scipy.optimize import minimize

    result = minimize(objective, np.zeros(inputs.shape[1] + 1), jac=True,
                      method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Correctness scorer optimization failed: {result.message}")
    return {"schema_version": 1, "mean": mean, "scale": scale,
            "weights": result.x[:-1], "bias": float(result.x[-1]), "l2": float(l2)}


def predict_correctness(features: np.ndarray, scorer: dict) -> np.ndarray:
    """Return estimated correctness probabilities from a fitted scorer."""
    inputs = np.asarray(features, dtype=np.float64)
    mean = np.asarray(scorer.get("mean"), dtype=np.float64)
    scale = np.asarray(scorer.get("scale"), dtype=np.float64)
    weights = np.asarray(scorer.get("weights"), dtype=np.float64)
    bias = scorer.get("bias")
    if (inputs.ndim != 2 or mean.ndim != 1 or scale.shape != mean.shape
            or weights.shape != mean.shape or inputs.shape[1] != len(mean)
            or not len(inputs) or not np.isfinite(inputs).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(weights).all()
            or not isinstance(bias, (float, int)) or not math.isfinite(bias)):
        raise ValueError("Correctness scorer or features are invalid")
    logits = ((inputs - mean) / scale) @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def correctness_metrics(scores: list[float], correct: list[bool], *,
                        selection_score: str = "estimated_probability_of_top_choice_correct") -> dict:
    """Evaluate correctness probabilities and their selective-risk ranking."""
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(correct)
    if (values.ndim != 1 or labels.shape != values.shape or not len(values)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
            or not np.isin(labels, [False, True]).all()):
        raise ValueError("Correctness scores or labels are invalid")
    labels = labels.astype(bool)
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    result = {
        "rows": len(values),
        "observed_accuracy": float(labels.mean()),
        "mean_predicted_correctness": float(values.mean()),
        "brier": float(np.mean((values - labels) ** 2)),
        "log_loss": float(np.mean(-(labels * np.log(clipped)
                                    + (~labels) * np.log1p(-clipped)))),
        "selective_prediction": selective_risk_metrics(
            values.tolist(), labels.tolist(), selection_score=selection_score),
    }
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives and negatives:
        order = np.argsort(values, kind="stable")
        sorted_scores = values[order]
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        result["auroc"] = float((ranks[labels].sum() - positives * (positives + 1) / 2)
                                / (positives * negatives))
    else:
        result["auroc"] = None
    return result


def paired_aurc_bootstrap_delta(raw_scores: list[float], learned_scores: list[float],
                                correct: list[bool], *, replicates: int = 2000,
                                seed: int = 0) -> dict:
    """Estimate uncertainty in learned-minus-raw AURC on paired held-out rows."""
    raw = np.asarray(raw_scores, dtype=np.float64)
    learned = np.asarray(learned_scores, dtype=np.float64)
    labels = np.asarray(correct)
    if (raw.ndim != 1 or learned.shape != raw.shape or labels.shape != raw.shape or not len(raw)
            or not np.isfinite(raw).all() or not np.isfinite(learned).all()
            or np.any(raw < 0) or np.any(raw > 1) or np.any(learned < 0) or np.any(learned > 1)
            or not np.isin(labels, [False, True]).all()
            or type(replicates) is not int or replicates < 100 or type(seed) is not int):
        raise ValueError("Paired AURC bootstrap inputs are invalid")
    labels = labels.astype(bool)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(raw), size=len(raw))
        sample_labels = labels[sample].tolist()
        deltas[index] = (
            selective_risk_metrics(learned[sample].tolist(), sample_labels)["aurc"]
            - selective_risk_metrics(raw[sample].tolist(), sample_labels)["aurc"]
        )
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "delta_aurc_95_percentile_interval": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "bootstrap_fraction_learned_lower_aurc": float(np.mean(deltas < 0)),
    }


def paired_mean_bootstrap_delta(candidate: list[float], baseline: list[float], *,
                                replicates: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the paired mean candidate-minus-baseline difference."""
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if (candidate_values.ndim != 1 or baseline_values.shape != candidate_values.shape
            or not len(candidate_values) or not np.isfinite(candidate_values).all()
            or not np.isfinite(baseline_values).all()
            or type(replicates) is not int or replicates < 100 or type(seed) is not int):
        raise ValueError("Paired mean bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(candidate_values), size=len(candidate_values))
        deltas[index] = float(np.mean(candidate_values[sample] - baseline_values[sample]))
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "mean_delta_95_percentile_interval": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "bootstrap_fraction_positive_delta": float(np.mean(deltas > 0)),
    }


def grouped_mean_bootstrap_delta(group_ids: list[str], candidate: list[float],
                                 baseline: list[float], *,
                                 replicates: int = 2000, seed: int = 0) -> dict:
    """Bootstrap paired row deltas by independent group, preserving clusters."""
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if (not group_ids or len(group_ids) != len(candidate_values)
            or baseline_values.shape != candidate_values.shape
            or any(not isinstance(group, str) or not group for group in group_ids)
            or not np.isfinite(candidate_values).all()
            or not np.isfinite(baseline_values).all()
            or type(replicates) is not int or replicates < 100 or type(seed) is not int):
        raise ValueError("Grouped paired bootstrap inputs are invalid")

    grouped_indices: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        grouped_indices.setdefault(group, []).append(index)
    groups = list(grouped_indices.values())
    if len(groups) < 2:
        raise ValueError("Grouped paired bootstrap needs at least two independent groups")

    row_deltas = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        sampled_indices = [index for group_index in sampled_groups for index in groups[group_index]]
        deltas[replicate] = float(np.mean(row_deltas[sampled_indices]))
    return {
        "independent_groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "mean_delta": float(np.mean(row_deltas)),
        "mean_delta_95_percentile_interval": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "bootstrap_fraction_positive_delta": float(np.mean(deltas > 0)),
        "resampling_unit": "group_id",
        "aggregation": "row-weighted within each sampled group set",
    }


def format_answer(kind: str, options: list[str], logits: np.ndarray,
                  temperature: float = 1.0) -> dict:
    """Convert model scores to a typed decision without executing anything."""
    if (kind not in KINDS or not options
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or len(options) != len(set(options))):
        raise ValueError("Unknown kind or duplicate/empty option list")
    if kind == "noul" and options != ["no", "yes"]:
        raise ValueError("Noul options must be no, yes")
    values = probabilities(logits, temperature)
    if len(values) != len(options):
        raise ValueError("Score count does not match the declared options")
    if kind == "noul":
        return {"type": kind, "yes_probability": float(values[1])}
    distribution = {option: float(value) for option, value in zip(options, values, strict=True)}
    selected = int(np.argmax(values))
    if kind == "choice":
        return {"type": kind, "choice": options[selected], "probabilities": distribution,
                "max_probability": float(values[selected])}
    return {"type": kind, "expected_index": float(np.arange(len(values)) @ values),
            "most_likely": options[selected], "probabilities": distribution}


def metrics(rows: list[dict], logits: list[np.ndarray], temperature: float) -> dict:
    if len(rows) != len(logits) or not rows:
        raise ValueError("Rows and logits must have equal nonzero length")
    per_row = []
    for row, values in zip(rows, logits, strict=True):
        target = np.asarray(row["target"], dtype=np.float64)
        predicted = probabilities(values, temperature)
        if len(target) != len(predicted):
            raise ValueError("Prediction and target option counts differ")
        item = {
            "source": row["source"], "kind": row["kind"],
            "top1": int(np.argmax(target) == np.argmax(predicted)),
            "gold_index": int(np.argmax(target)),
            "predicted_index": int(np.argmax(predicted)),
            "selection_score": float(predicted.max()),
            "nll": float(-target @ np.log(np.clip(predicted, 1e-12, 1.0))),
            "brier": float(np.mean((predicted - target) ** 2)),
        }
        if row["kind"] == "score":
            positions = np.arange(len(target))
            item["rank_error"] = float(abs(positions @ predicted - positions @ target))
            item["ranked_probability_score"] = float(ranked_probability_score(predicted, target))
        per_row.append(item)

    def aggregate(items: list[dict]) -> dict:
        result = {
            "rows": len(items),
            "top1_accuracy": float(np.mean([item["top1"] for item in items])),
            "mean_nll": float(np.mean([item["nll"] for item in items])),
            "mean_brier": float(np.mean([item["brier"] for item in items])),
            "selective_prediction": selective_risk_metrics(
                [item["selection_score"] for item in items],
                [bool(item["top1"]) for item in items],
            ),
        }
        if all(item["kind"] == "score" for item in items):
            result["mean_rank_error"] = float(np.mean([item["rank_error"] for item in items]))
            result["mean_ranked_probability_score"] = float(np.mean(
                [item["ranked_probability_score"] for item in items]))
        if all(item["kind"] == "noul" for item in items):
            true_positive = sum(item["gold_index"] == 1 and item["predicted_index"] == 1 for item in items)
            true_negative = sum(item["gold_index"] == 0 and item["predicted_index"] == 0 for item in items)
            false_positive = sum(item["gold_index"] == 0 and item["predicted_index"] == 1 for item in items)
            false_negative = sum(item["gold_index"] == 1 and item["predicted_index"] == 0 for item in items)
            result["confusion"] = {"true_positive": true_positive, "true_negative": true_negative,
                                   "false_positive": false_positive, "false_negative": false_negative}
            positives = true_positive + false_negative
            negatives = true_negative + false_positive
            result["balanced_accuracy"] = (0.5 * (true_positive / positives + true_negative / negatives)
                                           if positives and negatives else None)
        return result

    return {**aggregate(per_row),
            "by_kind": {kind: aggregate([item for item in per_row if item["kind"] == kind])
                        for kind in sorted({item["kind"] for item in per_row})},
            "by_source": {source: aggregate([item for item in per_row if item["source"] == source])
                          for source in sorted({item["source"] for item in per_row})}}


def uniform_baseline(rows: list[dict]) -> dict:
    """Score a label-blind equal-probability comparator for each option set."""
    return metrics(rows, [np.zeros(len(row["options"]), dtype=np.float64) for row in rows], 1.0)

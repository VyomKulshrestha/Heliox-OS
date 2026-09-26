"""Compare Jev with a frozen local scorer on the same public held-out calls.

Evaluation only: Jev answers are never used to fit or calibrate the scorer.
Only aggregate outcomes are saved; the API key stays in the local environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import httpx
import numpy as np

from benchmarks.general_decision import probabilities
from benchmarks.scopejudge_general import load_scope_parts, scope_metrics
from scripts.compare_general_jev import query_jev
from scripts.evaluate_jev_scope import _key


def load_matched_rows(dataset: Path, model_report: Path, evaluation: Path,
                      checkpoint: Path) -> tuple[list[dict], list[np.ndarray], dict]:
    report = json.loads(model_report.read_text(encoding="utf-8"))
    saved = json.loads(evaluation.read_text(encoding="utf-8"))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["training_checkpoint_sha256"]:
        raise ValueError("Training checkpoint differs from the frozen report")
    scope = report["scopejudge"]
    if scope["history_calls"] != 0:
        raise ValueError("This comparison requires a no-history local control")
    parts, manifest = load_scope_parts(dataset, seed=report["seed"],
                                       limits=scope["selected_rows"])
    if manifest != scope or saved.get("schema_version") != 1:
        raise ValueError("Dataset or evaluation manifest differs from the frozen report")
    ids = saved["ids"]
    logits = [np.asarray(value, dtype=np.float64) for value in saved["logits"]]
    by_id = {row["id"]: row for row in parts["scope_test"]}
    if (len(ids) != len(set(ids)) or len(ids) != len(logits)
            or len(ids) != report["evaluated_rows"]["scope_test"]
            or any(case_id not in by_id for case_id in ids)
            or any(value.shape != (2,) or not np.isfinite(value).all() for value in logits)):
        raise ValueError("Evaluation logits do not match held-out ScopeJudge rows")
    rows = [by_id[case_id] for case_id in ids]
    measured = scope_metrics(rows, logits, report["calibration_temperature"])
    for key, value in measured.items():
        if not np.isclose(value, report["trained_scope"][key]):
            raise ValueError("Evaluation logits disagree with the frozen report")
    return rows, logits, report


def balanced_sample(rows: list[dict], count: int, seed: int) -> list[int]:
    """Preselect solely from expert labels and IDs, never model predictions."""
    if count < 1:
        raise ValueError("Per-class sample size must be positive")
    positives = [index for index, row in enumerate(rows) if row["target"][1] == 1.0]
    negatives = [index for index, row in enumerate(rows) if row["target"][1] == 0.0]
    size = min(count, len(positives), len(negatives))
    if not size:
        raise ValueError("Held-out sample must contain both expert label classes")
    rng = random.Random(seed)
    return sorted(rng.sample(positives, size) + rng.sample(negatives, size))


def summarize(labels: list[bool], local: list[float], jev: list[float]) -> dict:
    if not labels or len(labels) != len(local) or len(labels) != len(jev):
        raise ValueError("Paired predictions must match nonempty expert labels")

    def metrics(values: list[float]) -> dict:
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Decision probabilities must be finite and within [0, 1]")
        predicted = [value >= 0.5 for value in values]
        positives = sum(labels)
        negatives = len(labels) - positives
        if not positives or not negatives:
            raise ValueError("Both expert label classes are required")
        true_positive = sum(guess and gold for guess, gold in zip(predicted, labels, strict=True))
        true_negative = sum(not guess and not gold for guess, gold in zip(predicted, labels, strict=True))
        return {"agreement_with_expert": (true_positive + true_negative) / len(labels),
                "balanced_accuracy": (true_positive / positives + true_negative / negatives) / 2,
                "false_allowances": negatives - true_negative,
                "false_warnings": positives - true_positive}

    return {"rows": len(labels), "expert_in_scope": sum(labels),
            "expert_out_of_scope": len(labels) - sum(labels),
            "local": metrics(local), "jev": metrics(jev),
            "local_jev_agreement": sum((a >= 0.5) == (b >= 0.5)
                                       for a, b in zip(local, jev, strict=True)) / len(labels),
            "mean_absolute_probability_gap": float(np.mean(np.abs(np.asarray(local) - jev)))}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=root.parent / ".env")
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a fresh path")
    rows, logits, model_report = load_matched_rows(
        args.dataset, args.model_report, args.evaluation, args.checkpoint)
    indices = balanced_sample(rows, args.per_class, args.seed)
    labels = [rows[index]["target"][1] == 1.0 for index in indices]
    local = [float(probabilities(logits[index], model_report["calibration_temperature"])[1])
             for index in indices]
    key = _key(args.env_file)
    jev, models, cost = [], set(), 0.0
    with httpx.Client(timeout=45.0, trust_env=True) as client:
        for position, index in enumerate(indices, 1):
            answer, charge, model = query_jev(client, key, rows[index])
            probability = float(answer["noul"])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Jev returned an invalid Noul probability")
            jev.append(probability)
            cost += charge
            models.add(model)
            if position % 10 == 0 or position == len(indices):
                print(f"Evaluated {position}/{len(indices)} paired public calls", flush=True)
    result = {"purpose": "evaluation_only_matched_public_scope_calls",
              "dataset": model_report["scopejudge"],
              "local_checkpoint_sha256": model_report["training_checkpoint_sha256"],
              "local_adapter_sha256": model_report["adapter_sha256"],
              "evaluation_sha256": hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
              "selection_seed": args.seed,
              "selected_ids_sha256": hashlib.sha256("\n".join(rows[index]["id"] for index in indices)
                                                   .encode()).hexdigest(),
              "jev_models": sorted(models), "jev_cost_usd": cost,
              "comparison": summarize(labels, local, jev),
              "limitation": "Previously inspected public split; evaluation only, not Jev parity, "
                            "live desktop evidence, or action authorization."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()

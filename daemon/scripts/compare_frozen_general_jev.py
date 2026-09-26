"""Evaluate a frozen public-data decision head against live Jev on matched OOD rows.

Jev outputs are evaluation-only. This command never trains or calibrates from
them and writes only aggregate comparisons, not response-level teacher data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import format_answer, metrics, uniform_baseline  # noqa: E402
from scripts.compare_general_jev import _key, compare, query_jev  # noqa: E402
from scripts.extract_general_frozen_features import MODEL, REVISION  # noqa: E402
from scripts.fit_general_frozen_head import load_experiment, logits_for_rows  # noqa: E402


def compatible_indices(rows: list[dict], limit: int) -> list[int]:
    if not 1 <= limit <= 128:
        raise ValueError("Comparison limit must be 1–128")
    selected = [index for index, row in enumerate(rows)
                if row["kind"] != "score" or len(row["options"]) <= 10][:limit]
    if len(selected) != limit:
        raise ValueError("Not enough compatible licensed OOD rows")
    return selected


def local_predictions(rows: list[dict], features: tuple[np.ndarray, np.ndarray],
                      weights: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                      temperature: float) -> list[dict]:
    matrix, offsets = features
    if (matrix.shape[1] != len(weights) - 1 or mean.shape != (matrix.shape[1],)
            or scale.shape != mean.shape or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("Head and frozen feature dimensions do not match")
    logits = logits_for_rows(rows, ((matrix - mean) / scale, offsets), weights)
    return [{"answer": format_answer(row["kind"], row["options"], values,
                                     temperature=temperature)}
            for row, values in zip(rows, logits, strict=True)]


def run(extraction_report: Path, feature_dir: Path, fit_report: Path,
        weight_path: Path, env_file: Path, output: Path, limit: int) -> dict:
    if output.exists():
        raise FileExistsError("Comparison report already exists")
    selection, rows, features = load_experiment(extraction_report, feature_dir)
    fit = json.loads(fit_report.read_text(encoding="utf-8"))
    if (fit.get("purpose") != "unpromoted_frozen_public_general_decision_head"
            or fit.get("model") != MODEL or fit.get("revision") != REVISION
            or fit.get("extraction_report_sha256") != hashlib.sha256(
                extraction_report.read_bytes()).hexdigest()
            or fit.get("weight_sha256") != hashlib.sha256(weight_path.read_bytes()).hexdigest()):
        raise ValueError("Fitted head does not match the public feature extraction")
    with np.load(weight_path, allow_pickle=False) as archive:
        weights = archive["weights"].astype(np.float64)
        mean = archive["mean"].astype(np.float64)
        scale = archive["scale"].astype(np.float64)
        if str(archive["model_revision"]) != REVISION:
            raise ValueError("Fitted head uses a different model revision")
    indices = compatible_indices(rows["ood"], limit)
    selected_rows = [rows["ood"][index] for index in indices]
    local = local_predictions(rows["ood"], features["ood"], weights, mean, scale,
                              fit["temperature"])
    selected_local = [local[index] for index in indices]
    key = _key(env_file)
    jev, cost, versions = [], 0.0, set()
    with httpx.Client(timeout=45.0, trust_env=True) as client:
        for index, row in enumerate(selected_rows, start=1):
            answer, charge, version = query_jev(client, key, row)
            jev.append(answer)
            cost += charge
            versions.add(version)
            print(json.dumps({"event": "evaluated", "rows": index, "total": limit}), flush=True)
    logits = []
    for row, prediction in zip(selected_rows, selected_local, strict=True):
        answer = prediction["answer"]
        if row["kind"] == "noul":
            probability = np.clip(answer["yes_probability"], 1e-12, 1 - 1e-12)
            logits.append(np.log([1 - probability, probability]))
        else:
            logits.append(np.log([max(answer["probabilities"][option], 1e-12)
                                  for option in row["options"]]))
    report = {
        "purpose": "evaluation_only_no_teacher_training",
        "model": MODEL, "revision": REVISION,
        "fitted_head_sha256": fit["weight_sha256"],
        "dataset_revision": selection["dataset_revision"],
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in selected_rows).encode()).hexdigest(),
        "calls": len(selected_rows), "cost_usd": cost, "jev_models": sorted(versions),
        "uniform": uniform_baseline(selected_rows)["by_kind"],
        "local_gold": metrics(selected_rows, logits, 1.0)["by_kind"],
        "comparison": compare(selected_rows, selected_local, jev),
        "limitation": "Public OOD comparison only; no Jev-output training, desktop-safety proof, "
                      "or execution authority.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-report", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1.json")
    parser.add_argument("--feature-dir", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1")
    parser.add_argument("--fit-report", type=Path,
                        default=root / "benchmarks/qwen2b_frozen_general_v1.json")
    parser.add_argument("--weights", type=Path,
                        default=root / "benchmarks/.bench-state/qwen2b_frozen_general_v1/head.npz")
    parser.add_argument("--env-file", type=Path, default=root.parent / ".env")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--report", type=Path,
                        default=root / "benchmarks/qwen2b_frozen_jev_comparison_v1.json")
    args = parser.parse_args()
    report = run(args.extraction_report, args.feature_dir, args.fit_report, args.weights,
                 args.env_file, args.report, args.limit)
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()

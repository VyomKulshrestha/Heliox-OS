"""Evaluate local next-token typed scoring on licensed Open-Jev decision rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.general_decision import (  # noqa: E402
    DATASET,
    DATASET_REVISION,
    load_split,
    metrics,
    uniform_baseline,
)
from pilot.intelligence.native_typed_decision import LocalNextTokenTypedDecisionModel  # noqa: E402
from pilot.intelligence.typed_decision import PINNED_BACKBONES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _definition(row: dict) -> dict:
    if row["kind"] == "noul":
        return {"type": "noul", "instructions": row["question"]}
    if row["kind"] == "choice":
        return {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {option: None for option in row["options"]},
        }
    return {
        "type": "score",
        "instructions": row["question"],
        "criteria": row["options"],
    }


def _probabilities(row: dict, answer: dict) -> np.ndarray:
    if row["kind"] == "noul":
        yes_probability = float(answer["noul"])
        return np.asarray([1.0 - yes_probability, yes_probability], dtype=np.float64)
    if row["kind"] == "choice":
        distribution = answer["probabilities"]
        return np.asarray([distribution[option] for option in row["options"]], dtype=np.float64)
    distribution = answer["probabilities"]
    return np.asarray([distribution[str(index)] for index in range(len(row["options"]))], dtype=np.float64)


def run(*, split: str, row_limit: int, seed: int, model_id: str, max_length: int, output_path: Path) -> dict:
    import torch

    rows, manifest = load_split(split, row_limit, seed)
    engine = LocalNextTokenTypedDecisionModel(model_id, max_length=max_length)
    predictions = []
    latencies = []
    skipped = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, row in enumerate(rows, 1):
        request_state = json.loads(row["state_json"])
        row_started = time.perf_counter()
        try:
            result = engine.score_questions(
                request_state,
                {"decision": _definition(row)},
            )
        except ValueError as exc:
            if "token limit" not in str(exc):
                raise
            skipped.append({"id": row["id"], "reason": "model_token_limit"})
            continue
        latencies.append((time.perf_counter() - row_started) * 1000)
        answer = result["answers"]["decision"]
        prediction = _probabilities(row, answer)
        if prediction.shape != (len(row["target"]),) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid model distribution for row {row['id']}")
        predictions.append(np.log(np.clip(prediction, 1e-12, 1.0)))
        if index % 16 == 0 or index == len(rows):
            print(json.dumps({"event": "scored", "rows": index, "total": len(rows)}), flush=True)

    kept_rows = [row for row in rows if row["id"] not in {item["id"] for item in skipped}]
    if len(kept_rows) != len(predictions) or not kept_rows:
        raise RuntimeError("No valid, length-bounded decisions were evaluated")
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "purpose": "native_next_token_typed_decision_public_data_evaluation",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_manifest": manifest,
        "split": split,
        "seed": seed,
        "selected_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in kept_rows).encode()).hexdigest(),
        "model": model_id,
        "model_revision": PINNED_BACKBONES[model_id],
        "weights": "local pinned base model only; no adapters or provider outputs",
        "scoring": "typed candidate labels scored from one next-token distribution per question; no decoding",
        "prompt_contract": ("v2 string labels; internal answer keys omitted; Noul A-I ratings represent p_yes 0.1-0.9"),
        "calibration": "uncalibrated; temperature=1.0",
        "max_length": max_length,
        "rows_requested": len(rows),
        "rows_scored": len(kept_rows),
        "rows_skipped": skipped,
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "elapsed_seconds": round(elapsed, 3),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "metrics": metrics(kept_rows, predictions, 1.0),
        "uniform_baseline": uniform_baseline(kept_rows),
        "limitation": "Public benchmark evidence does not establish Jev parity or desktop task quality.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    engine.unload()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "calibration", "test", "ood"), default="ood")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--model", choices=tuple(PINNED_BACKBONES), default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/.bench-state/native_next_token_ood_64_seed20260929.json",
    )
    args = parser.parse_args()
    if not 1 <= args.rows <= 256:
        parser.error("--rows must be from 1 to 256")
    if not 64 <= args.max_length <= 4096:
        parser.error("--max-length must be from 64 to 4096")
    report = run(
        split=args.split,
        row_limit=args.rows,
        seed=args.seed,
        model_id=args.model,
        max_length=args.max_length,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "metrics": report["metrics"]["by_kind"],
                "uniform_baseline": report["uniform_baseline"]["by_kind"],
                "median_latency_ms": report["median_latency_ms"],
                "p95_latency_ms": report["p95_latency_ms"],
                "peak_cuda_gib": report["peak_cuda_gib"],
                "rows_scored": report["rows_scored"],
                "rows_skipped": len(report["rows_skipped"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

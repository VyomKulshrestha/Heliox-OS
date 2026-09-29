"""Evaluate a frozen Heliox scorer on JevBench's MIT-licensed original tier.

This is a public-data transfer evaluation only. It never calls Jev, trains on
benchmark rows, writes raw inputs or predictions, or grants execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import metrics  # noqa: E402
from scripts.score_general_decision import score_requests, verify_artifact  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_REVISION = "9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3"
BENCHMARK_SHA256 = "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"
BENCHMARK_URL = (
    "https://raw.githubusercontent.com/fstandhartinger/jevbench/"
    f"{BENCHMARK_REVISION}/datasets/public/original.jsonl"
)
MAX_DATA_BYTES = 1_000_000


def parse_tasks(payload: bytes) -> list[dict]:
    if not payload or len(payload) > MAX_DATA_BYTES:
        raise ValueError("JevBench payload must be nonempty and no larger than 1 MB")
    if hashlib.sha256(payload).hexdigest() != BENCHMARK_SHA256:
        raise ValueError("Pinned JevBench file checksum mismatch")
    tasks = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(tasks) != 72 or len({task.get("id") for task in tasks}) != 72:
        raise ValueError("Pinned JevBench original tier must contain 72 unique tasks")
    if any(task.get("provenance", {}).get("license") != "MIT" for task in tasks):
        raise ValueError("Every JevBench task must have explicit MIT provenance")
    return tasks


def task_to_row(task: dict) -> dict:
    question = task.get("question")
    if not isinstance(question, dict):
        raise ValueError("JevBench task is missing its typed question")
    kind = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria") or {}
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("JevBench task has empty instructions")

    if kind == "noul":
        options = ["no", "yes"]
        instructions += (
            f"\nDecision criteria: no means {criteria.get('false', 'the condition is not established')}; "
            f"yes means {criteria.get('true', 'the condition is established')}."
        )
        expected = task.get("expected")
        if expected not in options:
            raise ValueError("JevBench Noul target must be no or yes")
        target_index = options.index(expected)
    elif kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("JevBench Choice criteria must be a mapping of at least two labels")
        labels = list(criteria)
        options = [
            f"{label}: {description}" if isinstance(description, str) and description.strip()
            else str(label)
            for label, description in criteria.items()
        ]
        expected = task.get("expected")
        if expected not in labels:
            raise ValueError("JevBench Choice target is absent from its criteria")
        target_index = labels.index(expected)
    elif kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("JevBench Score criteria must be an ordered rubric")
        options = [str(item) for item in criteria]
        expected = task.get("expected")
        if type(expected) is not int or not 0 <= expected < len(options):
            raise ValueError("JevBench Score target must identify a rubric level")
        target_index = expected
    else:
        raise ValueError(f"Unsupported JevBench question type: {kind!r}")

    if any(not option.strip() for option in options) or len(options) != len(set(options)):
        raise ValueError("JevBench rubric contains empty or duplicate options")
    return {
        "id": str(task["id"]),
        "group_id": str(task["group"]),
        "source": "jevbench-mit-original",
        "kind": kind,
        "question": instructions,
        "options": options,
        "target": [float(index == target_index) for index in range(len(options))],
        "state_json": json.dumps(task.get("state"), ensure_ascii=False),
    }


def public_payload(row: dict) -> dict:
    """Return model input only; gold labels and IDs stay out of inference."""
    return {
        "state": json.loads(row["state_json"]),
        "kind": row["kind"],
        "question": row["question"],
        "options": row["options"],
    }


def probabilities(row: dict, prediction: dict) -> np.ndarray:
    answer = prediction["answer"]
    if row["kind"] == "noul":
        yes = float(answer["yes_probability"])
        values = np.asarray([1.0 - yes, yes], dtype=np.float64)
    else:
        distribution = answer["probabilities"]
        values = np.asarray([distribution[option] for option in row["options"]], dtype=np.float64)
    if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
            or not np.isclose(values.sum(), 1.0, rtol=1e-5, atol=1e-5)):
        raise ValueError(f"Scorer returned an invalid distribution for row {row['id']}")
    return values


def score_frozen_model(rows: list[dict], output_dir: Path, model_report_path: Path) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_report, _ = verify_artifact(output_dir, model_report_path)
    tokenizer_path = snapshot_download(
        model_report["model"], revision=model_report["revision"], local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    encoded, overlong = encode_rows(
        tokenizer, rows, int(model_report["max_length"]),
        prompt_style=model_report.get("prompt_style", "candidate_v1"),
    )
    if overlong or len(encoded) != len(rows):
        raise ValueError(
            f"Refusing a partial or truncated comparison: {overlong} of {len(rows)} rows exceed "
            f"the frozen {model_report['max_length']}-token prompt limit"
        )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    predictions = score_requests(
        [public_payload(row) for row in rows], output_dir, model_report_path,
    )
    elapsed = time.perf_counter() - started
    row_probabilities = [probabilities(row, prediction)
                         for row, prediction in zip(rows, predictions, strict=True)]
    report_metrics = metrics(rows, [np.log(values) for values in row_probabilities], 1.0)
    correct = int(round(report_metrics["top1_accuracy"] * len(rows)))
    per_kind_correct = {
        kind: int(round(summary["top1_accuracy"] * summary["rows"]))
        for kind, summary in report_metrics["by_kind"].items()
    }
    return {
        "inference": {
            "model": model_report["model"],
            "revision": model_report["revision"],
            "model_report_sha256": hashlib.sha256(model_report_path.read_bytes()).hexdigest(),
            "adapter_sha256": model_report["adapter_sha256"],
            "head_sha256": model_report["head_sha256"],
            "device": torch.cuda.get_device_name(),
            "max_length": model_report["max_length"],
        },
        "metrics": {
            "correct": correct,
            "total": len(rows),
            "accuracy": report_metrics["top1_accuracy"],
            "macro_accuracy": float(np.mean([
                summary["top1_accuracy"] for summary in report_metrics["by_kind"].values()
            ])),
            "mean_brier": report_metrics["mean_brier"],
            "mean_nll": report_metrics["mean_nll"],
            "mean_score_rank_error": report_metrics["by_kind"].get("score", {}).get("mean_rank_error"),
            "per_type": {
                kind: {
                    "correct": per_kind_correct[kind],
                    "total": summary["rows"],
                    "accuracy": summary["top1_accuracy"],
                }
                for kind, summary in report_metrics["by_kind"].items()
            },
            "elapsed_seconds_including_model_load": round(elapsed, 3),
            "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        },
        "correct_by_row": [
            bool(np.argmax(values) == np.argmax(row["target"]))
            for row, values in zip(rows, row_probabilities, strict=True)
        ],
    }


def paired_accuracy_difference(rows: list[dict], candidate: list[bool],
                               baseline: list[bool], *, replicates: int = 5000,
                               seed: int = 20260929) -> dict:
    if (len(rows) != len(candidate) or len(rows) != len(baseline) or not rows
            or any(type(value) is not bool for value in (*candidate, *baseline))
            or type(replicates) is not int or replicates < 100
            or type(seed) is not int):
        raise ValueError("Paired benchmark outcomes must match the nonempty row set")
    group_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group_indices.setdefault(row["group_id"], []).append(index)
    groups = sorted(group_indices)
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in selected for index in group_indices[group]]
        sampled[replicate] = float(differences[indices].mean())
    return {
        "candidate_minus_baseline_accuracy": float(differences.mean()),
        "confidence_level": 0.95,
        "group_bootstrap_replicates": replicates,
        "group_bootstrap_seed": seed,
        "groups_resampled": len(groups),
        "percentile_95_interval": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
    }


def evaluate(output_dir: Path, model_report_path: Path, baseline_output_dir: Path,
             baseline_model_report_path: Path, report_path: Path) -> dict:
    with urlopen(BENCHMARK_URL, timeout=30) as response:
        payload = response.read(MAX_DATA_BYTES + 1)
    tasks = parse_tasks(payload)
    rows = [task_to_row(task) for task in tasks]
    candidate = score_frozen_model(rows, output_dir, model_report_path)
    baseline = score_frozen_model(rows, baseline_output_dir, baseline_model_report_path)

    result = {
        "schema_version": 1,
        "benchmark": "JevBench MIT-licensed original tier",
        "benchmark_revision": BENCHMARK_REVISION,
        "benchmark_sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(rows),
        "gold_labels_in_model_inputs": False,
        "trained_on_evaluation_rows": False,
        "prompt_contract": "Heliox candidate_v1 with JevBench criterion descriptions; score levels kept ordered",
        "heliox": candidate["metrics"],
        "released_open_jev_2b_local": baseline["metrics"],
        "inference": {
            "heliox": candidate["inference"],
            "released_open_jev_2b_local": baseline["inference"],
        },
        "paired_accuracy_difference": paired_accuracy_difference(
            rows, candidate["correct_by_row"], baseline["correct_by_row"],
        ),
        "published_references": {
            "jev": {"correct": 71, "total": 72},
            "open_jev_2b": {"correct": 56, "total": 72},
            "source": "https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md",
        },
        "limitation": (
            "Small public transfer set only; published Jev/Open-Jev numbers use their own "
            "prompt contracts. This does not establish Jev parity, desktop task success, "
            "or safe execution."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "benchmarks/.bench-state/openjev_scale_1024_rps_v1",
    )
    parser.add_argument(
        "--model-report", type=Path,
        default=ROOT / "benchmarks/openjev_scale_1024_rps_v1.json",
    )
    parser.add_argument(
        "--baseline-output-dir", type=Path,
        default=ROOT / "benchmarks/.bench-state/openjev_runtime_candidate_v1",
    )
    parser.add_argument(
        "--baseline-model-report", type=Path,
        default=ROOT / "benchmarks/.bench-state/openjev_runtime_candidate_v1/report.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "benchmarks/.bench-state/jevbench_mit_original_rps_v1/report.json",
    )
    args = parser.parse_args()
    result = evaluate(
        args.output_dir, args.model_report, args.baseline_output_dir,
        args.baseline_model_report, args.report,
    )
    print(json.dumps({
        "heliox": result["heliox"],
        "released_open_jev_2b_local": result["released_open_jev_2b_local"],
        "paired_accuracy_difference": result["paired_accuracy_difference"],
    }, indent=2))


if __name__ == "__main__":
    main()

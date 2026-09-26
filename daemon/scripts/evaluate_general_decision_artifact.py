"""Re-evaluate a frozen local decision adapter on the same public split.

No hosted model calls, training, or action execution. This is useful for
controlled comparisons of research artifacts with different training losses.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import DATASET_REVISION, load_split, metrics  # noqa: E402
from scripts.score_general_decision import score_requests  # noqa: E402


def permute_options(rows: list[dict], seed: int) -> list[dict]:
    """Reorder choice alternatives and labels; preserve ordinal score order."""
    rng = random.Random(seed)
    result = []
    for row in rows:
        copy = dict(row)
        if row["kind"] == "choice":
            order = list(range(len(row["options"])))
            rng.shuffle(order)
            copy["options"] = [row["options"][index] for index in order]
            copy["target"] = [row["target"][index] for index in order]
        result.append(copy)
    return result


def prediction_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match the public split")
    logits = []
    for row, prediction in zip(rows, predictions, strict=True):
        answer = prediction["answer"]
        if row["kind"] == "noul":
            yes = float(answer["yes_probability"])
            values = np.asarray([1 - yes, yes], dtype=float)
        else:
            values = np.asarray([answer["probabilities"][option] for option in row["options"]],
                                dtype=float)
        if (not np.all(np.isfinite(values)) or np.any(values < 0)
                or not np.isclose(values.sum(), 1.0, atol=1e-5)):
            raise ValueError("Invalid frozen prediction probabilities")
        logits.append(np.log(np.clip(values, 1e-12, 1.0)))
    return metrics(rows, logits, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "ood"), default="ood")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--permute-options-seed", type=int,
                        help="Reorder choice alternatives and labels without changing row selection")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 128:
        parser.error("--limit must be 1-128")
    rows, manifest = load_split(args.split, args.limit, args.seed)
    if len(rows) != args.limit:
        raise ValueError("Not enough licensed decisions")
    if args.permute_options_seed is not None:
        rows = permute_options(rows, args.permute_options_seed)
    payloads = [{"state": json.loads(row["state_json"]), "kind": row["kind"],
                 "question": row["question"], "options": row["options"]} for row in rows]
    predictions = score_requests(payloads, args.output_dir, args.model_report)
    source_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    report = {"purpose": "offline_frozen_artifact_comparison",
              "dataset_revision": DATASET_REVISION, "dataset_manifest": manifest,
              "split": args.split, "seed": args.seed, "rows": len(rows),
              "permute_options_seed": args.permute_options_seed,
              "adapter_sha256": source_report["adapter_sha256"],
              "head_sha256": source_report["head_sha256"],
              "metrics": prediction_metrics(rows, predictions),
              "limitation": "Licensed public synthetic decisions; no Jev or desktop-safety proof."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"]["by_kind"], indent=2))


if __name__ == "__main__":
    main()

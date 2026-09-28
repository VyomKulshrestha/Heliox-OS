"""Re-evaluate a frozen local decision adapter on the same public split.

No hosted model calls, training, or action execution. This is useful for
controlled comparisons of research artifacts with different training losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.general_decision import DATASET_REVISION, load_split, metrics  # noqa: E402
from scripts.score_general_decision import score_requests, verify_artifact  # noqa: E402
from scripts.train_general_decision import encode_rows  # noqa: E402


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


def retain_tokenizable_rows(rows: list[dict], encoded: list[dict]) -> tuple[list[dict], int]:
    """Exclude over-budget rows without truncation and preserve source order."""
    eligible_ids = {item["row"].get("id") for item in encoded}
    if None in eligible_ids:
        raise ValueError("Public decision rows must have stable IDs")
    retained = [row for row in rows if row.get("id") in eligible_ids]
    if len(retained) != len(encoded):
        raise ValueError("Tokenized rows do not match the selected public rows")
    return retained, len(rows) - len(retained)


def load_partial(path: Path, *, signature: str, ids: list[str], resume: bool) -> list[dict]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(f"Frozen evaluation cache already exists: {path}; pass --resume")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved.get("schema_version") != 1 or saved.get("signature") != signature
            or saved.get("ids") != ids or not isinstance(saved.get("predictions"), list)
            or len(saved["predictions"]) > len(ids)):
        raise ValueError("Frozen evaluation cache does not match selected rows or artifact")
    return saved["predictions"]


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
    parser.add_argument("--resume", action="store_true", help="Resume matching per-row GPU scoring cache")
    args = parser.parse_args()
    if not 1 <= args.limit <= 128:
        parser.error("--limit must be 1-128")
    selected_rows, manifest = load_split(args.split, args.limit, args.seed)
    if len(selected_rows) != args.limit:
        raise ValueError("Not enough licensed decisions")
    if args.permute_options_seed is not None:
        selected_rows = permute_options(selected_rows, args.permute_options_seed)
    source_report, _ = verify_artifact(args.output_dir, args.model_report)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(source_report["model"], revision=source_report["revision"],
                                   local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded, _ = encode_rows(tokenizer, selected_rows, source_report["max_length"],
                             prompt_style=source_report.get("prompt_style", "candidate_v1"))
    rows, overlong_skipped = retain_tokenizable_rows(selected_rows, encoded)
    if not rows:
        raise ValueError("No selected decisions fit the frozen token limit")
    payloads = [{"state": json.loads(row["state_json"]), "kind": row["kind"],
                 "question": row["question"], "options": row["options"]} for row in rows]
    ids = [row["id"] for row in rows]
    identity = {"split": args.split, "seed": args.seed, "ids": ids,
                "selected_ids": [row["id"] for row in selected_rows],
                "overlong_skipped": overlong_skipped,
                "dataset_sha256": manifest["sha256"],
                "permute_options_seed": args.permute_options_seed,
                "adapter_sha256": source_report["adapter_sha256"],
                "head_sha256": source_report["head_sha256"]}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache = args.report.with_suffix(".partial.json")
    predictions = load_partial(cache, signature=signature, ids=ids, resume=args.resume)
    start = len(predictions)

    def save(index: int, prediction: dict) -> None:
        if start + index != len(predictions):
            raise ValueError("Frozen predictions arrived out of order")
        predictions.append(prediction)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps({"schema_version": 1, "signature": signature,
                                         "ids": ids, "predictions": predictions}, allow_nan=False),
                             encoding="utf-8")
        os.replace(temporary, cache)

    if start < len(payloads):
        score_requests(payloads[start:], args.output_dir, args.model_report, on_prediction=save)
    report = {"purpose": "offline_frozen_artifact_comparison",
              "dataset_revision": DATASET_REVISION, "dataset_manifest": manifest,
              "split": args.split, "seed": args.seed,
              "selected_rows": len(selected_rows), "rows": len(rows),
              "overlong_skipped": overlong_skipped,
              "permute_options_seed": args.permute_options_seed,
              "adapter_sha256": source_report["adapter_sha256"],
              "head_sha256": source_report["head_sha256"],
              "metrics": prediction_metrics(rows, predictions),
              "limitation": "Licensed public synthetic decisions; no Jev or desktop-safety proof."}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"]["by_kind"], indent=2))


if __name__ == "__main__":
    main()

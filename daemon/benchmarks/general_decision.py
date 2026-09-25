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


def candidate_prompts(row: dict) -> list[str]:
    """Render only input fields; no label, source, ID, or adjacent option leaks."""
    state = json.loads(row["state_json"])
    rendered = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    prefix = f"Context:\n{rendered}\n\nQuestion: {row['question']}\n"
    if row["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
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
            "nll": float(-target @ np.log(np.clip(predicted, 1e-12, 1.0))),
            "brier": float(np.mean((predicted - target) ** 2)),
        }
        if row["kind"] == "score":
            positions = np.arange(len(target))
            item["rank_error"] = float(abs(positions @ predicted - positions @ target))
        per_row.append(item)

    def aggregate(items: list[dict]) -> dict:
        result = {"rows": len(items),
                  "top1_accuracy": float(np.mean([item["top1"] for item in items])),
                  "mean_nll": float(np.mean([item["nll"] for item in items])),
                  "mean_brier": float(np.mean([item["brier"] for item in items]))}
        if all(item["kind"] == "score" for item in items):
            result["mean_rank_error"] = float(np.mean([item["rank_error"] for item in items]))
        return result

    return {**aggregate(per_row),
            "by_kind": {kind: aggregate([item for item in per_row if item["kind"] == kind])
                        for kind in sorted({item["kind"] for item in per_row})},
            "by_source": {source: aggregate([item for item in per_row if item["source"] == source])
                          for source in sorted({item["source"] for item in per_row})}}

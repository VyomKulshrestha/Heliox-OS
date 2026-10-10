"""Train a small, unpromoted shared-head decision model on Open-Jev public data.

This is an architectural probe, not Open-Jev weight reproduction. A frozen
NLI encoder replaces Qwen+LoRA; the single learned scalar head and per-question
softmax match the published decision shape. Data and encoder stay out of the
Heliox package and the head never grants action authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.scope_candidate_encoder import HIDDEN_SIZE, MAX_LENGTH, CandidateEncoder  # noqa: E402
from benchmarks.scope_nli_features import BACKBONE, REVISION  # noqa: E402

DATASET = "ZefanCai/Open-Jev"
DATASET_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
CONFIG = "release-v2-redistributable"
SEED = 20260923
VERSION = "openjev-style-multidomain-v1"
SPLIT_LIMITS = {"train": 100, "calibration": 30, "test": 40, "ood": 40}


def candidate_prompts(row: dict) -> list[tuple[str, str]]:
    state = json.loads(row["state_json"])
    rendered = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    context = f"Context:\n{rendered}\n\nQuestion: {row['question']}"
    return [(context, f"Proposed answer: {option}\nIs this proposed answer correct?") for option in row["options"]]


def load_split(name: str, *, per_source: int) -> tuple[list[dict], str]:
    # On Windows, pyarrow's native DLLs can prevent a subsequent ONNX Runtime
    # import. Load ORT first; this probe uses both in the same process.
    import onnxruntime  # noqa: F401
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    filename = f"data/{CONFIG}/{name}-00000-of-00001.parquet"
    path = Path(
        hf_hub_download(DATASET, filename, repo_type="dataset", revision=DATASET_REVISION, local_files_only=True)
    )
    columns = ("id", "group_id", "source", "question", "options", "target", "state_json")
    rows = parquet.read_table(path, columns=list(columns)).to_pylist()
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        target = row["target"]
        if not 2 <= len(row["options"]) <= 16 or len(target) != len(row["options"]):
            continue
        if not np.all(np.isfinite(target)) or min(target) < 0 or abs(sum(target) - 1) > 1e-4:
            continue
        by_source[row["source"]].append(row)
    rng = random.Random(SEED)
    sample = [
        row
        for source in sorted(by_source)
        for row in rng.sample(
            sorted(by_source[source], key=lambda value: value["id"]), min(per_source, len(by_source[source]))
        )
    ]
    return sample, hashlib.sha256(path.read_bytes()).hexdigest()


def encode_rows(rows: list[dict], cache: Path) -> tuple[np.ndarray, np.ndarray]:
    identity = hashlib.sha256(
        json.dumps(
            [(row["id"], row["state_json"], row["question"], row["options"]) for row in rows],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    offsets = np.concatenate(([0], np.cumsum([len(row["options"]) for row in rows])))
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if (
                str(data["input_sha256"]) != identity
                or str(data["backbone_revision"]) != REVISION
                or int(data["max_length"]) != MAX_LENGTH
            ):
                raise ValueError(f"Stale candidate feature cache: {cache}")
            features = np.asarray(data["features"], dtype=np.float32)
            saved_offsets = np.asarray(data["offsets"], dtype=np.int64)
        if not np.array_equal(offsets, saved_offsets):
            raise ValueError("Candidate offsets changed")
    else:
        encoder = CandidateEncoder()
        prompts = [prompt for row in rows for prompt in candidate_prompts(row)]
        chunks = []
        for start in range(0, len(prompts), 64):
            chunks.append(encoder.encode_prompts(prompts[start : start + 64]))
            if start % 512 == 0:
                print(f"{cache.stem}: encoded {min(start + 64, len(prompts))}/{len(prompts)} candidates", flush=True)
        features = np.vstack(chunks)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            features=features,
            offsets=offsets,
            input_sha256=identity,
            backbone_revision=REVISION,
            max_length=MAX_LENGTH,
        )
    if features.shape != (int(offsets[-1]), HIDDEN_SIZE) or not np.all(np.isfinite(features)):
        raise ValueError("Invalid candidate representation cache")
    return features, offsets


def objective(
    weights: np.ndarray,
    features: np.ndarray,
    offsets: np.ndarray,
    targets: list[np.ndarray],
    *,
    regularization: float = 0.01,
) -> tuple[float, np.ndarray]:
    scores = features @ weights
    gradient = np.zeros_like(weights)
    loss = 0.0
    for index, target in enumerate(targets):
        begin, end = offsets[index : index + 2]
        row_scores = scores[begin:end]
        shifted = row_scores - row_scores.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        loss -= float(target @ np.log(np.clip(probabilities, 1e-12, 1.0)))
        gradient += features[begin:end].T @ (probabilities - target)
    count = len(targets)
    loss = loss / count + regularization * float(weights @ weights) / 2
    gradient = gradient / count + regularization * weights
    return loss, gradient


def predict(
    weights: np.ndarray, features: np.ndarray, offsets: np.ndarray, *, temperature: float = 1.0
) -> list[np.ndarray]:
    scores = features @ weights / temperature
    predictions = []
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        values = np.exp(scores[begin:end] - max(scores[begin:end]))
        predictions.append(values / values.sum())
    return predictions


def evaluate(rows: list[dict], predictions: list[np.ndarray]) -> dict:
    correct = [
        int(np.argmax(row["target"]) == np.argmax(probability))
        for row, probability in zip(rows, predictions, strict=True)
    ]
    brier = [
        float(np.mean((probability - row["target"]) ** 2)) for row, probability in zip(rows, predictions, strict=True)
    ]
    nll = [
        -float(np.asarray(row["target"]) @ np.log(np.clip(probability, 1e-12, 1.0)))
        for row, probability in zip(rows, predictions, strict=True)
    ]
    sources = sorted({row["source"] for row in rows})
    return {
        "rows": len(rows),
        "top1_accuracy": float(np.mean(correct)),
        "mean_brier": float(np.mean(brier)),
        "mean_nll": float(np.mean(nll)),
        "by_source": {
            source: {
                "rows": sum(row["source"] == source for row in rows),
                "top1_accuracy": float(
                    np.mean([hit for row, hit in zip(rows, correct, strict=True) if row["source"] == source])
                ),
            }
            for source in sources
        },
    }


def run(cache_dir: Path, candidate: Path, output: Path) -> dict:
    splits = {}
    hashes = {}
    features = {}
    offsets = {}
    for name, limit in SPLIT_LIMITS.items():
        splits[name], hashes[name] = load_split(name, per_source=limit)
        features[name], offsets[name] = encode_rows(splits[name], cache_dir / f"openjev_{name}_v1.npz")
    groups = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    if any(groups[left] & groups[right] for left in groups for right in groups if left < right):
        raise ValueError("Group leakage between sampled Open-Jev splits")
    target = [np.asarray(row["target"], dtype=np.float64) for row in splits["train"]]
    optimized = minimize(
        objective,
        np.zeros(HIDDEN_SIZE),
        args=(features["train"], offsets["train"], target),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 120},
    )
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        raise RuntimeError(f"Shared scalar head did not converge: {optimized.message}")
    weights = optimized.x
    calibration_targets = [np.asarray(row["target"], dtype=np.float64) for row in splits["calibration"]]

    def calibration_loss(temperature: float) -> float:
        probabilities = predict(weights, features["calibration"], offsets["calibration"], temperature=temperature)
        return float(
            np.mean(
                [
                    -target @ np.log(np.clip(probability, 1e-12, 1.0))
                    for target, probability in zip(calibration_targets, probabilities, strict=True)
                ]
            )
        )

    calibrated = minimize_scalar(calibration_loss, bounds=(0.25, 5.0), method="bounded")
    temperature = float(calibrated.x)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        candidate,
        weights=weights,
        temperature=temperature,
        version=VERSION,
        backbone_revision=REVISION,
        max_length=MAX_LENGTH,
    )
    report = {
        "version": VERSION,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "config": CONFIG,
        "dataset_sha256": hashes,
        "backbone": BACKBONE,
        "backbone_revision": REVISION,
        "seed": SEED,
        "sample_per_source": SPLIT_LIMITS,
        "train_rows": len(splits["train"]),
        "calibration_rows": len(splits["calibration"]),
        "head_parameters": HIDDEN_SIZE,
        "lora_rank": 0,
        "calibration_temperature": temperature,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "optimizer_iterations": int(optimized.nit),
        "test": evaluate(splits["test"], predict(weights, features["test"], offsets["test"], temperature=temperature)),
        "ood": evaluate(splits["ood"], predict(weights, features["ood"], offsets["ood"], temperature=temperature)),
        "limitation": "Small frozen NLI backbone and sampled synthetic data; not Jev/Open-Jev parity or action authority.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "benchmarks/.bench-state")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/opendec_multidomain_candidate_v1.npz")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/opendec_multidomain_probe_v1.json")
    args = parser.parse_args()
    print(json.dumps(run(args.cache_dir, args.candidate, args.output), indent=2))


if __name__ == "__main__":
    main()

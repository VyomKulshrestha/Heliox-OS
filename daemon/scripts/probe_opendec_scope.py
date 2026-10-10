"""Offline Open-Jev-style shared candidate-head probe for Heliox scope.

Requires cached public ScopeJudge and NLI model. The resulting student is NOT
installed in Heliox: external expert labels and synthetic Jev labels do not
establish authorization or real desktop reliability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_semantic_v4_holdout import CASES as JEV_HOLDOUT  # noqa: E402
from benchmarks.scope_candidate_encoder import (  # noqa: E402
    CANDIDATES,
    HIDDEN_SIZE,
    MAX_LENGTH,
    CandidateEncoder,
)
from benchmarks.scope_nli_features import BACKBONE, REVISION  # noqa: E402
from benchmarks.scope_semantic_head import guarded_probabilities  # noqa: E402
from scripts.evaluate_scopejudge_external import evaluate, load_cases  # noqa: E402
from scripts.probe_scopejudge_transfer import SEED, trajectory_split  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402
from scripts.train_scope_decision_semantic import SUITES  # noqa: E402

VERSION = "openjev-style-scope-v1"


def synthetic_cases(root: Path) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Exclude the previously authored v4 comparison suite from fitting."""
    pairs: list[tuple[str, str]] = []
    labels: list[bool] = []
    for suite, name in SUITES:
        if name == "jev_scope_semantic_v4_teacher.json":
            continue
        teacher = _rows(root / "benchmarks" / name, suite)
        for case in suite:
            pairs.append((case.request, case.action))
            labels.append(teacher[case.id]["jev_probability"] >= 0.5)
    return pairs, np.asarray(labels, dtype=bool)


def cached_features(pairs: list[tuple[str, str]], cache: Path) -> tuple[np.ndarray, str]:
    fingerprint = hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if (str(data["input_sha256"]) != fingerprint or str(data["backbone_revision"]) != REVISION
                    or int(data["max_length"]) != MAX_LENGTH):
                raise ValueError("Candidate feature cache does not match the pinned input")
            features = np.asarray(data["features"], dtype=np.float32)
    else:
        encoder = CandidateEncoder()
        chunks = []
        for offset in range(0, len(pairs), 128):
            chunks.append(encoder.transform(pairs[offset:offset + 128]))
            print(f"encoded {min(offset + 128, len(pairs))}/{len(pairs)} pairs", flush=True)
        features = np.vstack(chunks)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=features, input_sha256=fingerprint,
                            backbone_revision=REVISION, max_length=MAX_LENGTH)
    if features.shape != (len(pairs), HIDDEN_SIZE) or not np.all(np.isfinite(features)):
        raise ValueError("Invalid candidate feature matrix")
    return features, fingerprint


def probabilities(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    # A two-candidate softmax over one shared linear score is sigmoid(w @ (h0-h1)).
    logits = np.asarray(features @ model.coef_[0], dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def run(dataset: Path, root: Path, cache: Path, candidate: Path, output: Path) -> dict:
    public = load_cases(dataset)
    train, validation, test = trajectory_split(public)
    public_pairs = [(case.request, case.action) for case in public]
    synthetic_pairs, synthetic_labels = synthetic_cases(root)
    comparison_pairs = [(case.request, case.action) for case in JEV_HOLDOUT]
    all_pairs = public_pairs + synthetic_pairs + comparison_pairs
    features, input_hash = cached_features(all_pairs, cache)
    public_features = features[:len(public)]
    synthetic_features = features[len(public):len(public) + len(synthetic_pairs)]
    comparison_features = features[-len(JEV_HOLDOUT):]
    train_features = np.vstack((public_features[train], synthetic_features))
    train_labels = np.concatenate((
        np.asarray([public[index].in_scope for index in train], dtype=bool), synthetic_labels
    ))
    # fit_intercept=False preserves one shared scalar head: a shared bias cancels
    # when the two candidate scores are subtracted. No test-set tuning.
    model = LogisticRegression(C=0.1, class_weight="balanced", fit_intercept=False,
                               solver="liblinear", max_iter=1000, random_state=SEED)
    model.fit(train_features, train_labels)
    if not np.all(np.isfinite(model.coef_)):
        raise ValueError("Trained scorer is not finite")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate, weights=model.coef_[0], version=VERSION,
                        backbone_revision=REVISION, max_length=MAX_LENGTH,
                        candidates=np.asarray(CANDIDATES))
    reports = {}
    for name, indices in (("validation", validation), ("test", test)):
        cases = [public[index] for index in indices]
        raw = probabilities(model, public_features[indices])
        pairs = [public_pairs[index] for index in indices]
        reports[f"public_{name}_raw"] = evaluate(cases, raw)
        reports[f"public_{name}_guarded"] = evaluate(cases, guarded_probabilities(pairs, raw))
    jev_teacher = _rows(root / "benchmarks/jev_scope_semantic_v4_teacher.json", JEV_HOLDOUT)
    jev_raw = probabilities(model, comparison_features)
    reports["jev_previous_holdout_raw"] = _metrics(list(JEV_HOLDOUT), jev_teacher, jev_raw)
    reports["jev_previous_holdout_guarded"] = _metrics(
        list(JEV_HOLDOUT), jev_teacher, guarded_probabilities(comparison_pairs, jev_raw)
    )
    report = {
        "version": VERSION,
        "source_method": "Open-Jev shared candidate scalar head adapted to frozen NLI pooled embeddings; no LoRA",
        "dataset": "dreadnode/scopejudge",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "backbone": BACKBONE,
        "backbone_revision": REVISION,
        "feature_input_sha256": input_hash,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "public_train_calls": len(train),
        "public_validation_calls": len(validation),
        "public_test_calls": len(test),
        "synthetic_jev_train_pairs": len(synthetic_pairs),
        "jev_previous_holdout_pairs": len(JEV_HOLDOUT),
        "seed": SEED,
        "regularization_C": 0.1,
        "frozen_before_evaluation": True,
        **reports,
        "limitation": "Research-only; prior v4 Jev holdout is not fresh. Never grants execution authority.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use
    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=root / "benchmarks/.bench-state/openjev_scope_features.npz")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/openjev_scope_candidate_v1.npz")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/openjev_scope_probe_v1.json")
    args = parser.parse_args()
    report = run(args.dataset, root, args.cache, args.candidate, args.output)
    print(json.dumps({key: ({k: v for k, v in value.items() if k not in {"mismatch_ids", "mismatches"}}
                            if isinstance(value, dict) else value) for key, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()

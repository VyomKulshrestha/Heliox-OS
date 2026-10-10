"""Train an unpromoted semantic Jev-scope student on inspected synthetic pairs.

The NLI backbone is frozen and cached locally. A new unseen suite is required
after this script freezes its candidate; grouped CV is development evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_augmentation import CASES as AUGMENTATION_CASES  # noqa: E402
from benchmarks.refscope_challenge import CASES as CHALLENGE_CASES  # noqa: E402
from benchmarks.refscope_fresh_final import CASES as FRESH_CASES  # noqa: E402
from benchmarks.refscope_random_final import CASES as RANDOM_CASES  # noqa: E402
from benchmarks.refscope_semantic_holdout import CASES as SEMANTIC_CASES  # noqa: E402
from benchmarks.refscope_semantic_v3_holdout import CASES as SEMANTIC_V3_CASES  # noqa: E402
from benchmarks.refscope_semantic_v4_holdout import CASES as SEMANTIC_V4_CASES  # noqa: E402
from benchmarks.refscope_stress import CASES as STRESS_CASES  # noqa: E402
from benchmarks.refscope_suite import CASES as ORIGINAL_CASES  # noqa: E402
from benchmarks.refscope_unseen_final import CASES as UNSEEN_CASES  # noqa: E402
from benchmarks.scope_nli_features import BACKBONE, REVISION, NliFeatureExtractor, cached_paths  # noqa: E402
from benchmarks.scope_semantic_head import MODEL_VERSION, combined_features, guarded_probabilities  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402

SUITES = (
    (ORIGINAL_CASES, "jev_scope_teacher.json"),
    (CHALLENGE_CASES, "jev_scope_challenge_teacher.json"),
    (AUGMENTATION_CASES, "jev_scope_augmentation_teacher.json"),
    (STRESS_CASES, "jev_scope_stress_teacher.json"),
    (FRESH_CASES, "jev_scope_fresh_teacher.json"),
    (RANDOM_CASES, "jev_scope_random_teacher.json"),
    (UNSEEN_CASES, "jev_scope_unseen_teacher.json"),
    (SEMANTIC_CASES, "jev_scope_semantic_teacher.json"),
    (SEMANTIC_V3_CASES, "jev_scope_semantic_v3_teacher.json"),
    (SEMANTIC_V4_CASES, "jev_scope_semantic_v4_teacher.json"),
)


def _nli_features(cases: list, pairs: list[tuple[str, str]], cache: Path) -> tuple[np.ndarray, str]:
    fingerprint = hashlib.sha256(json.dumps([(case.id, *pair) for case, pair in zip(cases, pairs, strict=True)]).encode()).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["backbone_revision"]) != REVISION or str(data["input_sha256"]) != fingerprint:
                raise ValueError("NLI cache does not match the frozen backbone and pairs")
            features = np.asarray(data["features"], dtype=np.float64)
    else:
        extractor = NliFeatureExtractor(*cached_paths())
        features = extractor.transform(pairs)
        np.savez_compressed(cache, features=features, backbone_revision=REVISION, input_sha256=fingerprint)
    if features.shape != (len(cases), 6) or not np.all(np.isfinite(features)):
        raise ValueError("NLI cache has an invalid feature layout")
    return features, fingerprint


def train(root: Path, cache: Path, candidate: Path, metrics_path: Path) -> dict:
    cases = [case for suite, _ in SUITES for case in suite]
    teacher = {}
    for suite, name in SUITES:
        teacher.update(_rows(root / "benchmarks" / name, suite))
    pairs = [(case.request, case.action) for case in cases]
    nli, input_hash = _nli_features(cases, pairs, cache)
    matrix = combined_features(pairs, nli)
    labels = np.asarray([teacher[case.id]["jev_probability"] >= 0.5 for case in cases])
    sample_weights = np.asarray([max(0.1, abs(teacher[case.id]["jev_probability"] - 0.5) * 2) for case in cases])
    groups = [case.request for case in cases]
    folds = list(GroupKFold(n_splits=5).split(matrix, labels, groups))
    best = None
    fold_logits: dict[float, np.ndarray] = {}
    for regularization in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        logits = np.zeros(len(cases), dtype=float)
        for train_indices, test_indices in folds:
            model = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000)
            model.fit(matrix[train_indices], labels[train_indices], sample_weight=sample_weights[train_indices])
            logits[test_indices] = model.decision_function(matrix[test_indices])
        fold_logits[regularization] = logits
        for threshold in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
            adjusted = logits - np.log(threshold / (1 - threshold))
            raw = 1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0)))
            probabilities = guarded_probabilities(pairs, raw)
            metrics = _metrics(cases, teacher, probabilities)
            false_allowances = metrics["false_allowances"]
            false_warnings = metrics["false_warnings"]
            gate = false_allowances == 0 and false_warnings <= len(cases) * 0.05
            rank = (not gate, false_allowances, false_warnings, metrics["brier_to_teacher_labels"])
            if best is None or rank < best[0]:
                best = (rank, regularization, threshold, metrics)
    assert best is not None
    _, regularization, threshold, cross_validation = best
    adjusted = fold_logits[regularization] - np.log(threshold / (1 - threshold))
    raw_cv = 1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0)))
    guarded_cv = guarded_probabilities(pairs, raw_cv)
    guard_impact = {
        "false_allowances_prevented": int(np.sum((raw_cv >= 0.5) & (guarded_cv < 0.5) & ~labels)),
        "false_warnings_introduced": int(np.sum((raw_cv >= 0.5) & (guarded_cv < 0.5) & labels)),
    }
    model = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000)
    model.fit(matrix, labels, sample_weight=sample_weights)
    intercept = float(model.intercept_[0] - np.log(threshold / (1 - threshold)))
    weights = np.asarray(model.coef_[0], dtype=np.float64)
    np.savez_compressed(candidate, weights=weights, intercept=intercept, model_version=MODEL_VERSION, backbone_revision=REVISION)
    report = {
        "model_version": MODEL_VERSION,
        "backbone": BACKBONE,
        "backbone_revision": REVISION,
        "training_pairs": len(cases),
        "distinct_requests": len(set(groups)),
        "regularization": regularization,
        "threshold": threshold,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "nli_input_sha256": input_hash,
        "out_of_fold_raw": _metrics(cases, teacher, raw_cv),
        "out_of_fold": cross_validation,
        "guard_impact": guard_impact,
        "limitation": "Grouped synthetic CV is development evidence; this artifact is not in production.",
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use
    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / "benchmarks/scope_semantic_features_v5.npz")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/scope_semantic_candidate_v5.npz")
    parser.add_argument("--metrics", type=Path, default=root / "benchmarks/scope_semantic_v5_cv_metrics.json")
    args = parser.parse_args()
    report = train(root, args.cache, args.candidate, args.metrics)
    print(json.dumps({
        key: ({metric: value for metric, value in data.items() if metric != "mismatches"}
              if key in {"out_of_fold", "out_of_fold_raw"} else data)
        for key, data in report.items()
    }, indent=2))


if __name__ == "__main__":
    main()

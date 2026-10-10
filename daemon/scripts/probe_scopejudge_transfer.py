"""Offline transfer probe; public ScopeJudge plus synthetic Jev training only.

The public JSONL is never committed or sent to a service. Trajectories, not
calls, are split before fitting. This probe never writes a runnable model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_semantic_v4_holdout import CASES as JEV_HOLDOUT  # noqa: E402
from benchmarks.scope_decision_experiment import feature_matrix  # noqa: E402
from benchmarks.scope_semantic_head import guarded_probabilities  # noqa: E402
from scripts.evaluate_scopejudge_external import evaluate, load_cases  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402
from scripts.train_scope_decision_semantic import SUITES  # noqa: E402

SEED = 20260923


def trajectory_split(cases: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep every call of a trajectory in exactly one split."""
    indices = np.arange(len(cases))
    groups = np.asarray([case.case_id.split(":", 1)[0] for case in cases])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    development, test = next(splitter.split(indices, groups=groups))
    train_relative, validation_relative = next(splitter.split(
        development, groups=groups[development]
    ))
    train, validation = development[train_relative], development[validation_relative]
    train_groups, validation_groups, test_groups = (set(groups[part]) for part in (train, validation, test))
    if (train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups):
        raise ValueError("Trajectory leaked between splits")
    return train, validation, test


def synthetic_training(root: Path) -> tuple[list[tuple[str, str]], np.ndarray]:
    cases = [case for suite, _ in SUITES for case in suite]
    teacher = {}
    for suite, name in SUITES:
        teacher.update(_rows(root / "benchmarks" / name, suite))
    pairs = [(case.request, case.action) for case in cases]
    labels = np.asarray([teacher[case.id]["jev_probability"] >= 0.5 for case in cases], dtype=bool)
    return pairs, labels


def score(model: LogisticRegression, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    raw = model.predict_proba(feature_matrix(pairs))[:, 1]
    return raw, guarded_probabilities(pairs, raw)


def run(dataset: Path, root: Path) -> dict:
    public = load_cases(dataset)
    train_ids, validation_ids, test_ids = trajectory_split(public)
    public_pairs = [(case.request, case.action) for case in public]
    synthetic_pairs, synthetic_labels = synthetic_training(root)
    train_pairs = [public_pairs[index] for index in train_ids] + synthetic_pairs
    train_labels = np.concatenate((
        np.asarray([public[index].in_scope for index in train_ids], dtype=bool), synthetic_labels
    ))
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000)
    model.fit(feature_matrix(train_pairs), train_labels)
    validation = [public[index] for index in validation_ids]
    test = [public[index] for index in test_ids]
    validation_raw, validation_guarded = score(model, [public_pairs[index] for index in validation_ids])
    test_raw, test_guarded = score(model, [public_pairs[index] for index in test_ids])
    jev_teacher = _rows(root / "benchmarks/jev_scope_semantic_v4_teacher.json", JEV_HOLDOUT)
    jev_pairs = [(case.request, case.action) for case in JEV_HOLDOUT]
    jev_raw, jev_guarded = score(model, jev_pairs)
    return {
        "dataset": "dreadnode/scopejudge",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "seed": SEED,
        "features": "fixed-layout lexical relation and 2048-dimensional word hashing per field",
        "model": "balanced logistic regression, C=1, no threshold tuning",
        "public_train_calls": len(train_ids),
        "public_validation_calls": len(validation_ids),
        "public_test_calls": len(test_ids),
        "synthetic_train_pairs": len(synthetic_pairs),
        "public_validation_raw": evaluate(validation, validation_raw),
        "public_validation_guarded": evaluate(validation, validation_guarded),
        "public_test_raw": evaluate(test, test_raw),
        "public_test_guarded": evaluate(test, test_guarded),
        "jev_holdout_raw": _metrics(list(JEV_HOLDOUT), jev_teacher, jev_raw),
        "jev_holdout_guarded": _metrics(list(JEV_HOLDOUT), jev_teacher, jev_guarded),
        "limitation": "Offline cross-domain probe; neither expert labels nor Jev synthetic labels authorize execution.",
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use
    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Local public ScopeJudge train.jsonl")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/scopejudge_transfer_probe.json")
    args = parser.parse_args()
    report = run(args.dataset, root)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: ({metric: value for metric, value in data.items() if metric not in {"mismatch_ids", "mismatches"}}
                            if isinstance(data, dict) else data) for key, data in report.items()}, indent=2))


if __name__ == "__main__":
    main()

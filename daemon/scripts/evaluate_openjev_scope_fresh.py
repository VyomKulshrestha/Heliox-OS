"""Evaluate a frozen scope student on the new Jev-labeled synthetic suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_openjev_v1_holdout import CASES  # noqa: E402
from benchmarks.scope_candidate_encoder import HIDDEN_SIZE, MAX_LENGTH, CandidateEncoder  # noqa: E402
from benchmarks.scope_nli_features import REVISION  # noqa: E402
from benchmarks.scope_semantic_head import SemanticScopeHead, guarded_probabilities  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def evaluate(candidate: Path, teacher_path: Path, output: Path) -> dict:
    teacher = _rows(teacher_path, CASES)
    with np.load(candidate, allow_pickle=False) as data:
        if (str(data["version"]) != "openjev-style-scope-v1"
                or str(data["backbone_revision"]) != REVISION
                or int(data["max_length"]) != MAX_LENGTH):
            raise ValueError("Candidate identity does not match the frozen experiment")
        weights = np.asarray(data["weights"], dtype=np.float64)
    if weights.shape != (HIDDEN_SIZE,) or not np.all(np.isfinite(weights)):
        raise ValueError("Invalid candidate weights")
    pairs = [(case.request, case.action) for case in CASES]
    features = CandidateEncoder().transform(pairs)
    logits = features @ weights
    raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    guarded = guarded_probabilities(pairs, raw)
    report = {
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "candidate_raw": _metrics(list(CASES), teacher, raw),
        "candidate_guarded": _metrics(list(CASES), teacher, guarded),
    }
    baseline_path = Path(__file__).resolve().parents[1] / "benchmarks/scope_semantic_candidate_v5.npz"
    baseline = SemanticScopeHead(baseline_path)
    baseline_scores = baseline.predict_many_with_raw(pairs)
    if baseline_scores is not None:
        report["previous_v5_raw"] = _metrics(list(CASES), teacher, baseline_scores[0])
        report["previous_v5_guarded"] = _metrics(list(CASES), teacher, baseline_scores[1])
        report["previous_v5_sha256"] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/openjev_scope_candidate_v1.npz")
    parser.add_argument("--teacher", type=Path, default=root / "benchmarks/jev_scope_openjev_v1_teacher.json")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/openjev_scope_fresh_v1.json")
    args = parser.parse_args()
    report = evaluate(args.candidate, args.teacher, args.output)
    print(json.dumps({key: ({metric: value for metric, value in data.items() if metric != "mismatches"}
                            if isinstance(data, dict) else data) for key, data in report.items()}, indent=2))


if __name__ == "__main__":
    main()

"""Score a committed semantic scope candidate on the independent Jev suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jev_scope_semantic_holdout import CASES  # noqa: E402
from benchmarks.scope_nli_features import BACKBONE, REVISION  # noqa: E402
from benchmarks.scope_semantic_head import SemanticScopeHead  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=root / "benchmarks/jev_scope_semantic_teacher.json")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/scope_semantic_candidate.npz")
    parser.add_argument("--training-metrics", type=Path, default=root / "benchmarks/scope_semantic_cv_metrics.json")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/scope_semantic_holdout_metrics.json")
    args = parser.parse_args()
    training = json.loads(args.training_metrics.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    if digest != training["candidate_sha256"]:
        raise ValueError("Candidate differs from the frozen training report")
    teacher = _rows(args.teacher, CASES)
    head = SemanticScopeHead(args.candidate)
    if not head.available:
        raise ValueError("Frozen semantic candidate is unavailable or incompatible")
    pairs = [(case.request, case.action) for case in CASES]
    predictions = head.predict_many(pairs)
    assert predictions is not None
    report = {
        "teacher_model": "typesafe/jev-1.13",
        "candidate_sha256": digest,
        "backbone": BACKBONE,
        "backbone_revision": REVISION,
        "scenario_count": len(CASES) // 2,
        "held_out": _metrics(list(CASES), teacher, predictions),
        "limitation": "Synthetic request/action pairs; not general Jev parity or execution authorization.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Score a committed semantic scope candidate on the independent Jev suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_semantic_holdout import CASES  # noqa: E402
from benchmarks.refscope_semantic_v3_holdout import CASES as V3_CASES  # noqa: E402
from benchmarks.refscope_semantic_v4_holdout import CASES as V4_CASES  # noqa: E402
from benchmarks.scope_nli_features import BACKBONE, REVISION  # noqa: E402
from benchmarks.scope_semantic_head import SemanticScopeHead  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use
    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("v2", "v3", "v4"), default="v2")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    version = f"_{args.suite}" if args.suite != "v2" else ""
    teacher_path = args.teacher or root / f"benchmarks/jev_scope_semantic{version}_teacher.json"
    candidate_path = args.candidate or root / f"benchmarks/scope_semantic_candidate{version}.npz"
    training_path = args.training_metrics or root / (
        f"benchmarks/scope_semantic{version}_cv_metrics.json" if version else "benchmarks/scope_semantic_cv_metrics.json"
    )
    output_path = args.output or root / f"benchmarks/scope_semantic{version}_holdout_metrics.json"
    cases = {"v2": CASES, "v3": V3_CASES, "v4": V4_CASES}[args.suite]
    training = json.loads(training_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if digest != training["candidate_sha256"]:
        raise ValueError("Candidate differs from the frozen training report")
    teacher = _rows(teacher_path, cases)
    head = SemanticScopeHead(candidate_path)
    if not head.available:
        raise ValueError("Frozen semantic candidate is unavailable or incompatible")
    pairs = [(case.request, case.action) for case in cases]
    predictions = head.predict_many(pairs)
    assert predictions is not None
    report = {
        "teacher_model": "typesafe/jev-1.13",
        "candidate_sha256": digest,
        "backbone": BACKBONE,
        "backbone_revision": REVISION,
        "scenario_count": len(cases) // 2,
        "held_out": _metrics(list(cases), teacher, predictions),
        "limitation": "Synthetic request/action pairs; not general Jev parity or execution authorization.",
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

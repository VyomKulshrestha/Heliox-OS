"""Score a frozen local scope candidate against a separate Jev report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_fresh_final import CASES as FRESH_CASES  # noqa: E402
from benchmarks.refscope_random_final import CASES as RANDOM_CASES  # noqa: E402
from benchmarks.refscope_stress import CASES as STRESS_CASES  # noqa: E402
from benchmarks.refscope_unseen_final import CASES as UNSEEN_CASES  # noqa: E402
from benchmarks.scope_decision_experiment import LocalScopeDecisionHead  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.reference_output_policy import block_jev_output_use
    block_jev_output_use()

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=root / "benchmarks/jev_scope_stress_teacher.json")
    parser.add_argument("--candidate", type=Path, default=root / "benchmarks/scope_decision_candidate.npz")
    parser.add_argument("--output", type=Path, default=root / "benchmarks/scope_decision_stress_metrics.json")
    parser.add_argument("--suite", choices=("stress", "fresh", "random", "unseen"), default="stress")
    args = parser.parse_args()
    cases = {"stress": STRESS_CASES, "fresh": FRESH_CASES, "random": RANDOM_CASES, "unseen": UNSEEN_CASES}[args.suite]
    teacher = _rows(args.teacher, cases)
    head = LocalScopeDecisionHead(args.candidate)
    if not head.available:
        raise ValueError("Candidate artifact is unavailable or incompatible")
    predictions = np.asarray([head.predict(case.request, case.action).in_scope_probability for case in cases])
    report = {
        "teacher_model": "typesafe/jev-1.13",
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "scenario_count": len(cases) // 2,
        "candidate_evaluation": _metrics(list(cases), teacher, predictions),
        "limitation": "Synthetic de-identified scenarios; this score is not general Jev parity or an action-safety certification.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

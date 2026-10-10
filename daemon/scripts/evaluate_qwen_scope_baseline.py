"""Compare the existing frozen Heliox scope head on the new Jev suite."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.refscope_qwen_v1_holdout import CASES  # noqa: E402
from benchmarks.scope_semantic_head import SemanticScopeHead  # noqa: E402
from scripts.reference_output_policy import block_jev_output_use  # noqa: E402
from scripts.train_scope_decision_challenge import _metrics, _rows  # noqa: E402


def main() -> None:
    block_jev_output_use()
    root = Path(__file__).resolve().parents[1]
    teacher_path = root / "benchmarks/jev_scope_qwen_v1_teacher.json"
    baseline_path = root / "benchmarks/scope_semantic_candidate_v5.npz"
    teacher = _rows(teacher_path, CASES)
    pairs = [(case.request, case.action) for case in CASES]
    scores = SemanticScopeHead(baseline_path).predict_many_with_raw(pairs)
    if scores is None:
        raise RuntimeError("Existing semantic scope head unavailable")
    report = {
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "existing_v5_raw": _metrics(list(CASES), teacher, scores[0]),
        "existing_v5_guarded": _metrics(list(CASES), teacher, scores[1]),
    }
    (root / "benchmarks/qwen_scope_baseline_v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: {key: value for key, value in result.items() if key != "mismatches"}
                      for name, result in report.items() if isinstance(result, dict)}, indent=2))


if __name__ == "__main__":
    main()

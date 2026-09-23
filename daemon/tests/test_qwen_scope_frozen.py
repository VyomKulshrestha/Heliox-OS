"""Frozen GPU adapter integrity and independent-suite invariants."""

from __future__ import annotations

from pathlib import Path

from benchmarks.jev_scope_openjev_v1_holdout import CASES as PRIOR_CASES
from benchmarks.jev_scope_qwen_v1_holdout import CASES, SCENARIOS
from scripts.evaluate_qwen_scope_frozen import verify_artifact
from scripts.train_scope_decision_semantic import SUITES


def test_new_suite_is_balanced_and_disjoint() -> None:
    assert len(SCENARIOS) == 30
    assert len(CASES) == 60
    assert sum(case.in_scope for case in CASES) == 30
    pairs = {(case.request, case.action) for case in CASES}
    assert len(pairs) == len(CASES)
    earlier = {(case.request, case.action) for suite, _ in SUITES for case in suite}
    earlier |= {(case.request, case.action) for case in PRIOR_CASES}
    assert not pairs & earlier
    for direct, extra in zip(CASES[::2], CASES[1::2], strict=True):
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope


def test_frozen_artifact_matches_training_report() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = verify_artifact(root, root / "benchmarks/.bench-state/qwen_scope_v1")
    assert frozen["fit_rows"] == 520
    assert frozen["revision"]

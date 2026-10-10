"""The fresh Jev comparison is fixed and disjoint from student training."""

from __future__ import annotations

from benchmarks.refscope_opendec_v1_holdout import CASES, SCENARIOS
from scripts.train_scope_decision_semantic import SUITES


def test_fresh_holdout_is_balanced_and_disjoint() -> None:
    assert len(SCENARIOS) == 30
    assert len(CASES) == 60
    assert sum(case.in_scope for case in CASES) == 30
    pairs = {(case.request, case.action) for case in CASES}
    assert len(pairs) == len(CASES)
    earlier = {(case.request, case.action) for suite, _ in SUITES for case in suite}
    assert not pairs & earlier
    for index in range(len(SCENARIOS)):
        direct, extra = CASES[2 * index:2 * index + 2]
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope

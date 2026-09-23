from __future__ import annotations

from benchmarks.jev_scope_augmentation import CASES as AUGMENTATION_CASES
from benchmarks.jev_scope_challenge import CASES as CHALLENGE_CASES
from benchmarks.jev_scope_fresh_final import CASES as FRESH_CASES
from benchmarks.jev_scope_random_final import CASES as RANDOM_CASES
from benchmarks.jev_scope_semantic_holdout import CASES as SEMANTIC_CASES
from benchmarks.jev_scope_semantic_v3_holdout import CASES as SEMANTIC_V3_CASES
from benchmarks.jev_scope_semantic_v4_holdout import CASES, SCENARIOS
from benchmarks.jev_scope_stress import CASES as STRESS_CASES
from benchmarks.jev_scope_suite import CASES as ORIGINAL_CASES
from benchmarks.jev_scope_unseen_final import CASES as UNSEEN_CASES


def test_semantic_v4_holdout_is_balanced_and_disjoint():
    assert len(SCENARIOS) == 50
    assert len(CASES) == 100
    assert sum(case.in_scope for case in CASES) == 50
    pairs = [(case.request, case.action) for case in CASES]
    assert len(set(pairs)) == 100
    earlier = {
        (case.request, case.action)
        for case in (*ORIGINAL_CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES,
                     *STRESS_CASES, *FRESH_CASES, *RANDOM_CASES, *UNSEEN_CASES,
                     *SEMANTIC_CASES, *SEMANTIC_V3_CASES)
    }
    assert not earlier.intersection(pairs)


def test_each_v4_scenario_has_one_direct_and_one_extra_action():
    for index in range(50):
        direct, extra = CASES[2 * index : 2 * index + 2]
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope

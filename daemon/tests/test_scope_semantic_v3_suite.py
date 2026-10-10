from __future__ import annotations

from benchmarks.refscope_augmentation import CASES as AUGMENTATION_CASES
from benchmarks.refscope_challenge import CASES as CHALLENGE_CASES
from benchmarks.refscope_fresh_final import CASES as FRESH_CASES
from benchmarks.refscope_random_final import CASES as RANDOM_CASES
from benchmarks.refscope_semantic_holdout import CASES as SEMANTIC_CASES
from benchmarks.refscope_semantic_v3_holdout import CASES, SCENARIOS
from benchmarks.refscope_stress import CASES as STRESS_CASES
from benchmarks.refscope_suite import CASES as ORIGINAL_CASES
from benchmarks.refscope_unseen_final import CASES as UNSEEN_CASES


def test_semantic_v3_holdout_is_balanced_and_disjoint():
    assert len(SCENARIOS) == 50
    assert len(CASES) == 100
    assert sum(case.in_scope for case in CASES) == 50
    pairs = [(case.request, case.action) for case in CASES]
    assert len(set(pairs)) == 100
    earlier = {
        (case.request, case.action)
        for case in (*ORIGINAL_CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES,
                     *STRESS_CASES, *FRESH_CASES, *RANDOM_CASES, *UNSEEN_CASES, *SEMANTIC_CASES)
    }
    assert not earlier.intersection(pairs)


def test_each_v3_scenario_has_one_direct_and_one_extra_action():
    for index in range(50):
        direct, extra = CASES[2 * index : 2 * index + 2]
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope

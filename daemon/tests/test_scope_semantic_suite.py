from __future__ import annotations

from benchmarks.jev_scope_augmentation import CASES as AUGMENTATION_CASES
from benchmarks.jev_scope_challenge import CASES as CHALLENGE_CASES
from benchmarks.jev_scope_fresh_final import CASES as FRESH_CASES
from benchmarks.jev_scope_random_final import CASES as RANDOM_CASES
from benchmarks.jev_scope_semantic_holdout import CASES, POOL, SCENARIOS, SELECTED_INDICES
from benchmarks.jev_scope_stress import CASES as STRESS_CASES
from benchmarks.jev_scope_suite import CASES as ORIGINAL_CASES
from benchmarks.jev_scope_unseen_final import CASES as UNSEEN_CASES


def test_semantic_holdout_is_randomized_balanced_and_disjoint():
    assert len(POOL) == 60
    assert len(SCENARIOS) == 50
    assert len(set(SELECTED_INDICES)) == 50
    assert len(CASES) == 100
    assert sum(case.in_scope for case in CASES) == 50
    assert len({case.id for case in CASES}) == 100
    assert len({(case.request, case.action) for case in CASES}) == 100
    earlier = {
        (case.request, case.action)
        for case in (*ORIGINAL_CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES,
                     *STRESS_CASES, *FRESH_CASES, *RANDOM_CASES, *UNSEEN_CASES)
    }
    assert not earlier.intersection((case.request, case.action) for case in CASES)


def test_each_scenario_has_one_direct_and_one_extra_action():
    for index in range(50):
        direct, extra = CASES[2 * index : 2 * index + 2]
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope

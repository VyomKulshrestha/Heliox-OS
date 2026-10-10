from __future__ import annotations

from benchmarks.refscope_augmentation import CASES as AUGMENTATION_CASES
from benchmarks.refscope_challenge import CASES as CHALLENGE_CASES
from benchmarks.refscope_challenge import SCENARIOS
from benchmarks.refscope_fresh_final import CASES as FRESH_CASES
from benchmarks.refscope_fresh_final import SCENARIOS as FRESH_SCENARIOS
from benchmarks.refscope_random_final import CASES as RANDOM_CASES
from benchmarks.refscope_random_final import POOL as RANDOM_POOL
from benchmarks.refscope_random_final import SCENARIOS as RANDOM_SCENARIOS
from benchmarks.refscope_random_final import SELECTED_INDICES
from benchmarks.refscope_stress import CASES as STRESS_CASES
from benchmarks.refscope_stress import SCENARIOS as STRESS_SCENARIOS
from benchmarks.refscope_suite import CASES, jev_request
from benchmarks.refscope_unseen_final import CASES as UNSEEN_CASES
from benchmarks.refscope_unseen_final import SCENARIOS as UNSEEN_SCENARIOS


def test_suite_has_unique_balanced_cases_and_held_out_challenge():
    cases = CASES
    assert len(cases) >= 72
    assert len({case.id for case in cases}) == len(cases)
    assert {case.split for case in cases} == {"train", "test", "final"}
    for split in ("train", "test", "final"):
        labels = [case.in_scope for case in cases if case.split == split]
        assert labels.count(True) == labels.count(False)


def test_jev_requests_only_contain_synthetic_case_state():
    for case in CASES:
        request = jev_request(case)
        assert request["model"] == "typesafe/jev-1.13"
        assert request["state"] == {"user_request": case.request, "proposed_action": case.action}
        assert request["questions"]["in_scope"]["type"] == "noul"


def test_randomized_challenge_is_scenario_disjoint_and_balanced():
    assert len(SCENARIOS) == 60
    assert len(CHALLENGE_CASES) == 120
    assert len({case.id for case in CHALLENGE_CASES}) == len(CHALLENGE_CASES)
    assert {split: sum(case.split == split for case in CHALLENGE_CASES) for split in ("train", "test", "final")} == {
        "train": 72,
        "test": 24,
        "final": 24,
    }
    for index in range(60):
        direct, extra = CHALLENGE_CASES[2 * index : 2 * index + 2]
        assert direct.request == extra.request
        assert direct.split == extra.split
        assert direct.in_scope and not extra.in_scope
    old_pairs = {(case.request, case.action) for case in CASES}
    assert not old_pairs.intersection((case.request, case.action) for case in CHALLENGE_CASES)


def test_fresh_stress_set_is_disjoint_from_training_and_challenge():
    assert len(STRESS_SCENARIOS) == 50
    assert len(STRESS_CASES) == 100
    assert len({case.id for case in STRESS_CASES}) == 100
    earlier_pairs = {(case.request, case.action) for case in (*CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES)}
    assert not earlier_pairs.intersection((case.request, case.action) for case in STRESS_CASES)
    for index in range(50):
        direct, extra = STRESS_CASES[2 * index : 2 * index + 2]
        assert direct.request == extra.request
        assert direct.in_scope and not extra.in_scope


def test_second_fresh_final_is_disjoint_from_all_inspected_pairs():
    assert len(FRESH_SCENARIOS) == 50
    assert len(FRESH_CASES) == 100
    earlier = {(case.request, case.action) for case in (*CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES, *STRESS_CASES)}
    assert not earlier.intersection((case.request, case.action) for case in FRESH_CASES)
    assert len({(case.request, case.action) for case in FRESH_CASES}) == 100


def test_seeded_random_final_is_unseen_and_balanced():
    assert len(RANDOM_POOL) == 60
    assert len(SELECTED_INDICES) == 50
    assert len(set(SELECTED_INDICES)) == 50
    assert len(RANDOM_SCENARIOS) == 50
    assert len(RANDOM_CASES) == 100
    earlier = {(case.request, case.action) for case in (*CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES, *STRESS_CASES, *FRESH_CASES)}
    assert not earlier.intersection((case.request, case.action) for case in RANDOM_CASES)
    assert len({(case.request, case.action) for case in RANDOM_CASES}) == 100
    assert sum(case.in_scope for case in RANDOM_CASES) == 50


def test_third_unseen_set_is_disjoint_and_balanced():
    assert len(UNSEEN_SCENARIOS) == 50
    assert len(UNSEEN_CASES) == 100
    earlier = {(case.request, case.action) for case in (*CASES, *CHALLENGE_CASES, *AUGMENTATION_CASES, *STRESS_CASES, *FRESH_CASES, *RANDOM_CASES)}
    assert not earlier.intersection((case.request, case.action) for case in UNSEEN_CASES)
    assert len({(case.request, case.action) for case in UNSEEN_CASES}) == 100
    assert sum(case.in_scope for case in UNSEEN_CASES) == 50

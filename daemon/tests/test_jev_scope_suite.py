from __future__ import annotations

from benchmarks.jev_scope_suite import CASES, jev_request


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

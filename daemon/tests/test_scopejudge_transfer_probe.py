from __future__ import annotations

from scripts.evaluate_scopejudge_external import ExternalCase
from scripts.probe_scopejudge_transfer import trajectory_split


def test_trajectory_split_never_leaks_calls_across_splits():
    cases = [
        ExternalCase(f"trajectory-{group}:step:{call}", "request", "action", call == 0)
        for group in range(20)
        for call in range(2)
    ]
    train, validation, test = trajectory_split(cases)
    assert sorted([*train, *validation, *test]) == list(range(len(cases)))
    for left, right in ((train, validation), (train, test), (validation, test)):
        left_groups = {cases[index].case_id.split(":")[0] for index in left}
        right_groups = {cases[index].case_id.split(":")[0] for index in right}
        assert left_groups.isdisjoint(right_groups)

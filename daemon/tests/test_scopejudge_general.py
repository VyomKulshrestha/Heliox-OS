"""MIT ScopeJudge decisions remain label-balanced and trajectory-disjoint."""

from __future__ import annotations

import json

import numpy as np

from benchmarks.general_decision import candidate_prompts, verify_disjoint
from benchmarks.scopejudge_general import ExternalCase, balanced_cases, scope_metrics, split_cases


def cases() -> list[ExternalCase]:
    return [ExternalCase(f"trajectory-{group:02d}:{index}:call", f"Inspect target {group}",
                         f"action {index}", index % 2 == 0)
            for group in range(10) for index in range(4)]


def test_scope_rows_are_balanced_disjoint_and_prompt_only_inputs() -> None:
    samples = cases()
    selected = balanced_cases(samples, 12, 7)
    assert sum(case.in_scope for case in selected) == 6
    limits = {"scope_train": 12, "scope_calibration": 4, "scope_test": 4}
    parts = split_cases(samples, seed=7, limits=limits)
    verify_disjoint(parts)
    assert {name: len(rows) for name, rows in parts.items()} == limits
    for rows in parts.values():
        assert sum(row["target"][1] for row in rows) == len(rows) / 2
        for row in rows:
            assert json.loads(row["state_json"])["user_request"]
            assert all("trajectory-" not in prompt for prompt in candidate_prompts(row))
    summary = scope_metrics(parts["scope_test"],
                            [np.array([0.0, 2.0])] * len(parts["scope_test"]), 1.0)
    assert summary["false_allowances"] == len(parts["scope_test"]) // 2
    assert summary["balanced_accuracy"] == 0.5
    full = split_cases(samples, seed=7,
                       limits={**limits, "scope_test": 100})["scope_test"]
    assert len(full) == 8

"""MIT ScopeJudge decisions remain label-balanced and trajectory-disjoint."""

from __future__ import annotations

import json

import numpy as np

from benchmarks.general_decision import candidate_prompts, verify_disjoint
from benchmarks.scopejudge_general import ExternalCase, balanced_cases, decision_row, scope_metrics, split_cases
from scripts.evaluate_scopejudge_external import load_cases
from scripts.train_general_decision import encode_rows


def cases() -> list[ExternalCase]:
    return [
        ExternalCase(
            f"trajectory-{group:02d}:{index}:call", f"Inspect target {group}", f"action {index}", index % 2 == 0
        )
        for group in range(10)
        for index in range(4)
    ]


def test_optional_history_excludes_same_step_future_and_observations(tmp_path) -> None:
    steps = [
        {
            "step_id": "one",
            "tool_calls": [{"tool_call_id": "a", "function_name": "inspect", "arguments": {"target": "alpha"}}],
            "observation": "PRIVATE_OBSERVATION",
        },
        {
            "step_id": "two",
            "tool_calls": [
                {"tool_call_id": "b", "function_name": "read", "arguments": {"target": "beta"}},
                {"tool_call_id": "c", "function_name": "delete", "arguments": {"target": "gamma"}},
            ],
        },
        {"step_id": "three", "tool_calls": [{"tool_call_id": "d", "function_name": "summarize", "arguments": {}}]},
    ]
    labels = [
        {"step_id": step["step_id"], "tool_call_id": call["tool_call_id"], "golden_label": "in_scope"}
        for step in steps
        for call in step["tool_calls"]
    ]
    trajectory = {
        "steps": steps,
        "extra": {"scopejudge": {"trajectory_id": "test", "user_intent": "Inspect alpha", "labels": labels}},
    }
    path = tmp_path / "scope.jsonl"
    path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")

    default = load_cases(path)
    assert all(not case.prior_actions for case in default)
    contextual = load_cases(path, history_calls=2)
    assert contextual[0].prior_actions == ()
    assert contextual[1].prior_actions == contextual[2].prior_actions
    assert len(contextual[1].prior_actions) == 1
    assert contextual[3].prior_actions == (contextual[1].action, contextual[2].action)
    assert "PRIVATE_OBSERVATION" not in decision_row(contextual[3])["state_json"]
    assert json.loads(decision_row(contextual[3])["state_json"])["previous_tool_calls"] == [
        contextual[1].action,
        contextual[2].action,
    ]


def test_prompt_budget_drops_oldest_history_but_keeps_current_action() -> None:
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

        def __call__(self, value, **_kwargs):
            return {"input_ids": [1] * len(value)}

    current = ExternalCase(
        "trajectory:step:call", "Inspect alpha", "read alpha", True, ("inspect " + "old" * 100, "list alpha")
    )
    latest = decision_row(
        ExternalCase(current.case_id, current.request, current.action, current.in_scope, current.prior_actions[-1:])
    )
    token_limit = len(candidate_prompts(latest)[0])
    encoded, overlong = encode_rows(CharacterTokenizer(), [decision_row(current)], token_limit, fit_history=True)
    assert overlong == 0
    assert len(encoded) == 1
    assert encoded[0]["history_calls_dropped"] == 1
    assert encoded[0]["history_calls_retained"] == 1
    state = json.loads(encoded[0]["row"]["state_json"])
    assert state == {
        "user_request": current.request,
        "proposed_action": current.action,
        "previous_tool_calls": ["list alpha"],
    }


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
    summary = scope_metrics(parts["scope_test"], [np.array([0.0, 2.0])] * len(parts["scope_test"]), 1.0)
    assert summary["false_allowances"] == len(parts["scope_test"]) // 2
    assert summary["balanced_accuracy"] == 0.5
    full = split_cases(samples, seed=7, limits={**limits, "scope_test": 100})["scope_test"]
    assert len(full) == 8


def test_balanced_trajectory_split_is_deterministic_and_disjoint() -> None:
    limits = {"scope_train": 12, "scope_calibration": 4, "scope_test": 100}
    first = split_cases(cases(), seed=19, limits=limits, strategy="balanced_trajectory_v2")
    second = split_cases(cases(), seed=19, limits=limits, strategy="balanced_trajectory_v2")
    verify_disjoint(first)
    assert {name: [row["id"] for row in rows] for name, rows in first.items()} == {
        name: [row["id"] for row in rows] for name, rows in second.items()
    }
    assert all(
        any(row["target"][1] for row in rows) and any(not row["target"][1] for row in rows) for rows in first.values()
    )


def test_group_diverse_sampling_does_not_let_one_trajectory_fill_a_class() -> None:
    examples = [
        ExternalCase(f"group-{group}:{index}:call", "Review", "Read", index == 0)
        for group in range(10)
        for index in range(2)
    ]
    examples += [ExternalCase(f"group-0:{index}:call", "Review", "Inspect", False) for index in range(2, 50)]
    selected = balanced_cases(examples, 20, 9, group_diverse=True)
    assert len(selected) == 20
    assert len({case.case_id.split(":", 1)[0] for case in selected if not case.in_scope}) == 10


def test_diverse_trajectory_split_is_deterministic_and_disjoint() -> None:
    limits = {"scope_train": 12, "scope_calibration": 4, "scope_test": 100}
    first = split_cases(cases(), seed=19, limits=limits, strategy="diverse_trajectory_v3")
    second = split_cases(cases(), seed=19, limits=limits, strategy="diverse_trajectory_v3")
    reordered = split_cases(list(reversed(cases())), seed=19, limits=limits, strategy="diverse_trajectory_v3")
    verify_disjoint(first)
    assert {name: [row["id"] for row in rows] for name, rows in first.items()} == {
        name: [row["id"] for row in rows] for name, rows in second.items()
    }
    assert {name: [row["id"] for row in rows] for name, rows in first.items()} == {
        name: [row["id"] for row in rows] for name, rows in reordered.items()
    }
    assert {name: len(rows) for name, rows in first.items()} == {
        "scope_train": 12,
        "scope_calibration": 4,
        "scope_test": 8,
    }

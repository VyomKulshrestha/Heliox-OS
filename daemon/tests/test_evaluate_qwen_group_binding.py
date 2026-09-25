"""Grouped target review keeps ambiguous and malformed decisions conservative."""

import json

import pytest

from benchmarks.jev_scope_suite import ScopeCase
from scripts.evaluate_qwen_group_binding import (
    group_cases,
    load_partial,
    messages,
    parse_selection,
    save_partial,
    summarize,
)


def test_group_prompt_contains_all_candidates_but_not_labels():
    cases = (
        ScopeCase("task:blue", "Select blue", "Use blue", True),
        ScopeCase("task:red", "Select blue", "Use red", False),
    )
    assert group_cases(cases, "entity-binding") == [("task", cases)]
    payload = json.loads(messages(cases)[1]["content"])
    assert payload["candidate_actions"] == [
        {"number": 1, "action": "Use blue"}, {"number": 2, "action": "Use red"}]
    assert "in_scope" not in json.dumps(payload)
    with pytest.raises(ValueError, match="mismatched requests"):
        group_cases((cases[0], ScopeCase("task:red", "Select red", "Use red", False)),
                    "entity-binding")


@pytest.mark.parametrize(("raw", "count", "expected"), [
    ("[2,1]", 2, (1, 2)), ("[]", 2, ()), ("[0]", 2, None),
    ("[3]", 2, None), ("[1,1]", 2, None), ("[true]", 2, None),
    ("[1] because", 2, None),
])
def test_only_exact_valid_candidate_arrays_are_accepted(raw, count, expected):
    assert parse_selection(raw, count) == expected


def test_group_summary_and_checkpoint_integrity(tmp_path):
    members = (ScopeCase("task:right", "Choose right", "Use right", True),
               ScopeCase("task:wrong", "Choose right", "Use wrong", False))
    ambiguous = (ScopeCase("amb:a", "Unclear", "Use a", False),
                 ScopeCase("amb:b", "Unclear", "Use b", False))
    groups = [("task", members), ("amb", ambiguous)]
    rows = [{"id": "task", "gold_indices": [1], "selected_indices": [1], "status": "ok"},
            {"id": "amb", "gold_indices": [], "selected_indices": [], "status": "ok"}]
    result = summarize(groups, rows)
    assert result["exact_groups"] == 2
    assert result["ambiguous_groups"] == 1
    assert result["ambiguous_abstentions"] == 1
    assert result["false_allowances"] == result["false_warnings"] == 0
    path = tmp_path / "group.partial.json"
    save_partial(path, {"model": "pinned"}, rows)
    assert load_partial(path, {"model": "pinned"}, groups) == rows
    with pytest.raises(ValueError, match="does not match"):
        load_partial(path, {"model": "changed"}, groups)
    save_partial(path, {"model": "pinned"}, [{**rows[0], "selected_indices": [3]}])
    with pytest.raises(ValueError, match="does not match"):
        load_partial(path, {"model": "pinned"}, groups)

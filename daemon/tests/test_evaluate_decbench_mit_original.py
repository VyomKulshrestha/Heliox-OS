from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_decbench_mit_original.py"
SPEC = importlib.util.spec_from_file_location("evaluate_decbench_mit_original", SCRIPT)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def _task(kind: str, criteria, expected):
    return {
        "id": "fixture-1",
        "group": "fixture-group",
        "provenance": {"license": "MIT"},
        "state": "Known context",
        "question": {"type": kind, "instructions": "Make a decision", "criteria": criteria},
        "expected": expected,
    }


def test_choice_mapping_keeps_descriptions_and_hides_gold_from_input():
    row = evaluator.task_to_row(_task(
        "choice", {"alpha": "First action", "beta": "Second action"}, "beta",
    ))

    assert row["options"] == ["alpha: First action", "beta: Second action"]
    assert row["target"] == [0.0, 1.0]
    assert evaluator.public_payload(row) == {
        "state": "Known context",
        "kind": "choice",
        "question": "Make a decision",
        "options": ["alpha: First action", "beta: Second action"],
    }
    assert "target" not in evaluator.public_payload(row)
    assert "fixture-1" not in str(evaluator.public_payload(row))


def test_noul_mapping_preserves_both_rubric_definitions():
    row = evaluator.task_to_row(_task(
        "noul", {"false": "A requirement is missing", "true": "All requirements hold"}, "yes",
    ))

    assert row["options"] == ["no", "yes"]
    assert "no means A requirement is missing" in row["question"]
    assert "yes means All requirements hold" in row["question"]
    assert row["target"] == [0.0, 1.0]


def test_score_mapping_preserves_ordered_rubric_and_level():
    row = evaluator.task_to_row(_task("score", ["low", "medium", "high"], 2))

    assert row["options"] == ["low", "medium", "high"]
    assert row["target"] == [0.0, 0.0, 1.0]


def test_rejects_malformed_expected_score_and_duplicate_options():
    with pytest.raises(ValueError, match="Score target"):
        evaluator.task_to_row(_task("score", ["low", "high"], 2))
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.task_to_row(_task("score", ["same", "same"], 0))


def test_paired_accuracy_uses_group_bootstrap_and_is_reproducible():
    rows = [
        {"group_id": "a"},
        {"group_id": "a"},
        {"group_id": "b"},
    ]
    candidate = [True, False, True]
    baseline = [False, False, True]

    first = evaluator.paired_accuracy_difference(rows, candidate, baseline, replicates=500)
    second = evaluator.paired_accuracy_difference(rows, candidate, baseline, replicates=500)

    assert first == second
    assert first["candidate_minus_baseline_accuracy"] == pytest.approx(1 / 3)
    assert first["groups_resampled"] == 2
    assert len(first["percentile_95_interval"]) == 2
    with pytest.raises(ValueError, match="Paired benchmark outcomes"):
        evaluator.paired_accuracy_difference(rows, candidate, baseline, replicates=20)

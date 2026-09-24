"""General typed-decision data contract and leakage checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.general_decision import (
    candidate_prompts,
    format_answer,
    licensed_row,
    metrics,
    probabilities,
    select_balanced,
    validate_decision,
    validate_row,
    verify_disjoint,
)


def row(kind: str = "choice") -> dict:
    return {"id": "item", "group_id": "parent", "source": "source", "kind": kind,
            "state_json": json.dumps({"counter": 3}), "question": "Select a result.",
            "options": ["no", "yes"] if kind == "noul" else ["left", "right", "wait"],
            "target": [0.0, 1.0] if kind == "noul" else [0.0, 1.0, 0.0],
            "metadata_json": json.dumps({"provenance": {"license": "CC0-1.0"}})}


@pytest.mark.parametrize("kind,count", [("noul", 1), ("choice", 3), ("score", 3)])
def test_candidate_prompts_share_context_without_labels(kind: str, count: int) -> None:
    example = row(kind)
    validate_row(example)
    prompts = candidate_prompts(example)
    assert len(prompts) == count
    assert all("counter" in prompt and "Select a result" in prompt for prompt in prompts)
    assert all("parent" not in prompt and "source" not in prompt for prompt in prompts)
    if kind == "noul":
        assert "Proposed answer" not in prompts[0]
    else:
        assert all("Proposed answer: " + option in prompt
                   for prompt, option in zip(prompts, example["options"], strict=True))


def test_reject_unverified_rights_and_invalid_targets() -> None:
    example = row()
    assert licensed_row(example)
    example["metadata_json"] = json.dumps({"provenance": {
        "license": "Generated conversations: CC0-1.0; upstream question descriptions: license not verified"
    }})
    assert not licensed_row(example)
    example["target"] = [0.1, 0.8, 0.0]
    with pytest.raises(ValueError, match="sum to one"):
        validate_row(example)


def test_group_disjoint_and_balanced_sampling() -> None:
    examples = []
    for source in ("a", "b"):
        for kind in ("choice", "score"):
            for index in range(3):
                examples.append({**row(kind), "id": f"{source}:{kind}:{index}",
                                 "group_id": f"{source}:{kind}:{index}", "source": source})
    first = select_balanced(examples, 8, 12)
    second = select_balanced(examples, 8, 12)
    assert [entry["id"] for entry in first] == [entry["id"] for entry in second]
    assert {(entry["source"], entry["kind"]) for entry in first} == {
        ("a", "choice"), ("a", "score"), ("b", "choice"), ("b", "score")}
    verify_disjoint({"train": first, "test": [examples[-1]]})
    with pytest.raises(ValueError, match="Group leakage"):
        verify_disjoint({"train": first, "test": [first[0]]})


def test_variable_option_metrics_and_temperature() -> None:
    examples = [row("noul"), row("choice")]
    logits = [np.array([0.0, 2.0]), np.array([-1.0, 2.0, 0.0])]
    report = metrics(examples, logits, 1.0)
    assert report["top1_accuracy"] == 1.0
    assert set(report["by_kind"]) == {"noul", "choice"}
    assert probabilities(logits[1], 0.5)[1] > probabilities(logits[1], 2.0)[1]
    with pytest.raises(ValueError, match="option counts"):
        metrics(examples, [logits[0], logits[0]], 1.0)


def test_typed_answers_have_normalized_variable_option_probabilities() -> None:
    yes = format_answer("noul", ["no", "yes"], np.array([0.0, 1.0]))
    assert yes["yes_probability"] > 0.5
    choice = format_answer("choice", ["left", "right", "wait"],
                           np.array([0.0, 2.0, -1.0]))
    assert choice["choice"] == "right"
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)
    score = format_answer("score", ["low", "medium", "high"],
                          np.array([-2.0, 0.0, 2.0]))
    assert 1.0 < score["expected_index"] < 2.0
    with pytest.raises(ValueError, match="Score count"):
        format_answer("choice", ["left", "right"], np.array([1.0]))
    with pytest.raises(ValueError, match="option"):
        format_answer("choice", [["left"], "right"], np.array([0.0, 1.0]))


def test_invalid_option_types_fail_before_duplicate_check() -> None:
    example = row()
    example["options"] = [["left"], "right"]
    with pytest.raises(ValueError, match="options"):
        validate_row(example)
    validate_decision("choice", "Which route?", ["walk", "train", "drive", "cycle"])
    with pytest.raises(ValueError, match="Noul"):
        validate_decision("noul", "Can I?", ["yes", "no"])

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_general_decision_fresh_replication import (
    load_candidate_rows,
    paired_top1_correctness,
)


def test_load_candidate_rows_accepts_exact_ordered_public_logits() -> None:
    rows = [{"id": "one", "target": [1.0, 0.0]}, {"id": "two", "target": [0.0, 1.0]}]
    saved = {"schema_version": 1, "signature": "frozen", "ids": ["one", "two"],
             "logits": [[2.0, 0.0], [0.0, 3.0]]}

    actual = load_candidate_rows(rows, saved, signature="frozen")

    assert all(isinstance(item, np.ndarray) for item in actual)
    assert [item.argmax() for item in actual] == [0, 1]


@pytest.mark.parametrize("change", ["signature", "order", "shape", "nonfinite"])
def test_load_candidate_rows_rejects_unaligned_or_invalid_logits(change: str) -> None:
    rows = [{"id": "one", "target": [1.0, 0.0]}, {"id": "two", "target": [0.0, 1.0]}]
    saved = {"schema_version": 1, "signature": "frozen", "ids": ["one", "two"],
             "logits": [[2.0, 0.0], [0.0, 3.0]]}
    if change == "signature":
        saved["signature"] = "other"
    elif change == "order":
        saved["ids"] = ["two", "one"]
    elif change == "shape":
        saved["logits"][1] = [0.0]
    else:
        saved["logits"][1] = [float("nan"), 1.0]

    with pytest.raises(ValueError):
        load_candidate_rows(rows, saved, signature="frozen")


def test_paired_top1_correctness_aligns_noul_and_candidate_decisions() -> None:
    rows = [
        {"target": [0.0, 1.0], "kind": "noul", "options": ["no", "yes"]},
        {"target": [1.0, 0.0], "kind": "choice", "options": ["left", "right"]},
    ]
    logits = [np.asarray([0.0, 2.0]), np.asarray([3.0, 0.0])]
    predictions = [
        {"answer": {"yes_probability": 0.8}},
        {"answer": {"probabilities": {"left": 0.2, "right": 0.8}}},
    ]

    baseline, candidate = paired_top1_correctness(rows, logits, predictions)

    assert baseline == [1.0, 1.0]
    assert candidate == [1.0, 0.0]


def test_paired_top1_correctness_rejects_missing_typed_output() -> None:
    row = {"target": [1.0, 0.0], "kind": "choice", "options": ["left", "right"]}
    with pytest.raises(ValueError, match="no typed answer"):
        paired_top1_correctness([row], [np.asarray([1.0, 0.0])], [{}])


def test_paired_top1_correctness_rejects_unnormalized_candidate_probabilities() -> None:
    row = {"target": [1.0, 0.0], "kind": "choice", "options": ["left", "right"]}
    prediction = {"answer": {"probabilities": {"left": 0.9, "right": 0.9}}}
    with pytest.raises(ValueError, match="probabilities are invalid"):
        paired_top1_correctness([row], [np.asarray([1.0, 0.0])], [prediction])

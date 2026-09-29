"""Split-safety invariants for the correctness-head research evaluator."""

import pytest

from scripts.evaluate_general_decision_correctness_head import (
    select_unseen_group_rows,
    validate_split_policy,
)


def test_select_unseen_group_rows_preserves_order_and_skips_seen_groups() -> None:
    rows = [
        {"id": "seen", "group_id": "already-used"},
        {"id": "first", "group_id": "new-a"},
        {"id": "duplicate", "group_id": "new-a"},
        {"id": "second", "group_id": "new-b"},
    ]

    selected = select_unseen_group_rows(rows, {"already-used"}, 2)

    assert [row["id"] for row in selected] == ["first", "second"]


def test_select_unseen_group_rows_rejects_insufficient_unique_groups() -> None:
    rows = [{"id": "one", "group_id": "same"}, {"id": "two", "group_id": "same"}]

    with pytest.raises(ValueError, match="previously unseen groups"):
        select_unseen_group_rows(rows, set(), 2)


def test_split_policy_allows_only_disjoint_nontraining_partitions() -> None:
    validate_split_policy({
        "base_training_splits": ["train"],
        "fit_split": "calibration",
        "test_split": "ood",
    })


@pytest.mark.parametrize(("fit_split", "test_split"), [
    ("train", "ood"),
    ("calibration", "train"),
    ("calibration", "calibration"),
])
def test_split_policy_rejects_training_or_reused_partitions(fit_split: str,
                                                            test_split: str) -> None:
    with pytest.raises(ValueError, match="distinct from the base training mixture"):
        validate_split_policy({
            "base_training_splits": ["train"],
            "fit_split": fit_split,
            "test_split": test_split,
        })

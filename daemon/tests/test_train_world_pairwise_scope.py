"""Within-task ranking pairs never mix requests or reverse labels."""

import numpy as np
import pytest

from scripts.train_world_pairwise_scope import paired_indices, top_one_accuracy


def test_pairs_use_positive_and_negative_from_same_task():
    groups = ["a", "a", "a", "b", "b", "c"]
    labels = np.asarray([True, False, False, False, True, True])
    np.testing.assert_array_equal(paired_indices(groups, labels),
                                  [[0, 1], [0, 2], [4, 3]])
    with pytest.raises(ValueError, match="No within-task"):
        paired_indices(["a", "b"], np.asarray([True, False]))


def test_top_one_excludes_tasks_without_both_classes():
    groups = ["a", "a", "b", "b", "c"]
    labels = np.asarray([True, False, True, False, True])
    scores = np.asarray([0.9, 0.1, 0.2, 0.8, 0.99])
    assert top_one_accuracy(groups, labels, scores) == {
        "mixed_tasks": 2, "correct": 1, "accuracy": 0.5}

import numpy as np
import pytest

from scripts.derive_miniwob_within_task_validation import derived_splits


def test_derived_validation_keeps_pairs_and_external_task_separate() -> None:
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d",
                       "old", "old", "old2", "old2", "outside", "outside"])
    splits = np.array(["train"] * 8 + ["validation"] * 4 + ["eval"] * 2)
    tasks = np.array(["first"] * 4 + ["second"] * 4 + ["old"] * 4 + ["outside"] * 2)
    keep, recoded = derived_splits(groups, splits, tasks, 1, 7)
    assert not keep[8] and not keep[9]
    assert list(recoded).count("validation") == 4
    assert list(recoded).count("eval") == 2
    for group in ("a", "b", "c", "d"):
        assert len(set(recoded[groups[keep] == group])) == 1
    again = derived_splits(groups, splits, tasks, 1, 7)
    assert np.array_equal(again[0], keep) and np.array_equal(again[1], recoded)
    mixed_keep, mixed_splits = derived_splits(
        groups, splits, tasks, 1, 7, include_prior_validation=True)
    assert mixed_keep.all()
    assert list(mixed_splits).count("validation") == 6
    assert list(mixed_splits).count("eval") == 2


def test_derived_validation_rejects_task_leakage() -> None:
    groups = np.array(["a", "a", "b", "b", "v", "v", "e", "e"])
    splits = np.array(["train"] * 4 + ["validation"] * 2 + ["eval"] * 2)
    tasks = np.array(["same"] * 4 + ["old"] * 2 + ["same"] * 2)
    with pytest.raises(ValueError, match="separate"):
        derived_splits(groups, splits, tasks, 1, 7)

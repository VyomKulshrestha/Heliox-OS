"""Frozen visual transfer reports pair-level action evidence."""

import numpy as np
import pytest

from scripts.probe_world_visual_model_transfer import paired_action_diagnostic


def test_paired_action_diagnostic_counts_groups_not_rows() -> None:
    state = np.zeros((4, 2), dtype=np.float32)
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    correct = target.copy()
    swapped = target.copy()
    swapped[:2] = target[1::-1]
    result = paired_action_diagnostic(
        correct, swapped, state, target, np.array(["a", "a", "b", "b"])
    )

    assert result["groups"] == 2
    assert result["correct_action_wins"] == 1
    assert result["ties"] == 1
    assert result["mean_weighted_mse_advantage"] > 0
    with pytest.raises(ValueError, match="exactly two"):
        paired_action_diagnostic(correct, swapped, state, target,
                                 np.array(["a", "a", "a", "b"]))

"""Frozen visual transfer reports pair-level action evidence."""

import numpy as np
import pytest

from scripts.probe_world_visual_model_transfer import (
    observed_pair_signal,
    paired_action_diagnostic,
    paired_discriminative_diagnostic,
    task_masks,
)


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


def test_discriminative_diagnostic_skips_pairs_without_distinct_pixels() -> None:
    target = np.array([[1, 0], [0, 1], [0, 0], [0.001, 0]], dtype=np.float32)
    correct = target.copy()
    swapped = target[[1, 0, 3, 2]]
    result = paired_discriminative_diagnostic(
        correct, swapped, target, np.array(["a", "a", "b", "b"])
    )
    assert result["groups_with_distinct_pixels"] == 1
    assert result["correct_action_wins"] == 1
    assert result["mean_mse_advantage"] > 0


def test_task_masks_align_with_eval_rows(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(path, split=np.asarray(["train", "eval", "eval", "eval", "eval"]),
             task=np.asarray(["train", "a", "a", "b", "b"]))
    assert task_masks(path, 4)["b"].tolist() == [False, False, True, True]
    with pytest.raises(ValueError, match="align"):
        task_masks(path, 3)


def test_observed_pair_signal_reports_pixel_separation() -> None:
    target = np.asarray([[0, 0], [1, 0], [0, 0], [0, 0.2]], dtype=np.float32)
    signal = observed_pair_signal(target, np.asarray(["a", "a", "b", "b"]))
    assert signal["groups"] == 2
    assert signal["mean_pixel_mse"] == pytest.approx(0.26)

"""Inverse probe reports action and location errors separately."""

import numpy as np

from scripts.probe_world_visual_inverse import score_inverse


def test_inverse_scores_include_majority_and_location_controls() -> None:
    logits = np.zeros((3, 9), dtype=np.float32)
    logits[0, 0] = 1
    logits[1, 1] = 1
    logits[2, 0] = 1
    labels = np.array([0, 1, 1])
    coords = np.array([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])
    targets = np.array([[0.1, 0.1], [0.2, 0.8], [0.4, 0.6]])
    scores = score_inverse(logits, coords, labels, targets, majority=1)
    assert scores["action_accuracy"] == 2 / 3
    assert scores["action_macro_recall"] == 0.75
    assert scores["majority_action_accuracy"] == 2 / 3
    assert np.isclose(scores["click_coordinate_mae"], 0.1)
    assert np.isclose(scores["center_coordinate_mae"], 0.2)

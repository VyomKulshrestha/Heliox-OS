"""Inverse probe reports action and location errors separately."""

import numpy as np

from scripts.probe_world_visual_inverse import inverse_features, score_inverse


def test_inverse_features_keep_matched_input_width_without_after() -> None:
    partition = {
        "state": np.array([[0.2, 0.4]], dtype=np.float32),
        "target": np.array([[0.3, 0.1]], dtype=np.float32),
    }

    observed = inverse_features(partition, state_only=False)
    state_only = inverse_features(partition, state_only=True)
    assert observed.shape == state_only.shape == (1, 4)
    assert np.array_equal(observed[:, :2], state_only[:, :2])
    assert np.allclose(observed[:, 2:], [[0.1, -0.3]])
    assert np.array_equal(state_only[:, 2:], np.zeros((1, 2)))


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

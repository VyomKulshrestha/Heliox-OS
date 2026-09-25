"""The semantic task-state head keeps preprocessing and selection on training data."""

from __future__ import annotations

import numpy as np

from scripts.train_world_scope_semantic import metrics, normalize, select_head


def test_normalization_uses_only_training_split() -> None:
    transformed, mean, scale = normalize({
        "train": np.asarray([[0.0, 2.0], [2.0, 2.0]]),
        "test": np.asarray([[100.0, 2.0]]),
    })
    np.testing.assert_array_equal(mean, [1.0, 2.0])
    np.testing.assert_array_equal(scale, [1.0, 1.0])
    np.testing.assert_array_equal(transformed["test"], [[99.0, 0.0]])


def test_calibration_threshold_excludes_all_calibration_negatives() -> None:
    train = np.asarray([[-4.0], [-3.0], [-2.0], [-1.0],
                        [1.0], [2.0], [3.0], [4.0]])
    labels = np.asarray([False] * 4 + [True] * 4)
    calibration = np.asarray([[-2.5], [-0.5], [0.5], [2.5]])
    calibration_labels = np.asarray([False, False, True, True])
    head, threshold, trials = select_head(train, labels, calibration, calibration_labels)
    scores = head.predict_proba(calibration)[:, 1]
    assert len(trials) == 4
    assert threshold > scores[~calibration_labels].max()
    evidence = metrics(calibration_labels, scores, threshold)
    assert evidence["false_allowances"] == 0
    assert evidence["out_of_scope_rows"] == 2

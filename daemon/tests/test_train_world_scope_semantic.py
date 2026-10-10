"""The semantic task-state head keeps preprocessing and selection on training data."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.refscope_suite import ScopeCase
from scripts.train_world_scope_semantic import (
    feature_identity,
    load_feature_checkpoint,
    metrics,
    normalize,
    save_feature_checkpoint,
    select_head,
)


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


def test_feature_checkpoint_resumes_only_matching_case_content(tmp_path) -> None:
    cases = (ScopeCase("a", "Read file", "read", True, "test"),
             ScopeCase("b", "Read file", "delete", False, "test"),
             ScopeCase("c", "Open page", "open", True, "test"))
    identity = feature_identity(cases, 1024)
    path = tmp_path / "features.npz"
    save_feature_checkpoint(path, identity, 3, 1, [1, 3],
                            [np.array([1.0, 2.0]), np.array([3.0, 4.0])], [True, True])
    processed, overlong, indices, vectors, labels = load_feature_checkpoint(
        path, identity, cases)
    assert (processed, overlong, indices, labels) == (3, 1, [1, 3], [True, True])
    np.testing.assert_array_equal(vectors, [[1.0, 2.0], [3.0, 4.0]])
    assert feature_identity(cases, 768) != identity
    assert feature_identity(cases, 1024, batch_size=4) != identity
    with pytest.raises(ValueError, match="identity mismatch"):
        load_feature_checkpoint(path, feature_identity(cases, 768), cases)

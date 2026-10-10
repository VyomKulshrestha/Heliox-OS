"""Shape guards for offline semantic world scope artifact evaluation."""

import numpy as np
import pytest

from scripts.evaluate_world_scope_artifact import score_features


def test_score_features_uses_training_normalization():
    artifact = {
        "weights": np.array([[2.0, 0.0]]),
        "bias": np.array([0.0]),
        "mean": np.array([1.0, 0.0]),
        "scale": np.array([2.0, 1.0]),
    }
    assert score_features(artifact, np.array([[1.0, 3.0], [3.0, 3.0]])).tolist() == pytest.approx(
        [0.5, 0.8807970779778823]
    )


def test_score_features_rejects_mismatched_dimensions():
    artifact = {
        "weights": np.array([[1.0, 2.0]]),
        "bias": np.array([0.0]),
        "mean": np.array([0.0, 0.0]),
        "scale": np.array([1.0, 1.0]),
    }
    with pytest.raises(ValueError, match="dimensions"):
        score_features(artifact, np.array([[1.0, 2.0, 3.0]]))

"""Saved nonlinear heads are shape-checked before scoring."""

import numpy as np
import pytest
from scipy.special import expit, ndtr

from scripts.score_world_mlp_artifact import score_features


def test_score_saved_head_and_reject_bad_feature_width(tmp_path):
    path = tmp_path / "head.npz"
    np.savez_compressed(
        path,
        w1=np.asarray([[1.0]]),
        b1=np.asarray([0.0]),
        w2=np.asarray([[1.0]]),
        b2=np.asarray([0.0]),
        mean=np.asarray([0.0]),
        scale=np.asarray([1.0]),
    )
    with np.load(path, allow_pickle=False) as artifact:
        scores = score_features(artifact, np.asarray([[0.0], [1.0]]))
        np.testing.assert_allclose(scores, expit([0.0, ndtr(1.0)]))
        with pytest.raises(ValueError, match="dimensions"):
            score_features(artifact, np.asarray([[1.0, 2.0]]))

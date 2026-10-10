import numpy as np
import pytest

from scripts.evaluate_opendec_scope_head_transfer import score
from scripts.fit_opendec_scope_head import check_splits, fit, metrics


def test_metrics_separate_false_allowances_from_false_warnings():
    labels = np.asarray([True, True, False, False])
    probabilities = np.asarray([0.9, 0.2, 0.8, 0.1])
    result = metrics(labels, probabilities, 0.5)
    assert result["false_allowances"] == 1
    assert result["false_warnings"] == 1
    assert result["balanced_accuracy"] == 0.5


def test_split_guard_rejects_shared_security_trajectory():
    parts = {
        split: {"ids": np.asarray(ids), "labels": np.asarray([True, False])}
        for split, ids in {
            "train": ["group-a:call_1", "group-a:call_2"],
            "calibration": ["group-b:call_1", "group-b:call_2"],
            "test": ["group-a:call_3", "group-c:call_1"],
        }.items()
    }
    metadata = {
        split: {"source": {"checkpoint": "pinned"}, "scopejudge": {"sha256": "pinned"}, "feature_dimension": 2}
        for split in parts
    }
    with pytest.raises(ValueError, match="cross partitions"):
        check_splits(parts, metadata)


def test_transfer_scorer_uses_frozen_scaler_and_head():
    data = {"features": np.asarray([[3.0, 5.0], [1.0, 1.0]])}
    head = {
        "mean": np.asarray([1.0, 1.0]),
        "scale": np.asarray([2.0, 4.0]),
        "coefficient": np.asarray([[1.0, 1.0]]),
        "intercept": np.asarray([0.0]),
    }
    np.testing.assert_allclose(score(data, head), [1 / (1 + np.exp(-2)), 0.5])


def test_weighted_head_fits_without_using_test_for_selection(tmp_path):
    parts = {
        "train": {
            "ids": np.asarray(["a:1", "a:2", "b:1", "c:1", "d:1", "e:1"]),
            "labels": np.asarray([False, False, False, True, True, True]),
            "features": np.asarray([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]]),
        },
        "calibration": {
            "ids": np.asarray(["f:1", "g:1"]),
            "labels": np.asarray([False, True]),
            "features": np.asarray([[0.3], [0.7]]),
        },
        "test": {
            "ids": np.asarray(["h:1", "i:1"]),
            "labels": np.asarray([False, True]),
            "features": np.asarray([[0.25], [0.75]]),
        },
    }
    weighted = fit(parts, artifact=tmp_path / "weighted.npz", group_weighting="inverse_sqrt")
    assert weighted["group_weighting"] == "inverse_sqrt"
    assert len(weighted["train_weights_sha256"]) == 64
    assert weighted["test_at_default_threshold"]["rows"] == 2
    control = fit(parts, artifact=tmp_path / "control.npz")
    assert "train_weights_sha256" not in control
    with pytest.raises(ValueError, match="Unknown"):
        fit(parts, artifact=tmp_path / "invalid.npz", group_weighting="invalid")

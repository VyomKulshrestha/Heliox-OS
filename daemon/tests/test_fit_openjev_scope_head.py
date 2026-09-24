import numpy as np
import pytest

from scripts.evaluate_openjev_scope_head_transfer import score
from scripts.fit_openjev_scope_head import check_splits, metrics


def test_metrics_separate_false_allowances_from_false_warnings():
    labels = np.asarray([True, True, False, False])
    probabilities = np.asarray([0.9, 0.2, 0.8, 0.1])
    result = metrics(labels, probabilities, 0.5)
    assert result["false_allowances"] == 1
    assert result["false_warnings"] == 1
    assert result["balanced_accuracy"] == 0.5


def test_split_guard_rejects_shared_security_trajectory():
    parts = {split: {"ids": np.asarray(ids), "labels": np.asarray([True, False])}
             for split, ids in {
                 "train": ["group-a:call_1", "group-a:call_2"],
                 "calibration": ["group-b:call_1", "group-b:call_2"],
                 "test": ["group-a:call_3", "group-c:call_1"],
             }.items()}
    metadata = {split: {"source": {"checkpoint": "pinned"}, "scopejudge": {"sha256": "pinned"},
                        "feature_dimension": 2} for split in parts}
    with pytest.raises(ValueError, match="cross partitions"):
        check_splits(parts, metadata)


def test_transfer_scorer_uses_frozen_scaler_and_head():
    data = {"features": np.asarray([[3.0, 5.0], [1.0, 1.0]])}
    head = {"mean": np.asarray([1.0, 1.0]), "scale": np.asarray([2.0, 4.0]),
            "coefficient": np.asarray([[1.0, 1.0]]), "intercept": np.asarray([0.0])}
    np.testing.assert_allclose(score(data, head), [1 / (1 + np.exp(-2)), 0.5])

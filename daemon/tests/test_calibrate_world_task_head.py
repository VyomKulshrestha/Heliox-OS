"""Regression checks for research-only frozen-head calibration."""

import numpy as np
import pytest
from scipy.special import ndtr

from scripts.calibrate_world_task_head import (
    balanced_calibration_weights,
    calibration_metrics,
    fit_calibrators,
    infer_logits,
    weighted_nll,
)


def test_infer_logits_matches_linear_gelu_linear():
    features = np.asarray([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    weights = {
        "mean": np.asarray([1.0, 1.0], dtype=np.float32),
        "scale": np.asarray([2.0, 1.0], dtype=np.float32),
        "w1": np.asarray([[2.0, -1.0]], dtype=np.float32),
        "b1": np.asarray([0.5], dtype=np.float32),
        "w2": np.asarray([[3.0]], dtype=np.float32),
        "b2": np.asarray([-0.25], dtype=np.float32),
    }
    hidden = np.asarray([-0.5, 3.5])
    np.testing.assert_allclose(infer_logits(features, weights),
                               3.0 * hidden * ndtr(hidden) - 0.25, rtol=1e-6)
    with pytest.raises(ValueError, match="incompatible shapes"):
        infer_logits(features[:, :1], weights)


def test_calibration_weights_equalize_domains_and_classes():
    labels = {
        "a_calibration": np.asarray([True, False, False]),
        "b_calibration": np.asarray([True, True, False]),
    }
    weights = balanced_calibration_weights(labels, ("a", "b"))
    np.testing.assert_allclose([
        weights[0], weights[1:3].sum(), weights[3:5].sum(), weights[5],
    ], [0.25] * 4)
    with pytest.raises(ValueError, match="lacks a class"):
        balanced_calibration_weights({"a_calibration": np.asarray([True])}, ("a",))


def test_monotone_calibrators_reduce_fitting_nll_without_changing_ranking():
    logits = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.asarray([False, False, True, False, True, True])
    weights = np.full(len(logits), 1 / len(logits))
    maps = fit_calibrators(logits, labels, weights)
    baseline = weighted_nll(logits, labels, weights)
    assert set(maps) == {"raw", "temperature", "platt"}
    for scale, bias in maps.values():
        assert scale > 0
        assert weighted_nll(logits * scale + bias, labels, weights) <= baseline + 1e-6
        np.testing.assert_array_equal(np.argsort(logits * scale + bias), np.argsort(logits))


def test_fixed_acceptance_grid_counts_negative_examples():
    result = calibration_metrics(np.asarray([True, False, True, False]),
                                 np.asarray([0.95, 0.91, 0.8, 0.1]))
    assert result["false_positives_at_0.5"] == 1
    assert result["acceptance"]["0.9"] == {
        "accepted": 2, "positive_accepted": 1, "negative_accepted": 1,
    }
    with pytest.raises(ValueError, match="Invalid probability"):
        calibration_metrics(np.asarray([True]), np.asarray([1.2]))

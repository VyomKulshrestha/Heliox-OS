"""Threshold fitting must use only calibration probabilities and labels."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_general_binary_calibration import binary_metrics, choose_threshold


def test_threshold_is_selected_from_calibration_ranking() -> None:
    values = np.array([0.1, 0.2, 0.7, 0.8])
    labels = np.array([False, False, True, True])
    threshold, metrics = choose_threshold(values, labels)
    assert 0.2 < threshold <= 0.7
    assert metrics["balanced_accuracy"] == 1.0
    assert binary_metrics(values, labels, threshold)["confusion"] == {
        "true_positive": 2,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
    }


def test_binary_metrics_rejects_single_class_or_invalid_probability() -> None:
    with pytest.raises(ValueError, match="both classes"):
        binary_metrics(np.array([0.1, 0.9]), np.array([False, False]), 0.5)
    with pytest.raises(ValueError, match="Invalid binary"):
        binary_metrics(np.array([0.1, 1.1]), np.array([False, True]), 0.5)

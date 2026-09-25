"""Frozen artifact comparison is label-safe and checks typed probabilities."""

from __future__ import annotations

import pytest

from scripts.evaluate_general_decision_artifact import prediction_metrics


def test_prediction_metrics_includes_ordinal_rank_error() -> None:
    rows = [{"kind": "noul", "source": "public", "options": ["no", "yes"],
             "target": [0.0, 1.0]},
            {"kind": "score", "source": "public", "options": ["low", "mid", "high"],
             "target": [0.0, 0.0, 1.0]}]
    predictions = [{"answer": {"yes_probability": 0.8}},
                   {"answer": {"probabilities": {"low": 1 / 3, "mid": 1 / 3,
                                                  "high": 1 / 3}}}]
    report = prediction_metrics(rows, predictions)
    assert report["by_kind"]["noul"]["top1_accuracy"] == 1.0
    assert report["by_kind"]["score"]["mean_rank_error"] == pytest.approx(1.0)


def test_prediction_metrics_rejects_mismatched_or_invalid_outputs() -> None:
    rows = [{"kind": "noul", "source": "public", "options": ["no", "yes"],
             "target": [0.0, 1.0]}]
    with pytest.raises(ValueError, match="count"):
        prediction_metrics(rows, [])
    with pytest.raises(ValueError, match="probabilities"):
        prediction_metrics(rows, [{"answer": {"yes_probability": 1.2}}])

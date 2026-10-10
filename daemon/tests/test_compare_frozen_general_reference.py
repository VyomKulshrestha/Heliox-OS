"""Matched Jev evaluation uses only held-out public rows and local predictions."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import compare_frozen_general_reference as comparison
from scripts.compare_frozen_general_reference import compatible_indices, local_predictions


def test_compatible_rows_preserve_order_without_task_specific_labels() -> None:
    rows = [{"kind": "choice", "options": ["a", "b"]},
            {"kind": "score", "options": [str(i) for i in range(11)]},
            {"kind": "noul", "options": ["no", "yes"]}]
    assert compatible_indices(rows, 2) == [0, 2]
    with pytest.raises(ValueError, match="Not enough"):
        compatible_indices(rows, 3)


def test_local_predictions_keep_typed_option_semantics() -> None:
    rows = [{"kind": "noul", "options": ["no", "yes"]},
            {"kind": "choice", "options": ["left", "right"]},
            {"kind": "score", "options": ["low", "high"]}]
    features = (np.array([[2.0], [-1.0], [1.0], [-1.0], [1.0]]),
                np.array([0, 1, 3, 5]))
    predictions = local_predictions(rows, features, np.array([1.0, 0.0]),
                                    np.array([0.0]), np.array([1.0]), 1.0)
    assert predictions[0]["answer"]["yes_probability"] > 0.5
    assert predictions[1]["answer"]["choice"] == "right"
    assert predictions[2]["answer"]["expected_index"] > 0.5
    with pytest.raises(ValueError, match="dimensions"):
        local_predictions(rows, features, np.array([1.0, 0.0]),
                          np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0)


def test_archived_matched_comparison_fails_closed_before_reading_inputs_or_billing(tmp_path, monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("Archived Jev comparison must stop before reading inputs or billing")

    monkeypatch.setattr(comparison, "load_experiment", forbidden)
    monkeypatch.setattr(comparison, "_key", forbidden)
    monkeypatch.setattr(comparison, "query_jev", forbidden)
    output = tmp_path / "comparison.json"
    with pytest.raises(SystemExit, match="written TypeSafe authorization"):
        comparison.run(
            tmp_path / "extraction.json",
            tmp_path,
            tmp_path / "fit.json",
            tmp_path / "weights.npz",
            tmp_path / "key",
            output,
            2,
        )
    assert not output.exists()

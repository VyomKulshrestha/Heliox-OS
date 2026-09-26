"""Matched Jev evaluation uses only held-out public rows and local predictions."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts import compare_frozen_general_jev as comparison
from scripts.compare_frozen_general_jev import compatible_indices, local_predictions
from scripts.extract_general_frozen_features import MODEL, REVISION


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


def test_matched_comparison_writes_aggregates_not_teacher_answers(tmp_path, monkeypatch) -> None:
    rows = [
        {"id": "noul-1", "source": "public", "kind": "noul",
         "options": ["no", "yes"], "target": [0.0, 1.0]},
        {"id": "choice-1", "source": "public", "kind": "choice",
         "options": ["left", "right"], "target": [0.0, 1.0]},
    ]
    features = (np.array([[2.0], [-1.0], [1.0]]), np.array([0, 1, 3]))
    extraction = tmp_path / "extraction.json"
    extraction.write_text("{}", encoding="utf-8")
    weights = tmp_path / "weights.npz"
    with weights.open("wb") as stream:
        np.savez_compressed(stream, weights=np.array([1.0, 0.0]), mean=np.array([0.0]),
                            scale=np.array([1.0]), model_revision=np.asarray(REVISION))
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({
        "purpose": "unpromoted_frozen_public_general_decision_head",
        "model": MODEL, "revision": REVISION,
        "extraction_report_sha256": hashlib.sha256(extraction.read_bytes()).hexdigest(),
        "weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "temperature": 1.0,
    }), encoding="utf-8")
    monkeypatch.setattr(comparison, "load_experiment", lambda *_: (
        {"dataset_revision": "public-revision"}, {"ood": rows}, {"ood": features}))
    monkeypatch.setattr(comparison, "_key", lambda *_: "test-key")
    monkeypatch.setattr(comparison, "query_jev", lambda _client, _key, row: (
        {"noul": 0.8} if row["kind"] == "noul" else {"choice": "option_1"}, 0.0, "mock"))
    output = tmp_path / "comparison.json"
    report = comparison.run(extraction, tmp_path, fit, weights, tmp_path / "key", output, 2)
    assert report["calls"] == 2
    assert report["comparison"]["choice"]["agreement"] == 1.0
    assert "option_1" not in output.read_text(encoding="utf-8")
    assert "0.8" not in output.read_text(encoding="utf-8")

"""Frozen artifact comparison is label-safe and checks typed probabilities."""

from __future__ import annotations

import json

import pytest

from scripts.evaluate_general_decision_artifact import (
    load_partial,
    permute_options,
    prediction_metrics,
    retain_tokenizable_rows,
)


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


def test_option_permutation_preserves_labels_and_binary_order() -> None:
    rows = [{"kind": "choice", "options": ["a", "b", "c", "d"], "target": [0, 0, 1, 0]},
            {"kind": "noul", "options": ["no", "yes"], "target": [0, 1]},
            {"kind": "score", "options": ["low", "mid", "high"], "target": [0, 0, 1]}]
    first = permute_options(rows, 7)
    assert first == permute_options(rows, 7)
    assert first[0]["options"][first[0]["target"].index(1)] == "c"
    assert first[1] == rows[1]
    assert first[2] == rows[2]
    assert rows[0]["options"] == ["a", "b", "c", "d"]


def test_retain_tokenizable_rows_skips_overlong_without_reordering() -> None:
    rows = [{"id": "first"}, {"id": "too-long"}, {"id": "last"}]
    encoded = [{"row": rows[0]}, {"row": rows[2]}]

    retained, skipped = retain_tokenizable_rows(rows, encoded)

    assert [row["id"] for row in retained] == ["first", "last"]
    assert skipped == 1


def test_retain_tokenizable_rows_rejects_missing_or_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="stable IDs"):
        retain_tokenizable_rows([{"id": "row"}], [{"row": {}}])
    with pytest.raises(ValueError, match="do not match"):
        retain_tokenizable_rows([{"id": "row"}], [{"row": {"id": "other"}}])


def test_frozen_partial_cache_requires_exact_identity_and_explicit_resume(tmp_path) -> None:
    path = tmp_path / "evaluation.partial.json"
    assert load_partial(path, signature="pinned", ids=["a", "b"], resume=True) == []
    path.write_text(json.dumps({"schema_version": 1, "signature": "pinned",
                                "ids": ["a", "b"], "predictions": [{"answer": {}}]}))
    with pytest.raises(FileExistsError):
        load_partial(path, signature="pinned", ids=["a", "b"], resume=False)
    assert len(load_partial(path, signature="pinned", ids=["a", "b"], resume=True)) == 1
    with pytest.raises(ValueError, match="does not match"):
        load_partial(path, signature="other", ids=["a", "b"], resume=True)
    with pytest.raises(ValueError, match="does not match"):
        load_partial(path, signature="pinned", ids=["b", "a"], resume=True)

"""Matched Jev comparison selection and metrics remain label-only and paired."""

import pytest

from scripts.compare_scopejudge_reference import balanced_sample, summarize


def test_balanced_selection_uses_both_classes_deterministically() -> None:
    rows = [{"id": str(index), "target": [float(index % 2), float(not index % 2)]}
            for index in range(10)]
    first = balanced_sample(rows, 3, 17)
    assert first == balanced_sample(rows, 3, 17)
    assert len(first) == 6
    assert sum(rows[index]["target"][1] for index in first) == 3
    with pytest.raises(ValueError, match="positive"):
        balanced_sample(rows, 0, 17)


def test_paired_summary_exposes_false_allowances() -> None:
    result = summarize([True, True, False, False],
                       [0.9, 0.3, 0.7, 0.2], [0.8, 0.9, 0.1, 0.2])
    assert result["local"]["false_allowances"] == 1
    assert result["local"]["false_warnings"] == 1
    assert result["local"]["balanced_accuracy"] == 0.5
    assert result["jev"]["balanced_accuracy"] == 1.0
    assert result["local_jev_agreement"] == 0.5
    with pytest.raises(ValueError, match="within"):
        summarize([True, False], [0.5, 0.5], [1.1, 0.1])

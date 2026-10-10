"""The frozen decision head is generic and fits only supplied training labels."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.fit_general_frozen_head import fit_head, logits_for_rows, loss_gradient


def _row(kind: str, target: list[float]) -> dict:
    return {"kind": kind, "target": target, "options": ["no", "yes"] if kind == "noul" else ["left", "right"]}


def test_analytic_gradient_matches_finite_difference() -> None:
    rows = [_row("noul", [0.0, 1.0]), _row("choice", [1.0, 0.0])]
    data = (np.array([[0.4, 0.2], [0.7, -0.3], [-0.2, 0.5]]), np.array([0, 1, 3]))
    parameters = np.array([0.2, -0.4, 0.1])
    _, gradient = loss_gradient(parameters, rows, data, 0.01)
    for index in range(len(parameters)):
        shift = np.zeros(len(parameters))
        shift[index] = 1e-6
        plus = loss_gradient(parameters + shift, rows, data, 0.01)[0]
        minus = loss_gradient(parameters - shift, rows, data, 0.01)[0]
        assert gradient[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-7)


def test_shared_head_learns_binary_and_choice_without_test_labels() -> None:
    rows = [_row("noul", [0.0, 1.0]), _row("noul", [1.0, 0.0]), _row("choice", [1.0, 0.0]), _row("choice", [0.0, 1.0])]
    data = (np.array([[2.0], [-2.0], [2.0], [-2.0], [-2.0], [2.0]]), np.array([0, 1, 2, 4, 6]))
    parameters, _ = fit_head(rows, data, 0.01)
    predicted = logits_for_rows(rows, data, parameters)
    assert all(
        int(np.argmax(values)) == int(np.argmax(row["target"])) for row, values in zip(rows, predicted, strict=True)
    )

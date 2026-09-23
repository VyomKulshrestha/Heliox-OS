"""Contract checks for the unpromoted candidate-conditioned scope probe."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.scope_candidate_encoder import HIDDEN_SIZE, candidate_differences, candidate_prompts
from scripts.probe_openjev_scope import probabilities


def test_candidate_prompts_share_context_and_differ_only_in_candidate() -> None:
    prompts = candidate_prompts("Read CPU usage", "Inspect system metrics")
    assert prompts[0][0] == prompts[1][0]
    assert "Read CPU usage" in prompts[0][0]
    assert "Inspect system metrics" in prompts[0][0]
    assert prompts[0][1] != prompts[1][1]


def test_candidate_prompts_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        candidate_prompts(" ", "Inspect system metrics")


def test_candidate_differences_preserve_pair_order() -> None:
    hidden = np.zeros((4, HIDDEN_SIZE), dtype=np.float32)
    hidden[0, 0] = 2
    hidden[1, 0] = 1
    hidden[2, 0] = 1
    hidden[3, 0] = 2
    differences = candidate_differences(hidden)
    assert differences.shape == (2, HIDDEN_SIZE)
    assert differences[:, 0].tolist() == [1, -1]
    with pytest.raises(ValueError):
        candidate_differences(hidden[:3])


def test_shared_scalar_head_is_candidate_softmax() -> None:
    class Head:
        coef_ = np.array([[2.0, 0.0]])

    actual = probabilities(Head(), np.array([[1.0, 0.0], [-1.0, 0.0]]))
    expected = np.exp([2.0, -2.0]) / (1 + np.exp([2.0, -2.0]))
    np.testing.assert_allclose(actual, expected)

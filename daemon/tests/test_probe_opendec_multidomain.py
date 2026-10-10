"""Mathematical checks for the offline shared candidate head."""

from __future__ import annotations

import json

import numpy as np

from scripts.probe_opendec_multidomain import candidate_prompts, objective, predict


def test_candidate_prompts_keep_question_context_shared() -> None:
    row = {
        "state_json": json.dumps("The browser is on a settings page."),
        "question": "What should happen next?",
        "options": ["Read settings", "Delete settings"],
    }
    prompts = candidate_prompts(row)
    assert len(prompts) == 2
    assert prompts[0][0] == prompts[1][0]
    assert "Read settings" in prompts[0][1]
    assert "Delete settings" in prompts[1][1]


def test_shared_head_gradient_and_variable_candidates() -> None:
    features = np.array([[0.2, 0.3], [0.4, -0.1], [-0.1, 0.2], [0.5, 0.2], [0.1, 0.6]], dtype=np.float64)
    offsets = np.array([0, 2, 5])
    targets = [np.array([1.0, 0.0]), np.array([0.2, 0.3, 0.5])]
    weights = np.array([0.5, -0.3])
    loss, gradient = objective(weights, features, offsets, targets)
    for index in range(len(weights)):
        delta = np.zeros_like(weights)
        delta[index] = 1e-5
        forward = objective(weights + delta, features, offsets, targets)[0]
        backward = objective(weights - delta, features, offsets, targets)[0]
        np.testing.assert_allclose(gradient[index], (forward - backward) / (2e-5), atol=1e-7)
    assert loss > 0
    probabilities = predict(weights, features, offsets)
    assert [len(row) for row in probabilities] == [2, 3]
    np.testing.assert_allclose([sum(row) for row in probabilities], [1.0, 1.0])


def test_shared_score_is_invariant_to_common_candidate_shift() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    offsets = np.array([0, 2])
    weights = np.array([1.0, -1.0])
    base = predict(weights, features, offsets)[0]
    shifted = predict(weights, features + np.array([4.0, -2.0]), offsets)[0]
    np.testing.assert_allclose(base, shifted)

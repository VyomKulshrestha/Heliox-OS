"""Frozen reward-head fitting keeps group alignment and learns from labels."""

import numpy as np

from scripts.fit_miniwob_reward_head import (
    binary_loss_gradient,
    candidate_prompts,
    fit_head,
    group_metrics,
)


def test_interleaved_group_prompts_remain_aligned_without_reward_leakage() -> None:
    rows = [
        {"group": group, "goal": "Click ONE", "candidate": {"label": label},
         "split": "train", "reward": reward}
        for group, label, reward in (
            ("a", "ONE", 1), ("b", "TWO", -1),
            ("a", "TWO", -1), ("b", "ONE", 1),
        )
    ]
    prompts = candidate_prompts(rows)
    assert len(prompts) == 4
    assert "control 0" in prompts[0] and "control 1" in prompts[2]
    assert "control 0" in prompts[1] and "control 1" in prompts[3]
    assert all("reward" not in prompt for prompt in prompts)
    assert group_metrics(rows, np.array([2.0, -1.0, -2.0, 1.0]), "train")["correct"] == 2


def test_regularized_binary_head_learns_toy_signal() -> None:
    matrix = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    weights = fit_head(matrix, labels, 0.1)
    assert weights.shape == (2,)
    assert weights[0] > 0
    loss, gradient = binary_loss_gradient(weights, matrix, labels, 0.1)
    assert np.isfinite(loss) and np.linalg.norm(gradient) < 1e-4

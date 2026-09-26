"""WebWorld trajectory isolation and action-shuffle controls."""

import hashlib
import json

import numpy as np
import pytest

from scripts.probe_webworld_transition import (
    action_permutation,
    clustered_advantage,
    load_features,
    weighted_mse,
)


def _archive(tmp_path, *, leak: bool = False, state_leak: bool = False):
    path = tmp_path / "features.npz"
    groups = np.array([1, 2, 3, 1 if leak else 4])
    np.savez_compressed(
        path, state=np.eye(1024, dtype=np.float32)[[0, 1, 2, 0 if state_leak else 3]],
        target=np.ones((4, 1024), dtype=np.float32),
        action_type=np.eye(256, dtype=np.float32)[:4],
        action_bound=np.eye(256, dtype=np.float32)[:4],
        group=groups, split=np.array(["train", "train", "validation", "eval"]),
    )
    report = tmp_path / "source.json"
    report.write_text(json.dumps({"features_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    return path, report


def test_loader_rejects_trajectory_leakage_and_tampered_features(tmp_path) -> None:
    path, report = _archive(tmp_path)
    assert len(load_features(path, report)["eval"]["state"]) == 1
    report.write_text(json.dumps({"features_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="does not match"):
        load_features(path, report)
    path, report = _archive(tmp_path, leak=True)
    with pytest.raises(ValueError, match="leaked"):
        load_features(path, report)
    path, report = _archive(tmp_path, state_leak=True)
    with pytest.raises(ValueError, match="pre-action feature leaked"):
        load_features(path, report)


def test_shuffle_keeps_action_types_and_metric_rewards_correct_target() -> None:
    types = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    permutation = action_permutation(types, np.random.default_rng(7))
    assert np.array_equal(types, types[permutation])
    state = np.zeros((1, 4), dtype=np.float32)
    target = np.ones((1, 4), dtype=np.float32)
    assert weighted_mse(target, state, target) < weighted_mse(state, state, target)


def test_clustered_advantage_resamples_whole_trajectories() -> None:
    result = clustered_advantage(
        np.array([0.2, 0.2, 0.3]), np.array([0.1, 0.1, 0.2]),
        np.array([1, 1, 2]), seed=7,
    )
    assert result["groups"] == 2
    assert result["cluster_bootstrap_95_low"] > 0
    with pytest.raises(ValueError, match="align"):
        clustered_advantage(np.array([0.1]), np.array([0.2, 0.3]), np.array([1]), seed=7)

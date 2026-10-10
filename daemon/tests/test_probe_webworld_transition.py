"""WebWorld trajectory isolation and action-shuffle controls."""

import hashlib
import json

import numpy as np
import pytest

from scripts.probe_webworld_transition import (
    action_permutation,
    clustered_advantage,
    feature_equivalent_action_pairs,
    load_features,
    paired_outcome_diagnostic,
    weighted_mse,
)


def _archive(
    tmp_path, *, leak: bool = False, state_leak: bool = False, with_path: bool = False, with_digest: bool = False
):
    path = tmp_path / "features.npz"
    groups = np.array([1, 2, 3, 1 if leak else 4])
    optional = {"action_path": np.eye(256, dtype=np.float32)[:4]} if with_path else {}
    if with_digest:
        optional["state_digest"] = np.arange(128, dtype=np.uint8).reshape(4, 32)
    np.savez_compressed(
        path,
        state=np.eye(1024, dtype=np.float32)[[0, 1, 2, 0 if state_leak else 3]],
        target=np.ones((4, 1024), dtype=np.float32),
        action_type=np.eye(256, dtype=np.float32)[:4],
        action_bound=np.eye(256, dtype=np.float32)[:4],
        group=groups,
        split=np.array(["train", "train", "validation", "eval"]),
        **optional,
    )
    report = tmp_path / "source.json"
    report.write_text(json.dumps({"features_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    return path, report


def test_loader_rejects_trajectory_leakage_and_tampered_features(tmp_path) -> None:
    path, report = _archive(tmp_path)
    partition = load_features(path, report)
    assert len(partition["eval"]["state"]) == 1
    assert len(partition["eval"]["clusters"]) == 1
    report.write_text(json.dumps({"features_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="does not match"):
        load_features(path, report)
    path, report = _archive(tmp_path, leak=True)
    with pytest.raises(ValueError, match="leaked"):
        load_features(path, report)
    path, report = _archive(tmp_path, state_leak=True)
    with pytest.raises(ValueError, match="pre-action feature leaked"):
        load_features(path, report)
    path, report = _archive(tmp_path, with_path=True)
    assert load_features(path, report)["train"]["action_path"].shape == (2, 256)
    path, report = _archive(tmp_path, with_digest=True)
    assert load_features(path, report)["train"]["state_digest"].shape == (2, 32)


def test_shuffle_keeps_action_types_and_metric_rewards_correct_target() -> None:
    types = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    permutation = action_permutation(types, np.random.default_rng(7))
    assert np.array_equal(types, types[permutation])
    state = np.zeros((1, 4), dtype=np.float32)
    target = np.ones((1, 4), dtype=np.float32)
    assert weighted_mse(target, state, target) < weighted_mse(state, state, target)


def test_clustered_advantage_resamples_whole_trajectories() -> None:
    result = clustered_advantage(
        np.array([0.2, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.2]),
        np.array([1, 1, 2]),
        seed=7,
    )
    assert result["groups"] == 2
    assert result["cluster_bootstrap_95_low"] > 0
    with pytest.raises(ValueError, match="align"):
        clustered_advantage(np.array([0.1]), np.array([0.2, 0.3]), np.array([1]), seed=7)


def test_feature_equivalent_pairs_reward_the_right_action_outcome() -> None:
    partition = {
        "state": np.zeros((4, 2), dtype=np.float32),
        "action_type": np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32),
        "action_bound": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        "target": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
    }
    pairs = feature_equivalent_action_pairs(partition)
    assert pairs.tolist() == [[0, 1], [1, 2]]
    correct = paired_outcome_diagnostic(partition["target"], partition["target"], pairs)
    assert correct["correct_action_wins"] == 2
    assert correct["mean_swapped_minus_correct_mse"] > 0
    tied = paired_outcome_diagnostic(np.zeros((4, 2), dtype=np.float32), partition["target"], pairs)
    assert tied["ties"] == 2
    with pytest.raises(ValueError, match="outside"):
        paired_outcome_diagnostic(partition["target"], partition["target"], np.array([[0, 4]]))


def test_exact_identity_excludes_hashed_state_aliases() -> None:
    partition = {
        "state": np.zeros((2, 2), dtype=np.float32),
        "state_digest": np.array([[1, 0], [2, 0]], dtype=np.uint8),
        "action_type": np.ones((2, 2), dtype=np.float32),
        "action_bound": np.eye(2, dtype=np.float32),
        "target": np.eye(2, dtype=np.float32),
    }
    assert feature_equivalent_action_pairs(partition).shape == (0, 2)
    partition["state_digest"][1] = partition["state_digest"][0]
    assert feature_equivalent_action_pairs(partition).tolist() == [[0, 1]]

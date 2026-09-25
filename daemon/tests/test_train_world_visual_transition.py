"""Visual model training never mixes episodes or trusts altered features."""

import hashlib
import json

import numpy as np
import pytest

from scripts.train_world_visual_transition import load_features, metrics, within_action_permutation


def _write_features(tmp_path, groups=("a", "b", "c")):
    features = tmp_path / "features.npz"
    state = np.zeros((3, 256), dtype=np.float32)
    target = np.ones((3, 256), dtype=np.float32)
    np.savez_compressed(
        features, state=state, target=target,
        action=np.array([1, 2, 3]),
        coords=np.full((3, 2), 0.5, dtype=np.float32),
        group=np.array(groups), split=np.array(["train", "validation", "eval"]),
    )
    report = tmp_path / "source.json"
    report.write_text(json.dumps({
        "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return features, report


def test_valid_features_produce_episode_disjoint_action_vectors(tmp_path) -> None:
    features, report = _write_features(tmp_path)
    rows = load_features(features, report)
    assert rows["train"]["action"].shape == (1, 11)
    assert rows["train"]["action"][0, 1] == 1
    assert rows["eval"]["groups"].tolist() == ["c"]
    assert metrics(rows["train"]["target"], rows["train"]["state"],
                   rows["train"]["target"])["mse"] == 0


def test_rejects_episode_leakage_and_changed_source(tmp_path) -> None:
    features, report = _write_features(tmp_path, groups=("a", "a", "c"))
    with pytest.raises(ValueError, match="Episode leaked"):
        load_features(features, report)
    report.write_text('{"features_sha256": "not-the-file"}', encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_features(features, report)


def test_location_permutation_keeps_action_types() -> None:
    actions = np.zeros((8, 11), dtype=np.float32)
    actions[:4, 1] = 1
    actions[4:, 2] = 1
    actions[:, 9] = np.arange(8) / 8
    permutation = within_action_permutation(actions, np.random.default_rng(7))
    assert sorted(permutation.tolist()) == list(range(8))
    assert np.array_equal(actions[:, :9], actions[permutation, :9])
    assert not np.array_equal(actions[:, 9], actions[permutation, 9])

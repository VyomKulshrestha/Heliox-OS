"""Visual model training never mixes episodes or trusts altered features."""

import hashlib
import json

import numpy as np
import pytest

from scripts.train_world_visual_transition import (
    feature_shape,
    load_features,
    load_recorded_features,
    metrics,
    within_action_permutation,
)


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


def test_rejects_feature_shape_mismatch(tmp_path) -> None:
    features, report = _write_features(tmp_path)
    source = json.loads(report.read_text(encoding="utf-8"))
    source["feature_shape"] = [3, 32, 32]
    report.write_text(json.dumps(source), encoding="utf-8")
    assert feature_shape(report) == (3, 32, 32)
    with pytest.raises(ValueError, match="Invalid visual transition"):
        load_features(features, report)
    source["feature_shape"] = [100, 32, 32]
    report.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded visual feature shape"):
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


def test_private_samples_split_by_recording_without_plaintext_files() -> None:
    class Store:
        def load_research_samples(self):
            return [
                {
                    "workflow_id": f"recording-{group}",
                    "samples": [
                        {
                            "action_type": "mouse_click" if index % 2 else "mouse_move",
                            "before": [0] * 256,
                            "after": [255] * 256,
                            "coords": [0.25, 0.75],
                            "post_observed": True,
                        }
                        for index in range(12)
                    ],
                }
                for group in range(4)
            ]

    rows = load_recorded_features(Store(), seed=7)
    assert {key: len(value["state"]) for key, value in rows.items()} == {
        "train": 24, "validation": 12, "eval": 12,
    }
    assert set(rows["train"]["groups"]).isdisjoint(rows["eval"]["groups"])
    assert rows["train"]["action"].shape == (24, 11)
    assert np.all(rows["eval"]["target"] == 1)


def test_private_samples_require_four_independent_recordings() -> None:
    class Store:
        def load_research_samples(self):
            return [{"workflow_id": "one", "samples": []}]

    with pytest.raises(ValueError, match="four separate"):
        load_recorded_features(Store(), seed=7)

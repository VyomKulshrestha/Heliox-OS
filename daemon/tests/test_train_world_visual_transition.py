"""Visual model training never mixes episodes or trusts altered features."""

import hashlib
import json

import numpy as np
import pytest

from scripts.train_world_visual_transition import (
    action_shuffle_diagnostic,
    combine_training_features,
    feature_shape,
    load_features,
    load_recorded_features,
    merge_recordings_with_shared_states,
    metrics,
    paired_counterfactual_permutation,
    paired_group_indices,
    pairwise_transition_loss,
    within_action_permutation,
)


def test_action_shuffle_diagnostic_requires_observed_action_advantage() -> None:
    grounded = action_shuffle_diagnostic(0.1, [0.2, 0.3, 0.4])
    ignored = action_shuffle_diagnostic(0.3, [0.2, 0.3, 0.4])

    assert grounded["weighted_mse_advantage"] > 0
    assert grounded["one_sided_shuffle_rank"] == 0.25
    assert ignored["one_sided_shuffle_rank"] == 0.75
    with pytest.raises(ValueError, match="finite and nonempty"):
        action_shuffle_diagnostic(0.1, [float("nan")])


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


def test_additional_source_keeps_primary_eval_and_rejects_screen_leakage(tmp_path) -> None:
    features, report = _write_features(tmp_path)
    primary = load_features(features, report)
    for index, split in enumerate(("train", "validation", "eval")):
        primary[split]["state"][:] = (index + 1) / 10
    additional = {
        split: {key: value.copy() for key, value in partition.items()}
        for split, partition in primary.items()
    }
    for index, split in enumerate(("train", "validation", "eval")):
        additional[split]["state"][:] = (index + 4) / 10
    merged = combine_training_features(primary, additional)
    assert len(merged["train"]["state"]) == 2
    assert len(merged["validation"]["state"]) == 2
    assert len(merged["eval"]["state"]) == 1
    assert merged["train"]["groups"].tolist() == ["primary:a", "additional:a"]
    assert np.array_equal(merged["eval"]["target"], primary["eval"]["target"])
    additional["train"]["state"][:] = primary["eval"]["state"]
    with pytest.raises(ValueError, match="pre-action screen leaked"):
        combine_training_features(primary, additional)


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


def test_counterfactual_permutation_requires_same_state_distinct_outcomes() -> None:
    actions = np.zeros((4, 11), dtype=np.float32)
    actions[:, 1] = 1
    actions[:, 9] = [0.25, 0.75, 0.25, 0.75]
    partition = {
        "groups": np.array(["a", "a", "b", "b"]),
        "state": np.zeros((4, 256), dtype=np.float32),
        "target": np.tile(np.array([[0], [1], [0], [1]], dtype=np.float32), (1, 256)),
        "action": actions,
    }
    assert paired_counterfactual_permutation(partition).tolist() == [1, 0, 3, 2]
    assert paired_group_indices(partition).tolist() == [[0, 1], [2, 3]]
    partition["action"][1, 1] = 0
    partition["action"][1, 2] = 1
    assert paired_counterfactual_permutation(partition).tolist() == [1, 0, 3, 2]
    partition["state"][1, 0] = 0.5
    with pytest.raises(ValueError, match="one state"):
        paired_counterfactual_permutation(partition)


def test_pairwise_loss_requires_action_specific_outcome_difference() -> None:
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    matched = pairwise_transition_loss(target, target)
    ignored = pairwise_transition_loss(np.full_like(target, 0.5), target)
    assert matched == 0
    assert ignored > 0
    with pytest.raises(ValueError, match="aligned action pairs"):
        pairwise_transition_loss(target[:1], target[:1])


def test_private_samples_split_by_recording_without_plaintext_files() -> None:
    class Store:
        def load_research_samples(self):
            return [
                {
                    "workflow_id": f"recording-{group}",
                    "samples": [
                        {
                            "action_type": "mouse_click" if index % 2 else "mouse_move",
                            "before": [group] * 256,
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


def test_private_recordings_with_same_pre_screen_stay_in_one_partition() -> None:
    class Store:
        def load_research_samples(self):
            return [
                {
                    "workflow_id": f"recording-{group}",
                    "samples": [
                        {
                            "action_type": "mouse_click",
                            "before": [0 if group == 1 else group] * 256,
                            "after": [255] * 256,
                            "coords": [0.25, 0.75],
                            "post_observed": True,
                        }
                        for _ in range(12)
                    ],
                }
                for group in range(5)
            ]

    rows = load_recorded_features(Store(), seed=7)
    states = {split: set(partition["state"][:, 0]) for split, partition in rows.items()}
    assert all(states[left].isdisjoint(states[right]) for left, right in
               (("train", "validation"), ("train", "eval"), ("validation", "eval")))
    assert sorted(len(partition["state"]) for partition in rows.values()) == [12, 12, 36]


def test_shared_pre_screen_merging_is_transitive() -> None:
    def row(value: int):
        before = np.full(256, value / 255, dtype=np.float32)
        return before, before, 1, np.array([0.5, 0.5], dtype=np.float32)

    merged = merge_recordings_with_shared_states([
        ("first", [row(1)]),
        ("bridge", [row(1), row(2)]),
        ("third", [row(2)]),
        ("separate", [row(3)]),
    ])

    assert sorted(len(rows) for _, rows in merged) == [1, 4]


def test_private_recordings_reject_insufficient_independent_pre_screens() -> None:
    class Store:
        def load_research_samples(self):
            return [
                {
                    "workflow_id": f"recording-{group}",
                    "samples": [
                        {"action_type": "mouse_click", "before": [0] * 256,
                         "after": [255] * 256, "coords": [0.25, 0.75], "post_observed": True}
                    ],
                }
                for group in range(4)
            ]

    with pytest.raises(ValueError, match="four independent"):
        load_recorded_features(Store(), seed=7)


def test_private_samples_require_four_independent_recordings() -> None:
    class Store:
        def load_research_samples(self):
            return [{"workflow_id": "one", "samples": []}]

    with pytest.raises(ValueError, match="four independent"):
        load_recorded_features(Store(), seed=7)

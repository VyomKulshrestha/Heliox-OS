"""Frozen visual transfer reports pair-level action evidence."""

import hashlib

import numpy as np
import pytest

from scripts.probe_world_visual_model_transfer import (
    audit_source_overlap,
    observed_pair_signal,
    paired_action_diagnostic,
    paired_discriminative_diagnostic,
    paired_evaluation,
    task_masks,
)


def test_independence_audit_rejects_relabelled_development_screens(tmp_path) -> None:
    source = tmp_path / "source.npz"
    np.savez(source, state=np.asarray([[0, 1]], dtype=np.float32), group=np.asarray(["old"]))
    report = {"features_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    target = {"state": np.asarray([[1, 0]], dtype=np.float32), "groups": np.asarray(["new"])}
    assert audit_source_overlap(source, report, target)["verified"] is True
    with pytest.raises(ValueError, match="frozen model report"):
        audit_source_overlap(source, {"features_sha256": "wrong"}, target)
    with pytest.raises(ValueError, match="all training sources"):
        audit_source_overlap(source, {**report, "additional_training_features_sha256": "extra"}, target)
    target["state"] = np.asarray([[0, 1]], dtype=np.float32)
    with pytest.raises(ValueError, match="1 prestates"):
        audit_source_overlap(source, report, target)
    target["state"] = np.asarray([[1, 0]], dtype=np.float32)
    target["groups"] = np.asarray(["old"])
    with pytest.raises(ValueError, match="1 groups"):
        audit_source_overlap(source, report, target)


def test_paired_evaluation_includes_retrieval_and_group_uncertainty() -> None:
    target = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    partition = {"state": np.zeros_like(target), "target": target,
                 "groups": np.asarray(["a", "a", "b", "b"]),
                 "action": np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)}
    result = paired_evaluation(target, target[[1, 0, 3, 2]], partition, seed=17, replicates=100)
    assert result["successor_retrieval"]["mean_group_top1_accuracy"] == 1
    assert result["successor_retrieval"]["mean_group_chance_accuracy"] == 0.5
    assert result["paired_action_group_bootstrap"]["ci95_high"] < 0
    partition = {key: value[:2] for key, value in partition.items()}
    result = paired_evaluation(target[:2], target[1::-1], partition, seed=17, replicates=100)
    assert "uncertainty_unavailable" in result


def test_paired_action_diagnostic_counts_groups_not_rows() -> None:
    state = np.zeros((4, 2), dtype=np.float32)
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    correct = target.copy()
    swapped = target.copy()
    swapped[:2] = target[1::-1]
    result = paired_action_diagnostic(
        correct, swapped, state, target, np.array(["a", "a", "b", "b"])
    )

    assert result["groups"] == 2
    assert result["correct_action_wins"] == 1
    assert result["ties"] == 1
    assert result["mean_weighted_mse_advantage"] > 0
    with pytest.raises(ValueError, match="exactly two"):
        paired_action_diagnostic(correct, swapped, state, target,
                                 np.array(["a", "a", "a", "b"]))


def test_discriminative_diagnostic_skips_pairs_without_distinct_pixels() -> None:
    target = np.array([[1, 0], [0, 1], [0, 0], [0.001, 0]], dtype=np.float32)
    correct = target.copy()
    swapped = target[[1, 0, 3, 2]]
    result = paired_discriminative_diagnostic(
        correct, swapped, target, np.array(["a", "a", "b", "b"])
    )
    assert result["groups_with_distinct_pixels"] == 1
    assert result["correct_action_wins"] == 1
    assert result["mean_mse_advantage"] > 0


def test_task_masks_align_with_eval_rows(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(path, split=np.asarray(["train", "eval", "eval", "eval", "eval"]),
             task=np.asarray(["train", "a", "a", "b", "b"]))
    assert task_masks(path, 4)["b"].tolist() == [False, False, True, True]
    assert task_masks(path, 1, "train")["train"].tolist() == [True]
    with pytest.raises(ValueError, match="align"):
        task_masks(path, 3)


def test_observed_pair_signal_reports_pixel_separation() -> None:
    target = np.asarray([[0, 0], [1, 0], [0, 0], [0, 0.2]], dtype=np.float32)
    signal = observed_pair_signal(target, np.asarray(["a", "a", "b", "b"]))
    assert signal["groups"] == 2
    assert signal["mean_pixel_mse"] == pytest.approx(0.26)

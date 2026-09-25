"""Frozen-feature integrity and balanced multi-domain weighting."""

import hashlib

import numpy as np
import pytest

from scripts.train_world_multidomain_mlp import DOMAINS, balanced_weights, load_features


def test_balanced_weights_equalize_domain_and_class_cells():
    labels = {f"{name}_train": np.asarray([True] * (index + 1) + [False] * 3)
              for index, name in enumerate(DOMAINS)}
    weights = balanced_weights(labels)
    offset = 0
    cell_sums = []
    for name in DOMAINS:
        targets = labels[f"{name}_train"]
        part = weights[offset:offset + len(targets)]
        cell_sums.extend((part[targets].sum(), part[~targets].sum()))
        offset += len(targets)
    np.testing.assert_allclose(cell_sums, [cell_sums[0]] * len(cell_sums))


def test_load_features_rejects_changed_checkpoint(tmp_path):
    path = tmp_path / "agentic_train.npz"
    np.savez_compressed(path, processed=2, indices=np.asarray([1, 2]),
                        features=np.asarray([[1.0], [2.0]]),
                        labels=np.asarray([True, False]))
    source = {"rows": {"agentic_train": 2}, "excluded_overlong": {"agentic_train": 0},
              "feature_checkpoint_sha256": {
                  "agentic_train": hashlib.sha256(path.read_bytes()).hexdigest()}}
    features, labels = load_features(tmp_path, source)
    np.testing.assert_array_equal(features["agentic_train"], [[1.0], [2.0]])
    np.testing.assert_array_equal(labels["agentic_train"], [True, False])
    source["feature_checkpoint_sha256"]["agentic_train"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        load_features(tmp_path, source)

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.scope_decision_experiment import FEATURE_COUNT
from benchmarks.scope_nli_features import REVISION
from benchmarks.scope_semantic_head import (
    MODEL_VERSION,
    SemanticScopeHead,
    combined_features,
    semantic_guard,
)


class _FakeExtractor:
    def transform(self, pairs):
        return np.tile(np.asarray([0.1, 0.8, 0.1, 0.1, 0.8, 0.1]), (len(pairs), 1))


def test_semantic_features_have_fixed_finite_layout():
    pairs = [("Read the report", "Read the report")]
    features = combined_features(pairs, _FakeExtractor().transform(pairs))
    assert features.shape == (1, FEATURE_COUNT + 6)
    assert np.all(np.isfinite(features.data))
    with pytest.raises(ValueError):
        combined_features(pairs, np.zeros((1, 5)))


def test_semantic_guard_only_reduces_explicit_expansion():
    assert semantic_guard("Show config changes", "Revert config changes", 0.9) == 0.05
    assert semantic_guard("Turn off Bluetooth", "Turn off Wi-Fi", 0.9) == 0.05
    assert semantic_guard("Turn off Wi-Fi", "Turn off wifi", 0.9) == 0.9
    assert semantic_guard("Publish 2.4.1", "Publish 2.4.2", 0.9) == 0.05
    assert semantic_guard("Review issue 51", "Close issue 52", 0.9) == 0.05
    assert semantic_guard("Revert config changes", "Revert config changes", 0.9) == 0.9
    assert semantic_guard("Set cell D4 to complete", "Write complete across column D", 0.9) == 0.05
    assert semantic_guard("Move ticket 82 to backlog", "Move every ticket to backlog", 0.9) == 0.05
    assert semantic_guard("List active browser downloads", "Read active download entries", 0.9) == 0.9
    assert semantic_guard("List active browser downloads", "Read entries and download files", 0.9) == 0.05
    assert semantic_guard("List active browser downloads", "Please delete the download entries", 0.9) == 0.05
    assert semantic_guard("Set cell D4 to complete", "Write cell D4 in column D as complete", 0.9) == 0.9


def test_missing_or_wrong_backbone_weights_abstain(tmp_path):
    assert not SemanticScopeHead(tmp_path / "missing.npz").available
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, weights=np.zeros(FEATURE_COUNT + 6), intercept=0.0,
                        model_version=MODEL_VERSION, backbone_revision="wrong")
    assert not SemanticScopeHead(path).available


def test_numeric_candidate_predicts_without_approving(tmp_path):
    path = tmp_path / "student.npz"
    np.savez_compressed(path, weights=np.zeros(FEATURE_COUNT + 6), intercept=0.0,
                        model_version=MODEL_VERSION, backbone_revision=REVISION)
    head = SemanticScopeHead(path, extractor=_FakeExtractor())
    assert head.available
    assert head.predict_many([("Read the report", "Read the report")]).tolist() == [0.5]
    assert head.predict_many([("Show config changes", "Revert config changes")]).tolist() == [0.05]

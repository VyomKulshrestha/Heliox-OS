from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pilot.intelligence.scope_decision import (
    FEATURE_COUNT,
    MODEL_VERSION,
    LocalScopeDecisionHead,
    apply_target_guard,
    feature_matrix,
)
from scripts.train_scope_decision import train


def test_feature_layout_is_fixed_and_finite():
    matrix = feature_matrix([("Read the README", "Read the README"), ("Read the README", "Delete the README")])
    assert matrix.shape == (2, FEATURE_COUNT)
    assert np.all(np.isfinite(matrix.data))


def test_missing_or_malformed_weights_abstain(tmp_path):
    assert LocalScopeDecisionHead(tmp_path / "missing.npz").predict("read", "read") is None
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, weights=np.zeros(4), intercept=0.0, model_version=MODEL_VERSION)
    assert LocalScopeDecisionHead(bad).predict("read", "read") is None
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a zip archive")
    assert LocalScopeDecisionHead(corrupt).predict("read", "read") is None


def test_numeric_weights_return_typed_probability_without_approving_anything(tmp_path):
    path = tmp_path / "scope.npz"
    np.savez_compressed(path, weights=np.zeros(FEATURE_COUNT), intercept=0.0, model_version=MODEL_VERSION)
    head = LocalScopeDecisionHead(path)
    assert head.available
    decision = head.predict("Read the README", "Read the README")
    assert decision is not None
    assert decision.in_scope_probability == 0.5
    assert decision.confidence == 0.0


def test_named_target_mismatch_only_reduces_probability():
    assert apply_target_guard("Run Heliox tests", "Run Ferrum tests", 0.9) == 0.05
    assert apply_target_guard("Publish v0.14.0", "Publish v0.15.0", 0.9) == 0.05
    assert apply_target_guard("Delete report-a.txt", "Delete report-b.txt", 0.9) == 0.05
    assert apply_target_guard("Check Heliox and Ferrum", "Check Heliox", 0.9) == 0.9


def test_training_uses_only_authored_train_split():
    report_path = Path(__file__).resolve().parents[1] / "benchmarks" / "jev_scope_teacher.json"
    coefficients, intercept, metrics = train(json.loads(report_path.read_text(encoding="utf-8")))

    assert coefficients.shape == (FEATURE_COUNT,)
    assert np.all(np.isfinite(coefficients)) and np.isfinite(intercept)
    assert metrics["train_cases"] == 36
    assert metrics["development"]["cases"] == 24
    assert metrics["final"]["cases"] == 12

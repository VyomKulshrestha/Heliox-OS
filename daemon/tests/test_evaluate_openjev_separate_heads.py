import numpy as np

from scripts.evaluate_openjev_separate_heads import (
    fit_relevance,
    gate_metrics,
    gated_probabilities,
)


def test_relevance_fit_uses_only_training_and_calibration(tmp_path):
    def rows(size):
        labels = np.asarray([True, False] * size)
        features = np.asarray([[1.0, 0.0] if label else [0.0, 1.0]
                               for label in labels])
        return {"features": features, "labels": labels}

    artifact = tmp_path / "relevance.npz"
    result = fit_relevance(rows(10), rows(4), artifact)
    assert artifact.exists()
    assert result["calibration"]["false_positives"] == 0
    assert result["threshold"] > 0.0


def test_separate_gate_never_turns_scope_rejection_into_permission():
    data = {"features": np.asarray([[-1.0], [1.0], [2.0]])}
    scope = {"mean": np.zeros(1), "scale": np.ones(1), "coefficient": np.ones((1, 1)),
             "intercept": np.zeros(1), "threshold": np.asarray(0.5)}
    relevance = {**scope, "threshold": np.asarray(0.85)}
    scope_probability, gated = gated_probabilities(data, scope, relevance)
    assert list(scope_probability >= 0.5) == [False, True, True]
    assert list(gated) == [False, False, True]
    assert "auc" not in gate_metrics(np.asarray([False, True, True]), gated)

import math
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.world_scope_head import world_scope_features
from pilot.intelligence.scope_decision import FEATURE_COUNT
from scripts.train_world_scope_head import metrics, scores


def test_world_scope_features_keep_prior_calls_separate() -> None:
    base = SimpleNamespace(request="Read the report", action="Read report.txt", prior_tool_calls=())
    history = SimpleNamespace(request="Read the report", action="Read report.txt",
                              prior_tool_calls=("file_list directory=reports",))
    features = world_scope_features((base, history))
    assert features.shape[0] == 2
    assert np.array_equal(features[0, :FEATURE_COUNT], features[1, :FEATURE_COUNT])
    assert not np.array_equal(features[0, FEATURE_COUNT:-2],
                              features[1, FEATURE_COUNT:-2])
    assert features[0, -2:].tolist() == [0.0, 0.0]
    assert features[1, -2:].tolist() == [0.5, 1.0]


def test_empty_world_scope_features_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        world_scope_features(())


def test_world_scope_metrics_count_false_allowances() -> None:
    labels = np.asarray([False, False, True, True])
    values = scores(np.asarray([[0.0], [1.0], [2.0], [3.0]]),
                    np.asarray([1.0]), -1.5)
    report = metrics(labels, values, 0.5)
    assert report["rows"] == 4
    assert report["false_allowances"] == 0
    assert report["false_warnings"] == 0


def test_cautious_threshold_excludes_maximum_negative_score() -> None:
    values = scores(np.asarray([[1.0]], dtype=np.float32),
                    np.asarray([0.5], dtype=np.float32), 0.0)
    threshold = math.nextafter(float(values[0]), math.inf)
    assert values.dtype == np.float64
    assert not bool(values[0] >= threshold)

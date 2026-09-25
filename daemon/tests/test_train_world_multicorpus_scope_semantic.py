"""Cross-domain training weights and calibration selection stay split-safe."""

import numpy as np
import pytest

from scripts.train_world_multicorpus_scope_semantic import balanced_domain_weights, select_head


def test_weights_equalize_each_domain_and_class():
    labels = {"tbac": np.array([True, True, False, False]),
              "scope": np.array([True, True, True, False])}
    weights = balanced_domain_weights(labels)
    assert weights.sum() == pytest.approx(8)
    assert weights[:2].sum() == pytest.approx(2)
    assert weights[2:4].sum() == pytest.approx(2)
    assert weights[4:7].sum() == pytest.approx(2)
    assert weights[7] == pytest.approx(2)


def test_select_head_never_reads_test_labels():
    targets = {"tbac": np.array([False, True, False, True]),
               "scope": np.array([False, True, False, True])}
    train = np.array([[-2.0], [2.0], [-1.0], [1.0]] * 2)
    calibration = {"tbac": np.array([[-1.5], [1.5], [-0.5], [0.5]]),
                   "scope": np.array([[-2.5], [2.5], [-0.2], [0.2]])}
    head, threshold, trials = select_head(train, targets, calibration, targets)
    assert len(trials) == 4
    assert head.coef_.shape == (1, 1)
    assert threshold > max(head.predict_proba(calibration[name][~targets[name]])[:, 1].max()
                           for name in targets)

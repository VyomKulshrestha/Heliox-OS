"""Pure-data guards for the offline QLoRA world task-state experiment."""

import numpy as np
import pytest

from scripts.train_world_scope_qlora import calibration_choice, domain_class_weights


def test_domain_class_weights_balance_both_corpora():
    rows = {"tbac_train": [("a", [], True), ("b", [], False)],
            "scope_train": [("c", [], True), ("d", [], True), ("e", [], False)]}
    weights = domain_class_weights(rows)
    assert sum(weights[domain, row[2]] for domain in ("tbac", "scope")
               for row in rows[f"{domain}_train"]) == pytest.approx(5)
    assert weights["scope", False] == pytest.approx(1.25)


def test_calibration_choice_uses_both_negative_sets():
    calibration = {"tbac": (np.array([False, True]), np.array([0.1, 0.9])),
                   "scope": (np.array([False, True]), np.array([0.7, 0.8]))}
    threshold, auc = calibration_choice(calibration)
    assert threshold > 0.7
    assert auc == {"tbac": 1.0, "scope": 1.0}

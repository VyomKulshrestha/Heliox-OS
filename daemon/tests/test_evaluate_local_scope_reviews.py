from __future__ import annotations

import numpy as np

from scripts.evaluate_local_scope_reviews import summarize


def test_local_review_summary_counts_both_error_directions():
    result = summarize(
        np.asarray([True, True, False, False]),
        np.asarray([0.9, 0.1, 0.9, 0.1]),
    )
    assert result == {
        "cases": 4,
        "reviewed_in_scope": 2,
        "reviewed_out_of_scope": 2,
        "agreement": 0.5,
        "false_allowances": 1,
        "false_warnings": 1,
    }

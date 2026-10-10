"""Research features for a task-state head alongside the OS world model.

Prior calls are observations, not permission. This head cannot approve actions.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack

from pilot.intelligence.scope_decision import feature_matrix


def world_scope_features(cases) -> np.ndarray:
    """Encode request/action and bounded prior-call/action relations separately."""
    cases = tuple(cases)
    if not cases:
        raise ValueError("At least one scope case is required")
    request_action = feature_matrix([(case.request, case.action) for case in cases])
    history_action = feature_matrix([(" ".join(case.prior_tool_calls), case.action) for case in cases])
    counts = csr_matrix(
        np.asarray(
            [[min(len(case.prior_tool_calls), 2) / 2, float(bool(case.prior_tool_calls))] for case in cases],
            dtype=np.float32,
        )
    )
    combined = hstack((request_action, history_action, counts), format="csr")
    return np.asarray(combined.toarray(), dtype=np.float32)

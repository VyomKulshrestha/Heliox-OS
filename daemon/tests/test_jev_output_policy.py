from __future__ import annotations

import pytest

from scripts.evaluate_jev_scope import evaluate as evaluate_jev_scope
from scripts.jev_output_policy import block_jev_output_use


def test_historical_output_use_is_blocked_without_verified_authorization():
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        block_jev_output_use()


def test_live_provider_evaluator_is_blocked_before_network_access():
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        evaluate_jev_scope(key="not-a-real-key", repeats=1)

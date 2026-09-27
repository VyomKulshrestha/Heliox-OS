"""Archived provider-output comparisons must remain disabled."""

from __future__ import annotations

import pytest

from scripts import evaluate_joint_openjev_general as evaluator


def test_archived_comparison_stops_before_loading_model_or_report(monkeypatch):
    monkeypatch.setattr(evaluator, "verify_artifact", lambda *_: pytest.fail("model loaded"))
    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        evaluator.main()

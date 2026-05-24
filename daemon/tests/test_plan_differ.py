"""Tests for PlanDiffer — partial re-planning on retry."""

from __future__ import annotations

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    FileParams,
    VerificationResult,
)
from pilot.agents.plan_differ import PlanDiffer


def _make_action(action_type: ActionType = ActionType.FILE_READ, target: str = "test") -> Action:
    return Action(
        action_type=action_type,
        target=target,
        parameters=FileParams(path=target),
    )


def _make_result(action: Action, success: bool, error: str | None = None) -> ActionResult:
    return ActionResult(
        action=action,
        success=success,
        output="ok" if success else "",
        error=error,
    )


def _make_plan(*actions: Action) -> ActionPlan:
    return ActionPlan(actions=list(actions), explanation="test plan", raw_input="test")


def _make_verification(failed_indices: list[int], details: list[str] | None = None) -> VerificationResult:
    return VerificationResult(
        passed=len(failed_indices) == 0,
        details=details or [],
        failed_actions=failed_indices,
        rollback_triggered=False,
    )


def test_diff_returns_only_failed_actions():
    """PlanDiffer should return only failed actions in retry plan."""
    a1 = _make_action(target="file1.txt")
    a2 = _make_action(target="file2.txt")
    a3 = _make_action(target="file3.txt")
    plan = _make_plan(a1, a2, a3)

    results = [
        _make_result(a1, success=True),
        _make_result(a2, success=False, error="Permission denied"),
        _make_result(a3, success=True),
    ]
    verification = _make_verification(failed_indices=[1])

    retry_plan, successful_results = PlanDiffer.diff(plan, results, verification)

    assert len(retry_plan.actions) == 1
    assert retry_plan.actions[0].target == "file2.txt"
    assert len(successful_results) == 2


def test_diff_falls_back_to_full_replan_when_majority_fail():
    """PlanDiffer should fall back to full re-plan if >50% actions fail."""
    a1 = _make_action(target="file1.txt")
    a2 = _make_action(target="file2.txt")
    a3 = _make_action(target="file3.txt")
    plan = _make_plan(a1, a2, a3)

    results = [
        _make_result(a1, success=False, error="err"),
        _make_result(a2, success=False, error="err"),
        _make_result(a3, success=True),
    ]
    verification = _make_verification(failed_indices=[0, 1])

    retry_plan, successful_results = PlanDiffer.diff(plan, results, verification)

    # Should return original plan unchanged
    assert len(retry_plan.actions) == 3
    assert len(successful_results) == 0


def test_diff_empty_plan():
    """PlanDiffer should handle empty plans gracefully."""
    plan = _make_plan()
    results = []
    verification = _make_verification(failed_indices=[])

    retry_plan, successful_results = PlanDiffer.diff(plan, results, verification)

    assert retry_plan == plan
    assert successful_results == []


def test_diff_all_pass():
    """PlanDiffer should return empty retry plan when all actions pass."""
    a1 = _make_action(target="file1.txt")
    a2 = _make_action(target="file2.txt")
    plan = _make_plan(a1, a2)

    results = [
        _make_result(a1, success=True),
        _make_result(a2, success=True),
    ]
    verification = _make_verification(failed_indices=[])

    retry_plan, successful_results = PlanDiffer.diff(plan, results, verification)

    assert len(successful_results) == 2


def test_merge_results_preserves_order():
    """merge_results should return results in original plan order."""
    a1 = _make_action(target="file1.txt")
    a2 = _make_action(target="file2.txt")
    a3 = _make_action(target="file3.txt")
    plan = _make_plan(a1, a2, a3)

    successful_results = [
        _make_result(a1, success=True),
        _make_result(a3, success=True),
    ]
    retry_results = [
        _make_result(a2, success=True),
    ]
    verification = _make_verification(failed_indices=[1])

    merged = PlanDiffer.merge_results(successful_results, retry_results, plan, verification)

    assert len(merged) == 3
    assert merged[0].action.target == "file1.txt"
    assert merged[2].action.target == "file3.txt"

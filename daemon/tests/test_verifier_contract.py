"""Fail-closed contracts for plan-to-result verification."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.agents.verifier import Verifier


def _action(target: str) -> Action:
    return Action(
        action_type=ActionType.FILE_READ,
        target=target,
        parameters=FileParams(path=target),
    )


@pytest.mark.asyncio
async def test_missing_result_fails_verification() -> None:
    plan = ActionPlan(actions=[_action("one.txt")])

    verification = await Verifier(MagicMock()).verify(plan, [])

    assert verification.passed is False
    assert verification.failed_actions == [0]
    assert any("RESULT COUNT MISMATCH" in detail for detail in verification.details)
    assert any("MISSING" in detail for detail in verification.details)


@pytest.mark.asyncio
async def test_mismatched_result_action_fails_verification() -> None:
    plan = ActionPlan(actions=[_action("planned.txt")])
    result = ActionResult(action=_action("different.txt"), success=True, output="data")

    verification = await Verifier(MagicMock()).verify(plan, [result])

    assert verification.passed is False
    assert verification.failed_actions == [0]
    assert any("MISMATCH" in detail for detail in verification.details)


@pytest.mark.asyncio
async def test_extra_result_fails_verification() -> None:
    planned = _action("planned.txt")
    plan = ActionPlan(actions=[planned])
    results = [
        ActionResult(action=planned, success=True, output="planned"),
        ActionResult(action=_action("unexpected.txt"), success=True, output="unexpected"),
    ]

    verification = await Verifier(MagicMock()).verify(plan, results)

    assert verification.passed is False
    assert verification.failed_actions == [1]
    assert any("UNEXPECTED" in detail for detail in verification.details)


@pytest.mark.asyncio
async def test_only_independent_postconditions_are_marked_observed(tmp_path) -> None:
    written = tmp_path / "written.txt"
    written.write_text("expected", encoding="utf-8")
    write = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(written),
        parameters=FileParams(path=str(written), content="expected"),
    )
    read = _action(str(written))
    plan = ActionPlan(actions=[write, read])
    results = [ActionResult(action=action, success=True) for action in plan.actions]

    verification = await Verifier(MagicMock()).verify(plan, results)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]

    written.write_text("different", encoding="utf-8")
    failed = await Verifier(MagicMock()).verify(plan, results)
    assert failed.passed is False
    assert failed.failed_actions == [0]
    assert failed.independently_observed_actions == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", [ActionType.FILE_COPY, ActionType.FILE_MOVE])
async def test_missing_copy_or_move_destination_is_not_verified(action_type, tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    action = Action(
        action_type=action_type,
        target=str(source),
        parameters=FileParams(path=str(source)),
    )

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)]
    )

    assert verification.passed is False
    assert verification.failed_actions == [0]
    assert verification.independently_observed_actions == []

"""Fail-closed contracts for plan-to-result verification."""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.agents import verifier as verifier_module
from pilot.agents.verifier import Verifier
from pilot.intelligence.world_model import EffectDomain, StructuredTransitionPredictor, WorldState


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
async def test_success_without_independent_postcondition_is_reported_not_verified(tmp_path) -> None:
    action = _action(str(tmp_path / "read.txt"))
    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output="data")])

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED — no independent postcondition" in verification.details[0]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", [ActionType.FILE_COPY, ActionType.FILE_MOVE])
async def test_predicted_file_effects_match_independently_observed_sandbox_outcomes(action_type, tmp_path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("sample", encoding="utf-8")
    action = Action(action_type=action_type, target=str(source),
                    parameters=FileParams(path=str(source), destination=str(destination)))
    prediction = StructuredTransitionPredictor().predict(WorldState.empty(), action)

    if action_type == ActionType.FILE_COPY:
        shutil.copyfile(source, destination)
    else:
        shutil.move(source, destination)
    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])
    effects = [effect for effect in prediction.expected_effects if effect.domain == EffectDomain.FILESYSTEM]

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert destination.exists() is True
    assert any(effect.target == str(destination) and effect.after == "present_or_updated"
               for effect in effects)
    if action_type == ActionType.FILE_MOVE:
        assert source.exists() is False
        assert any(effect.target == str(source) and effect.after == "absent" for effect in effects)


@pytest.mark.asyncio
async def test_copy_with_wrong_content_is_not_verified(tmp_path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("correct", encoding="utf-8")
    destination.write_text("incorrect", encoding="utf-8")
    action = Action(action_type=ActionType.FILE_COPY, target=str(source),
                    parameters=FileParams(path=str(source), destination=str(destination)))

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])

    assert verification.passed is False
    assert verification.failed_actions == [0]
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_directory_copy_existence_does_not_count_as_independent_outcome(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    action = Action(action_type=ActionType.FILE_COPY, target=str(source),
                    parameters=FileParams(path=str(source), destination=str(destination), recursive=True))

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_large_copy_is_not_labeled_independently_verified(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(verifier_module, "MAX_COPY_CONTENT_VERIFY_BYTES", 4)
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("sample", encoding="utf-8")
    destination.write_text("sample", encoding="utf-8")
    action = Action(action_type=ActionType.FILE_COPY, target=str(source),
                    parameters=FileParams(path=str(source), destination=str(destination)))

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]

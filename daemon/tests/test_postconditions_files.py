"""file_write, file_copy and download_file are compared with the file on disk, also in the awkward cases."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.agents.executor import Executor
from pilot.agents.postconditions import files
from pilot.agents.verifier import Verifier
from pilot.system.filesystem import file_write


async def _verify(action: Action, output: str = "", success: bool = True, error: str | None = None):
    return await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=success, output=output, error=error)]
    )


def _write_action(path, content: str | None) -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE, target=str(path), parameters=FileParams(path=str(path), content=content)
    )


# ── file_write ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_written_file_verifies_by_the_hash_the_handler_recorded(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    output = await file_write(str(target), "line one\nline two\n")

    assert "sha256" in output and output.startswith("Written 18 bytes to")
    verification = await _verify(_write_action(target, "line one\nline two\n"), output)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_substituted_content_is_verified_not_flagged(tmp_path) -> None:
    """The plan says {OUTPUT}; the executor wrote the earlier step's text. That write is correct."""
    target = tmp_path / "summary.txt"
    action = _write_action(target, "Summary:\n{OUTPUT}")
    executor = SimpleNamespace(
        _OUTPUT_PLACEHOLDERS=Executor._OUTPUT_PLACEHOLDERS,
        _PLACEHOLDER_REGEX=Executor._PLACEHOLDER_REGEX,
        _largest_output="the extracted page text",
        _last_output="the extracted page text",
    )

    output = await Executor._exec_file_write(executor, action)  # type: ignore[arg-type]

    assert target.read_bytes() == "Summary:\nthe extracted page text".replace("\n", os.linesep).encode()
    verification = await _verify(action, output)
    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_carriage_returns_in_the_content_still_verify(tmp_path) -> None:
    target = tmp_path / "windows.txt"
    content = "a\r\nb\n\rc"
    output = await file_write(str(target), content)

    assert (await _verify(_write_action(target, content), output)).independently_observed_actions == [0]
    # Without a recorded hash (an "Already satisfied" result), the plan's text is compared byte for byte.
    assert (await _verify(_write_action(target, content), "")).independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_a_file_changed_after_the_write_is_a_mismatch(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    output = await file_write(str(target), "expected text")
    target.write_bytes(b"expected tex")

    verification = await _verify(_write_action(target, "expected text"), output)

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "12 bytes" in verification.details[0] and "13 bytes" in verification.details[0]

    target.write_bytes(b"EXPECTED TEXT")
    verification = await _verify(_write_action(target, "expected text"), output)
    assert verification.passed is False
    assert "differs from what was written" in verification.details[0]


@pytest.mark.asyncio
async def test_a_missing_file_is_a_mismatch(tmp_path) -> None:
    verification = await _verify(_write_action(tmp_path / "never.txt", "x"), "Written 1 bytes")

    assert verification.passed is False
    assert "does not exist" in verification.details[0]


@pytest.mark.asyncio
async def test_no_content_and_no_record_is_reported_and_never_skips_the_write(tmp_path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("old", encoding="utf-8")
    action = _write_action(target, None)

    verification = await _verify(action, "Written 0 bytes")
    precheck = await Verifier(MagicMock()).check_already_satisfied(action)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert precheck is None


@pytest.mark.asyncio
async def test_the_precheck_still_compares_the_plan_text(tmp_path) -> None:
    target = tmp_path / "ready.txt"
    await file_write(str(target), "ready\n")

    same = await Verifier(MagicMock()).check_already_satisfied(_write_action(target, "ready\n"))
    other = await Verifier(MagicMock()).check_already_satisfied(_write_action(target, "other"))

    assert same is not None and same[0] is True
    assert other is not None and other[0] is False


@pytest.mark.asyncio
async def test_a_failed_write_never_reaches_the_check(tmp_path) -> None:
    verification = await _verify(_write_action(tmp_path / "x.txt", "x"), success=False, error="disk full")

    assert verification.passed is False
    assert "FAILED — disk full" in verification.details[0]


def test_file_write_puts_the_same_bytes_as_text_mode(tmp_path) -> None:
    import asyncio

    content = "one\ntwo\r\nthree"
    asyncio.run(file_write(str(tmp_path / "new.txt"), content))
    (tmp_path / "old.txt").write_text(content, "utf-8")

    assert (tmp_path / "new.txt").read_bytes() == (tmp_path / "old.txt").read_bytes()
    assert files._expected_text_bytes(content) == (tmp_path / "old.txt").read_bytes()

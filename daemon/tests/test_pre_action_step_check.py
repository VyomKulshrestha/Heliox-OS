from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    EmptyParams,
    FileParams,
    PackageParams,
    ServiceParams,
)
from pilot.agents.executor import Executor
from pilot.agents.verifier import Verifier
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system import browser


class _Locator:
    def __init__(self, value: str, *, count: int = 1, visible: bool = True) -> None:
        self._value = value
        self._count = count
        self._visible = visible

    async def count(self) -> int:
        return self._count

    async def is_visible(self) -> bool:
        return self._visible

    async def input_value(self) -> str:
        return self._value


class _Page:
    def __init__(self, url: str, values: dict[str, str] | None = None) -> None:
        self.url = url
        self._values = values or {}

    def locator(self, selector: str) -> _Locator:
        return _Locator(self._values.get(selector, ""))


def _install_page(monkeypatch: pytest.MonkeyPatch, page: _Page, requested_tabs: list[int] | None = None) -> None:
    async def _get_page(tab_index: int = -1):
        if requested_tabs is not None:
            requested_tabs.append(tab_index)
        return page

    monkeypatch.setattr(browser, "_browser_context", object())
    monkeypatch.setattr(browser, "_get_page", _get_page)


@pytest.mark.asyncio
async def test_browser_step_check_proves_exact_navigation_state(monkeypatch):
    requested_tabs: list[int] = []
    _install_page(monkeypatch, _Page("https://example.com/"), requested_tabs)

    same = await browser.browser_action_already_satisfied(
        "browser_navigate",
        BrowserParams(url="example.com", tab_index=2),
    )
    different = await browser.browser_action_already_satisfied(
        "browser_navigate",
        BrowserParams(url="https://example.com/docs"),
    )

    assert same is not None and same[0] is True
    assert different is not None and different[0] is False
    assert requested_tabs == [2, -1]


@pytest.mark.asyncio
async def test_browser_step_check_skips_only_side_effect_free_field_equality(monkeypatch):
    _install_page(monkeypatch, _Page("https://example.com/", {"#name": "Vyom", "#role": "founder"}))

    typed = await browser.browser_action_already_satisfied(
        "browser_type",
        BrowserParams(selector="#name", text="Vyom"),
    )
    submitted = await browser.browser_action_already_satisfied(
        "browser_type",
        BrowserParams(selector="#name", text="Vyom", press_enter=True),
    )
    form = await browser.browser_action_already_satisfied(
        "browser_fill_form",
        BrowserParams(fields={"#name": "Vyom", "#role": "founder"}),
    )
    submitted_form = await browser.browser_action_already_satisfied(
        "browser_fill_form",
        BrowserParams(fields={"#name": "Vyom"}, submit_selector="#submit"),
    )

    assert typed is not None and typed[0] is True
    assert submitted is None
    assert form is not None and form[0] is True
    assert submitted_form is None


@pytest.mark.asyncio
async def test_browser_step_check_does_not_suppress_ambiguous_locator(monkeypatch):
    class _AmbiguousPage(_Page):
        def locator(self, _selector: str) -> _Locator:
            return _Locator("Vyom", count=2)

    _install_page(monkeypatch, _AmbiguousPage("https://example.com/"))

    state = await browser.browser_action_already_satisfied(
        "browser_type",
        BrowserParams(selector="#name", text="Vyom"),
    )

    assert state is None


@pytest.mark.asyncio
async def test_verifier_observes_browser_field_from_the_live_session(monkeypatch):
    _install_page(monkeypatch, _Page("https://example.com/", {"#name": "ready"}))
    action = Action(
        action_type=ActionType.BROWSER_TYPE, target="#name", parameters=BrowserParams(selector="#name", text="ready")
    )

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)]
    )

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_verifier_does_not_learn_from_redirected_browser_navigation(monkeypatch):
    _install_page(monkeypatch, _Page("https://example.com/login"))
    action = Action(
        action_type=ActionType.BROWSER_NAVIGATE,
        target="https://example.com/account",
        parameters=BrowserParams(url="https://example.com/account"),
    )

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)]
    )

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_precheck_does_not_skip_action_for_nondefault_browser_tab(monkeypatch):
    requested_tabs: list[int] = []
    _install_page(monkeypatch, _Page("https://example.com/"), requested_tabs)
    action = Action(
        action_type=ActionType.BROWSER_NAVIGATE,
        target="https://example.com/",
        parameters=BrowserParams(url="https://example.com/", tab_index=2),
    )

    precheck = await Verifier(MagicMock()).check_already_satisfied(action)

    assert precheck is None
    assert requested_tabs == []


@pytest.mark.asyncio
async def test_verifier_precheck_is_conservative(tmp_path):
    target = tmp_path / "ready.txt"
    target.write_text("ready", encoding="utf-8")
    verifier = Verifier(MagicMock())

    matching = await verifier.check_already_satisfied(
        Action(
            action_type=ActionType.FILE_WRITE,
            target=str(target),
            parameters=FileParams(path=str(target), content="ready"),
        )
    )
    mismatch = await verifier.check_already_satisfied(
        Action(
            action_type=ActionType.FILE_WRITE,
            target=str(target),
            parameters=FileParams(path=str(target), content="different"),
        )
    )
    restart = await verifier.check_already_satisfied(
        Action(
            action_type=ActionType.SERVICE_RESTART,
            target="daemon",
            parameters=ServiceParams(name="daemon"),
        )
    )
    versioned_package = await verifier.check_already_satisfied(
        Action(
            action_type=ActionType.PACKAGE_INSTALL,
            target="tool",
            parameters=PackageParams(name="tool", version="2.0"),
        )
    )
    unsupported = await verifier.check_already_satisfied(
        Action(action_type=ActionType.CPU_USAGE, target="", parameters=EmptyParams())
    )

    assert matching is not None and matching[0] is True
    assert mismatch is not None and mismatch[0] is False
    assert restart is None
    assert versioned_package is None
    assert unsupported is None


def _executor(tmp_path) -> Executor:
    config = PilotConfig()
    return Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )


@pytest.mark.asyncio
async def test_executor_skips_proven_redundant_step_without_training_world_model(tmp_path):
    target = tmp_path / "ready.txt"
    target.write_text("ready", encoding="utf-8")
    action = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(target),
        parameters=FileParams(path=str(target), content="ready"),
    )
    executor = _executor(tmp_path)
    executor.set_step_checker(Verifier(MagicMock()))
    effect = AsyncMock(return_value=ActionResult(action=action, success=True, output="written"))
    executor._execute_single = effect
    recorder = MagicMock()
    executor.set_world_model_outcome_recorder(recorder)
    starts = AsyncMock()
    completions = AsyncMock()

    results = await executor.execute(
        ActionPlan(actions=[action], explanation="write ready"),
        on_action_start=starts,
        on_action_complete=completions,
    )

    assert results[0].success is True
    assert results[0].executed is False
    assert results[0].skip_reason == "already_satisfied"
    effect.assert_not_awaited()
    starts.assert_not_awaited()
    completions.assert_awaited_once()
    recorder.assert_not_called()
    entries = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()]
    assert [entry["event_type"] for entry in entries] == ["action_complete"]


@pytest.mark.asyncio
async def test_executor_continues_when_step_state_does_not_match(tmp_path):
    target = tmp_path / "ready.txt"
    target.write_text("old", encoding="utf-8")
    action = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(target),
        parameters=FileParams(path=str(target), content="new"),
    )
    executor = _executor(tmp_path)
    executor.set_step_checker(Verifier(MagicMock()))
    effect = AsyncMock(return_value=ActionResult(action=action, success=True, output="written"))
    executor._execute_single = effect

    results = await executor.execute(ActionPlan(actions=[action], explanation="write new"))

    assert results[0].executed is True
    effect.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_continues_when_step_check_is_inconclusive(tmp_path):
    action = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(tmp_path / "ready.txt"),
        parameters=FileParams(path=str(tmp_path / "ready.txt"), content="ready"),
    )
    executor = _executor(tmp_path)
    checker = MagicMock()
    checker.check_already_satisfied = AsyncMock(side_effect=RuntimeError("probe unavailable"))
    executor.set_step_checker(checker)
    effect = AsyncMock(return_value=ActionResult(action=action, success=True, output="written"))
    executor._execute_single = effect

    results = await executor.execute(ActionPlan(actions=[action], explanation="write ready"))

    assert results[0].executed is True
    effect.assert_awaited_once()

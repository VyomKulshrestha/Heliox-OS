"""Tests for SimulationSandbox's pre-execution browser-target assessment.

Covers the new dom_diff.assess_target() integration into simulate() (now
async): when no browser session is open (the common case for a dry-run),
this must be a complete no-op; when one is open, an unresolved click/type
target should bump risk to HIGH and surface a predicted_issue.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.sandbox import RiskLevel, SimulationSandbox
from pilot.system.dom_diff import DomNode, DomSnapshot


def _plan(action_type: ActionType, target: str = "x") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=action_type, target=target, parameters=EmptyParams())], raw_input="test"
    )


def _snapshot_with(*nodes: DomNode) -> DomSnapshot:
    return DomSnapshot(nodes=list(nodes), url="https://example.com", title="Test")


def _button(id: str = "submit-btn", visible: bool = True) -> DomNode:
    key = f"button|{id}||"
    return DomNode(tag="button", id=id, cls="", text="", depth=0, visible=visible, x=0, y=0, w=80, h=30, key=key)


class TestNoActiveSession:
    @pytest.mark.asyncio
    async def test_no_browser_session_is_a_complete_noop(self):
        sandbox = SimulationSandbox()
        with patch("pilot.system.browser.has_active_session", return_value=False):
            report = await sandbox.simulate(_plan(ActionType.BROWSER_CLICK, target="#submit-btn"))
        impact = report.impacts[0]
        assert impact.predicted_issue == ""
        assert impact.predicted_issue_is_problem is False
        assert impact.risk == RiskLevel.MEDIUM  # unchanged from the existing browser-action default


class TestActiveSessionTargetFound:
    @pytest.mark.asyncio
    async def test_found_visible_target_does_not_raise_risk(self):
        from pilot.actions import BrowserParams

        sandbox = SimulationSandbox()
        with (
            patch("pilot.system.browser.has_active_session", return_value=True),
            patch(
                "pilot.system.browser.peek_current_dom_snapshot",
                new=AsyncMock(return_value=_snapshot_with(_button())),
            ),
        ):
            action = Action(
                action_type=ActionType.BROWSER_CLICK,
                target="#submit-btn",
                parameters=BrowserParams(selector="#submit-btn"),
            )
            report = await sandbox.simulate(ActionPlan(actions=[action], raw_input="test"))
        impact = report.impacts[0]
        assert "found and visible" in impact.predicted_issue
        assert impact.predicted_issue_is_problem is False


class TestActiveSessionTargetMissing:
    @pytest.mark.asyncio
    async def test_missing_target_bumps_risk_to_high_and_flags_problem(self):
        sandbox = SimulationSandbox()
        with (
            patch("pilot.system.browser.has_active_session", return_value=True),
            patch(
                "pilot.system.browser.peek_current_dom_snapshot",
                new=AsyncMock(return_value=_snapshot_with(_button(id="other-btn"))),
            ),
        ):
            from pilot.actions import BrowserParams

            action = Action(
                action_type=ActionType.BROWSER_CLICK,
                target="#submit-btn",
                parameters=BrowserParams(selector="#submit-btn"),
            )
            report = await sandbox.simulate(ActionPlan(actions=[action], raw_input="test"))
        impact = report.impacts[0]
        assert impact.predicted_issue_is_problem is True
        assert "not found" in impact.predicted_issue
        assert impact.risk == RiskLevel.HIGH
        assert any("predicted to fail" in w for w in report.warnings)

    @pytest.mark.asyncio
    async def test_snapshot_exception_degrades_to_noop(self):
        """A live-DOM snapshot failure must never break the dry-run itself."""
        sandbox = SimulationSandbox()
        with (
            patch("pilot.system.browser.has_active_session", return_value=True),
            patch(
                "pilot.system.browser.peek_current_dom_snapshot",
                new=AsyncMock(side_effect=RuntimeError("page crashed")),
            ),
        ):
            from pilot.actions import BrowserParams

            action = Action(
                action_type=ActionType.BROWSER_CLICK,
                target="#submit-btn",
                parameters=BrowserParams(selector="#submit-btn"),
            )
            report = await sandbox.simulate(ActionPlan(actions=[action], raw_input="test"))
        impact = report.impacts[0]
        assert impact.predicted_issue == ""
        assert impact.predicted_issue_is_problem is False


class TestFillFormMultiTarget:
    @pytest.mark.asyncio
    async def test_all_fields_found_reports_summary(self):
        from pilot.actions import BrowserParams

        sandbox = SimulationSandbox()
        with (
            patch("pilot.system.browser.has_active_session", return_value=True),
            patch(
                "pilot.system.browser.peek_current_dom_snapshot",
                new=AsyncMock(return_value=_snapshot_with(_button(id="email"), _button(id="submit"))),
            ),
        ):
            action = Action(
                action_type=ActionType.BROWSER_FILL_FORM,
                target="#login-form",
                parameters=BrowserParams(fields={"#email": "me@example.com"}, submit_selector="#submit"),
            )
            report = await sandbox.simulate(ActionPlan(actions=[action], raw_input="test"))
        impact = report.impacts[0]
        assert impact.predicted_issue_is_problem is False
        assert "2 form target(s) found" in impact.predicted_issue

    @pytest.mark.asyncio
    async def test_missing_field_flagged_as_problem(self):
        from pilot.actions import BrowserParams

        sandbox = SimulationSandbox()
        with (
            patch("pilot.system.browser.has_active_session", return_value=True),
            patch(
                "pilot.system.browser.peek_current_dom_snapshot",
                new=AsyncMock(return_value=_snapshot_with(_button(id="submit"))),
            ),
        ):
            action = Action(
                action_type=ActionType.BROWSER_FILL_FORM,
                target="#login-form",
                parameters=BrowserParams(fields={"#email": "me@example.com"}, submit_selector="#submit"),
            )
            report = await sandbox.simulate(ActionPlan(actions=[action], raw_input="test"))
        impact = report.impacts[0]
        assert impact.predicted_issue_is_problem is True
        assert impact.risk == RiskLevel.HIGH

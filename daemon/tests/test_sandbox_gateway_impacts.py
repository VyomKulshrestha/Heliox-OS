"""Tests for SimulationSandbox's browser/system-control impact modeling.

Before this, SimulationSandbox was shell/file-oriented only — a dry-run of a
plan containing browser or mouse/keyboard/registry actions produced a
generic fallback description instead of meaningful impact analysis. Also
covers the pre-existing dead-string-literal bug (risk sets referencing
ActionType values that don't actually exist, e.g. "power_action" instead of
"power_shutdown"/"power_restart") found while extending this module.
"""

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.sandbox import (
    ROOT_ACTIONS,
    SANDBOX_DESTRUCTIVE_ACTIONS,
    SANDBOX_HIGH_RISK_ACTIONS,
    RiskLevel,
    SimulationSandbox,
)


def _plan(action_type: ActionType, target: str = "x") -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=action_type, target=target, parameters=EmptyParams())], raw_input="test"
    )


class TestDeadLiteralsFixed:
    """The pre-existing sets used string literals that never matched a real
    ActionType.value — e.g. "power_action" when the real values are
    "power_shutdown"/"power_restart". These assert the real values are now
    present (and, implicitly via absence, that the fixed sets don't still
    carry the old dead strings)."""

    def test_destructive_actions_uses_real_power_values(self):
        assert "power_shutdown" in SANDBOX_DESTRUCTIVE_ACTIONS
        assert "power_restart" in SANDBOX_DESTRUCTIVE_ACTIONS
        assert "power_action" not in SANDBOX_DESTRUCTIVE_ACTIONS

    def test_destructive_actions_uses_real_package_value(self):
        assert "package_remove" in SANDBOX_DESTRUCTIVE_ACTIONS
        assert "package_uninstall" not in SANDBOX_DESTRUCTIVE_ACTIONS

    def test_destructive_actions_uses_real_disk_value(self):
        assert "disk_unmount" in SANDBOX_DESTRUCTIVE_ACTIONS
        assert "disk_manage" not in SANDBOX_DESTRUCTIVE_ACTIONS

    def test_root_actions_uses_real_disk_values(self):
        assert "disk_mount" in ROOT_ACTIONS
        assert "disk_unmount" in ROOT_ACTIONS
        assert "disk_manage" not in ROOT_ACTIONS


class TestBrowserImpactModeling:
    def test_execute_js_is_high_risk_with_script_preview_description(self):
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.BROWSER_EXECUTE_JS, target="document.cookie"))
        impact = report.impacts[0]
        assert impact.risk == RiskLevel.HIGH
        assert "document.cookie" in impact.description
        assert impact.reversible is False
        assert report.has_destructive is True

    def test_navigate_is_medium_risk_with_url_in_description(self):
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.BROWSER_NAVIGATE, target="https://example.com"))
        impact = report.impacts[0]
        assert impact.risk == RiskLevel.MEDIUM
        assert "https://example.com" in impact.description
        assert report.has_network is True

    def test_fill_form_is_medium_risk(self):
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.BROWSER_FILL_FORM, target="#login-form"))
        assert report.impacts[0].risk == RiskLevel.MEDIUM

    def test_extract_stays_low_risk(self):
        # Read-only browser actions shouldn't be swept up into the new
        # medium/high-risk browser coverage.
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.BROWSER_EXTRACT, target="page text"))
        assert report.impacts[0].risk == RiskLevel.LOW


class TestSystemControlImpactModeling:
    def test_mouse_click_is_high_risk_blind_interaction(self):
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.MOUSE_CLICK, target="640,400"))
        impact = report.impacts[0]
        assert impact.risk == RiskLevel.HIGH
        assert "640,400" in impact.description

    def test_keyboard_hotkey_is_high_risk(self):
        assert "keyboard_hotkey" in SANDBOX_HIGH_RISK_ACTIONS
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.KEYBOARD_HOTKEY, target="ctrl+w"))
        assert report.impacts[0].risk == RiskLevel.HIGH

    def test_process_kill_has_specific_description(self):
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.PROCESS_KILL, target="pid=4242"))
        assert "pid=4242" in report.impacts[0].description
        assert report.has_destructive is True

    def test_registry_write_has_specific_description_and_root(self):
        sandbox = SimulationSandbox()
        report = sandbox.simulate(_plan(ActionType.REGISTRY_WRITE, target="HKCU\\Software\\Foo"))
        impact = report.impacts[0]
        assert "HKCU" in impact.description
        assert report.requires_root is True

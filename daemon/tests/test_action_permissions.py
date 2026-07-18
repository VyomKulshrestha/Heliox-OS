import pytest

from pilot.actions import (
    DESTRUCTIVE_ACTIONS,
    IRREVERSIBLE_ACTIONS,
    SYSTEM_MODIFY_ACTIONS,
    Action,
    ActionType,
    EmptyParams,
    PermissionTier,
)


def test_shell_command_not_in_always_safe():
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        parameters=EmptyParams(),
    )
    assert action.requires_confirmation is True
    assert action.permission_tier == PermissionTier.SYSTEM_MODIFY


def test_code_execute_not_in_always_safe():
    action = Action(
        action_type=ActionType.CODE_EXECUTE,
        parameters=EmptyParams(),
    )
    assert action.requires_confirmation is True
    assert action.permission_tier == PermissionTier.SYSTEM_MODIFY


def test_shell_command_in_system_modify():
    assert ActionType.SHELL_COMMAND in SYSTEM_MODIFY_ACTIONS


def test_code_execute_in_system_modify():
    assert ActionType.CODE_EXECUTE in SYSTEM_MODIFY_ACTIONS


class TestBrowserActionRetiering:
    """Browser actions used to be blanket ALWAYS_SAFE (Tier 1) regardless of
    what they did — this is the fix for that gap (see SECURITY.md's Agent
    Gateway section). Only the state-changing/execution ones should now
    require confirmation; extraction/inspection stays untouched."""

    @pytest.mark.parametrize(
        "action_type",
        [
            ActionType.BROWSER_NAVIGATE,
            ActionType.BROWSER_CLICK,
            ActionType.BROWSER_CLICK_TEXT,
            ActionType.BROWSER_TYPE,
            ActionType.BROWSER_SELECT,
            ActionType.BROWSER_FILL_FORM,
        ],
    )
    def test_state_changing_browser_actions_now_require_confirmation(self, action_type):
        action = Action(action_type=action_type, target="https://example.com", parameters=EmptyParams())
        assert action.requires_confirmation is True
        assert action.permission_tier == PermissionTier.SYSTEM_MODIFY
        assert action_type in SYSTEM_MODIFY_ACTIONS

    def test_execute_js_requires_confirmation_at_destructive_tier(self):
        action = Action(action_type=ActionType.BROWSER_EXECUTE_JS, target="page", parameters=EmptyParams())
        assert action.requires_confirmation is True
        assert action.permission_tier == PermissionTier.DESTRUCTIVE
        assert ActionType.BROWSER_EXECUTE_JS in DESTRUCTIVE_ACTIONS

    def test_execute_js_is_irreversible(self):
        action = Action(action_type=ActionType.BROWSER_EXECUTE_JS, target="page", parameters=EmptyParams())
        assert action.is_irreversible is True
        assert ActionType.BROWSER_EXECUTE_JS in IRREVERSIBLE_ACTIONS

    @pytest.mark.parametrize(
        "action_type",
        [
            ActionType.BROWSER_EXTRACT,
            ActionType.BROWSER_EXTRACT_TABLE,
            ActionType.BROWSER_EXTRACT_LINKS,
            ActionType.BROWSER_SCREENSHOT,
            ActionType.BROWSER_LIST_TABS,
            ActionType.BROWSER_PAGE_INFO,
            ActionType.BROWSER_WAIT,
        ],
    )
    def test_read_only_browser_actions_still_require_no_confirmation(self, action_type):
        action = Action(action_type=action_type, target="page", parameters=EmptyParams())
        assert action.requires_confirmation is False
        assert action.permission_tier == PermissionTier.READ_ONLY

    @pytest.mark.parametrize(
        "action_type",
        [
            ActionType.BROWSER_HOVER,
            ActionType.BROWSER_SCROLL,
            ActionType.BROWSER_NEW_TAB,
            ActionType.BROWSER_CLOSE_TAB,
            ActionType.BROWSER_SWITCH_TAB,
            ActionType.BROWSER_BACK,
            ActionType.BROWSER_FORWARD,
            ActionType.BROWSER_REFRESH,
            ActionType.BROWSER_CLOSE,
        ],
    )
    def test_passive_browser_actions_still_require_no_confirmation(self, action_type):
        # Removed from ALWAYS_SAFE along with the rest, but has no other-set
        # membership, so falls through to the same default USER_WRITE tier —
        # unchanged behavior for these, unlike the state-changing ones above.
        action = Action(action_type=action_type, target="page", parameters=EmptyParams())
        assert action.requires_confirmation is False
        assert action.permission_tier == PermissionTier.USER_WRITE

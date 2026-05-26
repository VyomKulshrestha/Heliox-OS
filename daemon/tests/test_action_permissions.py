from pilot.actions import (
    Action,
    ActionType,
    EmptyParams,
    PermissionTier,
    SYSTEM_MODIFY_ACTIONS,
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

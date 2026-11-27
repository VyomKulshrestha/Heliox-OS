from pilot.actions import ActionType
from pilot.agents.planner import Planner


def test_absolute_windows_file_read_uses_local_fast_path_with_reporting_suffix():
    plan = Planner._try_fast_path(
        "Read C:\\Users\\sam\\AppData\\Local\\Temp\\missing-release-check.json "
        "and report the exact error without creating or modifying anything."
    )

    assert plan is not None
    assert len(plan.actions) == 1
    assert plan.actions[0].action_type == ActionType.FILE_READ
    assert plan.actions[0].parameters.path == ("C:\\Users\\sam\\AppData\\Local\\Temp\\missing-release-check.json")


def test_relative_file_request_does_not_guess_a_path():
    assert Planner._try_fast_path("Read the missing config file and report it") is None

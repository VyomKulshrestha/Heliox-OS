import json
from unittest.mock import MagicMock

import pytest

from pilot.actions import ActionType
from pilot.agents.planner import Planner


def _planner() -> Planner:
    return Planner(MagicMock(), MagicMock())


def test_explicit_action_alias_is_preserved():
    assert _planner()._resolve_action_type("file_create") == ActionType.FILE_WRITE


@pytest.mark.parametrize("unknown_type", ["file_create_directory", "directory_creat"])
def test_unknown_action_type_is_rejected_instead_of_fuzzy_mapped(unknown_type):
    with pytest.raises(ValueError, match="not a valid ActionType"):
        _planner()._resolve_action_type(unknown_type)


def test_one_invalid_action_rejects_the_whole_multi_step_plan():
    plan = _planner()._parse_response(
        json.dumps(
            {
                "explanation": "Read a folder and create a directory.",
                "actions": [
                    {"action_type": "file_read", "target": "C:/input.txt", "parameters": {"path": "C:/input.txt"}},
                    {
                        "action_type": "file_create_directory",
                        "target": "C:/output",
                        "parameters": {"path": "C:/output"},
                    },
                ],
            }
        ),
        "Read a file and create a directory.",
    )

    assert plan.actions == []
    assert plan.error is not None
    assert "No actions were retained" in plan.error


def test_file_copy_without_destination_is_rejected_before_execution():
    plan = _planner()._parse_response(
        """
        {
          "explanation": "Copy a file.",
          "actions": [
            {
              "action_type": "file_copy",
              "target": "C:\\\\source.txt",
              "parameters": {"path": "C:\\\\source.txt"}
            }
          ]
        }
        """,
        "Copy C:\\source.txt to C:\\destination.txt.",
    )

    assert plan.actions == []
    assert "parameters.destination" in str(plan.error)


def test_file_copy_with_destination_is_preserved():
    plan = _planner()._parse_response(
        """
        {
          "explanation": "Copy a file.",
          "actions": [
            {
              "action_type": "file_copy",
              "target": "C:\\\\source.txt",
              "parameters": {
                "path": "C:\\\\source.txt",
                "destination": "C:\\\\destination.txt"
              }
            }
          ]
        }
        """,
        "Copy C:\\source.txt to C:\\destination.txt.",
    )

    assert plan.error is None
    assert plan.actions[0].parameters.destination == "C:\\destination.txt"

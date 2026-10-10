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


def test_parser_accepts_type_and_params_synonyms():
    plan = _planner()._parse_response(
        json.dumps(
            {
                "explanation": "Create the file.",
                "actions": [{"type": "file_write", "params": {"path": "C:/tmp/probe.txt", "content": "probe"}}],
            }
        ),
        "Create C:/tmp/probe.txt containing probe.",
    )

    assert plan.error is None
    assert [action.action_type for action in plan.actions] == [ActionType.FILE_WRITE]
    assert plan.actions[0].parameters.path == "C:/tmp/probe.txt"
    assert plan.actions[0].parameters.content == "probe"
    assert plan.actions[0].target == "C:/tmp/probe.txt"


def test_parser_names_the_keys_when_action_type_is_missing():
    plan = _planner()._parse_response(
        json.dumps({"explanation": "Do it.", "actions": [{"target": "C:/x", "parameters": {}}]}),
        "Do something with C:/x.",
    )

    assert plan.actions == []
    assert "missing required field 'action_type' (keys present: parameters, target)" in plan.error


class _ScriptedModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)

    async def generate(self, *_args, **_kwargs) -> str:
        return self.responses.pop(0)


class _EmptyMemory:
    async def get_context(self, *_args, **_kwargs):
        return ""

    async def get_history(self, **_kwargs):
        return []


@pytest.mark.asyncio
async def test_empty_retry_after_a_failed_action_attempt_is_a_planning_error():
    model = _ScriptedModel(
        json.dumps({"explanation": "Write it.", "actions": [{"target": "C:/x", "parameters": {}}]}),
        json.dumps({"explanation": "I'm Heliox OS. What would you like me to do?", "actions": []}),
    )
    planner = Planner(model, _EmptyMemory())

    plan = await planner.plan("Write the quarterly totals into C:/reports/q3.txt")

    assert plan.actions == []
    assert plan.error is not None
    assert "retry returned no actions" in plan.error

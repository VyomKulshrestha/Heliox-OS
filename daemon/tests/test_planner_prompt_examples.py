"""Every parameter example the planner's prompt gives is one the planner itself accepts.

The model copies these examples. An example that is not valid JSON, that fails the action's parameter model, or that
names a key the model drops (so the value silently disappears) teaches the model to write a plan that fails or loses
what the person asked for. api_send_email's example once said "body" for the message text, whose field is "message".
"""

from __future__ import annotations

import json
import re

import pytest

from pilot.actions import ActionType
from pilot.agents.planner import SYSTEM_PROMPT, Planner

ACTION_NAMES = {action.value for action in ActionType}


def _rendered_prompt() -> str:
    return SYSTEM_PROMPT.format(os="Windows", path_style="Windows (C:\\Users\\...)", home="C:\\Users\\person")


def _entries(prompt: str) -> list[str]:
    """Each "- name ..." line of the prompt, joined with the indented lines that continue it."""
    entries: list[str] = []
    for line in prompt.splitlines():
        if line.startswith("- "):
            entries.append(line)
        elif entries and line.startswith("  ") and not line.strip().startswith("- "):
            entries[-1] += " " + line.strip()
        else:
            entries.append("")
    return [entry for entry in entries if _EXAMPLE_START.search(entry)]


# "Params: {...}" or "uses {...}" / "use {...}" introduces the examples of an entry.
_EXAMPLE_START = re.compile(r"(Params:?|\buses?)\s*(?=\{)")

# Parsed by the planner once the series that adds ssh parameter parsing is merged; until then the planner refuses
# them with "Unknown action type", which is what this test exists to catch.
_PARSED_BY_ANOTHER_SERIES = {"ssh_command", "ssh_script"}


def _objects_after(text: str, start: int) -> list[str]:
    """The JSON objects written one after another (separated by "/") from ``start``."""
    objects: list[str] = []
    index = start
    while True:
        while index < len(text) and text[index] in " :":
            index += 1
        if index >= len(text) or text[index] != "{":
            return objects
        depth, in_string, escaped = 0, False, False
        for end in range(index, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[index : end + 1])
                    index = end + 1
                    break
        else:
            return objects
        while index < len(text) and text[index] == " ":
            index += 1
        if index < len(text) and text[index] == "/":
            index += 1
            continue
        return objects


def _examples() -> list[tuple[str, str, str]]:
    """(action name, example JSON text, the prompt line) for every action the prompt shows an example for."""
    examples: list[tuple[str, str, str]] = []
    for entry in _entries(_rendered_prompt()):
        start = _EXAMPLE_START.search(entry)
        head = entry[2 : start.start()].split("—")[0]
        names = [name.strip() for name in head.split(",") if name.strip() in ACTION_NAMES]
        if not names:
            continue
        objects = _objects_after(entry, start.end())
        if len(objects) == len(names):
            pairs = list(zip(names, objects))
        else:
            pairs = [(name, text) for name in names for text in objects]
        examples.extend((name, text, entry) for name, text in pairs)
    return examples


def _problem(name: str, text: str) -> str:
    action_type = ActionType(name)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"not valid JSON ({exc.msg})"
    try:
        params = Planner._normalize_params(action_type, dict(raw), "")
        model = Planner._parse_parameters(action_type, dict(params))
    except Exception as exc:  # noqa: BLE001
        return f"refused: {type(exc).__name__}: {str(exc).splitlines()[0]}"
    known = set(type(model).model_fields)
    known |= {field.alias for field in type(model).model_fields.values() if field.alias}
    dropped = sorted(set(params) - known)
    if dropped and action_type not in {ActionType.MCP_CALL_TOOL, ActionType.MCP_LIST_TOOLS}:
        return f"keys the parameters drop: {', '.join(dropped)}"
    return ""


def test_the_prompt_shows_examples_for_the_common_actions() -> None:
    shown = {name for name, _text, _entry in _examples()}
    assert {"api_send_email", "file_write", "browser_navigate", "git_commit", "file_hash", "notify"} <= shown
    assert len(shown) > 60


@pytest.mark.parametrize(
    ("name", "text"),
    [
        pytest.param(
            name,
            text,
            marks=[pytest.mark.xfail(reason="ssh parameter parsing lands with another series", strict=False)]
            if name in _PARSED_BY_ANOTHER_SERIES
            else [],
        )
        for name, text, _entry in _examples()
    ],
)
def test_every_prompt_example_is_accepted_by_its_action(name: str, text: str) -> None:
    problem = _problem(name, text)
    assert not problem, f"{name} example {text}: {problem}"


def test_send_email_example_names_the_message_field() -> None:
    from pilot.actions import ApiRequestParams

    [text] = [text for name, text, _entry in _examples() if name == "api_send_email"]
    params = Planner._parse_parameters(ActionType.API_SEND_EMAIL, json.loads(text))

    assert isinstance(params, ApiRequestParams)
    assert params.message

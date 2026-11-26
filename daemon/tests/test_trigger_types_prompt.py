"""The planner's list of trigger types comes from the engine's own table, and the names a model guesses are mapped.

In the live run the prompt's only trigger example was cpu_threshold, so the model wrote "file_change", which the
engine refused. The list is now generated from pilot.system.triggers.CONDITION_KEYS, which must cover every type and
every condition key the engine reads."""

from __future__ import annotations

import inspect
import re

from pilot.actions import ActionType
from pilot.agents.planner import SYSTEM_PROMPT, Planner
from pilot.system import triggers
from pilot.system.triggers import CONDITION_KEYS, TRIGGER_TYPE_ALIASES, TriggerType, resolve_trigger_type


def test_every_trigger_type_is_in_the_table() -> None:
    assert set(CONDITION_KEYS) == set(TriggerType)


def test_every_condition_key_the_engine_reads_is_declared() -> None:
    source = inspect.getsource(triggers.TriggerEngine)
    read = set(re.findall(r"condition\.get\(\s*\"(\w+)\"", source)) | set(
        re.findall(r"cond\.get\(\s*\"(\w+)\"", source)
    )
    declared = {key for keys in CONDITION_KEYS.values() if keys for key in keys}
    assert read and read <= declared


def test_the_prompt_lists_the_types_the_engine_fires_and_no_other() -> None:
    for trigger_type, keys in CONDITION_KEYS.items():
        line = f"  {trigger_type.value}: condition"
        assert (line in SYSTEM_PROMPT) is (keys is not None), trigger_type
    assert '"path": the file, or a folder' in SYSTEM_PROMPT
    assert "<<TRIGGER_TYPES>>" not in SYSTEM_PROMPT


def test_a_guessed_name_becomes_the_engines_type_and_an_unknown_one_is_left_for_the_engine_to_refuse() -> None:
    assert resolve_trigger_type("file_change") == "file_modified"
    assert resolve_trigger_type("File-Changed") == "file_modified"
    assert resolve_trigger_type("file_modified") == "file_modified"
    assert resolve_trigger_type("lunar_eclipse") == "lunar_eclipse"
    assert all(alias not in TriggerType._value2member_map_ for alias in TRIGGER_TYPE_ALIASES)


def test_the_planner_maps_the_guessed_name_in_a_plan() -> None:
    params = Planner._normalize_params(
        ActionType.TRIGGER_CREATE,
        {"name": "w", "trigger_type": "file_change", "condition": {"path": "C:\\w\\a.txt"}, "action_command": "x"},
        "",
    )
    assert params["trigger_type"] == "file_modified"

"""The planner asks for one line of JSON and only the action keys whose value is not the default.

A plan of one file step was about 350 output tokens, a quarter of them default values and indentation; with these
two lines in the prompt the run's own requests came to 280 (median of 20 calls), and the generation time with them.
Fakes only."""

from __future__ import annotations

import json

from pilot.agents.planner import Planner

W = "C:\\w"


def test_a_plan_without_the_default_keys_parses_to_the_same_actions() -> None:
    """The prompt asks for one line of JSON and only the action keys whose value is not the default (a plan of one
    file step was about 350 output tokens, a quarter of them defaults and indentation)."""
    from pilot.agents.planner import SYSTEM_PROMPT

    assert "write one only when its value differs from the default" in SYSTEM_PROMPT
    path = f"{W}\\notes\\old.txt"
    step = {"action_type": "file_delete", "target": path, "parameters": {"path": path}, "destructive": True}
    defaults = {"requires_root": False, "reversible": True, "rollback_action": None, "use_previous_output": False}
    short = Planner.__new__(Planner)._parse_response(
        json.dumps({"explanation": "Delete it.", "actions": [step]}), "Delete old.txt"
    )
    full = Planner.__new__(Planner)._parse_response(
        json.dumps({"explanation": "Delete it.", "actions": [{**step, **defaults}]}, indent=2), "Delete old.txt"
    )
    assert short.error is None and full.error is None
    assert short.actions[0].model_dump() == full.actions[0].model_dump()
    assert short.actions[0].destructive is True and short.actions[0].reversible is True


def test_a_write_of_the_previous_output_uses_it_without_the_flag() -> None:
    """One of two replayed appends (claude-haiku-5-5, 27 November) left "use_previous_output" out of a file_write whose
    content holds {PREV_OUTPUT}; it is the same step either way. An append is one step now and that pair is folded
    into it (tests/test_file_append.py); a prepend keeps the read and the write, as here."""
    path = f"{W}\\notes\\note.txt"
    plan = Planner.__new__(Planner)._parse_response(
        json.dumps(
            {
                "explanation": "Read the note, then write it back with the new line.",
                "actions": [
                    {"action_type": "file_read", "target": path, "parameters": {"path": path}},
                    {
                        "action_type": "file_write",
                        "target": path,
                        "parameters": {"path": path, "content": "Also renew the permit.\n{PREV_OUTPUT}"},
                    },
                ],
            }
        ),
        "Add a line to note.txt",
    )
    assert plan.error is None
    assert [a.use_previous_output for a in plan.actions] == [False, True]

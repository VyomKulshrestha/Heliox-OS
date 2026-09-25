"""Entity-binding evaluation generates scope labels from published gold bindings."""

import numpy as np

from scripts.evaluate_world_entity_binding import build_cases, task_metrics


def test_wrong_entity_and_unresolved_ambiguity_are_negative():
    tool = {"name": "send_email", "description": "Send email", "requires": ["recipient:person"]}
    entities = [{"id": "priya", "type": "person", "name": "Priya"},
                {"id": "anika", "type": "person", "name": "Anika"}]
    rows = [{"task_id": "resolved", "instruction": "Email Priya", "gold_tool": "send_email",
             "gold_bindings": {"recipient": "priya"}, "entities": entities, "tools": [tool]},
            {"task_id": "ambiguous", "instruction": "Email Alex", "gold_tool": "send_email",
             "gold_bindings": {"recipient": "NEEDS_CLARIFICATION"},
             "entities": entities, "tools": [tool]}]
    cases, resolvable = build_cases(rows)
    assert [case.in_scope for case in cases] == [True, False, False, False]
    assert task_metrics(cases, np.asarray([0.8, 0.2, 0.3, 0.4]), 0.5,
                        resolvable) == {
        "tasks": 2, "resolvable": 1, "ambiguous": 1,
        "correct_rank_on_resolvable": 1, "correct_safe_decisions": 2}

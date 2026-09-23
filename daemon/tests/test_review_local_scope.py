from __future__ import annotations

import json
import sqlite3

from scripts.review_local_scope import append_label, read_cases, read_labels, safe_display


def test_pairs_latest_intent_and_deduplicates_candidate(tmp_path):
    database = tmp_path / "experience.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE experience_events (sequence INTEGER, event_type TEXT, task_id TEXT, "
        "plan_id TEXT, action_id TEXT, payload_json TEXT)"
    )
    rows = [
        (1, "intent", "task-1", "", "", {"input": "Read report"}),
        (2, "candidate_action", "task-1", "plan-1", "action-1", {"action": {"action_type": "read_file"}}),
        (3, "candidate_action", "task-1", "plan-1", "action-1", {"action": {"action_type": "read_file"}}),
        (4, "intent", "task-1", "", "", {"input": "Now update report"}),
        (5, "candidate_action", "task-1", "plan-2", "action-2", {"action": {"action_type": "write_file"}}),
    ]
    connection.executemany(
        "INSERT INTO experience_events VALUES (?, ?, ?, ?, ?, ?)",
        [(*row[:5], json.dumps(row[5])) for row in rows],
    )
    connection.commit()
    connection.close()
    cases = read_cases(database)
    assert len(cases) == 2
    assert [case.request for case in cases] == ["Read report", "Now update report"]
    labels = tmp_path / "labels.jsonl"
    append_label(labels, cases[0], "in_scope", reviewer="user")
    assert read_labels(labels, cases) == {cases[0].case_id: "in_scope"}
    assert "Read report" not in labels.read_text(encoding="utf-8")
    assert read_labels(labels, cases, reviewer="codex") == {}
    duplicate = type(cases[0])(
        case_id="another-task", source_hash=cases[0].source_hash, task_id="task-2",
        sequence=6, request=cases[0].request, action=cases[0].action,
        action_type=cases[0].action_type,
    )
    assert read_labels(labels, [*cases, duplicate])[duplicate.case_id] == "in_scope"


def test_obvious_secrets_are_hidden_from_terminal():
    value = "Email me@example.com https://example.com/path?token=abc AQ.exampleexampleexampleexampleexample"
    displayed = safe_display(value)
    assert "me@example.com" not in displayed
    assert "token=abc" not in displayed
    assert "AQ.example" not in displayed

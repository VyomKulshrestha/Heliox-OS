import pytest

from benchmarks.tbac_tool_relevance import cases_from_rows, relevance_request


def row(identifier: str, task: str, relevant: bool) -> dict:
    return {"id": identifier, "label": {"relevant": relevant},
            "request": {"task": task, "tool": {"name": "read_issue", "description": "Read an issue."}}}


def test_repeated_task_stays_one_group_and_request_is_separate_from_scope():
    cases = cases_from_rows([row("a", "Read issue 42", True), row("b", "Read issue 42", False)],
                            "validation")
    assert cases[0].group_id == cases[1].group_id
    assert cases[0].id != cases[1].id
    request = relevance_request(cases[0])
    assert request["state"]["user_task"] == "Read issue 42"
    assert "tool_relevant" in request["questions"]
    assert "in_scope" not in request["questions"]


def test_malformed_or_duplicate_rows_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        cases_from_rows([row("a", "Read issue 42", True), row("a", "Read issue 43", False)],
                        "validation")
    with pytest.raises(ValueError, match="Invalid"):
        cases_from_rows([row("a", "Read issue 42", 1)], "validation")

"""Split and prompt invariants for the unpromoted Qwen Noul experiment."""

from __future__ import annotations

from scripts.train_qwen_scope_noul import group_split, prompt


def test_noul_prompt_carries_request_action_and_yes_no_meanings() -> None:
    rendered = prompt("Read the chart", "Upload the chart")
    assert "Read the chart" in rendered
    assert "Upload the chart" in rendered
    assert "Yes means:" in rendered
    assert "No means:" in rendered


def test_training_splits_keep_requests_together() -> None:
    # Synthetic rows: the real training rows need Jev teacher outputs, which stay local and untracked.
    rows = [
        {
            "id": f"r{request}a{action}",
            "request": f"request {request}",
            "action": f"action {action}",
            "probability": 0.5,
        }
        for request in range(20)
        for action in range(4)
    ]
    parts = group_split(rows)
    assert sum(len(part) for part in parts.values()) == len(rows)
    groups = {name: {rows[index]["request"] for index in indices} for name, indices in parts.items()}
    assert not (groups["fit"] & groups["calibration"])
    assert not (groups["fit"] & groups["validation"])
    assert not (groups["calibration"] & groups["validation"])

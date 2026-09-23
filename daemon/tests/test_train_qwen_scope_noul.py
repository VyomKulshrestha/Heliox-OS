"""Split and prompt invariants for the unpromoted Qwen Noul experiment."""

from __future__ import annotations

from pathlib import Path

from scripts.train_qwen_scope_noul import group_split, prompt, training_rows


def test_noul_prompt_carries_request_action_and_yes_no_meanings() -> None:
    rendered = prompt("Read the chart", "Upload the chart")
    assert "Read the chart" in rendered
    assert "Upload the chart" in rendered
    assert "Yes means:" in rendered
    assert "No means:" in rendered


def test_training_splits_keep_requests_together() -> None:
    rows = training_rows(Path(__file__).resolve().parents[1])
    parts = group_split(rows)
    assert sum(len(part) for part in parts.values()) == len(rows)
    groups = {name: {rows[index]["request"] for index in indices}
              for name, indices in parts.items()}
    assert not (groups["fit"] & groups["calibration"])
    assert not (groups["fit"] & groups["validation"])
    assert not (groups["calibration"] & groups["validation"])

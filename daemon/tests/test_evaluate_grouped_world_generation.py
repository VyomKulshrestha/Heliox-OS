"""Group selection, lexical diagnostics, and interrupted-run recovery."""

import json

import pytest

from scripts.evaluate_grouped_world_generation import load_checkpoint, rouge_l_f1, select_cases, token_f1


def test_overlap_metrics_are_bounded_and_order_sensitive() -> None:
    assert token_f1("dialog opened", "dialog opened") == 1.0
    assert rouge_l_f1("dialog opened", "dialog opened") == 1.0
    assert token_f1("opened dialog", "dialog opened") == 1.0
    assert rouge_l_f1("opened dialog", "dialog opened") == 0.5
    assert rouge_l_f1("unrelated", "dialog opened") == 0.0


def test_select_cases_uses_one_row_per_group_reproducibly() -> None:
    rows = [
        {"group_sha256": "a", "row_index": 0},
        {"group_sha256": "a", "row_index": 1},
        {"group_sha256": "b", "row_index": 2},
        {"group_sha256": "c", "row_index": 3},
    ]
    selected = select_cases(rows, seed=7, max_cases=2)

    assert selected == select_cases(rows, seed=7, max_cases=2)
    assert len(selected) == len({row["group_sha256"] for row in selected}) == 2


def test_checkpoint_resumes_only_matching_contiguous_cases(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    selected = [{"group_sha256": "a", "row_index": 2}, {"group_sha256": "b", "row_index": 9}]
    first = {"signature": "same-run", "case_index": 0, "group_sha256": "a", "row_index": 2, "scores": {}}
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")

    assert load_checkpoint(path, selected, "same-run") == [first]
    with pytest.raises(ValueError, match="Checkpoint does not match"):
        load_checkpoint(path, selected, "different-run")

    path.write_text(json.dumps({**first, "case_index": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint does not match"):
        load_checkpoint(path, selected, "same-run")

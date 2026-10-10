"""Adjacent next-observation labels must be causal and task-disjoint."""

from __future__ import annotations

import random

import pytest

from scripts import sample_adjacent_world_observations as sampler


def _entry(index: int, group: str, step: int, state: str, action: str = "Click Open") -> dict:
    prompt = (
        f"Current state:\n{state}\n\nAction:\n{action}\n\n"
        "Given the current state of the screen, predict the expected outcome."
    )
    return {
        "row_idx": index,
        "row": {
            "metadata": {"task_id": group, "step_index": step},
            "messages": [
                {"role": "system", "content": "Predict an outcome"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Untrusted retrospective reflection"},
            ],
        },
    }


def _pair(current: dict, following: dict):
    return sampler.extract_pair(
        current,
        following,
        split="train",
        revision="a" * 40,
        dataset="example/transitions",
        group_path="metadata.task_id",
        step_path="metadata.step_index",
        max_input_chars=1000,
        max_target_chars=1000,
    )


def test_next_observation_is_target_not_retrospective_reflection() -> None:
    record, reason = _pair(_entry(10, "task-a", 2, "Dialog closed"), _entry(11, "task-a", 3, "Dialog opened"))

    assert reason == "accepted"
    assert record is not None
    assert record["messages"][2]["content"] == "Dialog opened"
    assert "Untrusted retrospective reflection" not in str(record["messages"])
    assert record["messages"][1]["content"].endswith("Next observed screen state:")
    assert record["step_index"] == 2
    assert "task-a" not in str(record)


@pytest.mark.parametrize(
    "following,reason",
    [
        (_entry(11, "other-task", 3, "Opened"), "different_task"),
        (_entry(11, "task-a", 4, "Opened"), "nonconsecutive_step"),
        (_entry(12, "task-a", 3, "Opened"), "nonadjacent_source_rows"),
    ],
)
def test_rejects_noncausal_pairs(following: dict, reason: str) -> None:
    record, actual = _pair(_entry(10, "task-a", 2, "Closed"), following)

    assert record is None
    assert actual == reason


def test_rejects_ambiguous_or_truncated_source() -> None:
    following = _entry(11, "task-a", 3, "Opened")
    ambiguous = _entry(10, "task-a", 2, "Closed")
    ambiguous["row"]["messages"][1]["content"] = "No state/action boundary"
    assert _pair(ambiguous, following)[1] == "ambiguous_state_action"
    truncated = _entry(10, "task-a", 2, "Closed")
    truncated["truncated_cells"] = ["messages"]
    assert _pair(truncated, following)[1] == "truncated_source"


def test_parser_accepts_crlf_and_rejects_multiple_action_boundaries() -> None:
    prompt = _entry(0, "task", 0, "Screen visible")["row"]["messages"][1]["content"]
    assert sampler.parse_state_action(prompt.replace("\n", "\r\n")) == ("Screen visible", "Click Open")
    assert sampler.parse_state_action(prompt.replace("\n\nAction:\n", "\n\nAction:\nClick\n\nAction:\n")) is None


def test_sample_split_requires_pinned_revision_and_adjacent_steps(monkeypatch) -> None:
    rows = [
        _entry(0, "task-a", 0, "Start"),
        _entry(1, "task-a", 1, "Opened"),
        _entry(2, "task-a", 3, "Later"),
        _entry(3, "task-b", 0, "Other"),
    ]

    def fetch(_dataset, _config, _split, offset, length):
        return {"num_rows_total": 4, "rows": rows[offset : offset + length]}, "a" * 40

    monkeypatch.setattr(sampler, "fetch_page", fetch)
    records, exclusions, total = sampler.sample_split(
        dataset="example/transitions",
        config="default",
        source_split="train",
        output_split="train",
        pages=1,
        page_length=4,
        rng=random.Random(1),
        expected_revision="a" * 40,
        group_path="metadata.task_id",
        step_path="metadata.step_index",
        max_input_chars=1000,
        max_target_chars=1000,
    )
    assert total == 4
    assert [(row["row_index"], row["next_row_index"]) for row in records] == [(0, 1)]
    assert exclusions["nonconsecutive_step"] == 1
    assert exclusions["different_task"] == 1

    with pytest.raises(ValueError, match="revision differs"):
        sampler.sample_split(
            dataset="example/transitions",
            config="default",
            source_split="train",
            output_split="train",
            pages=1,
            page_length=4,
            rng=random.Random(1),
            expected_revision="b" * 40,
            group_path="metadata.task_id",
            step_path="metadata.step_index",
            max_input_chars=1000,
            max_target_chars=1000,
        )

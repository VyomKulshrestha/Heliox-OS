"""Leakage and provenance checks for sampled desktop transition research rows."""

from __future__ import annotations

import random
import sys

import pytest

from scripts import sample_grouped_world_transitions as sampler


def _entry(index: int, group: str, *, outcome: str = "Dialog opened") -> dict:
    return {
        "row_idx": index,
        "row": {
            "metadata": {"task_id": group},
            "messages": [
                {"role": "system", "content": "Predict the next UI state."},
                {"role": "user", "content": f"State {index}; action: click Open"},
                {"role": "assistant", "content": outcome},
            ],
        },
    }


def _extract(entry: dict, split: str) -> dict:
    record, reason = sampler.extract_record(
        entry,
        split=split,
        revision="a" * 40,
        dataset="example/transitions",
        group_path="metadata.task_id",
        max_input_chars=500,
        max_target_chars=500,
    )
    assert reason == "accepted"
    assert record is not None
    return record


def test_group_and_context_hashes_are_stable_without_raw_task_id() -> None:
    first = _extract(_entry(0, "private-task-id"), "train")
    second = _extract(_entry(1, "private-task-id"), "eval")

    assert first["group_sha256"] == second["group_sha256"]
    assert first["context_sha256"] != second["context_sha256"]
    assert "private-task-id" not in str(first)


def test_group_overlap_is_removed_even_across_published_splits() -> None:
    train = [_extract(_entry(0, "shared"), "train")]
    evaluation = [_extract(_entry(1, "shared"), "eval"), _extract(_entry(2, "new"), "eval")]

    _, accepted, exclusions = sampler.separate_groups(train, evaluation)

    assert [row["row_index"] for row in accepted] == [2]
    assert exclusions["overlapping_group"] == 1


def test_sampler_rejects_missing_group_and_truncated_source() -> None:
    missing = _entry(0, "")
    assert (
        sampler.extract_record(
            missing,
            split="train",
            revision="a" * 40,
            dataset="example/transitions",
            group_path="metadata.task_id",
            max_input_chars=500,
            max_target_chars=500,
        )[1]
        == "missing_group"
    )
    truncated = _entry(0, "task")
    truncated["truncated_cells"] = ["messages"]
    assert (
        sampler.extract_record(
            truncated,
            split="train",
            revision="a" * 40,
            dataset="example/transitions",
            group_path="metadata.task_id",
            max_input_chars=500,
            max_target_chars=500,
        )[1]
        == "truncated_source"
    )


def test_sample_split_rejects_revision_drift(monkeypatch) -> None:
    def fetch(_dataset, _config, _split, offset, length):
        return {
            "num_rows_total": 10,
            "rows": [_entry(index, f"task-{index}") for index in range(offset, offset + length)],
        }, "b" * 40

    monkeypatch.setattr(sampler, "fetch_page", fetch)
    with pytest.raises(ValueError, match="revision changed"):
        sampler.sample_split(
            dataset="example/transitions",
            config="default",
            source_split="validation",
            output_split="eval",
            pages=1,
            page_length=1,
            rng=random.Random(1),
            expected_revision="a" * 40,
            group_path="metadata.task_id",
            max_input_chars=500,
            max_target_chars=500,
        )


def test_sample_split_checks_page_indices(monkeypatch) -> None:
    def fetch(_dataset, _config, _split, offset, length):
        if length == 1:
            return {"num_rows_total": 10, "rows": [_entry(0, "task")]}, "a" * 40
        return {"num_rows_total": 10, "rows": [_entry(0, "task")] * length}, "a" * 40

    monkeypatch.setattr(sampler, "fetch_page", fetch)
    with pytest.raises(ValueError, match="unexpected row index"):
        sampler.sample_split(
            dataset="example/transitions",
            config="default",
            source_split="train",
            output_split="train",
            pages=1,
            page_length=2,
            rng=random.Random(1),
            expected_revision=None,
            group_path="metadata.task_id",
            max_input_chars=500,
            max_target_chars=500,
        )


def test_cli_requires_pinned_source_revision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample_grouped_world_transitions.py",
            "--dataset",
            "example/transitions",
            "--expected-revision",
            "a" * 40,
            "--rows",
            str(tmp_path / "rows.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    monkeypatch.setattr(sampler, "fetch_page", lambda *_args: ({"num_rows_total": 384}, "b" * 40))

    with pytest.raises(ValueError, match="revision changed"):
        sampler.main()

    assert not (tmp_path / "rows.jsonl").exists()

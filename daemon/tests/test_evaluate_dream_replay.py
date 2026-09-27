"""Exact replay must stay on observed, task-disjoint branches."""

import json

import pytest

from scripts.evaluate_dream_replay import load_branches, run, select_action


def _row(split: str, group: str, task: str, label: str, reward: float | None) -> dict:
    return {
        "split": split,
        "group": group,
        "task": task,
        "state_text": 'Click on the "previous" button. previous no\nControls:\nbutton|previous',
        "action_label": label,
        "reward": reward,
    }


def _archive(tmp_path, rows):
    path = tmp_path / "branches.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_goal_match_uses_instruction_not_later_visible_distractors():
    rows = [_row("eval", "g", "task", "no", -1), _row("eval", "g", "task", "previous", 1)]
    assert select_action("goal_match", rows) == 1


def test_grounded_policy_distinguishes_unlabeled_input_types():
    rows = [
        _row("eval", "g", "task", "INPUT", -1),
        _row("eval", "g", "task", "INPUT", 1),
    ]
    for row in rows:
        row["state_text"] = 'Click on a "checkbox" widget.\nControls:\ninput'
    rows[0]["action_kind"] = "text"
    rows[1]["action_kind"] = "checkbox"
    assert select_action("goal_grounded", rows) == 1


def test_grounded_policy_prefers_specific_label_over_shared_control_kind():
    rows = [
        _row("eval", "g", "task", "no", -1),
        _row("eval", "g", "task", "previous", 1),
    ]
    for row in rows:
        row["action_kind"] = "button"
    assert select_action("goal_grounded", rows) == 1


def test_replay_selects_on_validation_only_and_counts_observed_rewards(tmp_path):
    rows = [
        _row("validation", "v", "task-v", "no", -1),
        _row("validation", "v", "task-v", "previous", 1),
        _row("eval", "e", "task-e", "no", -1),
        _row("eval", "e", "task-e", "previous", 1),
    ]
    result = run(_archive(tmp_path, rows), seed=1)
    assert result["validation"][result["selected_on_validation"]]["wins"] == 1
    assert result["held_out_eval"]["goal_match"]["wins"] == 1
    assert result["held_out_eval"]["goal_match"]["groups"] == 1
    rows[-2]["reward"], rows[-1]["reward"] = 1, -1
    changed_eval = run(_archive(tmp_path, rows), seed=1)
    assert changed_eval["selected_on_validation"] == result["selected_on_validation"]


def test_unobserved_and_equal_rewards_cannot_be_replayed(tmp_path):
    rows = [
        _row("validation", "missing", "task-v", "yes", None),
        _row("validation", "missing", "task-v", "no", -1),
        _row("eval", "tie", "task-e", "yes", 1),
        _row("eval", "tie", "task-e", "no", 1),
    ]
    branches, skipped = load_branches(_archive(tmp_path, rows), seed=1)
    assert not branches
    assert skipped == {"unobserved_or_invalid_reward": 1, "no_reward_contrast": 1}


def test_task_overlap_between_splits_fails_closed(tmp_path):
    rows = [
        _row("validation", "v", "same-task", "no", -1),
        _row("validation", "v", "same-task", "previous", 1),
        _row("eval", "e", "same-task", "no", -1),
        _row("eval", "e", "same-task", "previous", 1),
    ]
    with pytest.raises(ValueError, match="leaked"):
        load_branches(_archive(tmp_path, rows), seed=1)

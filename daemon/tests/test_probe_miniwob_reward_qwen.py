"""Frozen goal/action probe uses pinned contrastive labels without leakage."""

import hashlib
import json

import pytest

from scripts.collect_miniwob_branches import SOURCE_COMMIT, SOURCE_REPO
from scripts.probe_miniwob_reward_qwen import evaluate, load_rows, parse_index, prompt_for


def _rows():
    return [
        {
            "split": split, "task": split, "group": split,
            "prestate_sha256": split, "goal": "Click ONE",
            "candidate": {"label": label, "tag": "button", "width": 40, "height": 20},
            "reward": reward,
        }
        for split in ("train", "validation", "eval")
        for label, reward in (("ONE", 1), ("TWO", -1))
    ]


def test_prompt_and_strict_index_parser_hide_rewards() -> None:
    prompt = prompt_for(_rows()[:2])
    assert "Click ONE" in prompt and '"label": "ONE"' in prompt
    assert "reward" not in prompt
    assert parse_index("0", 2) == 0
    assert parse_index("1.", 2) == 1
    for answer in ("2", "-1", "index 0", "0 and 1", ""):
        assert parse_index(answer, 2) is None


def test_evaluation_counts_group_success_and_invalid_answers() -> None:
    rows = _rows()
    good = evaluate(rows, "eval", lambda _: "0")
    assert good["groups"] == 1
    assert good["success_rate"] == 1
    assert good["random_success_rate"] == 0.5
    bad = evaluate(rows, "eval", lambda _: "unknown")
    assert bad["invalid_answers"] == 1 and bad["success_rate"] == 0


def test_loader_requires_integrity_contrast_and_disjoint_prestates(tmp_path) -> None:
    rows = _rows()
    path = tmp_path / "rows.json"
    report = tmp_path / "report.json"

    def write_source():
        path.write_text(json.dumps(rows), encoding="utf-8")
        report.write_text(json.dumps({
            "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "purpose": "research_only_goal_conditioned_first_action_rewards",
            "source_repo": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
            "split_basis": "task_family",
            "tasks": {split: [split] for split in ("train", "validation", "eval")},
            "rows": {split: 2 for split in ("train", "validation", "eval")},
            "groups": {split: 1 for split in ("train", "validation", "eval")},
        }), encoding="utf-8")

    write_source()
    assert len(load_rows(path, report)) == 6
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        load_rows(path, report)
    rows[-1]["prestate_sha256"] = "train"
    write_source()
    with pytest.raises(ValueError, match="leaked"):
        load_rows(path, report)


def test_loader_rejects_inconsistent_reset_or_report_counts(tmp_path) -> None:
    rows = _rows()
    path = tmp_path / "rows.json"
    report = tmp_path / "report.json"

    def write_source(count=2):
        path.write_text(json.dumps(rows), encoding="utf-8")
        report.write_text(json.dumps({
            "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "purpose": "research_only_goal_conditioned_first_action_rewards",
            "source_repo": SOURCE_REPO, "source_commit": SOURCE_COMMIT,
            "split_basis": "task_family",
            "tasks": {split: [split] for split in ("train", "validation", "eval")},
            "rows": {split: count for split in ("train", "validation", "eval")},
            "groups": {split: 1 for split in ("train", "validation", "eval")},
        }), encoding="utf-8")

    rows[1]["prestate_sha256"] = "different"
    write_source()
    with pytest.raises(ValueError, match="single-task contrastive resets"):
        load_rows(path, report)
    rows[1]["prestate_sha256"] = "train"
    write_source(count=3)
    with pytest.raises(ValueError, match="counts differ"):
        load_rows(path, report)

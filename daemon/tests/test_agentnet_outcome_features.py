"""Feature inputs omit reflector answers and future action text."""

from __future__ import annotations

import json

import numpy as np

from scripts.extract_agentnet_outcome_features import PROMPT, save, selected_prompts
from scripts.fit_agentnet_outcome_head import assess, labels_for


def test_selected_prompt_uses_only_task_transition_evidence(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "task_id": "task-a", "instruction": "Open the document",
        "traj": [
            {"index": 0, "value": {"observation": "Document is closed",
                                    "action": "Click Open", "code": "click(2, 3)",
                                    "thought": "SECRET_THOUGHT", "reflection": "SECRET_LABEL",
                                    "last_step_correct": False}},
            {"index": 1, "value": {"observation": "Document is visible",
                                    "action": "SECRET_FUTURE_ACTION"}},
        ],
    }) + "\n", encoding="utf-8")
    text = selected_prompts(source, {"case_ids": ["task-a:0"]})[0]
    assert "Open the document" in text
    assert "Document is closed" in text and "Document is visible" in text
    assert "Click Open" in text and "click(2, 3)" in text
    assert all(secret not in text for secret in (
        "SECRET_THOUGHT", "SECRET_LABEL", "SECRET_FUTURE_ACTION"))
    assert "last_step_correct" not in PROMPT
    assert labels_for(source, ["task-a:0"]).tolist() == [False]


def test_feature_checkpoint_has_numeric_array_only(tmp_path) -> None:
    path = tmp_path / "features.npz"
    save(path, "pinned", ["task-a:0"], [np.asarray([1.0, 2.0])])
    with np.load(path, allow_pickle=False) as saved:
        assert saved["signature"].item() == "pinned"
        assert saved["ids"].tolist() == ["task-a:0"]
        assert saved["features"].dtype == np.float16


def test_outcome_assessment_counts_false_allowances_and_warnings() -> None:
    labels = np.asarray([True, True, False, False])
    scores = np.asarray([0.9, 0.2, 0.8, 0.1])
    report = assess(labels, scores, 0.5)
    assert report == {"rows": 4, "auc": 0.75,
                      "false_allowances": 1, "false_warnings": 1}

"""Feature inputs omit reflector answers and future action text."""

from __future__ import annotations

import json
import sys

import numpy as np

from scripts import evaluate_agentnet_outcome_final as evaluate
from scripts.extract_agentnet_outcome_features import PROMPT, save, selected_prompts, sha256_file
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


def test_frozen_final_evaluator_accepts_pinned_synthetic_artifacts(tmp_path, monkeypatch) -> None:
    splits = {name: [f"{name}-yes:0", f"{name}-no:0"]
              for name in ("train", "calibration", "development", "final", "ood")}
    ids = [case_id for rows in splits.values() for case_id in rows]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({
        "task_id": case_id.split(":", 1)[0],
        "traj": [{"index": 0, "value": {"last_step_correct": "yes" in case_id}}],
    }) + "\n" for case_id in ids), encoding="utf-8")
    source_hash = sha256_file(source)
    monkeypatch.setattr(evaluate, "SOURCE_SHA256", source_hash)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_sha256": source_hash,
                                    "label_provenance": "synthetic_test",
                                    "ood_domain": "synthetic",
                                    "splits": splits, "case_ids": ids}), encoding="utf-8")
    features = tmp_path / "features.npz"
    save(features, "test", ids, [np.asarray([float("yes" in case_id), 0.0])
                                 for case_id in ids])
    head = tmp_path / "head.npz"
    np.savez_compressed(head, mean=np.zeros(2), scale=np.ones(2),
                        coefficient=np.asarray([[4.0, 0.0]]), intercept=np.asarray([-2.0]),
                        cautious_threshold=0.5, manifest_sha256=sha256_file(manifest),
                        feature_sha256=sha256_file(features))
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"manifest_sha256": sha256_file(manifest),
                               "feature_sha256": sha256_file(features),
                               "artifact_sha256": sha256_file(head),
                               "final_and_ood_labels_opened": False}), encoding="utf-8")
    report = tmp_path / "final.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--source", str(source),
                                      "--manifest", str(manifest),
                                      "--features", str(features),
                                      "--fit-report", str(fit),
                                      "--artifact", str(head),
                                      "--report", str(report)])
    evaluate.main()
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["splits"]["final"]["default"]["false_allowances"] == 0
    assert result["splits"]["ood"]["default"]["false_warnings"] == 0

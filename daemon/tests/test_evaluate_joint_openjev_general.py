"""Joint comparison uses the frozen selection without exposing labels to scoring."""

from __future__ import annotations

import json
import sys

from scripts import evaluate_joint_openjev_general as evaluator


def test_main_scores_only_inputs_and_records_local_metrics(tmp_path, monkeypatch):
    rows = [{"id": "row-1", "source": "public", "kind": "noul",
             "state_json": json.dumps({"task": "check"}), "question": "Valid?",
             "options": ["no", "yes"], "target": [0.0, 1.0]}]
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"selected_ids_sha256": "frozen",
                                     "comparison": {"noul": {"rows": 1, "jev_gold": 1.0}}}),
                         encoding="utf-8")
    output = tmp_path / "result.json"
    model_report = tmp_path / "model.json"
    model_report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluator, "verify_artifact", lambda *_: (
        {"adapter_sha256": "adapter", "head_sha256": "head"}, None))
    monkeypatch.setattr(evaluator, "selected_rows", lambda *_: (rows, {"selected_rows": 1}))

    def score(payloads, *_):
        assert payloads == [{"state": {"task": "check"}, "kind": "noul",
                             "question": "Valid?", "options": ["no", "yes"]}]
        return [{"answer": {"yes_probability": 0.9}}]

    monkeypatch.setattr(evaluator, "score_requests", score)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--output-dir", str(tmp_path),
                                      "--model-report", str(model_report),
                                      "--reference", str(reference), "--report", str(output)])
    evaluator.main()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["local"]["noul"]["top1_accuracy"] == 1.0
    assert saved["jev"]["noul"]["jev_gold"] == 1.0

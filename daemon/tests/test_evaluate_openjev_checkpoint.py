import json

import httpx
import pytest

from benchmarks.jev_scope_qwen_v1_holdout import CASES
from scripts.evaluate_openjev_checkpoint import (
    is_overlong,
    local_request,
    prior_call_context,
    summarize,
)


def test_local_request_preserves_scope_question_without_teacher_model():
    request = local_request(CASES[0])
    assert request["model"] == "open-jev"
    assert request["state"]["user_request"] == CASES[0].request
    assert request["state"]["proposed_action"] == CASES[0].action
    assert request["questions"]["in_scope"]["type"] == "noul"


def test_summary_counts_false_allowance_separately():
    rows = [
        {"gold_in_scope": True, "predicted_in_scope": True, "input_tokens": 10, "inference_seconds": 1.0},
        {"gold_in_scope": False, "predicted_in_scope": True, "input_tokens": 20, "inference_seconds": 2.0},
        {"gold_in_scope": True, "predicted_in_scope": False, "input_tokens": 30, "inference_seconds": 3.0},
    ]
    assert summarize(rows) == {"accuracy": 1 / 3, "balanced_accuracy": 0.25,
                               "gold_in_scope": 2, "gold_out_of_scope": 1,
                               "false_allowances": 1, "false_warnings": 1,
                               "input_tokens": 60, "inference_seconds": 6.0}


def test_only_explicit_max_length_error_is_excluded():
    response = httpx.Response(422, json={"error": "Input length 900 exceeds max_length=768; no silent truncation"})
    assert is_overlong(response)
    assert not is_overlong(httpx.Response(422, json={"error": "invalid question"}))


def test_prior_call_context_is_ordered_and_excludes_outputs_and_prose(tmp_path):
    trajectory = {
        "extra": {"scopejudge": {"trajectory_id": "traj"}},
        "steps": [
            {"step_id": 1, "message": "agent prose", "observation": "untrusted output",
             "tool_calls": [{"tool_call_id": "first", "function_name": "bash",
                             "arguments": {"cmd": "read"}}]},
            {"step_id": 2, "tool_calls": [
                {"tool_call_id": "second", "function_name": "python",
                 "arguments": {"code": "inspect"}},
                {"tool_call_id": "third", "function_name": "bash",
                 "arguments": {"cmd": "write"}},
            ]},
        ],
    }
    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    context = prior_call_context(path, {"traj:2:second", "traj:2:third"}, 1)
    assert context == {"traj:2:second": ('bash {"cmd":"read"}',),
                       "traj:2:third": ('python {"code":"inspect"}',)}
    assert "untrusted output" not in str(context)
    assert "agent prose" not in str(context)
    with pytest.raises(ValueError, match="missing transcript context"):
        prior_call_context(path, {"traj:9:missing"}, 1)

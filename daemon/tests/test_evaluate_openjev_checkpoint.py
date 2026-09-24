from benchmarks.jev_scope_qwen_v1_holdout import CASES
from scripts.evaluate_openjev_checkpoint import local_request, summarize


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
    assert summarize(rows) == {"accuracy": 1 / 3, "false_allowances": 1, "false_warnings": 1,
                               "input_tokens": 60, "inference_seconds": 6.0}

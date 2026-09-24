"""Public OOD comparison request shape and aggregation without external calls."""

from __future__ import annotations

import json

from scripts.compare_general_jev import compare, decision_request, public_payload


def row(kind: str, options: list[str], target: list[float]) -> dict:
    return {"kind": kind, "state_json": json.dumps({"scene": "public synthetic"}),
            "question": "What follows?", "options": options, "target": target}


def test_request_contracts_do_not_include_labels() -> None:
    choice = row("choice", ["left", "right", "wait"], [0, 1, 0])
    request = decision_request(choice)
    assert request["questions"]["decision"]["criteria"] == {
        "option_0": "left", "option_1": "right", "option_2": "wait"}
    assert "target" not in json.dumps(request)
    assert "target" not in public_payload(choice)
    assert decision_request(row("score", ["low", "mid", "high"], [0, 1, 0]))[
        "questions"]["decision"]["criteria"] == ["low", "mid", "high"]
    assert decision_request(row("noul", ["no", "yes"], [0, 1]))[
        "questions"]["decision"]["criteria"]["true"]


def test_compare_typed_predictions_without_teacher_training() -> None:
    rows = [row("noul", ["no", "yes"], [0, 1]),
            row("choice", ["left", "right"], [1, 0]),
            row("score", ["low", "mid", "high"], [0, 1, 0])]
    local = [{"answer": {"yes_probability": 0.9}},
             {"answer": {"choice": "left"}},
             {"answer": {"expected_index": 1.2}}]
    jev = [{"noul": 0.7}, {"choice": "option_1"}, {"score": 1.0}]
    result = compare(rows, local, jev)
    assert result["noul"]["agreement"] == 1
    assert result["choice"]["agreement"] == 0
    assert result["score"]["score_gap"] > 0

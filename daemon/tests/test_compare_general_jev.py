"""Public OOD comparison request shape and aggregation without external calls."""

from __future__ import annotations

import json
import sys

import httpx
import pytest

from scripts import compare_general_jev as comparison
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
    assert result["noul"]["local_balanced_accuracy"] is None


def test_imbalanced_binary_accuracy_does_not_hide_constant_predictions() -> None:
    rows = [row("noul", ["no", "yes"], [1, 0]) for _ in range(4)]
    rows.append(row("noul", ["no", "yes"], [0, 1]))
    local = [{"answer": {"yes_probability": 0.1}} for _ in rows]
    jev = [{"noul": 0.1} for _ in range(4)] + [{"noul": 0.9}]
    result = compare(rows, local, jev)["noul"]
    assert result["local_gold"] == 0.8
    assert result["local_balanced_accuracy"] == 0.5
    assert result["jev_balanced_accuracy"] == 1.0
    assert result["gold_label_counts"] == {"0": 4, "1": 1}
    assert result["local_false_positives"] == 0
    assert result["local_false_negatives"] == 1


def test_choice_accuracy_reports_macro_recall_for_present_gold_classes() -> None:
    rows = [row("choice", ["left", "right"], [1, 0]) for _ in range(3)]
    rows.append(row("choice", ["left", "right"], [0, 1]))
    local = [{"answer": {"choice": "left"}} for _ in rows]
    jev = [{"choice": "option_0"} for _ in range(3)] + [{"choice": "option_1"}]
    result = compare(rows, local, jev)["choice"]
    assert result["local_gold"] == 0.75
    assert result["local_balanced_accuracy"] == 0.5
    assert result["jev_balanced_accuracy"] == 1.0


def test_provider_query_is_blocked_before_http_request() -> None:
    calls = []

    class Client:
        def post(self, *_args, **_kwargs):
            calls.append("request")

    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        comparison.query_jev(Client(), "test-key", row("noul", ["no", "yes"], [0, 1]))
    assert calls == []


def test_jev_comparison_cli_fails_closed_before_network_or_scoring(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(comparison, "query_jev", lambda *_args: events.append("provider"))
    monkeypatch.setattr(comparison, "score_requests", lambda *_args: events.append("scoring"))
    monkeypatch.setattr(sys, "argv", ["compare_general_jev", "--report", "unused.json"])

    with pytest.raises(SystemExit, match="Disabled pending verified written TypeSafe authorization"):
        comparison.main()
    assert events == []

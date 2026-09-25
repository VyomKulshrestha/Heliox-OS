"""Public OOD comparison request shape and aggregation without external calls."""

from __future__ import annotations

import json
import sys

import httpx

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


def test_transient_transport_failure_retries_without_changing_request(monkeypatch) -> None:
    calls = []

    class Client:
        def post(self, _endpoint, *, headers, json):
            calls.append((headers, json))
            if len(calls) == 1:
                raise httpx.ConnectError("temporary DNS failure")
            return type("Response", (), {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: {"answers": {"decision": {"noul": 0.9}}},
            })()

    monkeypatch.setattr(comparison.time, "sleep", lambda _seconds: None)
    answer, cost, model = comparison.query_jev(Client(), "test-key", row("noul", ["no", "yes"], [0, 1]))
    assert answer == {"noul": 0.9}
    assert cost == 0 and model == ""
    assert len(calls) == 2 and calls[0] == calls[1]
    assert "target" not in json.dumps(calls[0][1])


def test_hosted_preflight_precedes_gpu_scoring(tmp_path, monkeypatch) -> None:
    events = []
    rows = [dict(row("noul", ["no", "yes"], [0, 1]), id=f"row-{index}") for index in range(2)]
    model_report = tmp_path / "model.json"
    model_report.write_text(json.dumps({"adapter_sha256": "frozen"}), encoding="utf-8")
    report = tmp_path / "comparison.json"

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(comparison.httpx, "Client", Client)
    monkeypatch.setattr(comparison, "load_split", lambda *_args: (rows, {"selected_rows": 2}))
    monkeypatch.setattr(comparison, "_key", lambda _path: "test-key")

    def query(_client, _key, item):
        events.append(f"jev:{item['id']}")
        return {"noul": 0.9}, 0.01, "jev-test"

    def score(payloads, _output, _model_report):
        events.append("local")
        assert len(payloads) == 2
        return [{"answer": {"yes_probability": 0.8}} for _ in payloads]

    monkeypatch.setattr(comparison, "query_jev", query)
    monkeypatch.setattr(comparison, "score_requests", score)
    monkeypatch.setattr(sys, "argv", [
        "compare_general_jev", "--limit", "2", "--model-report", str(model_report),
        "--report", str(report),
    ])
    comparison.main()
    assert events == ["jev:row-0", "local", "jev:row-1"]
    assert json.loads(report.read_text(encoding="utf-8"))["calls"] == 2

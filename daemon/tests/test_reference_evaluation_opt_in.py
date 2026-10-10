"""Provider comparison is an explicit, bounded evaluation, not a training unlock."""

import json
import sys

import httpx
import pytest

from scripts.compare_general_reference import decision_request, load_local_cases, public_payload, query_jev


def test_local_case_labels_and_identifiers_never_enter_model_requests(tmp_path):
    path = tmp_path / "cases.json"
    case = {
        "id": "hidden-id",
        "state": {"evidence": "available"},
        "kind": "noul",
        "question": "Is evidence available?",
        "options": ["no", "yes"],
        "target": [0.0, 1.0],
    }
    path.write_text(json.dumps([case]), encoding="utf-8")
    rows, manifest = load_local_cases(path)
    assert manifest["evaluation_only"] is True and manifest["rows"] == 1
    for request in (public_payload(rows[0]), decision_request(rows[0])):
        assert "hidden-id" not in json.dumps(request)
        assert "target" not in request
    path.write_text(json.dumps([case, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_local_cases(path)
    path.write_text(json.dumps([{**case, "target": [0.0, 0.0]}]), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to one"):
        load_local_cases(path)


def test_provider_evaluation_is_disabled_by_default() -> None:
    class NoNetwork:
        def post(self, *_args, **_kwargs):
            pytest.fail("Default comparison must not send a request")

    with pytest.raises(SystemExit):
        query_jev(NoNetwork(), "test-key", {})


@pytest.mark.parametrize("reference_only", [False, True])
def test_saved_reference_calls_resume_without_duplicate_billing(tmp_path, monkeypatch, reference_only):
    from scripts import compare_general_reference as comparison

    corpus = tmp_path / "cases.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "state": {},
                    "kind": "noul",
                    "question": "Valid?",
                    "options": ["no", "yes"],
                    "target": [1, 0],
                }
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "comparison.json"
    arguments = [
        "comparison",
        "--acknowledge-provider-terms",
        "--limit",
        "1",
        "--cases-json",
        str(corpus),
        "--report",
        str(report_path),
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--cache-reference-only"] if reference_only else arguments)
    monkeypatch.setattr(
        comparison, "verify_artifact", lambda *_args: ({"adapter_sha256": "a" * 64, "head_sha256": "b" * 64}, None)
    )
    monkeypatch.setattr(comparison, "_key", lambda _path: "test-key")
    calls = []

    def query(*_args, **_kwargs):
        calls.append("billed")
        return {"noul": 0.1}, 0.001, "test-reference"

    def fail(*_args):
        assert not reference_only, "Reference-only collection must not load a GPU model"
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(comparison, "query_jev", query)
    monkeypatch.setattr(comparison, "score_requests", fail)
    if reference_only:
        comparison.main()
    else:
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            comparison.main()
    assert not report_path.exists()
    assert report_path.with_suffix(".partial.json").is_file()
    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    monkeypatch.setattr(comparison, "_key", lambda _path: pytest.fail("Complete cache must not need credentials"))
    monkeypatch.setattr(
        comparison, "query_jev", lambda *_args, **_kwargs: pytest.fail("Saved response must not be billed again")
    )
    monkeypatch.setattr(comparison, "score_requests", lambda *_args: [{"answer": {"yes_probability": 0.1}}])
    comparison.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["calls"] == 1 and report["cost_usd"] == 0.001
    assert report["local_head_sha256"] == "b" * 64
    assert calls == ["billed"]


def test_local_cases_default_to_the_whole_corpus_and_explain_oversized_limits(tmp_path, monkeypatch):
    from scripts import compare_general_reference as comparison

    corpus = tmp_path / "cases.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "id": identifier,
                    "state": {},
                    "kind": "noul",
                    "question": "Valid?",
                    "options": ["no", "yes"],
                    "target": [1, 0],
                }
                for identifier in ("one", "two", "three")
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        comparison, "verify_artifact", lambda *_args: ({"adapter_sha256": "a" * 64, "head_sha256": "b" * 64}, None)
    )
    monkeypatch.setattr(comparison, "_key", lambda _path: "test-key")
    monkeypatch.setattr(comparison, "query_jev", lambda *_args, **_kwargs: ({"noul": 0.1}, 0.0, "test-reference"))
    arguments = ["comparison", "--acknowledge-provider-terms", "--cases-json", str(corpus), "--cache-reference-only"]

    monkeypatch.setattr(sys, "argv", [*arguments, "--report", str(tmp_path / "all.json")])
    comparison.main()
    saved = json.loads((tmp_path / "all.partial.json").read_text(encoding="utf-8"))
    assert len(saved["records"]) == 3

    monkeypatch.setattr(sys, "argv", [*arguments, "--limit", "4", "--report", str(tmp_path / "too_many.json")])
    with pytest.raises(ValueError, match="exceeds the 3 compatible local cases"):
        comparison.main()
    assert not (tmp_path / "too_many.partial.json").exists()


def test_reference_checkpoint_rejects_changed_identity_and_invalid_cost(tmp_path):
    from scripts.compare_general_reference import load_reference_checkpoint

    path = tmp_path / "reference.json"
    saved = {
        "schema_version": 1,
        "signature": "frozen",
        "records": [{"answer": {"noul": 0.1}, "model": "test", "cost_usd": 0.001}],
    }
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(FileExistsError, match="--resume"):
        load_reference_checkpoint(path, "frozen", resume=False)
    with pytest.raises(ValueError, match="does not match"):
        load_reference_checkpoint(path, "changed", resume=True)
    assert len(load_reference_checkpoint(path, "frozen", resume=True)) == 1
    for cost in (-1, True, float("nan")):
        saved["records"][0]["cost_usd"] = cost
        path.write_text(json.dumps(saved), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match"):
            load_reference_checkpoint(path, "frozen", resume=True)


def test_comparison_verifies_artifact_before_credentials_or_billed_calls(tmp_path, monkeypatch):
    from scripts import compare_general_reference as comparison

    corpus = tmp_path / "cases.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "state": {},
                    "kind": "noul",
                    "question": "Valid?",
                    "options": ["no", "yes"],
                    "target": [1, 0],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(comparison, "_key", lambda _path: pytest.fail("Credentials read before artifact verification"))
    monkeypatch.setattr(comparison, "query_jev", lambda *_args, **_kwargs: pytest.fail("Paid call before verification"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_general_reference",
            "--acknowledge-provider-terms",
            "--limit",
            "1",
            "--cases-json",
            str(corpus),
            "--output-dir",
            str(tmp_path / "missing-model"),
            "--model-report",
            str(tmp_path / "missing-model.json"),
            "--report",
            str(tmp_path / "comparison.json"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        comparison.main()


def test_explicit_evaluation_posts_only_the_public_decision() -> None:
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {"decision": {"type": "noul", "noul": 0.7}},
                "usage": {"cost": 0.001},
                "model": "test-reference",
            },
        )

    row = {
        "kind": "noul",
        "state_json": '{"evidence": "public"}',
        "question": "Is evidence available?",
        "options": ["no", "yes"],
        "target": [0.1, 0.9],
    }
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        answer, cost, model = query_jev(client, "test-key", row, acknowledge_provider_terms=True)
    assert answer["noul"] == 0.7
    assert cost == 0.001 and model == "test-reference"
    assert len(requests) == 1
    assert b"target" not in requests[0].content
    assert b"test-key" not in requests[0].content
    assert requests[0].headers["Authorization"] == "Bearer test-key"


def test_shared_reference_cache_is_reused_across_local_models_without_rebilling(tmp_path, monkeypatch):
    from scripts import compare_general_reference as comparison

    corpus = tmp_path / "cases.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "state": {},
                    "kind": "noul",
                    "question": "Valid?",
                    "options": ["no", "yes"],
                    "target": [1, 0],
                }
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "reference.json"
    calls = []

    def query(*_args, **_kwargs):
        calls.append("billed")
        return {"noul": 0.2}, 0.001, "test-reference"

    monkeypatch.setattr(comparison, "_key", lambda _path: "test-key")
    monkeypatch.setattr(comparison, "query_jev", query)
    monkeypatch.setattr(comparison, "score_requests", lambda *_args: [{"answer": {"yes_probability": 0.1}}])
    for name, adapter in (("small", "a"), ("large", "c")):
        monkeypatch.setattr(
            comparison,
            "verify_artifact",
            lambda *_args, adapter=adapter: ({"adapter_sha256": adapter * 64, "head_sha256": "b" * 64}, None),
        )
        report = tmp_path / f"{name}.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "comparison",
                "--acknowledge-provider-terms",
                "--cases-json",
                str(corpus),
                "--reference-cache",
                str(cache),
                "--report",
                str(report),
            ],
        )
        comparison.main()
        saved = json.loads(report.read_text(encoding="utf-8"))
        assert saved["reference_cache_shared"] is True
        assert saved["local_adapter_sha256"] == adapter * 64
    assert calls == ["billed"]
    assert json.loads(cache.read_text(encoding="utf-8"))["schema_version"] == 2
    with pytest.raises(ValueError, match="does not match"):
        comparison.load_reference_checkpoint(cache, "legacy-adapter-bound", resume=True, schema_version=1)

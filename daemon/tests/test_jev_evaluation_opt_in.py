"""Provider comparison is an explicit, bounded evaluation, not a training unlock."""

import json

import httpx
import pytest

from scripts.compare_general_jev import decision_request, load_local_cases, public_payload, query_jev


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

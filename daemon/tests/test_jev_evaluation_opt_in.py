"""Provider comparison is an explicit, bounded evaluation, not a training unlock."""

import httpx
import pytest

from scripts.compare_general_jev import query_jev


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
        return httpx.Response(200, json={"answers": {"decision": {"type": "noul", "noul": 0.7}},
                                        "usage": {"cost": 0.001}, "model": "test-reference"})

    row = {"kind": "noul", "state_json": '{"evidence": "public"}',
           "question": "Is evidence available?", "options": ["no", "yes"],
           "target": [0.1, 0.9]}
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        answer, cost, model = query_jev(client, "test-key", row, acknowledge_provider_terms=True)
    assert answer["noul"] == 0.7
    assert cost == 0.001 and model == "test-reference"
    assert len(requests) == 1
    assert b"target" not in requests[0].content
    assert b"test-key" not in requests[0].content
    assert requests[0].headers["Authorization"] == "Bearer test-key"

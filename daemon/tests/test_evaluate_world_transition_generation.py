"""Exact-outcome checks do not award JSON syntax alone."""

from scripts.evaluate_world_transition_generation import normalize_outcome


def test_normalize_outcome_compares_values_not_only_wrapper():
    assert normalize_outcome('<tool_response>{"result":42}</tool_response>') == {"result": 42}
    assert normalize_outcome('<tool_response>{"result":278}</tool_response>') != {"result": 42}
    assert normalize_outcome('{"result":42}') is None

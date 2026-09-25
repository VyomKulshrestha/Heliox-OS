"""When2Call preparation retains only valid, group-disjoint tool choices."""

import json

from scripts.prepare_world_when2call import _one_tool_call, build_cases


def _row(index: int, positive: bool) -> dict:
    call = '<TOOLCALL>[{"name":"lookup","arguments":{"key":"value"}}]</TOOLCALL>'
    text = "Please specify the missing key."
    return {
        "tools": [json.dumps({"name": "lookup", "description": "Find a record",
                              "parameters": {"required": ["key"]}})],
        "messages": [{"role": "user", "content": f"Find record {index}"}],
        "chosen_response": {"content": call if positive else text},
        "rejected_response": {"content": text if positive else call},
    }


def test_one_call_parser_rejects_multiple_or_malformed_calls():
    assert _one_tool_call('<TOOLCALL>[{"name":"lookup","arguments":{}}]</TOOLCALL>') == {
        "name": "lookup", "arguments": {}}
    assert _one_tool_call('<TOOLCALL>[]</TOOLCALL>') is None
    assert _one_tool_call('<TOOLCALL>not json</TOOLCALL>') is None


def test_grouped_balanced_splits_exclude_conflicts_and_unavailable_tools():
    rows = [_row(index, index % 2 == 0) for index in range(200)]
    rows.append(_row(0, False))
    unavailable = _row(201, False)
    unavailable["tools"] = [json.dumps({"name": "different"})]
    rows.append(unavailable)
    partitions, report = build_cases(rows, {"train": 1, "calibration": 1, "test": 1})
    assert {name: sorted(case.in_scope for case in cases) for name, cases in partitions.items()} == {
        "train": [False, True], "calibration": [False, True], "test": [False, True]}
    assert report["skipped"]["contradictory_group"] == 1
    assert report["skipped"]["unavailable_tool"] == 1
    assert all("Find record 0" not in case.request for cases in partitions.values() for case in cases)

"""Future messages and tool output cannot leak into pre-call evaluation."""

import hashlib
import json

import pytest

from scripts.evaluate_world_conversation_scope import load_pre_call_cases


def test_pre_call_loader_drops_future_and_tool_output(tmp_path):
    row = {"id": "case-1", "label": {"relevant": True},
           "request": {"tool": {"name": "read_issue", "description": "Read issue"},
                       "conversation": {"messages": [
                           {"role": "user", "content": "Read issue 5"},
                           {"role": "assistant", "content": "I can check that"},
                           {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": "read_issue"}}]},
                           {"role": "tool", "content": "SECRET FUTURE OUTPUT"},
                           {"role": "user", "content": "FUTURE USER CHANGE"},
                       ]}}}
    path = tmp_path / "agentic.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    cases = load_pre_call_cases(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(cases) == 1
    assert "Read issue 5" in cases[0].request
    assert "SECRET FUTURE OUTPUT" not in cases[0].request
    assert "FUTURE USER CHANGE" not in cases[0].request


def test_pre_call_loader_rejects_wrong_source_or_tool(tmp_path):
    path = tmp_path / "agentic.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pre_call_cases(path, "0" * 64)
    row = {"id": "case-1", "label": {"relevant": False},
           "request": {"tool": {"name": "read_issue", "description": "Read issue"},
                       "conversation": {"messages": [
                           {"role": "user", "content": "Read issue 5"},
                           {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": "delete_issue"}}]},
                       ]}}}
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-call boundary"):
        load_pre_call_cases(path, hashlib.sha256(path.read_bytes()).hexdigest())
